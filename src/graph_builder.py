"""Turns the relational database into a heterogeneous temporal graph.

Row -> node, table -> node type, foreign key -> typed edge (both directions), timestamp ->
node time. Writes data/graph_stats.json: nodes and edges per type, average and maximum
degree per foreign key, time ranges.

    python src/graph_builder.py --dataset rel-f1 --out data/graph_stats.json

This is plain numpy, no torch, because Intel macOS has no recent torch wheels. The edge type
names and node indexing are the same as what relbench.modeling.graph.make_pkey_fkey_graph
builds with PyG on the GPU side. If they differed, these statistics would describe a
different graph from the one that gets trained on.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import relbench

EdgeType = tuple[str, str, str]


def to_unix_time(series: pd.Series) -> np.ndarray:
    """UNIX seconds. A copy of relbench.modeling.utils.to_unix_time.

    Copied rather than imported because that module pulls in torch at the top of the file.
    It has to return the same number as the original: node time is what the sampler compares
    against, so any difference here changes what the model is allowed to see.
    """
    ts = pd.to_datetime(series, utc=True)
    return (ts.astype("int64").to_numpy(copy=False) // 1_000_000_000).astype(np.int64)


@dataclass
class HeteroGraph:
    """A heterogeneous temporal graph with no framework dependency."""

    n_nodes: dict[str, int]
    # None means a static node type (no time column); such a neighbour is always visible.
    node_time: dict[str, np.ndarray | None]
    edges: dict[EdgeType, tuple[np.ndarray, np.ndarray]]
    dropped_dangling: dict[str, int]

    @property
    def n_edges_directed(self) -> int:
        return sum(len(src) for src, _ in self.edges.values())


def build_graph(db) -> HeteroGraph:
    """Turns the primary-key / foreign-key structure into nodes and typed edges."""
    n_nodes: dict[str, int] = {}
    node_time: dict[str, np.ndarray | None] = {}

    for name, table in db.table_dict.items():
        df = table.df
        if table.pkey_col is not None:
            # Foreign key values index straight into the parent, here and in PyG alike.
            # That only holds if the primary key runs 0..n-1. Better to fail here than to
            # have edges quietly point at the wrong row.
            assert (df[table.pkey_col].to_numpy() == np.arange(len(df))).all(), (
                f"{name}.{table.pkey_col}: primary key is not 0..n-1"
            )
        n_nodes[name] = len(df)
        node_time[name] = to_unix_time(df[table.time_col]) if table.time_col else None

    edges: dict[EdgeType, tuple[np.ndarray, np.ndarray]] = {}
    dropped: dict[str, int] = {}

    for name, table in db.table_dict.items():
        df = table.df
        for fkey_col, parent in table.fkey_col_to_pkey_table.items():
            values = df[fkey_col]
            keep = ~values.isna()
            dropped[f"{name}.{fkey_col}"] = int((~keep).sum())

            child_idx = np.arange(len(df), dtype=np.int64)[keep.to_numpy()]
            parent_idx = values[keep].astype("int64").to_numpy()
            assert (parent_idx < n_nodes[parent]).all(), (
                f"{name}.{fkey_col}: index out of range for {parent}"
            )

            # Both directions. Without the reverse edge drivers cannot see its own results:
            # the signal runs from children up to the parent, but the foreign key points down.
            edges[(name, f"f2p_{fkey_col}", parent)] = (child_idx, parent_idx)
            edges[(parent, f"rev_f2p_{fkey_col}", name)] = (parent_idx, child_idx)

    return HeteroGraph(n_nodes, node_time, edges, dropped)


def degree_stats(graph: HeteroGraph, edge_type: EdgeType) -> dict:
    """Degree distribution from the source node type.

    For rev_f2p_* this is fan-out, how many children one parent has, which is what drives
    the cost of neighbour sampling during training.
    """
    src_type = edge_type[0]
    src, _ = graph.edges[edge_type]
    counts = np.bincount(src, minlength=graph.n_nodes[src_type])
    return {
        "n_edges": int(len(src)),
        "mean": round(float(counts.mean()), 2),
        "median": float(np.median(counts)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
        "n_isolated": int((counts == 0).sum()),
    }


def edge_time_consistency(graph: HeteroGraph, edge_type: EdgeType) -> dict | None:
    """When both ends carry time, is the child before, at, or after the parent.

    Not an assertion. In rel-f1 results.date is the race date, so nearly everything should
    come out as "same time". A deviation would mean some table's timestamp means something
    other than I think, which changes what counts as a valid neighbour.
    """
    src_type, _, dst_type = edge_type
    src_time, dst_time = graph.node_time[src_type], graph.node_time[dst_type]
    if src_time is None or dst_time is None:
        return None
    src, dst = graph.edges[edge_type]
    delta = src_time[src] - dst_time[dst]
    return {
        "src_before_dst": int((delta < 0).sum()),
        "same_time": int((delta == 0).sum()),
        "src_after_dst": int((delta > 0).sum()),
    }


def visible_neighbour_counts(
    graph: HeteroGraph, edge_type: EdgeType, seed_nodes: np.ndarray, seed_time: np.ndarray
) -> np.ndarray:
    """How many neighbours of `edge_type` are visible at each seed node's time.

    The bound is <= t, the same rule the tabular features use. All three tasks build their
    label from rows with date > t, so a row sitting exactly at t is not part of the answer
    and is legitimate input. Sort each parent's neighbours by time, then searchsorted.
    """
    src_type, _, dst_type = edge_type
    src, dst = graph.edges[edge_type]
    dst_time = graph.node_time[dst_type]
    assert dst_time is not None, f"{dst_type} has no time, nothing to cut on"

    # Edges sorted first by parent, then by the child's time.
    order = np.lexsort((dst_time[dst], src))
    parent_of_edge, time_of_edge = src[order], dst_time[dst][order]

    n_parents = graph.n_nodes[src_type]
    lo = np.searchsorted(parent_of_edge, np.arange(n_parents), side="left")
    hi = np.searchsorted(parent_of_edge, np.arange(n_parents), side="right")

    counts = np.empty(len(seed_nodes), dtype=np.int64)
    for i, (node, t) in enumerate(zip(seed_nodes, seed_time, strict=True)):
        window = time_of_edge[lo[node] : hi[node]]
        k = int(np.searchsorted(window, t, side="right"))
        # The last included neighbour must not be after the seed. If this fails, sampling is
        # letting the future through and every later number is invalid.
        assert k == 0 or window[k - 1] <= t, f"{edge_type}: neighbour after the seed time"
        assert k == len(window) or window[k] > t, f"{edge_type}: dropped a visible neighbour"
        counts[i] = k
    return counts


def seed_visibility(graph: HeteroGraph, task, split: str) -> dict:
    """What the model sees one hop from a seed node at prediction time.

    This is the computation graph the network builds for that entity, one layer deep.
    """
    entity_table = task.entity_table
    table = task.get_table(split)
    seed_nodes = table.df[task.entity_col].astype("int64").to_numpy()
    seed_time = to_unix_time(table.df[task.time_col])

    per_relation = {}
    total = np.zeros(len(seed_nodes), dtype=np.int64)
    for edge_type in graph.edges:
        src_type, rel, dst_type = edge_type
        if src_type != entity_table or not rel.startswith("rev_f2p_"):
            continue
        if graph.node_time[dst_type] is None:
            continue  # static neighbour: always visible, no time window to apply
        counts = visible_neighbour_counts(graph, edge_type, seed_nodes, seed_time)
        total += counts
        per_relation[f"{src_type} -{rel}-> {dst_type}"] = {
            "mean": round(float(counts.mean()), 1),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "frac_zero": round(float((counts == 0).mean()), 3),
        }

    return {
        "n_seeds": int(len(seed_nodes)),
        "per_relation": per_relation,
        "total_1hop": {
            "mean": round(float(total.mean()), 1),
            "median": float(np.median(total)),
            "max": int(total.max()),
            "frac_zero": round(float((total == 0).mean()), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--out", type=Path, default=Path("data/graph_stats.json"))
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["driver-position", "driver-top3", "driver-dnf"],
        help="Entity tasks whose seed nodes get profiled.",
    )
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument(
        "--full-db",
        action="store_true",
        help="Do not mask after test_timestamp. For inspection only, not for training.",
    )
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    # RelBench masks the database after test_timestamp by default. A driver who debuts after
    # 2010 therefore has no history at all. That is the benchmark's setup, not a bug, and it
    # explains most of the seeds with no visible neighbours.
    db = dataset.get_db(upto_test_timestamp=not args.full_db)
    graph = build_graph(db)

    node_types = {}
    for name in sorted(graph.n_nodes):
        time = graph.node_time[name]
        node_types[name] = {
            "n_nodes": graph.n_nodes[name],
            "has_time": time is not None,
            "time_range": (
                None
                if time is None
                else {
                    "min": str(pd.Timestamp(int(time.min()), unit="s")),
                    "max": str(pd.Timestamp(int(time.max()), unit="s")),
                }
            ),
        }

    edge_types = {}
    for edge_type in sorted(graph.edges):
        src, rel, dst = edge_type
        entry = {"src": src, "rel": rel, "dst": dst, **degree_stats(graph, edge_type)}
        consistency = edge_time_consistency(graph, edge_type)
        if consistency is not None:
            entry["time_consistency"] = consistency
        edge_types[f"{src} -{rel}-> {dst}"] = entry

    visibility = {}
    for task_name in args.tasks:
        task = dataset.load_task(task_name)
        visibility[task_name] = {s: seed_visibility(graph, task, s) for s in args.splits}

    static_types = sorted(n for n, t in graph.node_time.items() if t is None)
    stats = {
        "database": args.dataset,
        "relbench_version": relbench.__version__,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "db_upto_test_timestamp": not args.full_db,
        "test_timestamp": str(dataset.test_timestamp),
        "n_node_types": len(node_types),
        "n_nodes": sum(graph.n_nodes.values()),
        "n_edge_types": len(edge_types),
        "n_edges_directed": graph.n_edges_directed,
        "n_edges_undirected": graph.n_edges_directed // 2,
        "static_node_types": static_types,
        "dangling_fkeys_dropped": graph.dropped_dangling,
        "node_types": node_types,
        "edge_types": edge_types,
        "seed_visibility": visibility,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(f"{args.out}")
    print(
        f"  {stats['n_node_types']} node types, {stats['n_nodes']} nodes; "
        f"{stats['n_edge_types']} edge types, {stats['n_edges_undirected']} edges "
        f"({stats['n_edges_directed']} directed)"
    )
    print(f"  static types (always visible): {', '.join(static_types)}")
    if not args.full_db:
        print(f"  database masked after {dataset.test_timestamp}")
    for task_name, splits in visibility.items():
        for split, info in splits.items():
            t = info["total_1hop"]
            print(
                f"  {task_name}/{split}: {info['n_seeds']} seeds, visible 1-hop neighbours "
                f"— mean {t['mean']}, median {t['median']}, max {t['max']}, "
                f"none at all {t['frac_zero']:.1%}"
            )


if __name__ == "__main__":
    main()
