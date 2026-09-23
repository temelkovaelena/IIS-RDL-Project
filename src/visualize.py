"""Schema graph plus a sampled subgraph around a single prediction.

Writes output/schema.gexf, output/subgraph.gexf (for Gephi) and output/subgraph.html (pyvis).

    python src/visualize.py --task driver-top3 --split val --seed-row 0 --depth 2

The subgraph uses the same time rule as the features: a neighbour is included only if it is
not newer than the seed node. That is the point of the picture, so it must not contain a
single node dated after the prediction. This is the computation graph for one prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import relbench
from pyvis.network import Network

from graph_builder import EdgeType, HeteroGraph, build_graph, to_unix_time

NODE_COLOURS = {
    "drivers": "#e15759",
    "results": "#4e79a7",
    "qualifying": "#76b7b2",
    "standings": "#59a14f",
    "races": "#f28e2b",
    "constructors": "#b07aa1",
    "constructor_results": "#9c755f",
    "constructor_standings": "#bab0ac",
    "circuits": "#edc948",
}


def schema_graph(db, schema: dict) -> nx.DiGraph:
    """One node per table and one edge per foreign key."""
    g = nx.DiGraph()
    for name, table in db.table_dict.items():
        info = schema["tables"][name]
        g.add_node(
            name,
            label=name,
            n_rows=info["n_rows"],
            time_col=table.time_col or "",
            is_static=table.time_col is None,
            hops_from_drivers=info["hops_from"].get("drivers", -1),
            colour=NODE_COLOURS.get(name, "#cccccc"),
        )
    for name, table in db.table_dict.items():
        for fkey_col, parent in table.fkey_col_to_pkey_table.items():
            g.add_edge(name, parent, label=fkey_col, fkey=fkey_col, n_edges=len(table.df))
    return g


class NeighbourIndex:
    """For one edge type: each source's neighbours, sorted by time.

    Same preparation as graph_builder.visible_neighbour_counts, sort once then searchsorted
    per source. Here the neighbours themselves are needed, not just how many there are.
    """

    def __init__(self, graph: HeteroGraph, edge_type: EdgeType):
        src_type, _, dst_type = edge_type
        src, dst = graph.edges[edge_type]
        dst_time = graph.node_time[dst_type]
        order = np.lexsort((dst if dst_time is None else dst_time[dst], src))
        self.src_sorted = src[order]
        self.dst_sorted = dst[order]
        self.time_sorted = None if dst_time is None else dst_time[dst][order]
        n_src = graph.n_nodes[src_type]
        self.lo = np.searchsorted(self.src_sorted, np.arange(n_src), side="left")
        self.hi = np.searchsorted(self.src_sorted, np.arange(n_src), side="right")

    def before(self, src_idx: int, t: int) -> np.ndarray:
        """Neighbours of src_idx visible at time t. Static ones are always visible."""
        lo, hi = self.lo[src_idx], self.hi[src_idx]
        if self.time_sorted is None:
            return self.dst_sorted[lo:hi]
        k = int(np.searchsorted(self.time_sorted[lo:hi], t, side="right"))
        assert k == 0 or self.time_sorted[lo + k - 1] <= t, "neighbour after the seed"
        return self.dst_sorted[lo : lo + k]


def sample_subgraph(
    graph: HeteroGraph,
    seed_type: str,
    seed_idx: int,
    seed_time: int,
    num_neighbours: list[int],
    rng: np.random.Generator,
) -> tuple[set[tuple[str, int]], list[tuple]]:
    """Time-correct neighbour sampling, hop by hop.

    num_neighbours[i] caps how many neighbours per edge type are taken at hop i, the same
    shape as num_neighbors in PyG's NeighborLoader, so the picture reflects what the model
    actually sees.
    """
    index: dict[EdgeType, NeighbourIndex] = {}
    nodes: set[tuple[str, int]] = {(seed_type, seed_idx)}
    edges: list[tuple] = []
    frontier = [(seed_type, seed_idx)]

    for hop, budget in enumerate(num_neighbours):
        next_frontier: list[tuple[str, int]] = []
        for node_type, node_idx in frontier:
            for edge_type in graph.edges:
                src_type, rel, dst_type = edge_type
                if src_type != node_type:
                    continue
                if edge_type not in index:
                    index[edge_type] = NeighbourIndex(graph, edge_type)
                candidates = index[edge_type].before(node_idx, seed_time)
                if len(candidates) == 0:
                    continue
                if len(candidates) > budget:
                    candidates = rng.choice(candidates, size=budget, replace=False)
                for dst_idx in candidates:
                    dst_node = (dst_type, int(dst_idx))
                    edges.append((node_type, node_idx, rel, dst_type, int(dst_idx), hop + 1))
                    if dst_node not in nodes:
                        nodes.add(dst_node)
                        next_frontier.append(dst_node)
        frontier = next_frontier
    return nodes, edges


def subgraph_to_networkx(
    db, graph: HeteroGraph, nodes: set, edges: list, seed: tuple[str, int], seed_time: int
) -> nx.DiGraph:
    g = nx.DiGraph()
    for node_type, node_idx in sorted(nodes):
        time = graph.node_time[node_type]
        node_time = None if time is None else int(time[node_idx])
        g.add_node(
            f"{node_type}:{node_idx}",
            node_type=node_type,
            row=int(node_idx),
            is_seed=(node_type, node_idx) == seed,
            time=("" if node_time is None else str(pd.Timestamp(node_time, unit="s"))),
            seconds_before_seed=(0 if node_time is None else int(seed_time - node_time)),
            colour=NODE_COLOURS.get(node_type, "#cccccc"),
        )
    for src_type, src_idx, rel, dst_type, dst_idx, hop in edges:
        g.add_edge(f"{src_type}:{src_idx}", f"{dst_type}:{dst_idx}", label=rel, hop=hop)
    return g


def write_pyvis(g: nx.DiGraph, out: Path, title: str) -> None:
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut(spring_length=140)
    for name, attrs in g.nodes(data=True):
        net.add_node(
            name,
            label=name if attrs["is_seed"] else attrs["node_type"],
            color="#000000" if attrs["is_seed"] else attrs["colour"],
            size=34 if attrs["is_seed"] else 12,
            title=f"{name}\ntime: {attrs['time'] or 'static node'}",
        )
    for src, dst, attrs in g.edges(data=True):
        net.add_edge(src, dst, title=attrs["label"], label="", width=1)
    out.write_text(net.generate_html(notebook=False).replace("<body>", f"<body><h3>{title}</h3>"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed-row", type=int, default=0, help="Which row of the label table.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=8, help="Max neighbours per edge type.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    db = dataset.get_db()
    graph = build_graph(db)
    args.outdir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(args.schema.read_text())
    sg = schema_graph(db, schema)
    nx.write_gexf(sg, args.outdir / "schema.gexf")
    print(
        f"{args.outdir / 'schema.gexf'}: {sg.number_of_nodes()} tables, "
        f"{sg.number_of_edges()} foreign keys"
    )

    task = dataset.load_task(args.task)
    label = task.get_table(args.split, mask_input_cols=False).df
    row = label.iloc[args.seed_row]
    seed_idx = int(row[task.entity_col])
    seed_time = int(to_unix_time(pd.Series([row[task.time_col]])).item())

    rng = np.random.default_rng(args.random_seed)
    nodes, edges = sample_subgraph(
        graph, task.entity_table, seed_idx, seed_time, [args.fanout] * args.depth, rng
    )
    g = subgraph_to_networkx(db, graph, nodes, edges, (task.entity_table, seed_idx), seed_time)

    # The check that justifies the picture: nothing in it postdates the seed.
    for name, attrs in g.nodes(data=True):
        assert attrs["seconds_before_seed"] >= 0, f"{name} is after the prediction time"

    nx.write_gexf(g, args.outdir / "subgraph.gexf")
    when = pd.Timestamp(seed_time, unit="s").date()
    target = row.get(task.target_col, "?")
    title = (
        f"{args.task} / {args.split}: {task.entity_table} {seed_idx} on {when} "
        f"({task.target_col} = {target}) — {args.depth} hops, max {args.fanout} per edge type"
    )
    write_pyvis(g, args.outdir / "subgraph.html", title)

    by_type: dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        by_type[attrs["node_type"]] = by_type.get(attrs["node_type"], 0) + 1
    print(f"{args.outdir / 'subgraph.gexf'} and subgraph.html")
    print(f"  seed: {task.entity_table} {seed_idx} on {when}, {task.target_col} = {target}")
    print(f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("  by type: " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))


if __name__ == "__main__":
    main()
