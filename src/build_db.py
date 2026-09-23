"""Reads rel-f1 and writes down its schema.

Produces data/schema.json: per table the row count, primary key, foreign keys, time column
and the type of every column. Nothing else should run before this is correct.

    python src/build_db.py --dataset rel-f1 --out data/schema.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import relbench

# Threshold for object columns: below it a category, above it free text. It is a guess,
# not a fact, so it is named in one place and easy to change once the encoders are in.
CATEGORICAL_MAX_UNIQUE = 100


def column_kind(series: pd.Series, name: str, pkey_col: str | None, fkey_cols: set[str]) -> str:
    """Semantic type of a column, which is what a column encoder needs, not the dtype."""
    if name == pkey_col:
        return "pkey"
    if name in fkey_cols:
        return "fkey"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical" if series.nunique(dropna=True) <= CATEGORICAL_MAX_UNIQUE else "text"


def describe_column(
    series: pd.Series, name: str, pkey_col: str | None, fkey_cols: set[str]
) -> dict:
    kind = column_kind(series, name, pkey_col, fkey_cols)
    info: dict = {
        "dtype": str(series.dtype),
        "kind": kind,
        "n_null": int(series.isna().sum()),
        "n_unique": int(series.nunique(dropna=True)),
    }
    if kind == "timestamp":
        info["min"] = str(series.min())
        info["max"] = str(series.max())
    elif kind == "numeric":
        info["min"] = float(series.min()) if series.notna().any() else None
        info["max"] = float(series.max()) if series.notna().any() else None
    return info


def fk_adjacency(db) -> dict[str, set[str]]:
    """Table adjacency over foreign keys, undirected.

    Messages travel both ways (child -> parent and parent -> child), so hop distance has to
    be measured undirected. Otherwise circuits looks unreachable from drivers, and half the
    signal runs through it.
    """
    adj: dict[str, set[str]] = {name: set() for name in db.table_dict}
    for name, table in db.table_dict.items():
        for parent in table.fkey_col_to_pkey_table.values():
            adj[name].add(parent)
            adj[parent].add(name)
    return adj


def hops_from(adj: dict[str, set[str]], source: str) -> dict[str, int]:
    """Breadth-first search: how many foreign-key hops each table is from `source`."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adj[node]):
            if neighbour not in dist:
                dist[neighbour] = dist[node] + 1
                queue.append(neighbour)
    return dist


def check_table_keys(name: str, table, table_pkeys: dict[str, set]) -> dict:
    """Key checks. If the schema is wrong, nothing later is worth running."""
    df = table.df
    if table.pkey_col is not None:
        pkey = df[table.pkey_col]
        assert pkey.notna().all(), f"{name}.{table.pkey_col}: primary key has empty values"
        assert pkey.is_unique, f"{name}.{table.pkey_col}: primary key is not unique"
    dangling = {}
    for fkey_col, parent in table.fkey_col_to_pkey_table.items():
        parent_pkey = table_pkeys[parent]
        present = df[fkey_col].dropna()
        missing = int((~present.isin(parent_pkey)).sum())
        assert missing == 0, f"{name}.{fkey_col}: {missing} values missing from {parent}"
        dangling[fkey_col] = missing
    if table.time_col is not None:
        time = df[table.time_col]
        assert time.notna().all(), f"{name}.{table.time_col}: empty timestamps"
    return dangling


def check_split_boundaries(task, val_timestamp, test_timestamp) -> dict:
    """Checks the label tables against the split boundaries.

    If train holds a row at or after val_timestamp, the model trains on the future and every
    later number is invalid. RelBench builds these tables, so this is not checking our own
    code, but it is the first place the assumption gets tested instead of trusted.
    """
    bounds = {}
    for split, upper in (("train", val_timestamp), ("val", test_timestamp), ("test", None)):
        table = task.get_table(split)
        time = table.df[table.time_col]
        assert time.notna().all(), f"{task.name}/{split}: empty timestamps"
        if upper is not None:
            late = int((time >= upper).sum())
            assert late == 0, f"{task.name}/{split}: {late} rows at or after {upper}"
        bounds[split] = {
            "n_rows": int(len(table.df)),
            "min": str(time.min()),
            "max": str(time.max()),
            "columns": list(table.df.columns),
            # The test split hides the target column, so its column list is shorter.
            "has_target": getattr(task, "target_col", None) in table.df.columns,
        }
    lower = {"train": None, "val": val_timestamp, "test": test_timestamp}
    for split, floor in lower.items():
        if floor is not None:
            early = pd.Timestamp(bounds[split]["min"])
            assert early >= floor, f"{task.name}/{split}: starts {early}, before {floor}"
    return bounds


def describe_task(task) -> dict:
    """Only the fields that affect the seed nodes; the rest is in the RelBench manifest."""
    return {
        "task_type": str(task.task_type).removeprefix("TaskType."),
        "kind": task.kind,
        "entity_table": getattr(task, "entity_table", None),
        "entity_col": getattr(task, "entity_col", None),
        "src_entity_table": getattr(task, "src_entity_table", None),
        "dst_entity_table": getattr(task, "dst_entity_table", None),
        "target_col": getattr(task, "target_col", None),
        "time_col": getattr(task, "time_col", None),
        "timedelta": str(getattr(task, "timedelta", "")) or None,
        "eval_k": getattr(task, "eval_k", None),
        "metrics": [m.__name__ for m in task.metrics],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--out", type=Path, default=Path("data/schema.json"))
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["driver-position", "driver-top3", "driver-dnf", "driver-circuit-compete"],
        help="Tasks whose label tables get checked and whose entity tables root the hops.",
    )
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    # upto_test_timestamp=True is the RelBench default: everything after test_timestamp is
    # hidden. That is why results.date stops in 2009 rather than 2023.
    db = dataset.get_db()

    table_pkeys = {
        name: set(table.df[table.pkey_col]) if table.pkey_col else set()
        for name, table in db.table_dict.items()
    }

    adj = fk_adjacency(db)
    tasks = {name: dataset.load_task(name) for name in args.tasks}

    # Hops are measured from the tasks' entity tables, since those hold the seed nodes.
    roots = sorted(
        {
            t
            for task in tasks.values()
            for t in (
                getattr(task, "entity_table", None),
                getattr(task, "src_entity_table", None),
                getattr(task, "dst_entity_table", None),
            )
            if t is not None
        }
    )
    hops = {root: hops_from(adj, root) for root in roots}

    tables = {}
    for name, table in sorted(db.table_dict.items()):
        fkey_cols = set(table.fkey_col_to_pkey_table)
        dangling = check_table_keys(name, table, table_pkeys)
        tables[name] = {
            "n_rows": int(len(table.df)),
            "pkey_col": table.pkey_col,
            "time_col": table.time_col,
            "time_range": (
                {"min": str(table.min_timestamp), "max": str(table.max_timestamp)}
                if table.time_col
                else None
            ),
            "fkeys": dict(table.fkey_col_to_pkey_table),
            "referenced_by": sorted(
                f"{child}.{col}"
                for child, child_table in db.table_dict.items()
                for col, parent in child_table.fkey_col_to_pkey_table.items()
                if parent == name
            ),
            "dangling_fkeys": dangling,
            "hops_from": {root: hops[root].get(name) for root in roots},
            "columns": {
                col: describe_column(table.df[col], col, table.pkey_col, fkey_cols)
                for col in table.df.columns
            },
        }

    task_info = {}
    for name, task in tasks.items():
        info = describe_task(task)
        info["splits"] = check_split_boundaries(task, dataset.val_timestamp, dataset.test_timestamp)
        task_info[name] = info

    schema = {
        "database": args.dataset,
        "relbench_version": relbench.__version__,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "db_upto_test_timestamp": True,
        "val_timestamp": str(dataset.val_timestamp),
        "test_timestamp": str(dataset.test_timestamp),
        "n_tables": len(tables),
        "n_rows_total": sum(t["n_rows"] for t in tables.values()),
        "available_tasks": dataset.get_task_names(),
        "hop_roots": roots,
        "tables": tables,
        "tasks": task_info,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")

    print(f"{args.out}: {schema['n_tables']} tables, {schema['n_rows_total']} rows")
    for root in roots:
        by_hop: dict[int, list[str]] = {}
        for name, dist in sorted(hops[root].items()):
            by_hop.setdefault(dist, []).append(name)
        summary = "; ".join(f"{d} hops: {', '.join(t)}" for d, t in sorted(by_hop.items()))
        print(f"  from {root} -> {summary}")


if __name__ == "__main__":
    main()
