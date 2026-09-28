"""Checks that nothing from the future reaches a prediction.

Instead of inventing rows, this truncates the real database at a cutoff and compares. If a
feature is computed only from the past, then deleting everything after the cutoff must leave
it unchanged. Real data, no synthetic rows.

    python src/check_leakage.py --task driver-top3

Three checks:
  1. the splits are ordered in time and do not overlap
  2. features for seeds before a cutoff survive truncating the database at that cutoff
  3. the same, per seed, truncating at each seed's own timestamp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import relbench

from baseline_tabular import engineered_features
from graph_builder import build_graph, to_unix_time, visible_neighbour_counts


def report_splits(dataset, task) -> list[str]:
    """Train, val and test must be ordered in time and must not overlap."""
    lines = [
        f"val_timestamp:  {dataset.val_timestamp}",
        f"test_timestamp: {dataset.test_timestamp}",
        "",
    ]
    bounds = {}
    for split in ("train", "val", "test"):
        df = task.get_table(split, mask_input_cols=False).df
        lo, hi = df[task.time_col].min(), df[task.time_col].max()
        bounds[split] = (lo, hi)
        lines.append(f"  {split:5s} {len(df):6d} rows   {lo.date()} .. {hi.date()}")

    assert bounds["train"][1] < dataset.val_timestamp, "train reaches into the val period"
    assert bounds["val"][1] < dataset.test_timestamp, "val reaches into the test period"
    assert bounds["val"][0] >= dataset.val_timestamp, "val starts before its period"
    assert bounds["test"][0] >= dataset.test_timestamp, "test starts before its period"
    lines.append("")
    lines.append("  splits are ordered in time and do not overlap")
    return lines


def truncation_test(dataset, task, schema, split: str) -> list[str]:
    """Delete everything after the last seed, and check the features do not move."""
    db = dataset.get_db()
    label = task.get_table(split, mask_input_cols=False).df
    cutoff = pd.Timestamp(label[task.time_col].max())

    full = engineered_features(db, task, schema, label)
    truncated = engineered_features(db.upto(cutoff), task, schema, label)

    removed = sum(
        len(t.df) - len(db.upto(cutoff).table_dict[name].df)
        for name, t in db.table_dict.items()
    )
    same = full.equals(truncated)
    return [
        f"  cutoff {cutoff.date()} removes {removed} rows from the database",
        f"  {len(label)} seeds x {full.shape[1]} features recomputed",
        f"  identical after truncation: {same}",
    ]


def per_seed_test(dataset, task, schema, split: str, sample: int, rng) -> list[str]:
    """The strongest form: truncate at each seed's own time, one seed at a time."""
    db = dataset.get_db()
    label = task.get_table(split, mask_input_cols=False).df
    idx = rng.choice(len(label), size=min(sample, len(label)), replace=False)

    mismatches = 0
    for i in sorted(idx):
        row = label.iloc[[i]]
        seed_time = pd.Timestamp(row[task.time_col].iloc[0])
        full = engineered_features(db, task, schema, row)
        truncated = engineered_features(db.upto(seed_time), task, schema, row)
        if not full.equals(truncated):
            mismatches += 1
    return [
        f"  {len(idx)} seeds, each with the database cut at its own timestamp",
        f"  features that changed: {mismatches}",
    ]


def graph_test(dataset, task, split: str, sample: int) -> list[str]:
    """Graph side, checked against a plain pandas count rather than against itself.

    Truncating the database is not an option here: db.upto() drops rows from parent tables,
    which breaks the positional primary keys the graph is built on. So instead the visible
    neighbour count is recomputed naively from the dataframe and the two are compared.
    """
    db = dataset.get_db()
    label = task.get_table(split, mask_input_cols=False).df
    nodes = label[task.entity_col].astype("int64").to_numpy()
    times = to_unix_time(label[task.time_col])
    stamps = pd.to_datetime(label[task.time_col])
    edge_type = (task.entity_table, "rev_f2p_driverId", "results")

    counts = visible_neighbour_counts(build_graph(db), edge_type, nodes, times)

    results = db.table_dict["results"].df[["driverId", "date"]]
    oracle = np.array(
        [
            int(((results["driverId"] == d) & (results["date"] <= t)).sum())
            for d, t in zip(nodes[:sample], stamps[:sample], strict=False)
        ]
    )
    return [
        f"  {edge_type[0]} -> {edge_type[2]}, {len(nodes)} seeds",
        f"  graph matches a plain pandas count on {len(oracle)} seeds: "
        f"{np.array_equal(counts[: len(oracle)], oracle)}",
        f"  mean visible neighbours: {counts.mean():.1f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--split", default="val", help="Split whose seeds are checked.")
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    parser.add_argument("--out", type=Path, default=Path("output/leakage_check.txt"))
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    schema = json.loads(args.schema.read_text())
    rng = np.random.default_rng(args.random_seed)

    lines = [f"leakage check: {args.dataset} / {args.task} / {args.split}", ""]
    lines += ["1. temporal splits", ""] + report_splits(dataset, task)
    lines += ["", "2. truncate the database at the last seed", ""]
    lines += truncation_test(dataset, task, schema, args.split)
    lines += ["", "3. truncate at each seed's own timestamp", ""]
    lines += per_seed_test(dataset, task, schema, args.split, args.sample, rng)
    lines += ["", "4. graph side: visible neighbours against a plain pandas count", ""]
    lines += graph_test(dataset, task, args.split, args.sample * 4)

    text = "\n".join(lines) + "\n"
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
