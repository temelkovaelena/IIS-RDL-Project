"""Trivial baselines for the driver-constructor task.

persistence  keep the driver's most recent constructors, most recent first
popularity   the constructors with the most results in the season before the seed

Most drivers stay put, so persistence is strong on the full set of seeds. That is exactly
why MAP is reported per slice: on `movers` persistence is useless by construction, because a
mover is defined as someone whose next season does not include their last constructor.

    python src/baseline_recommend.py --k 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import relbench
from relbench import metrics

EVAL_COLUMNS = [
    "database", "task", "task_type", "arm", "split", "slice",
    "metric", "value", "seed", "train_minutes", "engineering_minutes",
]

SEASON = pd.Timedelta(days=365)


def popularity_ranking(results: pd.DataFrame, t: pd.Timestamp, k: int) -> list[int]:
    """Constructors with the most results in the season before the seed."""
    recent = results[(results["date"] > t - SEASON) & (results["date"] <= t)]
    if recent.empty:
        recent = results[results["date"] <= t]
    return recent["constructorId"].astype(int).value_counts().index[:k].tolist()


def driver_ranking(results: pd.DataFrame, t: pd.Timestamp, driver: int, k: int) -> list[int]:
    """The driver's own constructors before the seed, most recent first."""
    past = results[(results["driverId"] == driver) & (results["date"] <= t)]
    if past.empty:
        return []
    ordered = past.sort_values("date", ascending=False)["constructorId"].astype(int)
    seen: list[int] = []
    for c in ordered:
        if c not in seen:
            seen.append(c)
        if len(seen) == k:
            break
    return seen


def pad(ranking: list[int], fallback: list[int], k: int) -> list[int]:
    out = list(ranking)
    for c in fallback:
        if len(out) == k:
            break
        if c not in out:
            out.append(c)
    return (out + [-1] * k)[:k]


def predict(labels: pd.DataFrame, results: pd.DataFrame, arm: str, k: int) -> np.ndarray:
    """A [n_seeds, k] matrix of ranked constructor ids."""
    preds = []
    for t, group in labels.groupby("date", sort=True):
        popular = popularity_ranking(results, t, k)
        for driver in group["driverId"]:
            if arm == "popularity":
                preds.append(pad(popular, [], k))
            else:
                # A rookie has no history, so persistence has nothing to keep and falls
                # back to popularity. That is the honest behaviour, not a special case.
                preds.append(pad(driver_ranking(results, t, int(driver), k), popular, k))
    return np.array(preds, dtype=np.int64)


def map_at_k(pred: np.ndarray, truth: pd.Series) -> float:
    """MAP@k through relbench.metrics, so it matches how the built-in tasks are scored."""
    pred_isin = np.zeros(pred.shape, dtype=np.int64)
    dst_count = np.zeros(len(pred), dtype=np.int64)
    for i, actual in enumerate(truth):
        target = set(int(c) for c in actual)
        dst_count[i] = len(target)
        for j, c in enumerate(pred[i]):
            pred_isin[i, j] = int(c) in target
    return float(metrics.map(pred_isin, dst_count))


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--labels", type=Path, default=Path("data/driver_constructor"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("output/evaluation.csv"))
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    # Features come from the masked database, the one a model is allowed to see. Only the
    # labels were built from the full one.
    results = dataset.get_db().table_dict["results"].df[["driverId", "constructorId", "date"]]

    rows = []
    for split in ("val", "test"):
        labels = pd.read_pickle(args.labels / f"{split}.pkl").reset_index(drop=True)
        for arm in ("persistence", "popularity"):
            pred = predict(labels, results, arm, args.k)
            for name, mask in (
                ("all", np.ones(len(labels), dtype=bool)),
                ("movers", labels["is_mover"].to_numpy()),
                ("rookies", labels["is_rookie"].to_numpy()),
            ):
                if mask.sum() == 0:
                    continue
                value = map_at_k(pred[mask], labels.loc[mask, "constructorId"])
                rows.append({
                    "database": args.dataset,
                    "task": "driver-constructor",
                    "task_type": "RECOMMENDATION",
                    "arm": arm,
                    "split": split,
                    "slice": name,
                    "metric": f"map@{args.k}",
                    "value": round(value, 6),
                    # Both baselines are deterministic: more seeds would repeat the number.
                    "seed": 0,
                    "train_minutes": 0,
                    "engineering_minutes": 0,
                })
                print(
                    f"  {split:5s} {arm:12s} {name:8s} "
                    f"n={int(mask.sum()):4d}  map@{args.k} {value:.4f}"
                )

    append_rows(args.out, rows)
    print(f"\n{len(rows)} rows appended to {args.out}")


if __name__ == "__main__":
    main()
