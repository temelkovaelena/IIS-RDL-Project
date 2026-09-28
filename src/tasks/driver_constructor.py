"""Custom task: recommend a constructor for a driver for the next season.

RelBench ships three entity tasks for rel-f1; this is a fourth, of the recommendation kind,
scored with MAP@k. It follows the shape of driver-circuit-compete: one row per
(date, driver), with the target a list of destination ids.

    python src/tasks/driver_constructor.py --out data/driver_constructor

Seed:  a driver, at a season boundary.
Label: the set of constructorId the driver has results with in the following season.
Split: by season, reusing val_timestamp / test_timestamp so the numbers stay comparable.

Labels are built from the FULL database, because test seeds sit after test_timestamp and the
masked database stops there. The model never sees the full database; it only ever gets the
masked one. Keeping these apart is the whole reason the split is honest.

Most drivers stay with the same constructor, so a "predict last season's team" baseline is
strong. MAP is therefore reported three times:
  all      every seed
  movers   drivers whose next season does not include their most recent constructor
  rookies  drivers with no results at all before the seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import relbench

SEASON = pd.Timedelta(days=365)


def season_boundaries(results: pd.DataFrame) -> list[pd.Timestamp]:
    """One seed timestamp per season: the first of January."""
    years = sorted(results["date"].dt.year.unique())
    return [pd.Timestamp(year=int(y), month=1, day=1) for y in years]


def build_labels(results: pd.DataFrame, timedelta: pd.Timedelta = SEASON) -> pd.DataFrame:
    """One row per (season boundary, driver) with the constructors of the next season.

    The window is (t, t + timedelta], strictly after the seed, which is the same convention
    the RelBench tasks use. A row dated exactly at t belongs to the past, not the answer.
    """
    rows = []
    for t in season_boundaries(results):
        upcoming = results[(results["date"] > t) & (results["date"] <= t + timedelta)]
        if upcoming.empty:
            continue
        grouped = upcoming.groupby("driverId")["constructorId"].apply(
            lambda s: np.array(sorted(set(s.astype(int))), dtype=np.int64)
        )
        for driver, constructors in grouped.items():
            rows.append({"date": t, "driverId": int(driver), "constructorId": constructors})
    return pd.DataFrame(rows).sort_values(["date", "driverId"]).reset_index(drop=True)


def add_slices(labels: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Mark which seeds are movers and which are rookies.

    Both are reporting slices, computed at evaluation time. Neither is ever given to a model
    as a feature: `is_mover` is derived from the label itself, so using it as input would be
    handing over the answer.
    """
    history = results[["driverId", "constructorId", "date"]].sort_values("date")
    is_mover, is_rookie = [], []

    # build_labels returns rows sorted by (date, driverId), and groupby("date") walks the
    # dates in the same order, so the lists below line up with the frame row for row.
    labels = labels.sort_values(["date", "driverId"]).reset_index(drop=True)

    for t, group in labels.groupby("date", sort=True):
        past = history[history["date"] <= t]
        last_constructor = past.groupby("driverId")["constructorId"].last()
        seen = set(past["driverId"].astype(int))
        for driver, constructors in zip(group["driverId"], group["constructorId"], strict=True):
            rookie = driver not in seen
            if rookie:
                mover = False
            else:
                mover = int(last_constructor.loc[driver]) not in set(constructors.tolist())
            is_rookie.append(rookie)
            is_mover.append(mover)

    out = labels.copy()
    out["is_mover"] = pd.Series(is_mover, dtype=bool)
    out["is_rookie"] = pd.Series(is_rookie, dtype=bool)
    return out


def split_labels(labels, val_timestamp, test_timestamp, max_test_years: int = 4) -> dict:
    """Split by season, with the test window capped.

    The model is given the database masked at test_timestamp, so its information ends in
    2009. A seed in 2023 would be asking for a prediction thirteen years ahead on stale data,
    which measures something other than this task. The cap keeps the test window close to the
    span the RelBench tasks use for rel-f1.
    """
    test_end = test_timestamp + pd.DateOffset(years=max_test_years)
    return {
        "train": labels[labels["date"] < val_timestamp],
        "val": labels[(labels["date"] >= val_timestamp) & (labels["date"] < test_timestamp)],
        "test": labels[(labels["date"] >= test_timestamp) & (labels["date"] < test_end)],
    }


def check_no_future(labels: pd.DataFrame, results: pd.DataFrame, timedelta=SEASON) -> None:
    """Every labelled constructor must come from the window after the seed, and only there."""
    by_driver = results.groupby("driverId")
    for _, row in labels.sample(min(200, len(labels)), random_state=0).iterrows():
        t = row["date"]
        rows = by_driver.get_group(row["driverId"])
        window = rows[(rows["date"] > t) & (rows["date"] <= t + timedelta)]
        expected = set(window["constructorId"].astype(int))
        assert set(row["constructorId"].tolist()) == expected, (
            f"driver {row['driverId']} at {t.date()}: label does not match the next season"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--out", type=Path, default=Path("data/driver_constructor"))
    parser.add_argument(
        "--max-test-years",
        type=int,
        default=4,
        help="Seasons of test seeds after test_timestamp. Beyond that the model's data is stale.",
    )
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    # Labels need the future, so they come from the unmasked database. Training never does.
    results = dataset.get_db(upto_test_timestamp=False).table_dict["results"].df
    results = results[["driverId", "constructorId", "date"]].copy()

    labels = add_slices(build_labels(results), results)
    check_no_future(labels, results)

    splits = split_labels(
        labels, dataset.val_timestamp, dataset.test_timestamp, args.max_test_years
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_pickle(args.out / f"{name}.pkl")

    print(f"seeds: {len(labels)}  |  seasons: {labels['date'].nunique()}")
    print(
        f"constructors per seed: mean {labels['constructorId'].apply(len).mean():.2f}, "
        f"max {labels['constructorId'].apply(len).max()}"
    )
    print()
    for name, df in splits.items():
        if df.empty:
            print(f"  {name:5s}      0 seeds")
            continue
        print(
            f"  {name:5s} {len(df):6d} seeds   {df['date'].min().date()} .. "
            f"{df['date'].max().date()}   movers {int(df['is_mover'].sum()):4d} "
            f"({df['is_mover'].mean():.1%})   rookies {int(df['is_rookie'].sum()):4d} "
            f"({df['is_rookie'].mean():.1%})"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
