"""Tabular baselines with LightGBM.

--mode flat        the main table only, no aggregates
--mode engineered  plus hand-written aggregates over the related tables

The time and lines of code spent on --mode engineered go into the engineering_minutes
column. That is a result, not overhead.

    python src/baseline_tabular.py --mode flat --tasks driver-top3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import relbench

EVAL_COLUMNS = [
    "database",
    "task",
    "task_type",
    "arm",
    "split",
    "slice",
    "metric",
    "value",
    "seed",
    "train_minutes",
    "engineering_minutes",
]

# Which columns of the main table become features, by the semantic type in schema.json.
# text is left out: a tree cannot use free text, and in drivers those columns (driverRef,
# forename, surname) are the driver's identity rather than data about them.
FLAT_COLUMN_KINDS = {"numeric", "categorical", "timestamp"}

# Windows (how many recent events) for the aggregates in --mode engineered.
WINDOWS = (5, 20)

# statusId == 1 means "Finished". Taken from the driver-dnf task SQL
# (MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END)), not assumed.
FINISHED_STATUS_ID = 1


class HistoryIndex:
    """Rows of a child table, grouped by entity and sorted by time.

    Returns the rows visible at prediction time. The bound is <= t, which does not leak: all
    three tasks build their label from rows with date > t.timestamp, so a row sitting exactly
    at t is not part of the answer.
    """

    def __init__(self, df: pd.DataFrame, entity_col: str, time_col: str, n_entities: int):
        entity = df[entity_col].astype("int64").to_numpy()
        time = pd.to_datetime(df[time_col]).astype("int64").to_numpy() // 10**9
        order = np.lexsort((time, entity))
        self.entity, self.time, self.order = entity[order], time[order], order
        self.df = df
        self.lo = np.searchsorted(self.entity, np.arange(n_entities), side="left")
        self.hi = np.searchsorted(self.entity, np.arange(n_entities), side="right")
        self.columns: dict[str, np.ndarray] = {}

    def column(self, name: str) -> np.ndarray:
        """A column reordered to match self.time, cached."""
        if name not in self.columns:
            self.columns[name] = self.df[name].to_numpy()[self.order]
        return self.columns[name]

    def window(self, entity_idx: int, t: int, last_k: int | None = None) -> tuple[int, int]:
        """Half-open range [start, end) of the entity's rows with time <= t."""
        lo, hi = self.lo[entity_idx], self.hi[entity_idx]
        k = int(np.searchsorted(self.time[lo:hi], t, side="right"))
        end = lo + k
        # If the last included row is after t, the window is reading the future.
        assert k == 0 or self.time[end - 1] <= t, "history after the prediction time"
        assert end == hi or self.time[end] > t, "dropped a row from before the prediction"
        start = lo if last_k is None else max(lo, end - last_k)
        return start, end


def _mean(values: np.ndarray, start: int, end: int) -> float:
    """Mean that returns NaN on an empty window; LightGBM reads NaN as "no history"."""
    return float(np.nanmean(values[start:end])) if end > start else np.nan


def _last(values: np.ndarray, start: int, end: int) -> float:
    return float(values[end - 1]) if end > start else np.nan


def engineered_features(db, task, schema: dict, label_df: pd.DataFrame) -> pd.DataFrame:
    """Hand-written aggregates over the related tables.

    This is the arm the graph model has to beat. Every aggregate below is a join that I write
    and time-limit by hand, which is exactly the work message passing learns on its own.
    """
    features = flat_features(db, task, schema, label_df)
    n_drivers = len(db.table_dict["drivers"].df)
    n_constructors = len(db.table_dict["constructors"].df)

    results = db.table_dict["results"].df
    by_driver = HistoryIndex(results, "driverId", "date", n_drivers)
    by_constructor = HistoryIndex(results, "constructorId", "date", n_constructors)
    quali = HistoryIndex(db.table_dict["qualifying"].df, "driverId", "date", n_drivers)
    standings = HistoryIndex(db.table_dict["standings"].df, "driverId", "date", n_drivers)

    seeds = label_df[task.entity_col].astype("int64").to_numpy()
    seed_time = pd.to_datetime(label_df[task.time_col]).astype("int64").to_numpy() // 10**9

    res_pos = by_driver.column("positionOrder")
    res_points = by_driver.column("points")
    res_grid = by_driver.column("grid")
    res_dnf = (by_driver.column("statusId") != FINISHED_STATUS_ID).astype(float)
    res_constructor = by_driver.column("constructorId")
    con_points = by_constructor.column("points")
    con_dnf = (by_constructor.column("statusId") != FINISHED_STATUS_ID).astype(float)
    qua_pos = quali.column("position")
    std_pos = standings.column("position")
    std_points = standings.column("points")
    std_wins = standings.column("wins")

    rows = []
    for driver, t in zip(seeds, seed_time, strict=True):
        r_start, r_end = by_driver.window(driver, t)
        q_start, q_end = quali.window(driver, t)
        s_start, s_end = standings.window(driver, t)

        row = {
            # --- one hop: results
            "res_n_career": r_end - r_start,
            "res_mean_position_all": _mean(res_pos, r_start, r_end),
            "res_dnf_rate_all": _mean(res_dnf, r_start, r_end),
            "res_days_since_last": (
                (t - by_driver.time[r_end - 1]) / 86400 if r_end > r_start else np.nan
            ),
            # --- one hop: qualifying
            "qua_n_career": q_end - q_start,
            "qua_mean_position_all": _mean(qua_pos, q_start, q_end),
            # --- one hop: standings (last known championship state)
            "std_last_position": _last(std_pos, s_start, s_end),
            "std_last_points": _last(std_points, s_start, s_end),
            "std_last_wins": _last(std_wins, s_start, s_end),
        }
        for k in WINDOWS:
            rk_start, _ = by_driver.window(driver, t, last_k=k)
            qk_start, _ = quali.window(driver, t, last_k=k)
            row[f"res_mean_position_{k}"] = _mean(res_pos, rk_start, r_end)
            row[f"res_mean_points_{k}"] = _mean(res_points, rk_start, r_end)
            row[f"res_mean_grid_{k}"] = _mean(res_grid, rk_start, r_end)
            row[f"res_dnf_rate_{k}"] = _mean(res_dnf, rk_start, r_end)
            row[f"res_best_position_{k}"] = (
                float(np.nanmin(res_pos[rk_start:r_end])) if r_end > rk_start else np.nan
            )
            row[f"qua_mean_position_{k}"] = _mean(qua_pos, qk_start, q_end)

        # --- two hops: driver -> last result -> constructor -> all of that team's results.
        # This is the path a second message-passing layer would traverse.
        constructor = int(res_constructor[r_end - 1]) if r_end > r_start else -1
        if constructor >= 0:
            c_start, c_end = by_constructor.window(constructor, t, last_k=20)
            _, c_end_all = by_constructor.window(constructor, t)
            row["con_mean_points_20"] = _mean(con_points, c_start, c_end_all)
            row["con_dnf_rate_20"] = _mean(con_dnf, c_start, c_end_all)
            row["con_n_results"] = c_end_all - by_constructor.lo[constructor]
        else:
            row["con_mean_points_20"] = np.nan
            row["con_dnf_rate_20"] = np.nan
            row["con_n_results"] = 0
        row["con_current"] = constructor
        rows.append(row)

    engineered = pd.DataFrame(rows, index=features.index)
    engineered["con_current"] = engineered["con_current"].astype("category")
    return pd.concat([features, engineered], axis=1)


def flat_features(db, task, schema: dict, label_df: pd.DataFrame) -> pd.DataFrame:
    """Features from the main table only, plus the prediction timestamp.

    No join to any other table. That is the whole definition of this arm, and why it is a
    lower bound: all it knows about a driver is what the drivers table says.
    """
    entity = db.table_dict[task.entity_table]
    columns = schema["tables"][task.entity_table]["columns"]
    keep = [c for c, info in columns.items() if info["kind"] in FLAT_COLUMN_KINDS]

    rows = entity.df.iloc[label_df[task.entity_col].to_numpy()][keep].reset_index(drop=True)
    seed_time = pd.to_datetime(label_df[task.time_col]).reset_index(drop=True)

    features = pd.DataFrame(index=rows.index)
    for column in keep:
        kind = columns[column]["kind"]
        if kind == "timestamp":
            values = pd.to_datetime(rows[column])
            features[column] = values.astype("int64") // 10**9
            # A within-row transform, no join: age on the day of the prediction. Without it
            # the tree has to learn the relationship between two dates by splitting on both.
            features[f"{column}_days_before_seed"] = (seed_time - values).dt.days
        elif kind == "categorical":
            categories = pd.CategoricalDtype(sorted(entity.df[column].dropna().unique()))
            features[column] = rows[column].astype(categories)
        else:
            features[column] = pd.to_numeric(rows[column])

    features["seed_unix"] = seed_time.astype("int64") // 10**9
    features["seed_year"] = seed_time.dt.year
    features["seed_month"] = seed_time.dt.month
    features["seed_dayofweek"] = seed_time.dt.dayofweek
    return features


def fit_predict(
    task, train_x, train_y, val_x, val_y, eval_x, seed: int
) -> tuple[np.ndarray, float]:
    """Fits on train, early-stops on val, returns predictions for eval_x and the minutes."""
    is_regression = str(task.task_type).endswith("REGRESSION")
    params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        # Subsampling is what makes different seeds give different numbers. Without it GBDT
        # is deterministic and the standard deviation would be 0, which looks like a bug.
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    model = (
        lgb.LGBMRegressor(objective="l1", **params)
        if is_regression
        else lgb.LGBMClassifier(objective="binary", **params)
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        train_y,
        eval_X=val_x,
        eval_y=val_y,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    minutes = (time.perf_counter() - started) / 60
    pred = model.predict(eval_x) if is_regression else model.predict_proba(eval_x)[:, 1]
    return pred, minutes


def run_task(
    dataset, task_name: str, schema: dict, mode: str, seeds: list[int], engineering_minutes: float
) -> list[dict]:
    task = dataset.load_task(task_name)
    tables = {s: task.get_table(s, mask_input_cols=False) for s in ("train", "val", "test")}
    db = dataset.get_db()
    build = flat_features if mode == "flat" else engineered_features
    features = {s: build(db, task, schema, t.df) for s, t in tables.items()}
    targets = {s: tables[s].df[task.target_col].to_numpy() for s in tables}

    rows = []
    for seed in seeds:
        for split in ("val", "test"):
            pred, minutes = fit_predict(
                task,
                features["train"],
                targets["train"],
                features["val"],
                targets["val"],
                features[split],
                seed,
            )
            scores = task.evaluate(pred, tables[split])
            for metric, value in scores.items():
                rows.append(
                    {
                        "database": dataset.name_or_path,
                        "task": task_name,
                        "task_type": str(task.task_type).removeprefix("TaskType."),
                        "arm": f"lgbm-{mode}",
                        "split": split,
                        "slice": "all",
                        "metric": metric,
                        "value": round(float(value), 6),
                        "seed": seed,
                        "train_minutes": round(minutes, 3),
                        "engineering_minutes": engineering_minutes,
                    }
                )
    return rows


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
    parser.add_argument("--mode", choices=["flat", "engineered"], default="flat")
    parser.add_argument(
        "--tasks", nargs="*", default=["driver-position", "driver-top3", "driver-dnf"]
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--schema", type=Path, default=Path("data/schema.json"))
    parser.add_argument("--out", type=Path, default=Path("output/evaluation.csv"))
    parser.add_argument(
        "--engineering-minutes",
        type=float,
        default=0.0,
        help="Measured minutes spent building the aggregates. Zero for --mode flat.",
    )
    args = parser.parse_args()

    dataset = relbench.load_dataset(args.dataset)
    schema = json.loads(args.schema.read_text())

    all_rows = []
    for task_name in args.tasks:
        rows = run_task(dataset, task_name, schema, args.mode, args.seeds, args.engineering_minutes)
        all_rows.extend(rows)
        frame = pd.DataFrame(rows)
        for (split, metric), group in frame.groupby(["split", "metric"]):
            mean, std = group["value"].mean(), group["value"].std(ddof=0)
            print(f"  {task_name:18s} {split:5s} {metric:10s} {mean:.4f} ± {std:.4f}")

    append_rows(args.out, all_rows)
    print(f"\n{len(all_rows)} rows appended to {args.out}")


if __name__ == "__main__":
    main()
