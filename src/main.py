"""Runs the local half of the pipeline, in order.

    python src/main.py
    python src/main.py --only schema graph
    python src/main.py --skip visualisation

The graph neural network is not here. Intel macOS has no recent PyTorch, so that half runs
on Kaggle through `notebooks/kaggle_runner.ipynb`. Everything below runs on a laptop.

The steps depend on each other: the graph reads `data/schema.json`, and so do both tabular
models. Running them out of order fails with a missing-file error that does not explain
itself, which is the reason this file exists.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Name -> command. The order is the dependency order.
STEPS: list[tuple[str, list[str]]] = [
    ("schema", ["src/build_db.py"]),
    ("graph", ["src/graph_builder.py"]),
    ("visualisation", ["src/visualize.py"]),
    ("flat", ["src/baseline_tabular.py", "--mode", "flat"]),
    (
        "engineered",
        ["src/baseline_tabular.py", "--mode", "engineered", "--engineering-minutes", "120"],
    ),
    ("labels", ["src/tasks/driver_constructor.py"]),
    ("recommend", ["src/baseline_recommend.py"]),
    ("leakage", ["src/check_leakage.py", "--task", "driver-top3"]),
]

# Arms these steps produce. They append to evaluation.csv, so their old rows are removed
# first; otherwise running this twice would double them. Rows from the graph model, which
# is trained elsewhere, are left alone.
LOCAL_ARMS = {"lgbm-flat", "lgbm-engineered", "persistence", "popularity"}
RESULTS = Path("output/evaluation.csv")


def clear_local_arms() -> None:
    if not RESULTS.exists():
        return
    d = pd.read_csv(RESULTS)
    keep = d[~d["arm"].isin(LOCAL_ARMS)]
    if len(keep) != len(d):
        keep.to_csv(RESULTS, index=False, lineterminator="\n")
        print(f"removed {len(d) - len(keep)} rows from a previous run, {len(keep)} kept\n")


def run(name: str, command: list[str]) -> bool:
    print(f"=== {name} " + "=" * (60 - len(name)))
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *command])
    seconds = time.perf_counter() - started
    if result.returncode != 0:
        print(f"--- {name} FAILED after {seconds:.0f}s")
        return False
    print(f"--- {name} done in {seconds:.0f}s\n")
    return True


def main() -> None:
    names = [name for name, _ in STEPS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=names, help="Run only these steps.")
    parser.add_argument("--skip", nargs="*", choices=names, default=[], help="Skip these.")
    args = parser.parse_args()

    chosen = [
        (name, cmd)
        for name, cmd in STEPS
        if (args.only is None or name in args.only) and name not in args.skip
    ]
    if any(name in {"flat", "engineered", "recommend"} for name, _ in chosen):
        clear_local_arms()

    failed = []
    for name, command in chosen:
        if not run(name, command):
            failed.append(name)
            break  # later steps read what earlier ones write

    print("=" * 66)
    if failed:
        print(f"stopped at: {failed[0]}")
        sys.exit(1)
    print(f"finished {len(chosen)} steps: {', '.join(n for n, _ in chosen)}")


if __name__ == "__main__":
    main()
