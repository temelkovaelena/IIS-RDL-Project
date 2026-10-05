"""Training. Must be callable from the command line:

    python src/train.py --task driver-top3 --seed 0 --out output/evaluation.csv

Neighbour sampling is time-limited: no sampled node may be newer than the seed node it was
sampled for. That is asserted on real batches, not assumed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import relbench
import torch
import torch.nn.functional as F
from relbench.base import TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_geometric.loader import NeighborLoader

from features import drop_text_columns, text_embedder_config
from model_gnn import Model

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

# Written only to the sweep file, so output/evaluation.csv keeps the agreed columns.
SWEEP_COLUMNS = EVAL_COLUMNS + [
    "lr",
    "channels",
    "num_layers",
    "num_neighbors",
    "epochs",
    "temporal_strategy",
]


def check_no_future(batch, entity_table: str) -> int:
    """Every sampled node must be at or before the seed time it was sampled for.

    The sampler is supposed to guarantee this. Checking it on real batches is what turns
    that into evidence: a silent failure here would make every number below meaningless.
    Returns how many node types were checked.
    """
    seed_time = batch[entity_table].seed_time
    checked = 0
    for node_type, node_time in batch.time_dict.items():
        owner = seed_time[batch.batch_dict[node_type]]
        late = int((node_time > owner).sum())
        assert late == 0, f"{node_type}: {late} sampled nodes are newer than their seed"
        checked += 1
    return checked


def build_loaders(data, task, args, device):
    loaders = {}
    for split in ("train", "val", "test"):
        table = task.get_table(split, mask_input_cols=False)
        table_input = get_node_train_table_input(table=table, task=task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=[args.num_neighbors] * args.num_layers,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            temporal_strategy=args.temporal_strategy,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    return loaders


def train_epoch(model, loader, optimizer, loss_fn, entity_table, device):
    """One pass over the training set.

    The prediction is deliberately NOT clamped here. Clamping before the loss gives a zero
    gradient to anything outside the range, so once the outputs drift out they can never come
    back and the model freezes on a constant. Clamping belongs at prediction time only.
    """
    model.train()
    total_loss = total_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, entity_table).squeeze(-1)
        target = batch[entity_table].y.float()
        loss = loss_fn(pred.float(), target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
        total_count += pred.size(0)
    return total_loss / total_count


@torch.no_grad()
def predict(model, loader, entity_table, device, task_type, clamp, verify: bool):
    model.eval()
    out = []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        if verify and i < 5:
            check_no_future(batch, entity_table)
        pred = model(batch, entity_table).squeeze(-1).detach()
        if task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)
        elif clamp is not None:
            pred = torch.clamp(pred, *clamp)
        out.append(pred.cpu())
    return torch.cat(out, dim=0).numpy()


def append_rows(path: Path, rows: list[dict]) -> None:
    columns = SWEEP_COLUMNS if any("lr" in row for row in rows) else EVAL_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-neighbors", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--aggr", default="sum")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--temporal-strategy",
        default="uniform",
        choices=["uniform", "last"],
        help="How neighbours are chosen among those older than the seed. uniform draws at "
        "random; last takes the most recent ones, which is a choice rather than a draw.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--arm",
        default="gnn",
        help="Label written to the arm column, e.g. gnn-hops1 for an ablation run.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["val", "test"],
        help="Which splits to score. Use only val while searching hyperparameters, so the "
        "test split is untouched during model selection.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Drop text columns instead of embedding them. Not comparable to published numbers.",
    )
    parser.add_argument("--out", type=Path, default=Path("output/evaluation.csv"))
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also record the hyperparameters on each row. For the sweep file, not results.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = relbench.load_dataset(args.dataset)
    task = dataset.load_task(args.task)
    db = dataset.get_db()

    # Text columns (driver, constructor and circuit names) need an embedder supplied up
    # front, otherwise torch_frame refuses to build the dataset.
    col_to_stype = get_stype_proposal(db)
    if args.no_text:
        col_to_stype = drop_text_columns(col_to_stype)
        text_cfg = None
    else:
        text_cfg = text_embedder_config(device=device)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        text_embedder_cfg=text_cfg,
        cache_dir=args.cache_dir,
        remove_columns=task.hidden_columns(),
    )
    print(f"{len(data.node_types)} node types, {len(data.edge_types)} edge types")

    # The metric and the loss follow the task type, not a guess.
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels, higher_is_better, clamp = 1, True, None
        loss_fn = F.binary_cross_entropy_with_logits
    elif task.task_type == TaskType.REGRESSION:
        train_target = task.get_table("train", mask_input_cols=False).df[task.target_col]
        clamp = (float(train_target.min()), float(train_target.max()))
        out_channels, loss_fn, higher_is_better = 1, F.l1_loss, False
    else:
        raise NotImplementedError(f"unsupported task type: {task.task_type}")

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        aggr=args.aggr,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loaders = build_loaders(data, task, args, device)
    entity_table = task.entity_table

    # One batch before training: if sampling leaks, stop now rather than after an hour.
    first = next(iter(loaders["train"])).to(device)
    n_checked = check_no_future(first, entity_table)
    print(f"temporal check on first batch: {n_checked} node types clean")

    val_table = task.get_table("val", mask_input_cols=False)
    best_metric = None
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, loss_fn, entity_table, device)
        val_pred = predict(
            model, loaders["val"], entity_table, device, task.task_type, clamp, epoch == 1
        )
        scores = task.evaluate(val_pred, val_table)
        metric_name = task.metrics[0].__name__
        value = scores[metric_name]
        better = best_metric is None or (
            value > best_metric if higher_is_better else value < best_metric
        )
        if better:
            best_metric, best_state = value, copy.deepcopy(model.state_dict())
        mark = "  *" if better else ""
        print(f"epoch {epoch:2d}  loss {loss:.4f}  val {metric_name} {value:.4f}{mark}")

    train_minutes = (time.perf_counter() - started) / 60
    model.load_state_dict(best_state)

    rows = []
    for split in args.splits:
        table = task.get_table(split, mask_input_cols=False)
        pred = predict(model, loaders[split], entity_table, device, task.task_type, clamp, True)
        for metric, value in task.evaluate(pred, table).items():
            rows.append(
                {
                    "database": args.dataset,
                    "task": args.task,
                    "task_type": str(task.task_type).removeprefix("TaskType."),
                    "arm": args.arm,
                    "split": split,
                    "slice": "all",
                    "metric": metric,
                    "value": round(float(value), 6),
                    "seed": args.seed,
                    "train_minutes": round(train_minutes, 3),
                    "engineering_minutes": 0,
                }
            )
            if args.sweep:
                rows[-1].update(
                    lr=args.lr,
                    channels=args.channels,
                    num_layers=args.num_layers,
                    num_neighbors=args.num_neighbors,
                    epochs=args.epochs,
                    temporal_strategy=args.temporal_strategy,
                )
            print(f"  {split:5s} {metric:10s} {value:.4f}")

    append_rows(args.out, rows)
    print(f"\n{len(rows)} rows appended to {args.out}")


if __name__ == "__main__":
    main()
