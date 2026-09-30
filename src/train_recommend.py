"""Training for the custom driver-constructor task.

    python src/train_recommend.py --seed 0 --epochs 20

The task is recommendation, not classification, so the setup differs from train.py:

- Two towers sharing one encoder. The driver subgraph and the constructor subgraph both go
  through the same GNN; the score is the dot product of the two embeddings.
- BPR loss over one sampled negative constructor per positive.
- At evaluation every one of the 211 constructors is scored for each driver, which is cheap
  enough to do exactly rather than by sampling.

MAP@k is reported three times: all seeds, drivers who changed constructor, and drivers with
no history. Most drivers stay put, so the `all` number is dominated by persistence and the
interesting question lives in `movers`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import relbench
import torch
import torch.nn.functional as F
from relbench.metrics import map as map_metric
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.loader import LinkNeighborLoader
from relbench.modeling.utils import get_stype_proposal, to_unix_time
from torch_geometric.loader import NeighborLoader

from features import text_embedder_config
from model_gnn import Model

SRC_TABLE, DST_TABLE = "drivers", "constructors"

EVAL_COLUMNS = [
    "database", "task", "task_type", "arm", "split", "slice",
    "metric", "value", "seed", "train_minutes", "engineering_minutes",
]


def load_labels(directory: Path, split: str) -> pd.DataFrame:
    return pd.read_pickle(directory / f"{split}.pkl").reset_index(drop=True)


def positives_csr(labels: pd.DataFrame, num_dst: int) -> torch.Tensor:
    """Sparse matrix of the true constructors per seed, in the layout the loader wants.

    Built the same way as relbench.modeling.graph.get_link_train_table_input, so the loader
    receives exactly what it does for the built-in recommendation tasks.
    """
    exploded = labels["constructorId"].explode()
    coo = torch.from_numpy(
        np.stack([exploded.index.values, exploded.values.astype(int)])
    )
    sparse = torch.sparse_coo_tensor(
        coo, torch.ones(coo.size(1), dtype=torch.bool), (len(labels), num_dst)
    )
    return sparse.to_sparse_csr()


def train_epoch(model, loader, optimizer, device) -> float:
    """One pass, BPR: the true constructor should score above a random one."""
    model.train()
    total, count = 0.0, 0
    for src_batch, pos_batch, neg_batch in loader:
        src_batch, pos_batch, neg_batch = (
            src_batch.to(device), pos_batch.to(device), neg_batch.to(device)
        )
        optimizer.zero_grad()
        src = model(src_batch, SRC_TABLE)
        pos = model(pos_batch, DST_TABLE)
        neg = model(neg_batch, DST_TABLE)
        pos_score = (src * pos).sum(dim=-1)
        neg_score = (src * neg).sum(dim=-1)
        loss = F.softplus(neg_score - pos_score).mean()
        loss.backward()
        optimizer.step()
        total += float(loss) * src.size(0)
        count += src.size(0)
    return total / max(count, 1)


@torch.no_grad()
def embed(model, data, node_type, indices, times, args, device) -> torch.Tensor:
    """Embeddings for the given nodes, each as seen at its own timestamp."""
    loader = NeighborLoader(
        data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        time_attr="time",
        input_nodes=(node_type, indices),
        input_time=times,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    out = []
    for batch in loader:
        batch = batch.to(device)
        out.append(model(batch, node_type).detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def rank(model, data, labels, num_dst, args, device, k) -> np.ndarray:
    """Top-k constructors per seed.

    Constructor embeddings depend on the seed time, because a constructor's neighbourhood is
    everything it has done up to that point. Seeds are grouped by season so all 211 are
    embedded once per distinct timestamp rather than once per driver.
    """
    model.eval()
    preds = np.full((len(labels), k), -1, dtype=np.int64)
    all_dst = torch.arange(num_dst)

    for t, group in labels.groupby("date", sort=True):
        stamp = int(to_unix_time(pd.Series([t])).item())
        dst_emb = embed(
            model, data, DST_TABLE, all_dst,
            torch.full((num_dst,), stamp, dtype=torch.long), args, device,
        )
        rows = group.index.to_numpy()
        src_emb = embed(
            model, data, SRC_TABLE,
            torch.from_numpy(group["driverId"].astype(int).to_numpy()),
            torch.full((len(group),), stamp, dtype=torch.long), args, device,
        )
        scores = src_emb @ dst_emb.T
        preds[rows] = scores.topk(k, dim=1).indices.numpy()
    return preds


def map_at_k(pred: np.ndarray, truth: pd.Series) -> float:
    pred_isin = np.zeros(pred.shape, dtype=np.int64)
    dst_count = np.zeros(len(pred), dtype=np.int64)
    for i, actual in enumerate(truth):
        target = {int(c) for c in actual}
        dst_count[i] = len(target)
        for j, c in enumerate(pred[i]):
            pred_isin[i, j] = int(c) in target
    return float(map_metric(pred_isin, dst_count))


def score_slices(pred: np.ndarray, labels: pd.DataFrame, k: int) -> dict[str, float]:
    masks = {
        "all": np.ones(len(labels), dtype=bool),
        "movers": labels["is_mover"].to_numpy(),
        "rookies": labels["is_rookie"].to_numpy(),
    }
    return {
        name: map_at_k(pred[m], labels.loc[m, "constructorId"])
        for name, m in masks.items()
        if m.sum() > 0
    }


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-neighbors", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--arm", default="gnn")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", type=Path, default=Path("output/evaluation.csv"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = relbench.load_dataset(args.dataset)
    # The masked database, the one a model is allowed to see. Only the labels were built
    # from the full one.
    db = dataset.get_db()
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=get_stype_proposal(db),
        text_embedder_cfg=text_embedder_config(device=device),
        cache_dir=args.cache_dir,
    )
    num_dst = len(db.table_dict[DST_TABLE].df)
    print(f"{len(data.node_types)} node types, {len(data.edge_types)} edge types, "
          f"{num_dst} constructors")

    splits = {s: load_labels(args.labels, s) for s in ("train", "val", "test")}
    for name, df in splits.items():
        print(f"  {name}: {len(df)} seeds")

    train_labels = splits["train"]
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[args.num_neighbors] * args.num_layers,
        time_attr="time",
        src_nodes=(SRC_TABLE, torch.from_numpy(train_labels["driverId"].astype(int).to_numpy())),
        dst_nodes=(DST_TABLE, positives_csr(train_labels, num_dst)),
        num_dst_nodes=num_dst,
        src_time=torch.from_numpy(to_unix_time(train_labels["date"])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = Model(
        data=data,
        col_stats_dict=col_stats,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=args.channels,  # the head produces the embedding, not a prediction
        aggr="sum",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best, best_state = None, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        pred = rank(model, data, splits["val"], num_dst, args, device, args.k)
        value = score_slices(pred, splits["val"], args.k)["all"]
        better = best is None or value > best
        if better:
            best, best_state = value, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:2d}  loss {loss:.4f}  val map@{args.k} {value:.4f}"
              + ("  *" if better else ""))

    train_minutes = (time.perf_counter() - started) / 60
    model.load_state_dict(best_state)

    rows = []
    for split in ("val", "test"):
        labels = splits[split]
        pred = rank(model, data, labels, num_dst, args, device, args.k)
        for name, value in score_slices(pred, labels, args.k).items():
            rows.append({
                "database": args.dataset,
                "task": "driver-constructor",
                "task_type": "RECOMMENDATION",
                "arm": args.arm,
                "split": split,
                "slice": name,
                "metric": f"map@{args.k}",
                "value": round(float(value), 6),
                "seed": args.seed,
                "train_minutes": round(train_minutes, 3),
                "engineering_minutes": 0,
            })
            print(f"  {split:5s} {name:8s} map@{args.k} {value:.4f}")

    append_rows(args.out, rows)
    print(f"\n{len(rows)} rows appended to {args.out}")


if __name__ == "__main__":
    main()
