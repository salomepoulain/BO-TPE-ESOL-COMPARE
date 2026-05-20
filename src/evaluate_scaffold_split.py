"""Scaffold-split fresh evaluation of one chosen config.

MoleculeNet comparators (Chemprop, AttentiveFP, GROVER, TChemGNN) all report
scaffold-RMSE; the random-split protocol used by
``evaluate_hpo_winner_fresh_splits.py`` is systematically easier, so its
RMSE numbers are not directly comparable to the leaderboard. This script
provides the matching scaffold-split measurement for one selected config,
using the standard MurckoScaffold grouping (`bemis1996scaffold`).

The split is **scaffold-balanced** rather than scaffold-random: scaffolds are
sorted largest-first and packed greedily into train, then val, then test
(matching `wu2018moleculenet`'s prescription). This keeps the size targets
(default 80/10/10) close to exact while ensuring no scaffold spans two
splits. ``--split-seed`` re-shuffles same-size scaffolds, so multiple seeds
explore different scaffold-tie-breaks while keeping the protocol fixed.

Run example (marginal-rec config across 5 scaffold seeds, deterministic init):
    uv run python src/evaluate_scaffold_split.py \\
        --config-json output/test_eval/marginal_rec_config.json \\
        --n-splits 5 \\
        --split-seed-start 2001 \\
        --output-csv output/test_eval/marginal_rec_scaffold_splits.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from helpers.models import ActivationType, LayerConfig, LayerType, ModelConfig, PoolingType
from helpers.training import OptimizerConfig, OptimizerType


def seed_everything(seed: int) -> None:
    """Pin every RNG seed so the same (split, init) re-runs are bit-identical."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def scaffold_groups(smiles_list: list[str]) -> list[list[int]]:
    """Return one index group per unique Murcko scaffold."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        sc = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if mol is not None else ""
        )
        groups[sc].append(i)
    return list(groups.values())


def scaffold_split(smiles_list: list[str], *, train_fraction: float, val_fraction: float, split_seed: int) -> tuple[list[int], list[int], list[int]]:
    """Largest-scaffold-first scaffold split (`wu2018moleculenet` style).

    ``split_seed`` is used only to break ties between equal-size scaffolds,
    so re-runs at the same seed produce identical splits.
    """
    rng = np.random.default_rng(split_seed)
    groups = scaffold_groups(smiles_list)
    rng.shuffle(groups)
    groups.sort(key=lambda g: (-len(g), rng.random()))
    n = len(smiles_list)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(val_idx) + len(group) <= n_val:
            val_idx += group
        else:
            test_idx += group
    return train_idx, val_idx, test_idx


def build_model_from_config(config: dict[str, Any], input_dim: int) -> torch.nn.Module:
    return ModelConfig(
        input_dim=input_dim,
        layers=[
            LayerConfig(
                layer_type=LayerType[config["layer_type"]],
                output_dim=int(config["hidden_dim"]),
                activation=ActivationType[config["activation"]],
                dropout=float(config["dropout"]),
            )
            for _ in range(int(config["num_layers"]))
        ],
        pooling=PoolingType[config["pooling"]],
    ).build()


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            total += float(loss_fn(pred, batch.y.float()).item())
            n_batches += 1
    return total / max(n_batches, 1)


def train_one_scaffold_split(
    *,
    dataset: MoleculeNet,
    smiles: list[str],
    config: dict[str, Any],
    split_seed: int,
    init_seed: int,
    train_fraction: float,
    val_fraction: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, float | int]:
    train_idx, val_idx, test_idx = scaffold_split(
        smiles, train_fraction=train_fraction, val_fraction=val_fraction, split_seed=split_seed,
    )
    seed_everything(init_seed)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(init_seed),
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    model = build_model_from_config(config, dataset[0].x.shape[1]).to(device)
    optimizer = OptimizerConfig(
        optimizer_type=OptimizerType[config["optimizer_type"]],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    ).build_for(model)
    loss_fn = torch.nn.MSELoss()

    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    started = time.perf_counter()
    for _epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        val_loss = evaluate_loader(model, val_loader, loss_fn, device)
        if math.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss = evaluate_loader(model, test_loader, loss_fn, device)
    train_loss = evaluate_loader(model, train_loader, loss_fn, device)
    return {
        "split_seed": split_seed,
        "init_seed": init_seed,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_rmse": math.sqrt(test_loss),
        "train_loss_at_best": train_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-name", default="ESOL")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed-start", type=int, default=2001)
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Single init seed reused across scaffold splits (set per-split for k-init sweeps).")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = json.loads(args.config_json.read_text())
    required = {"layer_type", "pooling", "activation", "num_layers",
                "hidden_dim", "batch_size", "learning_rate", "weight_decay",
                "dropout", "optimizer_type"}
    missing = required - set(config.keys())
    if missing:
        raise SystemExit(f"config-json missing required fields: {sorted(missing)}")

    device = (
        torch.device(args.device)
        if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")
    dataset = MoleculeNet(root=args.data_root, name=args.dataset_name)
    smiles = [d.smiles for d in dataset]
    print(f"dataset: {args.dataset_name} ({len(dataset)} molecules)")

    rows = []
    for i in range(args.n_splits):
        split_seed = args.split_seed_start + i
        metrics = train_one_scaffold_split(
            dataset=dataset, smiles=smiles, config=config,
            split_seed=split_seed, init_seed=args.init_seed,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            max_epochs=args.max_epochs, patience=args.patience, device=device,
        )
        print(f"scaffold split {split_seed}: "
              f"n_tr={metrics['n_train']} n_va={metrics['n_val']} n_te={metrics['n_test']}  "
              f"val={metrics['best_val_loss']:.4f}  test={metrics['test_loss']:.4f}  "
              f"rmse={metrics['test_rmse']:.3f}")
        rows.append(metrics)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nwrote {args.output_csv}")
    print(f"test MSE  mean={df['test_loss'].mean():.4f}  std={df['test_loss'].std():.4f}")
    print(f"test RMSE mean={df['test_rmse'].mean():.4f}  std={df['test_rmse'].std():.4f}")


if __name__ == "__main__":
    main()
