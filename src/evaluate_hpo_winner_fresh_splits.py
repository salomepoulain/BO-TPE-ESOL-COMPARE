"""Fresh-split robustness evaluation for one selected HPO configuration.

Hyperparameter search can overfit a fixed validation split. This script
quantifies that effect by retraining the chosen configuration on multiple fresh
data permutations and reporting validation-vs-test behavior per split.

Concretely, it measures the val->test gap by:

  1. Re-permuting ESOL with N different split seeds.
  2. Carving train / val / test = 80 / 10 / 10 from each permutation.
  3. Retraining the recorded config from scratch on `train`, monitoring
     `val` for early stopping (same patience as HPO), reporting the
     `test` loss at the best-val checkpoint.
  4. Writing a CSV with per-split val_loss and test_loss.

Run example (BO winner, 5 fresh splits):
    uv run python src/evaluate_hpo_winner_fresh_splits.py \
        --config-json output/fine_best_config_for_test_eval.json \
        --n-splits 5 \
        --output-csv output/test_eval/bo_winner_fresh_splits.csv
Running this file directly executes the CLI workflow and writes a CSV summary.

CLI Arguments
-------------
--config-json : pathlib.Path, required
    JSON file containing winner configuration fields.
--data-root : str, default="data"
    MoleculeNet dataset cache root.
--dataset-name : str, default="ESOL"
    MoleculeNet dataset name.
--n-splits : int, default=5
    Number of fresh split seeds evaluated.
--split-seed-start : int, default=1001
    First split seed; each next split adds +1.
--shuffle-seed : int, default=42
    Train-loader shuffle seed.
--train-fraction : float, default=0.8
    Training split fraction.
--val-fraction : float, default=0.1
    Validation split fraction.
--max-epochs : int, default=2000
    Maximum training epochs per split.
--patience : int, default=30
    Early-stopping patience in epochs.
--output-csv : pathlib.Path, required
    Destination CSV for split-level metrics.
--device : str | None, default=None
    Torch device override (e.g. "cpu", "cuda", "cuda:0").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs and enable deterministic CUDA kernels.

    Sets the workspace config required by cuBLAS for deterministic matmul and
    enables PyTorch's deterministic-algorithm guard (warn-only so the script
    does not crash on ops without a deterministic implementation; the result
    is still reproducible for our model). Must be called before any model
    construction or forward pass on a given (split, init) combination.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

from helpers.data_utils import dataset_input_dim
from helpers.models import (
    ActivationType,
    LayerConfig,
    LayerType,
    ModelConfig,
    PoolingType,
)
from helpers.training import OptimizerConfig, OptimizerType


@dataclass(frozen=True)
class SplitLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def build_three_way_split(
    dataset: MoleculeNet,
    *,
    train_fraction: float,
    val_fraction: float,
    batch_size: int,
    split_seed: int,
    shuffle_seed: int | None = None,
) -> SplitLoaders:
    """Permute ESOL with `split_seed` then carve train / val / test slices.

    Parameters
    ----------
    dataset : MoleculeNet
        Full dataset to split.
    train_fraction : float
        Fraction used for the training split.
    val_fraction : float
        Fraction used for the validation split.
    batch_size : int
        Batch size for all generated loaders.
    split_seed : int
        Seed that controls dataset permutation.
    shuffle_seed : int | None, default=None
        Seed for train-loader shuffling.

    Returns
    -------
    SplitLoaders
        Train, validation, and test dataloaders.
    """
    n_total = len(dataset)
    split_gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=split_gen).tolist()
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    shuffle_gen = None
    if shuffle_seed is not None:
        shuffle_gen = torch.Generator().manual_seed(shuffle_seed)

    return SplitLoaders(
        train_loader=DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=shuffle_gen
        ),
        val_loader=DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False),
    )


def build_model_from_config(config: dict[str, Any], input_dim: int) -> torch.nn.Module:
    """Construct a ``MolecularGNN`` from a serialized config.

    Parameters
    ----------
    config : dict[str, Any]
        Serialized architecture and optimizer-related fields.
    input_dim : int
        Number of input node features.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """
    layer_type = LayerType[config["layer_type"]]
    activation = ActivationType[config["activation"]]
    pooling = PoolingType[config["pooling"]]
    return ModelConfig(
        input_dim=input_dim,
        layers=[
            LayerConfig(
                layer_type=layer_type,
                output_dim=int(config["hidden_dim"]),
                activation=activation,
                dropout=float(config["dropout"]),
            )
            for _ in range(int(config["num_layers"]))
        ],
        pooling=pooling,
    ).build()


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Compute mean loss on one dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        Data loader for the split to evaluate.
    loss_fn : torch.nn.Module
        Loss function used for scoring.
    device : torch.device
        Device for evaluation.

    Returns
    -------
    float
        Mean loss across all batches.
    """
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


def train_one_split(
    *,
    dataset: MoleculeNet,
    config: dict[str, Any],
    split_seed: int,
    shuffle_seed: int,
    init_seed: int,
    max_epochs: int,
    patience: int,
    train_fraction: float,
    val_fraction: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one (split, init) combination and return per-run metrics.

    Determinism: `seed_everything(init_seed)` is called immediately before
    model construction so model init, dropout RNG, and any CUDA non-determinism
    are pinned. Re-running with the same (split_seed, shuffle_seed, init_seed)
    should reproduce the same test_loss bit-for-bit.

    Returns
    -------
    dict[str, float | int]
        Per-run metrics and metadata, including the three seeds used.
    """
    splits = build_three_way_split(
        dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        batch_size=int(config["batch_size"]),
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
    )

    # Pin model init, dropout, and CUDA RNGs for this (split, init) run.
    seed_everything(init_seed)

    model = build_model_from_config(config, input_dim=dataset_input_dim(dataset)).to(device)
    optimizer = OptimizerConfig(
        optimizer_type=OptimizerType[config["optimizer_type"]],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    ).build_for(model)
    loss_fn = torch.nn.MSELoss()

    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    best_epoch = -1
    started = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for batch in splits.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_loss = evaluate_loader(model, splits.val_loader, loss_fn, device)
        if math.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss = evaluate_loader(model, splits.test_loader, loss_fn, device)
    train_loss = evaluate_loader(model, splits.train_loader, loss_fn, device)
    elapsed = time.perf_counter() - started

    return {
        "split_seed": split_seed,
        "shuffle_seed": shuffle_seed,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "train_loss_at_best": train_loss,
        "best_epoch": best_epoch + 1,
        "epochs_run": epoch + 1,
        "elapsed_seconds": elapsed,
        "n_train": len(splits.train_loader.dataset),
        "n_val": len(splits.val_loader.dataset),
        "n_test": len(splits.test_loader.dataset),
        "init_seed": init_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, type=Path,
                        help="Path to a JSON file with the config to retrain (e.g. fine_best_config_for_test_eval.json)")
    parser.add_argument("--data-root", default="data", help="Where ESOL is cached.")
    parser.add_argument("--dataset-name", default="ESOL")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of fresh permutation seeds to evaluate.")
    parser.add_argument("--split-seed-start", type=int, default=1001,
                        help="First split_seed. Subsequent splits use split_seed_start + i.")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Train-loader shuffle seed (kept constant across splits).")
    parser.add_argument("--n-inits-per-split", type=int, default=1,
                        help="Number of independent model-init reruns per split. "
                             "k>1 measures the within-(config,split) noise floor.")
    parser.add_argument("--init-seed-start", type=int, default=0,
                        help="First init_seed. Inits use init_seed_start + j.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda / cpu (default = autodetect).")
    args = parser.parse_args()

    config = json.loads(args.config_json.read_text())
    # The JSON may have wrapping keys from the HPO run (pass_label, method, ...).
    # We only need the config-knob fields below.
    required = {"layer_type", "pooling", "activation", "num_layers",
                "hidden_dim", "batch_size", "learning_rate", "weight_decay",
                "dropout", "optimizer_type"}
    missing = required - set(config.keys())
    if missing:
        raise SystemExit(f"config-json missing required fields: {sorted(missing)}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: {json.dumps({k: config[k] for k in sorted(required)}, default=str, indent=2)}")

    dataset = MoleculeNet(root=args.data_root, name=args.dataset_name)
    print(f"dataset: {args.dataset_name} ({len(dataset)} molecules)")

    rows: list[dict[str, float | int]] = []
    for i in range(args.n_splits):
        split_seed = args.split_seed_start + i
        for j in range(args.n_inits_per_split):
            init_seed = args.init_seed_start + j
            tag = (f"split {i + 1}/{args.n_splits} "
                   f"(split_seed={split_seed}, init_seed={init_seed})")
            print(f"\n=== {tag} ===")
            metrics = train_one_split(
                dataset=dataset,
                config=config,
                split_seed=split_seed,
                shuffle_seed=args.shuffle_seed,
                init_seed=init_seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                device=device,
            )
            print(f"  best_val_loss = {metrics['best_val_loss']:.4f}  "
                  f"test_loss = {metrics['test_loss']:.4f}  "
                  f"train_loss = {metrics['train_loss_at_best']:.4f}  "
                  f"best_epoch = {metrics['best_epoch']}  "
                  f"epochs_run = {metrics['epochs_run']}  "
                  f"elapsed = {metrics['elapsed_seconds']:.1f}s")
            rows.append(metrics)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nwrote {args.output_csv}")

    print("\n=== summary ===")
    print(f"n_splits = {len(df)}")
    print(f"val_loss : mean = {df['best_val_loss'].mean():.4f}  std = {df['best_val_loss'].std():.4f}  "
          f"min = {df['best_val_loss'].min():.4f}  max = {df['best_val_loss'].max():.4f}")
    print(f"test_loss: mean = {df['test_loss'].mean():.4f}  std = {df['test_loss'].std():.4f}  "
          f"min = {df['test_loss'].min():.4f}  max = {df['test_loss'].max():.4f}")
    print(f"train_loss at best-val: mean = {df['train_loss_at_best'].mean():.4f}  "
          f"std = {df['train_loss_at_best'].std():.4f}")
    print()
    print(f"val->test gap (mean test - mean val): {df['test_loss'].mean() - df['best_val_loss'].mean():+.4f}")
    print("reported HPO winner val_loss        : 0.3670")
    print(f"observed mean test_loss             : {df['test_loss'].mean():.4f}")
    print(f"-> reusable-holdout penalty estimate: {df['test_loss'].mean() - 0.3670:+.4f}")


if __name__ == "__main__":
    main()
