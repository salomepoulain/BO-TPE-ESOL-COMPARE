"""Lasso-on-RDKit-descriptors baseline on ESOL across N fresh splits.

Provides the classical-ML anchor for the GNN-RMSE comparison in §16 of
`fine_pass_discussion.md`. Per-split deterministic 80/10/10 permutation
(same `split_seed` convention as `evaluate_hpo_winner_fresh_splits.py`),
sklearn `LassoCV` with internal 5-fold for the alpha sweep, evaluation on
the held-out test slice.

Note on outliers: at very small alpha, Lasso can extrapolate wildly on
molecules whose descriptor values lie outside the training range — split
1005 produced an RMSE > 25 because of one such molecule. The
recommended summary is **median RMSE** (robust to such outliers) or
**mean excluding outliers**, not the raw mean.

Run example (10 fresh splits matching evaluate_hpo_winner_fresh_splits.py):
    uv run python src/lasso_baseline.py \\
        --n-splits 10 \\
        --split-seed-start 1001 \\
        --output-csv output/test_eval/lasso_rdkit_baseline.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import MoleculeNet


def build_rdkit_features(smiles_list: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute the 200-ish RDKit descriptors, drop constant/inf columns.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings, one per molecule.

    Returns
    -------
    (X, kept_names)
        ``X`` is a finite-valued 2D array, shape ``(n_molecules, n_features)``.
        ``kept_names`` is the names of the columns that survived filtering.
    """
    desc_names = [n for n, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(desc_names))
            continue
        rows.append(list(calc.CalcDescriptors(mol)))
    X = np.array(rows, dtype=float)
    mask = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 1e-9)
    X = X[:, mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    kept = [name for name, keep in zip(desc_names, mask, strict=False) if keep]
    return X, kept


def split_indices(n: int, train_fraction: float, val_fraction: float, split_seed: int) -> tuple[list[int], list[int], list[int]]:
    """Permutation-based 3-way split that matches the GNN evaluation protocol."""
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def evaluate_split(X: np.ndarray, y: np.ndarray, *, train_fraction: float, val_fraction: float, split_seed: int, cv_folds: int) -> dict[str, float | int]:
    """Train LassoCV on the train slice and report MSE/RMSE on the test slice."""
    train_idx, _val_idx, test_idx = split_indices(len(y), train_fraction, val_fraction, split_seed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv_folds, max_iter=20_000, random_state=0)),
    ])
    pipe.fit(X[train_idx], y[train_idx])
    pred = pipe.predict(X[test_idx])
    mse = float(np.mean((pred - y[test_idx]) ** 2))
    return {
        "split_seed": split_seed,
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "alpha": float(pipe.named_steps["lasso"].alpha_),
        "n_active_features": int((pipe.named_steps["lasso"].coef_ != 0).sum()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-name", default="ESOL")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--split-seed-start", type=int, default=1001)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--cv-folds", type=int, default=5, help="LassoCV internal alpha-sweep folds.")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    dataset = MoleculeNet(root=args.data_root, name=args.dataset_name)
    smiles = [d.smiles for d in dataset]
    y = np.array([float(d.y.flatten()[0]) for d in dataset])
    print(f"loaded {len(smiles)} molecules from {args.dataset_name}")

    X, names = build_rdkit_features(smiles)
    print(f"feature matrix: {X.shape} ({len(names)} descriptors survived filtering)")

    rows = []
    for i in range(args.n_splits):
        split_seed = args.split_seed_start + i
        metrics = evaluate_split(
            X, y,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_seed=split_seed,
            cv_folds=args.cv_folds,
        )
        print(f"split {split_seed}: rmse={metrics['test_rmse']:.3f}  "
              f"alpha={metrics['alpha']:.4g}  active={metrics['n_active_features']}")
        rows.append(metrics)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nwrote {args.output_csv}")

    print("\n=== Lasso-on-RDKit summary ===")
    print(f"all splits:                mean RMSE = {df['test_rmse'].mean():.3f}  median = {df['test_rmse'].median():.3f}  std = {df['test_rmse'].std():.3f}")
    rmse = df["test_rmse"].to_numpy()
    iqr_low, iqr_high = np.percentile(rmse, [25, 75])
    iqr = iqr_high - iqr_low
    mask_robust = (rmse >= iqr_low - 1.5 * iqr) & (rmse <= iqr_high + 1.5 * iqr)
    if (~mask_robust).any():
        excluded = df.loc[~mask_robust, "split_seed"].tolist()
        print(f"excluding IQR outliers {excluded}: "
              f"mean RMSE = {df.loc[mask_robust, 'test_rmse'].mean():.3f}  "
              f"std = {df.loc[mask_robust, 'test_rmse'].std():.3f}")


if __name__ == "__main__":
    main()
