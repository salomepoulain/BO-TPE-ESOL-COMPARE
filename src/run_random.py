"""Resumable random-search driver for molecular GNN hyperparameters.

This script samples the configured search space uniformly at random, evaluates
each candidate with the shared training objective, and saves trial outcomes in
the same schema used by other search backends.

Using the same evaluation/persistence path as BoTorch and Optuna keeps
cross-method comparisons focused on sampler behavior, not pipeline differences.

Running this file directly starts a command-line random search run.

CLI Arguments
-------------
--output-dir : str, default="output/simulation"
    Directory where run artifacts are written.
--run-name : str | None, default=None
    Optional run folder name under `output-dir`.
--data-root : str, default="data"
    MoleculeNet dataset cache root.
--dataset-name : str, default="ESOL"
    MoleculeNet dataset name.
--seed : int, default=42
    Global RNG seed.
--search-epochs : int, default=10
    Training epochs per trial.
--train-fraction : float, default=0.8
    Fraction of dataset used for training split.
--trials : int, default=75
    Number of random candidates to evaluate.
--batch-size : int, default=1
    Number of candidates evaluated per loop iteration.
--parallel-trials : int, default=1
    Number of trials evaluated concurrently.
--search-space : {"coarse", "fine"}, default="coarse"
    Hyperparameter space variant.
--device : str | None, default=None
    Torch device override (e.g. "cpu", "cuda", "cuda:0").
--fast-trainer : flag
    Use `TrainerBatch` instead of `Trainer`.
--patience : int | None, default=None
    Early-stopping patience in epochs.
--gradient-clip-max-norm : float | None, default=None
    Global gradient-norm clipping threshold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import time

import torch
from torch_geometric.datasets import MoleculeNet

from helpers.data_utils import dataset_input_dim
from helpers.storage import OptimizationStorage
from helpers.rng import set_seed
from helpers.search_shared import (
    OUTPUT_DIR,
    PARALLEL_TRIALS,
    SEARCH_EPOCHS,
    SEARCH_SPACE_BUILDERS,
    SEED,
    TRAIN_FRACTION,
    TrialJob,
    configure_torch_runtime,
    evaluate_jobs,
    next_trial_index,
    print_runtime_diagnostics,
    trial_parameters,
)
from helpers.training import default_device


N_TRIALS = 75
BATCH_SIZE = 1


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_runtime()
    set_seed(args.seed)

    dataset_started = time.perf_counter()
    dataset = MoleculeNet(root=args.data_root, name=args.dataset_name)
    dataset_seconds = time.perf_counter() - dataset_started
    input_dim = dataset_input_dim(dataset)
    device = default_device() if args.device is None else torch.device(args.device)
    search_space = SEARCH_SPACE_BUILDERS[args.search_space]()
    print_runtime_diagnostics("Random", device, args.parallel_trials)
    print(
        "search config | "
        f"trials={args.trials} batch_size={args.batch_size} "
        f"parallel_trials={args.parallel_trials} search_epochs={args.search_epochs}",
        flush=True,
    )
    print(
        f"dataset loaded | name={args.dataset_name} size={len(dataset)} "
        f"input_dim={input_dim} seconds={dataset_seconds:.2f}",
        flush=True,
    )

    run_started_wall = time.perf_counter()
    start_time_iso = datetime.now(timezone.utc).isoformat()

    storage = OptimizationStorage(
        method="Random",
        output_dir=args.output_dir,
        run_name=args.run_name,
        metadata={
            "seed": args.seed,
            "dataset_name": args.dataset_name,
            "data_root": args.data_root,
            "input_dim": input_dim,
            "device": str(device),
            "search_epochs": args.search_epochs,
            "train_fraction": args.train_fraction,
            "trials": args.trials,
            "batch_size": args.batch_size,
            "parallel_trials": args.parallel_trials,
            "gradient_clip_max_norm": args.gradient_clip_max_norm,
            "start_time": start_time_iso,
        },
    )

    candidates = search_space.sample(args.trials, seed=args.seed)
    print(f"sampled {args.trials} random candidates", flush=True)

    while len(storage.trials()) < args.trials:
        next_index = next_trial_index(storage.trials())
        stop = min(args.trials, next_index + args.batch_size)
        jobs: list[TrialJob] = []

        for trial_index in range(next_index, stop):
            x = candidates[trial_index]
            parameters = search_space.decode(x)
            x_values = x.detach().cpu().flatten().tolist()
            job = TrialJob(
                trial_index=trial_index,
                parameters=parameters,
                x_values=x_values,
            )
            jobs.append(job)
            storage.save_pending(
                trial=trial_index,
                parameters=trial_parameters(parameters),
                x=x_values,
            )

        print(f"evaluating random trial batch {[job.trial_index for job in jobs]}")
        evaluations = evaluate_jobs(
            jobs,
            dataset=dataset,
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            input_dim=input_dim,
            search_epochs=args.search_epochs,
            train_fraction=args.train_fraction,
            seed=args.seed,
            device=device,
            parallel_trials=args.parallel_trials,
            fast_trainer=args.fast_trainer,
            patience=args.patience,
            gradient_clip_max_norm=args.gradient_clip_max_norm,
        )

        for evaluation in evaluations:
            job = evaluation.job
            parameters = job.parameters
            if evaluation.error is not None:
                storage.save_failed(
                    trial=job.trial_index,
                    parameters=trial_parameters(parameters),
                    error=evaluation.error,
                    elapsed_seconds=evaluation.elapsed_seconds,
                    x=job.x_values,
                )
                print(f"trial {job.trial_index} failed | {evaluation.error}")
                continue

            if evaluation.validation_loss is None or not math.isfinite(evaluation.validation_loss):
                error = (
                    "trial finished without a validation loss"
                    if evaluation.validation_loss is None
                    else f"trial produced non-finite validation loss ({evaluation.validation_loss!r})"
                )
                storage.save_failed(
                    trial=job.trial_index,
                    parameters=trial_parameters(parameters),
                    error=error,
                    elapsed_seconds=evaluation.elapsed_seconds,
                    x=job.x_values,
                )
                print(f"trial {job.trial_index} failed | {error}")
                continue

            validation_loss = evaluation.validation_loss
            storage.save_completed(
                trial=job.trial_index,
                parameters=trial_parameters(parameters),
                validation_loss=validation_loss,
                elapsed_seconds=evaluation.elapsed_seconds,
                x=job.x_values,
            )
            completed_rows = storage.completed_trials()
            best = min(completed_rows, key=lambda row: row["validation_loss"])
            print(
                f"trial {job.trial_index} completed | "
                f"validation loss {validation_loss:.4f} | "
                f"best {best['validation_loss']:.4f}"
            )

        trials = storage.trials()
        completed_rows = storage.completed_trials()
        total_trial_seconds = sum(float(r.get("elapsed_seconds") or 0.0) for r in storage.trials())
        metadata = {
            "attempted_trials": len(trials),
            "completed_trials": len(completed_rows),
            "failed_trials": sum(1 for row in trials if row["status"] == "failed"),
            "total_trial_seconds": total_trial_seconds,
            "wall_seconds_so_far": time.perf_counter() - run_started_wall,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
        }
        if completed_rows:
            best = min(completed_rows, key=lambda row: row["validation_loss"])
            metadata.update(
                {
                    "best_validation_loss": best["validation_loss"],
                    "best_trial": best["trial"],
                }
            )
        storage.update_metadata(metadata)

    completed_rows = storage.completed_trials()
    if not completed_rows:
        storage.update_metadata({"end_time": datetime.now(timezone.utc).isoformat()})
        print(f"saved empty random-search results to {storage.run_folder}")
        return

    best = min(completed_rows, key=lambda row: row["validation_loss"])
    total_trial_seconds = sum(float(r.get("elapsed_seconds") or 0.0) for r in storage.trials())
    wall_seconds_total = time.perf_counter() - run_started_wall
    storage.update_metadata(
        {
            "end_time": datetime.now(timezone.utc).isoformat(),
            "attempted_trials": len(storage.trials()),
            "completed_trials": len(completed_rows),
            "failed_trials": sum(1 for row in storage.trials() if row["status"] == "failed"),
            "total_trial_seconds": total_trial_seconds,
            "wall_seconds_total": wall_seconds_total,
        }
    )
    print(f"saved random-search results to {storage.run_folder}")
    print(f"best validation loss: {best['validation_loss']:.4f}")
    print(f"wall time: {wall_seconds_total:.1f}s | trial time sum: {total_trial_seconds:.1f}s")
    print(best)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for random search runs.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-name", default="ESOL")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--search-epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--parallel-trials", type=int, default=PARALLEL_TRIALS)
    parser.add_argument(
        "--search-space",
        choices=("coarse", "fine"),
        default="coarse",
        help="Which search space to sample from. 'coarse' uses the original wide "
             "space; 'fine' uses a narrowed space derived from coarse-pass evidence.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--fast-trainer",
        action="store_true",
        help="Use TrainerBatch (pre-collated batches on device, tensor-accumulated loss).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience in actual epochs. Default: disabled.",
    )
    parser.add_argument(
        "--gradient-clip-max-norm",
        type=float,
        default=None,
        help="If set, clip the global gradient norm before optimizer.step(). "
             "Use 5.0 for the coarse-pass stability setting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate runtime arguments before starting the search.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    if args.trials <= 0:
        raise ValueError(f"trials must be positive, got {args.trials}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.parallel_trials <= 0:
        raise ValueError(f"parallel_trials must be positive, got {args.parallel_trials}")
    if args.search_epochs <= 0:
        raise ValueError(f"search_epochs must be positive, got {args.search_epochs}")
    if args.gradient_clip_max_norm is not None and args.gradient_clip_max_norm <= 0:
        raise ValueError(
            "gradient_clip_max_norm must be positive if set, "
            f"got {args.gradient_clip_max_norm}"
        )


if __name__ == "__main__":
    main()
