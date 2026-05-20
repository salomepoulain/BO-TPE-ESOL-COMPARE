"""Resumable BoTorch search driver for molecular GNN hyperparameters.

This script orchestrates an end-to-end Bayesian optimization run:

- loads the ESOL dataset,
- proposes candidates in a normalized search space,
- evaluates candidates with the project training loop, and
- persists trial status/results so interrupted runs can resume.

The implementation separates orchestration from model/training details so the
same evaluation pipeline can be reused by other search methods.

Running this file directly starts a command-line BoTorch optimization run.

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
--initial-trials : int, default=3
    Number of random warmup trials before BO.
--bo-trials : int, default=2
    Number of BO-guided trials after warmup.
--bo-batch-size : int, default=1
    Number of candidates proposed per BO step.
--parallel-trials : int, default=1
    Number of trials evaluated concurrently.
--raw-samples : int, default=16
    Raw samples used during acquisition optimization.
--num-restarts : int, default=2
    Restart count for acquisition optimization.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import multiprocessing as mp
import os
from typing import Any
import time

import torch
from torch_geometric.datasets import MoleculeNet

from helpers.bayes import ChoiceDimension, HyperparameterSearchSpace, LinearRangeDimension, LogRangeDimension
from helpers.data_utils import dataset_input_dim
from helpers.models import ActivationType, LayerConfig, LayerType, ModelConfig, PoolingType
from helpers.storage import OptimizationStorage
from helpers.rng import set_seed
from helpers.training import OptimizerConfig, OptimizerType, SamplerConfig, Trainer, TrainerBatch, default_device


SEED = 42
SEARCH_EPOCHS = 10
TRAIN_FRACTION = 0.8
INITIAL_TRIALS = 3
BO_TRIALS = 2
RAW_SAMPLES = 16
NUM_RESTARTS = 2
BO_BATCH_SIZE = 1
PARALLEL_TRIALS = 1
OUTPUT_DIR = "output/simulation"


@dataclass(frozen=True)
class TrialJob:
    """One hyperparameter trial ready to evaluate."""

    trial_index: int
    parameters: dict[str, Any]
    x_values: list[float]


@dataclass(frozen=True)
class TrialEvaluation:
    """Result from evaluating one trial."""

    job: TrialJob
    validation_loss: float | None
    elapsed_seconds: float
    error: str | None = None


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
    total_trials = args.initial_trials + args.bo_trials
    print_runtime_diagnostics("BoTorch", device, args.parallel_trials)
    print(
        "search config | "
        f"total_trials={total_trials} initial_trials={args.initial_trials} "
        f"bo_trials={args.bo_trials} bo_batch_size={args.bo_batch_size} "
        f"parallel_trials={args.parallel_trials} search_epochs={args.search_epochs} "
        f"raw_samples={args.raw_samples} num_restarts={args.num_restarts}",
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
        method="BoTorch",
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
            "initial_trials": args.initial_trials,
            "bo_trials": args.bo_trials,
            "raw_samples": args.raw_samples,
            "num_restarts": args.num_restarts,
            "bo_batch_size": args.bo_batch_size,
            "parallel_trials": args.parallel_trials,
            "gradient_clip_max_norm": args.gradient_clip_max_norm,
            "total_trials": total_trials,
            "start_time": start_time_iso,
        },
    )

    while len(storage.trials()) < total_trials:
        candidate_started = time.perf_counter()
        completed = storage.completed_trials()
        attempted = len(storage.trials())
        batch_size = min(args.bo_batch_size, total_trials - attempted)
        candidates = next_candidates(
            search_space,
            completed,
            initial_trials=args.initial_trials,
            batch_size=batch_size,
            raw_samples=args.raw_samples,
            num_restarts=args.num_restarts,
            seed=args.seed,
        )
        print(
            f"candidate generation finished | completed={len(completed)} "
            f"attempted={attempted} batch_size={batch_size} "
            f"seconds={time.perf_counter() - candidate_started:.2f}",
            flush=True,
        )

        next_index = next_trial_index(storage.trials())
        jobs: list[TrialJob] = []
        for offset, x in enumerate(candidates):
            trial_index = next_index + offset
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

        print(f"evaluating trial batch {[job.trial_index for job in jobs]}")
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
        print(f"saved empty BoTorch results to {storage.run_folder}")
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
    print(f"saved BoTorch results to {storage.run_folder}")
    print(f"best validation loss: {best['validation_loss']:.4f}")
    print(f"wall time: {wall_seconds_total:.1f}s | trial time sum: {total_trial_seconds:.1f}s")
    print(best)


def evaluate_jobs(
    jobs: list[TrialJob],
    *,
    dataset: MoleculeNet,
    dataset_name: str,
    data_root: str,
    input_dim: int,
    search_epochs: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
    parallel_trials: int,
    fast_trainer: bool = False,
    patience: int | None = None,
    gradient_clip_max_norm: float | None = None,
) -> list[TrialEvaluation]:
    """Evaluate a batch of trial jobs, optionally in separate processes."""
    if parallel_trials <= 1 or len(jobs) <= 1:
        evaluations: list[TrialEvaluation] = []
        for job in jobs:
            evaluation = evaluate_job(
                job,
                dataset=dataset,
                input_dim=input_dim,
                search_epochs=search_epochs,
                train_fraction=train_fraction,
                seed=seed,
                device=device,
                fast_trainer=fast_trainer,
                patience=patience,
                gradient_clip_max_norm=gradient_clip_max_norm,
            )
            evaluations.append(evaluation)
            log_evaluation_finished(evaluation)
        return evaluations

    max_workers = min(parallel_trials, len(jobs))
    context = mp.get_context("spawn")
    parallel_evaluations: list[TrialEvaluation] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = []
        for worker_index, job in enumerate(jobs):
            worker_device = parallel_device(device, worker_index)
            futures.append(
                executor.submit(
                    evaluate_job_in_subprocess,
                    job,
                    dataset_name,
                    data_root,
                    input_dim,
                    search_epochs,
                    train_fraction,
                    seed,
                    str(worker_device),
                    fast_trainer,
                    patience,
                    gradient_clip_max_norm,
                )
            )
        for future in as_completed(futures):
            evaluation = future.result()
            parallel_evaluations.append(evaluation)
            log_evaluation_finished(evaluation)

    return sorted(parallel_evaluations, key=lambda evaluation: evaluation.job.trial_index)


def log_evaluation_finished(evaluation: TrialEvaluation) -> None:
    """Print immediate progress when one trial finishes evaluating."""
    trial = evaluation.job.trial_index
    elapsed = evaluation.elapsed_seconds
    if evaluation.error is not None:
        print(f"trial {trial} evaluation failed after {elapsed:.1f}s | {evaluation.error}", flush=True)
        return
    if evaluation.validation_loss is None:
        print(f"trial {trial} evaluation finished after {elapsed:.1f}s without validation loss", flush=True)
        return
    print(
        f"trial {trial} evaluation finished after {elapsed:.1f}s | "
        f"validation loss {evaluation.validation_loss:.4f}",
        flush=True,
    )


def evaluate_job(
    job: TrialJob,
    *,
    dataset: MoleculeNet,
    input_dim: int,
    search_epochs: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
    fast_trainer: bool = False,
    patience: int | None = None,
    gradient_clip_max_norm: float | None = None,
) -> TrialEvaluation:
    """Evaluate one trial against an already loaded dataset."""
    started = time.perf_counter()
    print_trial_started(job, device, search_epochs)
    try:
        validation_loss = objective(
            job.parameters,
            job.trial_index,
            dataset=dataset,
            input_dim=input_dim,
            search_epochs=search_epochs,
            train_fraction=train_fraction,
            seed=seed,
            device=device,
            fast_trainer=fast_trainer,
            patience=patience,
            gradient_clip_max_norm=gradient_clip_max_norm,
        )
    except Exception as exc:
        return TrialEvaluation(
            job=job,
            validation_loss=None,
            elapsed_seconds=time.perf_counter() - started,
            error=repr(exc),
        )

    return TrialEvaluation(
        job=job,
        validation_loss=validation_loss,
        elapsed_seconds=time.perf_counter() - started,
    )


def print_runtime_diagnostics(label: str, device: torch.device, parallel_trials: int) -> None:
    """Print runtime resource diagnostics for Slurm logs."""
    print(f"=== {label} runtime diagnostics ===", flush=True)
    print(f"requested device: {device}", flush=True)
    print(f"parallel_trials: {parallel_trials}", flush=True)
    print(f"torch version: {torch.__version__}", flush=True)
    print(f"torch cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"torch cuda device count: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gib = props.total_memory / 1024**3
            print(
                f"cuda:{index}: {props.name} | capability={props.major}.{props.minor} "
                f"| memory={memory_gib:.1f} GiB",
                flush=True,
            )
    print(f"torch num threads: {torch.get_num_threads()}", flush=True)
    print(f"torch interop threads: {torch.get_num_interop_threads()}", flush=True)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS",
        "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        print(f"{name}={os.environ.get(name, '<unset>')}", flush=True)
    print("=== end runtime diagnostics ===", flush=True)


def configure_torch_runtime() -> None:
    """Tune PyTorch runtime defaults for small Slurm GPU jobs."""
    num_threads = positive_int_from_env(
        ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"),
        default=1,
    )
    interop_threads = positive_int_from_env(
        ("ML4CHEM_INTEROP_THREADS",),
        default=1,
    )
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as exc:
        print(f"could not set torch interop threads: {exc}", flush=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(
        "configured torch runtime | "
        f"num_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"matmul_precision={torch.get_float32_matmul_precision()}",
        flush=True,
    )


def positive_int_from_env(names: tuple[str, ...], *, default: int) -> int:
    """Read the first positive integer from a list of environment variables."""
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except ValueError:
            print(f"ignoring non-integer {name}={value!r}", flush=True)
            continue
        if parsed > 0:
            return parsed
        print(f"ignoring non-positive {name}={value!r}", flush=True)
    return default


def print_trial_started(job: TrialJob, device: torch.device, search_epochs: int) -> None:
    """Print trial start details before training begins."""
    print(
        f"trial {job.trial_index} starting | device={device} | "
        f"epochs={search_epochs} | parameters={trial_parameters(job.parameters)}",
        flush=True,
    )


def evaluate_job_in_subprocess(
    job: TrialJob,
    dataset_name: str,
    data_root: str,
    input_dim: int,
    search_epochs: int,
    train_fraction: float,
    seed: int,
    device: str,
    fast_trainer: bool = False,
    patience: int | None = None,
    gradient_clip_max_norm: float | None = None,
) -> TrialEvaluation:
    """Load data inside a worker process and evaluate one trial."""
    configure_torch_runtime()
    dataset = MoleculeNet(root=data_root, name=dataset_name)
    print_runtime_diagnostics(f"trial {job.trial_index} worker", torch.device(device), 1)
    return evaluate_job(
        job,
        dataset=dataset,
        input_dim=input_dim,
        search_epochs=search_epochs,
        train_fraction=train_fraction,
        seed=seed,
        device=torch.device(device),
        fast_trainer=fast_trainer,
        patience=patience,
        gradient_clip_max_norm=gradient_clip_max_norm,
    )


def parallel_device(device: torch.device, worker_index: int) -> torch.device:
    """Assign one CUDA device per worker when a generic CUDA device is requested."""
    if device.type != "cuda":
        return device
    if device.index is not None:
        return device
    return torch.device(f"cuda:{worker_index}")


def objective(
    parameters: dict[str, Any],
    trial_index: int,
    *,
    dataset: MoleculeNet,
    input_dim: int,
    search_epochs: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
    fast_trainer: bool = False,
    patience: int | None = None,
    gradient_clip_max_norm: float | None = None,
) -> float:
    objective_started = time.perf_counter()
    set_seed(seed + trial_index)

    build_started = time.perf_counter()
    model = build_model_config(parameters, input_dim=input_dim).build()
    sampler = SamplerConfig(
        train_fraction=train_fraction,
        batch_size=parameters["batch_size"],
    ).build(dataset)
    optimizer = OptimizerConfig(
        optimizer_type=parameters["optimizer_type"],
        learning_rate=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    ).build_for(model)
    print(
        f"trial {trial_index} setup finished | "
        f"model_parameters={sum(p.numel() for p in model.parameters())} "
        f"train_batches={len(sampler.train_loader)} val_batches={len(sampler.val_loader)} "
        f"batch_size={parameters['batch_size']} seconds={time.perf_counter() - build_started:.2f}",
        flush=True,
    )

    trainer: Trainer | TrainerBatch
    if fast_trainer:
        trainer = TrainerBatch(
            sampler=sampler,
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.MSELoss(),
            device=device,
            shuffle_seed=trial_index,
            gradient_clip_max_norm=gradient_clip_max_norm,
        )
    else:
        trainer = Trainer(
            sampler=sampler,
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.MSELoss(),
            device=device,
            gradient_clip_max_norm=gradient_clip_max_norm,
        )
    train_started = time.perf_counter()
    epoch_log_every = epoch_log_interval(search_epochs)
    trainer.run_experiment(
        max_epochs=search_epochs,
        log_every=epoch_log_every,
        progress_log_every=epoch_log_every,
        patience=patience,
        show_progress=False,
    )
    train_seconds = time.perf_counter() - train_started
    print(f"trial {trial_index} training finished | seconds={train_seconds:.1f}", flush=True)

    validation_started = time.perf_counter()
    finite_val_losses = [v for v in trainer.val_losses if math.isfinite(v)]
    final_val = trainer.validation_loss()
    if finite_val_losses:
        validation_loss = min(min(finite_val_losses), final_val if math.isfinite(final_val) else float("inf"))
    else:
        validation_loss = final_val
    validation_seconds = time.perf_counter() - validation_started
    print(
        f"trial {trial_index} validation finished | "
        f"best_val_loss={validation_loss:.4f} final_val_loss={final_val:.4f} "
        f"seconds={validation_seconds:.2f} "
        f"objective_seconds={time.perf_counter() - objective_started:.1f}",
        flush=True,
    )
    return validation_loss


def epoch_log_interval(search_epochs: int) -> int | None:
    """Return the configured epoch logging interval for debug runs."""
    raw_value = os.environ.get("ML4CHEM_EPOCH_LOG_EVERY")
    if raw_value is None or raw_value == "":
        return None
    try:
        interval = int(raw_value)
    except ValueError:
        print(f"ignoring invalid ML4CHEM_EPOCH_LOG_EVERY={raw_value!r}", flush=True)
        return None
    if interval <= 0:
        return None
    return min(interval, search_epochs)


def next_candidates(
    search_space: HyperparameterSearchSpace,
    completed_trials: list[dict[str, Any]],
    *,
    initial_trials: int,
    batch_size: int,
    raw_samples: int,
    num_restarts: int,
    seed: int,
) -> torch.Tensor:
    """Generate the next candidate tensor(s) for BO.

    Parameters
    ----------
    search_space : HyperparameterSearchSpace
        Search space definition.
    completed_trials : list[dict[str, Any]]
        Completed trial rows with decoded vectors and losses.
    initial_trials : int
        Number of random warmup trials.
    batch_size : int
        Number of candidates to propose.
    raw_samples : int
        Number of raw samples for acquisition optimization.
    num_restarts : int
        Number of restart points for acquisition optimization.
    seed : int
        RNG seed.

    Returns
    -------
    torch.Tensor
        Candidate tensor in normalized coordinates.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if len(completed_trials) < initial_trials:
        initial_x = search_space.sample(initial_trials, seed=seed)
        start = len(completed_trials)
        stop = min(initial_trials, start + batch_size)
        return initial_x[start:stop]

    try:
        from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        try:
            from botorch.acquisition import LogExpectedImprovement
            from botorch.acquisition.logei import qLogExpectedImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from botorch.optim import optimize_acqf
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError:
            raise ImportError(
                "BoTorch optimization requires botorch and gpytorch. "
                "Install them with: uv add botorch gpytorch"
            ) from exc

    train_x = torch.tensor([row["x"] for row in completed_trials], dtype=torch.double)
    train_y = torch.tensor(
        [[-float(row["validation_loss"])] for row in completed_trials],
        dtype=torch.double,
    )
    gp = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    if batch_size == 1:
        acquisition = LogExpectedImprovement(
            model=gp,
            best_f=train_y.max(),
        )
    else:
        acquisition = qLogExpectedImprovement(
            model=gp,
            best_f=train_y.max(),
        )

    next_x, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=search_space.bounds.to(dtype=torch.double),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return next_x.detach()


def next_candidate(
    search_space: HyperparameterSearchSpace,
    completed_trials: list[dict[str, Any]],
    *,
    initial_trials: int,
    raw_samples: int,
    num_restarts: int,
    seed: int,
) -> torch.Tensor:
    """Generate one BO candidate point."""
    return next_candidates(
        search_space,
        completed_trials,
        initial_trials=initial_trials,
        batch_size=1,
        raw_samples=raw_samples,
        num_restarts=num_restarts,
        seed=seed,
    ).squeeze(0)


def build_search_space() -> HyperparameterSearchSpace:
    """Build the coarse hyperparameter search space."""
    return HyperparameterSearchSpace(
        dimensions=(
            ChoiceDimension("num_layers", (2, 3, 4, 5)),
            ChoiceDimension("hidden_dim", (32, 64, 128, 256, 512)),
            ChoiceDimension("layer_type", tuple(LayerType)),
            ChoiceDimension("activation", tuple(ActivationType)),
            ChoiceDimension("pooling", tuple(PoolingType)),
            ChoiceDimension("optimizer_type", tuple(OptimizerType)),
            LogRangeDimension("learning_rate", 1e-5, 1e-2),
            ChoiceDimension("weight_decay", (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)),
            ChoiceDimension("batch_size", (16, 32, 64, 128, 256)),
            LinearRangeDimension("dropout", 0.0, 0.5),
        )
    )


def build_search_space_fine() -> HyperparameterSearchSpace:
    """Build the narrowed fine-pass hyperparameter search space.

    Pinned dimensions (optimizer_type, weight_decay) are encoded as
    single-element ChoiceDimensions so the trial CSV schema stays identical
    to the coarse pass. The BoTorch GP sees them as constant inputs; the
    ARD-Matérn kernel handles constant dimensions gracefully but the cost
    is a small extra fit overhead per BO step.
    """
    return HyperparameterSearchSpace(
        dimensions=(
            ChoiceDimension("num_layers", (3, 4, 5)),
            ChoiceDimension("hidden_dim", (64, 128)),
            ChoiceDimension("layer_type", (LayerType.GATED, LayerType.GAT)),
            ChoiceDimension("activation", (ActivationType.RELU, ActivationType.TANH)),
            ChoiceDimension("pooling", (PoolingType.ADD, PoolingType.MEAN)),
            ChoiceDimension("optimizer_type", (OptimizerType.ADAM,)),
            LogRangeDimension("learning_rate", 3e-4, 3e-3),
            ChoiceDimension("weight_decay", (1e-5,)),
            ChoiceDimension("batch_size", (16, 32)),
            LinearRangeDimension("dropout", 0.0, 0.25),
        )
    )


SEARCH_SPACE_BUILDERS = {
    "coarse": build_search_space,
    "fine": build_search_space_fine,
}


def build_model_config(parameters: dict[str, Any], *, input_dim: int) -> ModelConfig:
    """Build a model config from decoded hyperparameters."""
    return ModelConfig(
        input_dim=input_dim,
        layers=[
            LayerConfig(
                layer_type=parameters["layer_type"],
                output_dim=parameters["hidden_dim"],
                activation=parameters["activation"],
                dropout=parameters["dropout"],
            )
            for _ in range(parameters["num_layers"])
        ],
        pooling=parameters["pooling"],
    )


def trial_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Serialize trial parameters for persistence."""
    return {
        "num_layers": parameters["num_layers"],
        "hidden_dim": parameters["hidden_dim"],
        "layer_type": parameters["layer_type"].name,
        "activation": parameters["activation"].name,
        "pooling": parameters["pooling"].name,
        "optimizer_type": parameters["optimizer_type"].name,
        "learning_rate": parameters["learning_rate"],
        "weight_decay": parameters["weight_decay"],
        "batch_size": parameters["batch_size"],
        "dropout": parameters["dropout"],
    }


def next_trial_index(trials: list[dict[str, Any]]) -> int:
    """Return the next sequential trial index."""
    if not trials:
        return 0
    return max(int(trial["trial"]) for trial in trials) + 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BoTorch search runs.

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
    parser.add_argument("--initial-trials", type=int, default=INITIAL_TRIALS)
    parser.add_argument("--bo-trials", type=int, default=BO_TRIALS)
    parser.add_argument("--bo-batch-size", type=int, default=BO_BATCH_SIZE)
    parser.add_argument("--parallel-trials", type=int, default=PARALLEL_TRIALS)
    parser.add_argument("--raw-samples", type=int, default=RAW_SAMPLES)
    parser.add_argument("--num-restarts", type=int, default=NUM_RESTARTS)
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
        help="Early-stopping patience in actual epochs (val loss must improve "
             "within this many epochs). Default: disabled. AttentiveFP uses 30.",
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
    if args.initial_trials <= 0:
        raise ValueError(f"initial_trials must be positive, got {args.initial_trials}")
    if args.bo_trials < 0:
        raise ValueError(f"bo_trials cannot be negative, got {args.bo_trials}")
    if args.bo_batch_size <= 0:
        raise ValueError(f"bo_batch_size must be positive, got {args.bo_batch_size}")
    if args.parallel_trials <= 0:
        raise ValueError(f"parallel_trials must be positive, got {args.parallel_trials}")
    if args.search_epochs <= 0:
        raise ValueError(f"search_epochs must be positive, got {args.search_epochs}")
    if args.gradient_clip_max_norm is not None and args.gradient_clip_max_norm <= 0:
        raise ValueError(
            "gradient_clip_max_norm must be positive if set, "
            f"got {args.gradient_clip_max_norm}"
        )
    if args.raw_samples <= 0:
        raise ValueError(f"raw_samples must be positive, got {args.raw_samples}")
    if args.num_restarts <= 0:
        raise ValueError(f"num_restarts must be positive, got {args.num_restarts}")


if __name__ == "__main__":
    main()
