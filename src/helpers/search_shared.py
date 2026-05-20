"""Shared search runtime utilities for HPO driver scripts.

This module centralizes common trial-evaluation logic used by multiple
entrypoint scripts so they can depend on `helpers` only.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import time
from typing import Any

import torch
from torch_geometric.datasets import MoleculeNet

from helpers.bayes import ChoiceDimension, HyperparameterSearchSpace, LinearRangeDimension, LogRangeDimension
from helpers.models import ActivationType, LayerConfig, LayerType, ModelConfig, PoolingType
from helpers.rng import set_seed
from helpers.training import OptimizerConfig, OptimizerType, SamplerConfig, Trainer, TrainerBatch


SEED = 42
SEARCH_EPOCHS = 10
TRAIN_FRACTION = 0.8
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
    """Print runtime resource diagnostics for logs."""
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
    """Tune PyTorch runtime defaults for small jobs."""
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
    """Read the first positive integer from environment variables."""
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
    """Assign one CUDA device per worker when generic CUDA requested."""
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
    """Evaluate one parameter set and return validation loss."""
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
        f"model_parameters={sum(param.numel() for param in model.parameters())} "
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
    finite_val_losses = [value for value in trainer.val_losses if math.isfinite(value)]
    final_val = trainer.validation_loss()
    if finite_val_losses:
        validation_loss = min(
            min(finite_val_losses),
            final_val if math.isfinite(final_val) else float("inf"),
        )
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
    """Return configured epoch logging interval for debug runs."""
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
    """Build the narrowed fine-pass hyperparameter search space."""
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
