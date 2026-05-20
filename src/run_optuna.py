"""Resumable Optuna search driver for molecular GNN hyperparameters.

This module mirrors the BoTorch/random runners but delegates candidate
generation to Optuna samplers (TPE, random, GP, or CMA-ES). Trials are
persisted to disk so failures or preemptions do not lose progress.

The architecture keeps candidate sampling separate from objective evaluation,
which makes search-method comparisons fair and easier to reason about.

Running this file directly starts a command-line Optuna optimization run.

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
--trials : int, default=5
    Target number of completed Optuna trials.
--sampler : {"tpe", "random", "gp", "cmaes"}, default="tpe"
    Optuna sampler backend.
--search-space : {"coarse", "fine"}, default="coarse"
    Hyperparameter space variant.
--n-startup-trials : int, default=20
    Random warmup trial count for model-based samplers.
--study-name : str, default="gnn-hyperparameters"
    Optuna study name.
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
import os
from typing import Any
import time

import optuna
from optuna.trial import TrialState
import torch
from torch_geometric.datasets import MoleculeNet

from helpers.data_utils import dataset_input_dim
from helpers.models import ActivationType, LayerConfig, LayerType, ModelConfig, PoolingType
from helpers.storage import OptimizationStorage
from helpers.rng import set_seed
from helpers.search_shared import configure_torch_runtime, epoch_log_interval
from helpers.training import OptimizerConfig, OptimizerType, SamplerConfig, Trainer, TrainerBatch, default_device


SEED = 42
SEARCH_EPOCHS = 10
TRAIN_FRACTION = 0.8
N_TRIALS = 5
SAMPLER_NAME = "tpe"
OUTPUT_DIR = "output/simulation"


def main() -> None:
    args = parse_args()
    if args.gradient_clip_max_norm is not None and args.gradient_clip_max_norm <= 0:
        raise ValueError(
            "gradient_clip_max_norm must be positive if set, "
            f"got {args.gradient_clip_max_norm}"
        )
    configure_torch_runtime()
    set_seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dataset_started = time.perf_counter()
    dataset = MoleculeNet(root=args.data_root, name=args.dataset_name)
    dataset_seconds = time.perf_counter() - dataset_started
    input_dim = dataset_input_dim(dataset)
    device = default_device() if args.device is None else torch.device(args.device)
    method = f"Optuna-{args.sampler}"
    print_runtime_diagnostics(method, device)
    print(
        "search config | "
        f"trials={args.trials} sampler={args.sampler} "
        f"search_epochs={args.search_epochs}",
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
        method=method,
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
            "sampler": args.sampler,
            "target_trials": args.trials,
            "gradient_clip_max_norm": args.gradient_clip_max_norm,
            "start_time": start_time_iso,
        },
    )

    def objective(trial: optuna.Trial) -> float:
        objective_started = time.perf_counter()
        parameters = SUGGEST_PARAMETERS_BY_SPACE[args.search_space](trial)
        set_seed(args.seed + trial.number)
        print(
            f"trial {trial.number} starting | device={device} | "
            f"epochs={args.search_epochs} | parameters={trial_parameters(parameters)}",
            flush=True,
        )

        build_started = time.perf_counter()
        model = build_model_config(parameters, input_dim=input_dim).build()
        sampler = SamplerConfig(
            train_fraction=args.train_fraction,
            batch_size=parameters["batch_size"],
        ).build(dataset)
        optimizer = OptimizerConfig(
            optimizer_type=parameters["optimizer_type"],
            learning_rate=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        ).build_for(model)
        print(
            f"trial {trial.number} setup finished | "
            f"model_parameters={sum(p.numel() for p in model.parameters())} "
            f"train_batches={len(sampler.train_loader)} val_batches={len(sampler.val_loader)} "
            f"batch_size={parameters['batch_size']} seconds={time.perf_counter() - build_started:.2f}",
            flush=True,
        )

        trainer: Trainer | TrainerBatch
        if args.fast_trainer:
            trainer = TrainerBatch(
                sampler=sampler,
                model=model,
                optimizer=optimizer,
                loss_fn=torch.nn.MSELoss(),
                device=device,
                shuffle_seed=trial.number,
                gradient_clip_max_norm=args.gradient_clip_max_norm,
            )
        else:
            trainer = Trainer(
                sampler=sampler,
                model=model,
                optimizer=optimizer,
                loss_fn=torch.nn.MSELoss(),
                device=device,
                gradient_clip_max_norm=args.gradient_clip_max_norm,
            )
        train_started = time.perf_counter()
        epoch_log_every = epoch_log_interval(args.search_epochs)
        trainer.run_experiment(
            max_epochs=args.search_epochs,
            log_every=epoch_log_every,
            progress_log_every=epoch_log_every,
            patience=args.patience,
            show_progress=False,
        )
        train_seconds = time.perf_counter() - train_started
        print(f"trial {trial.number} training finished | seconds={train_seconds:.1f}", flush=True)
        validation_started = time.perf_counter()
        finite_val_losses = [v for v in trainer.val_losses if math.isfinite(v)]
        final_val = trainer.validation_loss()
        if finite_val_losses:
            validation_loss = min(min(finite_val_losses), final_val if math.isfinite(final_val) else float("inf"))
        else:
            validation_loss = final_val
        validation_seconds = time.perf_counter() - validation_started
        print(
            f"trial {trial.number} validation finished | "
            f"best_val_loss={validation_loss:.4f} final_val_loss={final_val:.4f} "
            f"seconds={validation_seconds:.2f} "
            f"objective_seconds={time.perf_counter() - objective_started:.1f}",
            flush=True,
        )
        if not math.isfinite(validation_loss):
            print(
                f"trial {trial.number} produced non-finite validation loss "
                f"({validation_loss!r}); marking trial as failed.",
                flush=True,
            )
            raise optuna.TrialPruned(f"non-finite validation loss: {validation_loss!r}")
        return validation_loss

    sampler_seed = args.seed + len(storage.completed_trials())
    study = optuna.create_study(
        direction="minimize",
        sampler=build_sampler(args.sampler, sampler_seed, n_startup_trials=args.n_startup_trials),
        study_name=args.study_name,
        storage=f"sqlite:///{storage.run_folder / 'optuna.sqlite'}",
        load_if_exists=True,
    )

    started = time.perf_counter()
    completed_before = count_completed(study)
    remaining_trials = max(args.trials - completed_before, 0)
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[lambda study, trial: export_trial(study, trial, storage, method, log=True)],
        )
    export_study(study, storage, method)
    elapsed_seconds = time.perf_counter() - started

    total_trial_seconds = sum(float(r.get("elapsed_seconds") or 0.0) for r in storage.trials())
    wall_seconds_total = time.perf_counter() - run_started_wall

    if count_completed(study):
        storage.update_metadata(
            {
                "completed_trials": count_completed(study),
                "elapsed_seconds_this_run": elapsed_seconds,
                "best_validation_loss": study.best_value,
                "best_trial": study.best_trial.number,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_trial_seconds": total_trial_seconds,
                "wall_seconds_total": wall_seconds_total,
            }
        )
        print(f"saved Optuna results to {storage.run_folder}")
        print(f"completed trials: {count_completed(study)}")
        print(f"best validation loss: {study.best_value:.4f}")
        print(f"wall time: {wall_seconds_total:.1f}s | trial time sum: {total_trial_seconds:.1f}s")
        print(study.best_params)
    else:
        storage.update_metadata(
            {
                "attempted_trials": len(study.trials),
                "completed_trials": 0,
                "failed_trials": sum(1 for row in storage.trials() if row["status"] == "failed"),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_trial_seconds": total_trial_seconds,
                "wall_seconds_total": wall_seconds_total,
            }
        )
        print(f"saved empty Optuna study to {storage.run_folder}")


def build_model_config(parameters: dict[str, Any], *, input_dim: int) -> ModelConfig:
    """Build a model config from trial parameters.

    Parameters
    ----------
    parameters : dict[str, Any]
        Decoded trial hyperparameters.
    input_dim : int
        Number of node features.

    Returns
    -------
    ModelConfig
        Validated model configuration.
    """
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


def print_runtime_diagnostics(label: str, device: torch.device) -> None:
    """Print runtime resource diagnostics for Slurm logs."""
    print(f"=== {label} runtime diagnostics ===", flush=True)
    print(f"requested device: {device}", flush=True)
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


def suggest_enum(trial: optuna.Trial, name: str, enum_type: Any) -> Any:
    """Sample one enum value from Optuna categorical suggestions."""
    choice = trial.suggest_categorical(name, [value.name for value in enum_type])
    return enum_type[choice]


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one parameter set from the coarse Optuna search space."""
    return {
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512]),
        "layer_type": suggest_enum(trial, "layer_type", LayerType),
        "activation": suggest_enum(trial, "activation", ActivationType),
        "pooling": suggest_enum(trial, "pooling", PoolingType),
        "optimizer_type": suggest_enum(trial, "optimizer_type", OptimizerType),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def suggest_parameters_fine(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one parameter set from the narrowed fine-pass search space."""
    return {
        "num_layers": trial.suggest_int("num_layers", 3, 5),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128]),
        "layer_type": LayerType[trial.suggest_categorical("layer_type", ["GATED", "GAT"])],
        "activation": ActivationType[trial.suggest_categorical("activation", ["RELU", "TANH"])],
        "pooling": PoolingType[trial.suggest_categorical("pooling", ["ADD", "MEAN"])],
        "optimizer_type": OptimizerType[trial.suggest_categorical("optimizer_type", ["ADAM"])],
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
    }


SUGGEST_PARAMETERS_BY_SPACE = {
    "coarse": suggest_parameters,
    "fine": suggest_parameters_fine,
}


def decode_trial_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Decode string-valued Optuna params back into enum values."""
    return {
        **parameters,
        "layer_type": LayerType[parameters["layer_type"]],
        "activation": ActivationType[parameters["activation"]],
        "pooling": PoolingType[parameters["pooling"]],
        "optimizer_type": OptimizerType[parameters["optimizer_type"]],
    }


def build_sampler(
    name: str,
    seed: int,
    *,
    n_startup_trials: int,
) -> optuna.samplers.BaseSampler:
    match name:
        case "tpe":
            return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        case "random":
            return optuna.samplers.RandomSampler(seed=seed)
        case "gp":
            return optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup_trials)
        case "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=n_startup_trials)
        case _:
            raise ValueError(f"unknown sampler: {name}")


def export_trial(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    storage: OptimizationStorage,
    method: str,
    *,
    log: bool = False,
) -> None:
    if trial.state == TrialState.COMPLETE and trial.value is not None:
        storage.save_completed(
            trial=trial.number,
            parameters=trial_parameters(decode_trial_parameters(trial.params)),
            validation_loss=float(trial.value),
            elapsed_seconds=trial.duration.total_seconds() if trial.duration else 0.0,
        )
        if log:
            elapsed_seconds = trial.duration.total_seconds() if trial.duration else 0.0
            print(
                f"trial {trial.number} completed after {elapsed_seconds:.1f}s | "
                f"validation loss {float(trial.value):.4f}",
                flush=True,
            )
    elif trial.state in {TrialState.FAIL, TrialState.PRUNED}:
        storage.save_failed(
            trial=trial.number,
            parameters=trial.params,
            error=f"Optuna trial {trial.state.name.lower()}",
            elapsed_seconds=trial.duration.total_seconds() if trial.duration else 0.0,
        )
        if log:
            elapsed_seconds = trial.duration.total_seconds() if trial.duration else 0.0
            print(
                f"trial {trial.number} {trial.state.name.lower()} after {elapsed_seconds:.1f}s",
                flush=True,
            )

    trials = storage.trials()
    metadata = {
        "attempted_trials": len(study.trials),
        "completed_trials": count_completed(study),
        "failed_trials": sum(1 for row in trials if row["status"] == "failed"),
        "last_update_time": datetime.now(timezone.utc).isoformat(),
    }
    if count_completed(study):
        metadata.update(
            {
                "best_validation_loss": study.best_value,
                "best_trial": study.best_trial.number,
            }
        )
    storage.update_metadata(metadata)


def export_study(study: optuna.Study, storage: OptimizationStorage, method: str) -> None:
    """Export all trials in the study into persistent storage."""
    for trial in study.trials:
        export_trial(study, trial, storage, method)


def trial_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Serialize enum values in trial parameters into strings."""
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


def count_completed(study: optuna.Study) -> int:
    """Count completed trials in an Optuna study."""
    return sum(trial.state == TrialState.COMPLETE for trial in study.trials)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Optuna search runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-name", default="ESOL")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--search-epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--sampler", choices=("tpe", "random", "gp", "cmaes"), default=SAMPLER_NAME)
    parser.add_argument(
        "--search-space",
        choices=("coarse", "fine"),
        default="coarse",
        help="Which search space to sample from. 'coarse' uses the original wide "
             "space; 'fine' uses a narrowed space derived from coarse-pass evidence.",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=20,
        help="Random warmup trials before the sampler starts modelling (TPE/GP/CMA-ES).",
    )
    parser.add_argument("--study-name", default="gnn-hyperparameters")
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


if __name__ == "__main__":
    main()
