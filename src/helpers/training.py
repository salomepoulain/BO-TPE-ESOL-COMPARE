"""Training and evaluation utilities for molecular GNN experiments.

The module provides:

- optimizer and sampler configuration objects with input validation,
- two trainer implementations (standard and pre-collated batch mode), and
- consistent logging/early-stopping behavior for search experiments.

The split between config objects and trainer classes keeps the runtime path
predictable while allowing different search scripts to reuse the same training
semantics.

Running this file directly executes a small local training smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import time
from collections.abc import Iterator

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from helpers.data_utils import dataset_input_dim
from helpers.models import MolecularGNN
from helpers.rng import default_seed, set_seed


def default_device() -> torch.device:
    """Return the default training device.

    Returns
    -------
    torch.device
        CUDA device when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptimizerType(IntEnum):
    """Optimizers supported by the training loop."""

    ADAM = 1
    ADAMW = 2
    RMSPROP = 3
    SGD = 4

    def build(
        self,
        parameters: Iterator[torch.nn.Parameter],
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Build the selected torch optimizer.

        Parameters
        ----------
        parameters : Iterator[torch.nn.Parameter]
            Trainable model parameters.
        learning_rate : float
            Optimizer learning rate.
        weight_decay : float
            Weight decay coefficient.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
        match self:
            case OptimizerType.ADAM:
                return torch.optim.Adam(parameters, **kwargs)
            case OptimizerType.ADAMW:
                return torch.optim.AdamW(parameters, **kwargs)
            case OptimizerType.RMSPROP:
                return torch.optim.RMSprop(parameters, **kwargs)
            case OptimizerType.SGD:
                return torch.optim.SGD(parameters, momentum=0.9, **kwargs)


@dataclass(frozen=True)
class OptimizerConfig:
    """Settings for building an optimizer for a specific model."""

    optimizer_type: OptimizerType
    learning_rate: float
    weight_decay: float = 0.0

    def build_for(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Build an optimizer attached to a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose parameters are optimized.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")

        return self.optimizer_type.build(
            model.parameters(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class Sampler:
    """Splits a dataset into train/val loaders."""

    def __init__(
        self,
        dataset: MoleculeNet,
        *,
        train_fraction: float,
        batch_size: int,
        seed: int | None = None,
    ) -> None:
        train_size = int(len(dataset) * train_fraction)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        self.train_loader = DataLoader(dataset[:train_size], batch_size=batch_size, shuffle=True, generator=generator)
        self.val_loader = DataLoader(dataset[train_size:], batch_size=batch_size, shuffle=False)


@dataclass(frozen=True)
class SamplerConfig:
    """Settings for building train/validation data loaders."""

    train_fraction: float
    batch_size: int
    seed: int | None = None

    def build(self, dataset: MoleculeNet) -> Sampler:
        """Build a sampler for a dataset.

        Parameters
        ----------
        dataset : MoleculeNet
            Dataset to split into train/validation loaders.

        Returns
        -------
        Sampler
            Sampler with initialized loaders.
        """
        if not 0 < self.train_fraction < 1:
            raise ValueError(
                f"train_fraction must be between 0 and 1, got {self.train_fraction}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        seed = default_seed() if self.seed is None else self.seed

        return Sampler(
            dataset,
            train_fraction=self.train_fraction,
            batch_size=self.batch_size,
            seed=seed,
        )


class Trainer:
    """Encapsulates one training epoch."""

    def __init__(
        self,
        *,
        sampler: Sampler,
        model: MolecularGNN,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device | str | None = None,
        console: Console | None = None,
        gradient_clip_max_norm: float | None = None,
    ) -> None:
        training_device = default_device() if device is None else torch.device(device)
        if gradient_clip_max_norm is not None and gradient_clip_max_norm <= 0:
            raise ValueError(
                f"gradient_clip_max_norm must be positive if set, got {gradient_clip_max_norm}"
            )

        self.model = model.to(training_device)
        self.train_loader = sampler.train_loader
        self.val_loader = sampler.val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = training_device
        self.console = Console() if console is None else console
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.losses: list[float] = []
        self.val_losses: list[float] = []
        self.logged_epochs: list[int] = []

    def _do_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            prediction, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
            loss = self.loss_fn(prediction, batch.y.float())
            loss.backward()
            if self.gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_max_norm,
                )
            self.optimizer.step()
            total_loss += float(loss.item())

        return total_loss / len(self.train_loader)

    def validation_loss(self) -> float:
        """Evaluate average validation loss.

        Returns
        -------
        float
            Mean validation loss across batches.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.inference_mode():
            for batch in self.val_loader:
                batch = batch.to(self.device, non_blocking=True)
                prediction, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
                loss = self.loss_fn(prediction, batch.y.float())
                total_loss += float(loss.item())

        return total_loss / len(self.val_loader)

    def run_experiment(
        self,
        *,
        max_epochs: int,
        log_every: int | None = None,
        progress_log_every: int | None = None,
        patience: int | None = None,
        min_delta: float = 1e-6,
        show_progress: bool = True,
    ) -> tuple[MolecularGNN, list[float]]:
        """Train for max_epochs. Returns model and train-loss history.

        Side effect: populates self.losses (train), self.val_losses, and
        self.logged_epochs at the same `log_every` cadence so callers can
        plot a paired train/val curve via self.val_losses.

        Early stopping: if `patience` is set, training stops after `patience`
        actual epochs with no validation improvement of at least `min_delta`.
        Patience is checked at logged epochs only (convert to logged-step
        count internally). Disabled when patience is None.

        After training, `self.early_stopped` (bool), `self.best_val_loss`,
        and `self.best_val_epoch` reflect what happened.
        """
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if patience is not None and patience <= 0:
            raise ValueError(f"patience must be positive if set, got {patience}")

        self.losses = []
        self.val_losses = []
        self.logged_epochs = []
        self.early_stopped = False
        self.best_val_loss = float("inf")
        self.best_val_epoch: int | None = None

        step_size = log_every if (log_every is not None and log_every > 0) else 1
        patience_steps: int | None = (
            None if patience is None else max(1, math.ceil(patience / step_size))
        )
        logged_steps_since_improvement = 0

        def should_log(epoch: int) -> bool:
            return log_every is None or epoch % log_every == 0 or epoch == max_epochs - 1

        def record_and_check(epoch: int, loss: float) -> bool:
            """Record losses; return True if early stop triggered this step."""
            nonlocal logged_steps_since_improvement
            val = self.validation_loss()
            self.losses.append(loss)
            self.val_losses.append(val)
            self.logged_epochs.append(epoch)

            if math.isfinite(val) and val < self.best_val_loss - min_delta:
                self.best_val_loss = val
                self.best_val_epoch = epoch
                logged_steps_since_improvement = 0
            else:
                logged_steps_since_improvement += 1

            if patience_steps is not None and logged_steps_since_improvement >= patience_steps:
                self.early_stopped = True
                return True
            return False

        if not show_progress:
            started = time.perf_counter()
            for epoch in range(max_epochs):
                epoch_started = time.perf_counter()
                loss = self._do_epoch()
                stopped = False
                if should_log(epoch):
                    stopped = record_and_check(epoch, loss)
                if progress_log_every is not None and (
                    epoch == 0
                    or (epoch + 1) % progress_log_every == 0
                    or epoch == max_epochs - 1
                    or stopped
                ):
                    elapsed = time.perf_counter() - started
                    epoch_elapsed = time.perf_counter() - epoch_started
                    val_str = (
                        f" | val_loss={self.val_losses[-1]:.4f}"
                        if self.val_losses and self.logged_epochs[-1] == epoch
                        else ""
                    )
                    print(
                        f"epoch {epoch + 1}/{max_epochs} | "
                        f"train_loss={loss:.4f}{val_str} | "
                        f"epoch_seconds={epoch_elapsed:.3f} | "
                        f"elapsed_seconds={elapsed:.1f}",
                        flush=True,
                    )
                if stopped:
                    print(
                        f"early stopping at epoch {epoch + 1}/{max_epochs} | "
                        f"no val improvement for {patience} epochs | "
                        f"best_val_loss={self.best_val_loss:.4f} "
                        f"at epoch {self.best_val_epoch + 1 if self.best_val_epoch is not None else 'n/a'}",
                        flush=True,
                    )
                    break
            return self.model, self.losses

        # Display a nice Rich Console
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("train loss: {task.fields[train_loss]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        with progress:
            task_id = progress.add_task("Training", total=max_epochs, train_loss="--")
            last_loss = "--"
            for epoch in range(max_epochs):
                loss = self._do_epoch()
                last_loss = f"{loss:.4f}"
                stopped = False
                if should_log(epoch):
                    stopped = record_and_check(epoch, loss)
                progress.update(task_id, advance=1, train_loss=last_loss)
                if stopped:
                    progress.console.print(
                        f"[yellow]early stopping at epoch {epoch + 1}/{max_epochs} "
                        f"(no val improvement for {patience} epochs; "
                        f"best_val_loss={self.best_val_loss:.4f})[/yellow]"
                    )
                    break

            progress.update(task_id, train_loss=last_loss, refresh=True)

        return self.model, self.losses


class TrainerBatch:
    """Fast trainer that pre-collates the entire dataset onto the target device.

    Same math as Trainer (same loss, same model, same optimizer), but:
    - Each split is collated into Batch objects once and moved to device once.
    - Loss is accumulated as a device-side tensor (no per-batch .item() sync).
    - Per-epoch shuffling reorders the pre-built batches rather than re-collating.

    Suitable when the dataset fits comfortably in device memory.
    """

    def __init__(
        self,
        *,
        sampler: Sampler,
        model: MolecularGNN,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device | str | None = None,
        shuffle_seed: int | None = None,
        gradient_clip_max_norm: float | None = None,
    ) -> None:
        training_device = default_device() if device is None else torch.device(device)
        if gradient_clip_max_norm is not None and gradient_clip_max_norm <= 0:
            raise ValueError(
                f"gradient_clip_max_norm must be positive if set, got {gradient_clip_max_norm}"
            )

        self.model = model.to(training_device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = training_device
        self.gradient_clip_max_norm = gradient_clip_max_norm

        self.train_batches: list[Batch] = [
            batch.to(training_device, non_blocking=True) for batch in sampler.train_loader
        ]
        self.val_batches: list[Batch] = [
            batch.to(training_device, non_blocking=True) for batch in sampler.val_loader
        ]

        self._generator = torch.Generator(device="cpu")
        if shuffle_seed is not None:
            self._generator.manual_seed(shuffle_seed)

        self.losses: list[float] = []
        self.val_losses: list[float] = []
        self.logged_epochs: list[int] = []

    def _do_epoch(self) -> torch.Tensor:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)

        order = torch.randperm(len(self.train_batches), generator=self._generator).tolist()
        for index in order:
            batch = self.train_batches[index]
            self.optimizer.zero_grad(set_to_none=True)
            prediction, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
            loss = self.loss_fn(prediction, batch.y.float())
            loss.backward()
            if self.gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_max_norm,
                )
            self.optimizer.step()
            total_loss += loss.detach()

        return total_loss / len(self.train_batches)

    def validation_loss(self) -> float:
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)

        with torch.inference_mode():
            for batch in self.val_batches:
                prediction, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
                loss = self.loss_fn(prediction, batch.y.float())
                total_loss += loss

        return float((total_loss / len(self.val_batches)).item())

    def run_experiment(
        self,
        *,
        max_epochs: int,
        log_every: int | None = None,
        progress_log_every: int | None = None,
        patience: int | None = None,
        min_delta: float = 1e-6,
        show_progress: bool = False,
    ) -> tuple[MolecularGNN, list[float]]:
        """Train for max_epochs. Returns model and train-loss history.

        Side effect: populates self.losses (train), self.val_losses, and
        self.logged_epochs at the same `log_every` cadence so callers can
        plot a paired train/val curve via self.val_losses.

        Early stopping: if `patience` is set, training stops after `patience`
        actual epochs with no validation improvement of at least `min_delta`.
        Patience is checked at logged epochs only.

        After training, `self.early_stopped`, `self.best_val_loss`, and
        `self.best_val_epoch` reflect what happened.
        """
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if patience is not None and patience <= 0:
            raise ValueError(f"patience must be positive if set, got {patience}")

        self.losses = []
        self.val_losses = []
        self.logged_epochs = []
        self.early_stopped = False
        self.best_val_loss = float("inf")
        self.best_val_epoch: int | None = None

        step_size = log_every if (log_every is not None and log_every > 0) else 1
        patience_steps: int | None = (
            None if patience is None else max(1, math.ceil(patience / step_size))
        )
        logged_steps_since_improvement = 0

        def should_log(epoch: int) -> bool:
            return log_every is None or epoch % log_every == 0 or epoch == max_epochs - 1

        started = time.perf_counter()
        for epoch in range(max_epochs):
            epoch_started = time.perf_counter()
            loss_tensor = self._do_epoch()
            logged = should_log(epoch)
            stopped = False

            if logged:
                train_loss = float(loss_tensor.item())
                val_loss = self.validation_loss()
                self.losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.logged_epochs.append(epoch)

                if math.isfinite(val_loss) and val_loss < self.best_val_loss - min_delta:
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch
                    logged_steps_since_improvement = 0
                else:
                    logged_steps_since_improvement += 1

                if patience_steps is not None and logged_steps_since_improvement >= patience_steps:
                    self.early_stopped = True
                    stopped = True

            if progress_log_every is not None and (
                epoch == 0
                or (epoch + 1) % progress_log_every == 0
                or epoch == max_epochs - 1
                or stopped
            ):
                loss_value = self.losses[-1] if logged else float(loss_tensor.item())
                val_str = (
                    f" | val_loss={self.val_losses[-1]:.4f}"
                    if logged
                    else ""
                )
                elapsed = time.perf_counter() - started
                epoch_elapsed = time.perf_counter() - epoch_started
                print(
                    f"epoch {epoch + 1}/{max_epochs} | "
                    f"train_loss={loss_value:.4f}{val_str} | "
                    f"epoch_seconds={epoch_elapsed:.3f} | "
                    f"elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )

            if stopped:
                print(
                    f"early stopping at epoch {epoch + 1}/{max_epochs} | "
                    f"no val improvement for {patience} epochs | "
                    f"best_val_loss={self.best_val_loss:.4f} "
                    f"at epoch {self.best_val_epoch + 1 if self.best_val_epoch is not None else 'n/a'}",
                    flush=True,
                )
                break

        return self.model, self.losses


if __name__ == "__main__":
    from torch_geometric.datasets import MoleculeNet
    from helpers.models import (
        ActivationType, LayerConfig,
        LayerType, ModelConfig, PoolingType,
    )

    set_seed()

    dataset = MoleculeNet(root="data", name="ESOL")
    num_features = dataset_input_dim(dataset)
    
    # build the sampler
    sampler = SamplerConfig(
        train_fraction=0.8,
        batch_size=32,
    ).build(dataset)
    
    # build the model
    model = ModelConfig(
        input_dim=num_features,
        layers=[
            LayerConfig(LayerType.GCN, output_dim=64, activation=ActivationType.RELU),
            LayerConfig(LayerType.GCN, output_dim=64, activation=ActivationType.RELU),
        ],
        pooling=PoolingType.MEAN_MAX, # 3 types
    ).build()

    # build the optimizer
    optimizer = OptimizerConfig(
        optimizer_type=OptimizerType.ADAM,
        learning_rate=0.001,
        weight_decay=1e-4,
    ).build_for(model)

    # plug into the trainer
    trainer = Trainer(
        sampler=sampler,
        model=model,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
    )

    # train
    trained_model, losses = trainer.run_experiment(max_epochs=10, log_every=2)
    print(f"final loss: {losses[-1]:.4f}")
