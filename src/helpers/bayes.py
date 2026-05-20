"""Core Bayesian-optimization primitives used by search runners.

This module defines a typed search-space abstraction that maps normalized
vectors to concrete Python hyperparameters, plus a compact BoTorch optimizer
wrapper used for standalone experiments and tests.

The design keeps search-space decoding explicit and deterministic so trial
logging, replay, and cross-run comparisons remain straightforward.

Running this file directly executes a tiny synthetic usage example.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import torch

from helpers.rng import set_seed


@dataclass(frozen=True)
class ChoiceDimension[T]:
    """One discrete hyperparameter dimension."""

    name: str
    values: tuple[T, ...]

    def decode(self, coordinate: torch.Tensor) -> T:
        """Map one normalized coordinate to one configured value.

        Parameters
        ----------
        coordinate : torch.Tensor
            Scalar coordinate in normalized space.

        Returns
        -------
        T
            Decoded categorical value.
        """
        if not self.values:
            raise ValueError(f"{self.name} must have at least one choice")
        index = min(int(float(coordinate) * len(self.values)), len(self.values) - 1)
        return self.values[index]


@dataclass(frozen=True)
class LogRangeDimension:
    """One positive continuous hyperparameter dimension sampled on a log scale."""

    name: str
    low: float
    high: float

    def decode(self, coordinate: torch.Tensor) -> float:
        """Map one normalized coordinate to a log-scaled value.

        Parameters
        ----------
        coordinate : torch.Tensor
            Scalar coordinate in normalized space.

        Returns
        -------
        float
            Decoded value on the configured log range.
        """
        if self.low <= 0 or self.high <= 0 or self.low >= self.high:
            raise ValueError(f"{self.name} has invalid log range: {(self.low, self.high)}")

        log_low = math.log10(self.low)
        log_high = math.log10(self.high)
        return 10 ** (log_low + float(coordinate) * (log_high - log_low))


@dataclass(frozen=True)
class LinearRangeDimension:
    """One continuous hyperparameter dimension sampled uniformly on a linear scale."""

    name: str
    low: float
    high: float

    def decode(self, coordinate: torch.Tensor) -> float:
        """Map one normalized coordinate to a linear-scaled value.

        Parameters
        ----------
        coordinate : torch.Tensor
            Scalar coordinate in normalized space.

        Returns
        -------
        float
            Decoded value on the configured linear range.
        """
        if self.low >= self.high:
            raise ValueError(f"{self.name} has invalid linear range: {(self.low, self.high)}")
        return self.low + float(coordinate) * (self.high - self.low)



@dataclass(frozen=True)
class HyperparameterSearchSpace:
    """Search space that maps BoTorch tensors to named Python values."""

    dimensions: tuple[ChoiceDimension[Any] | LogRangeDimension | LinearRangeDimension, ...]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("HyperparameterSearchSpace needs at least one dimension")

    @property
    def dim(self) -> int:
        """Return the number of optimized normalized dimensions.

        Returns
        -------
        int
            Dimensionality of the normalized search space.
        """
        return len(self.dimensions)

    @property
    def bounds(self) -> torch.Tensor:
        """Return unit-cube bounds for BoTorch.

        Returns
        -------
        torch.Tensor
            ``(2, dim)`` tensor with lower and upper bounds.
        """
        return torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])

    def decode(self, x: torch.Tensor) -> dict[str, Any]:
        """Convert a normalized BoTorch vector into named values.

        Parameters
        ----------
        x : torch.Tensor
            Candidate point in normalized coordinates.

        Returns
        -------
        dict[str, Any]
            Decoded hyperparameter mapping.
        """
        x = x.detach().cpu().flatten()
        if x.numel() != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {x.numel()}")
        x = x.clamp(0.0, 1.0)

        return {
            dimension.name: dimension.decode(coordinate)
            for dimension, coordinate in zip(self.dimensions, x, strict=True)
        }

    def sample(self, n: int, *, seed: int | None = None) -> torch.Tensor:
        """Draw random normalized candidates.

        Parameters
        ----------
        n : int
            Number of candidates to sample.
        seed : int | None, default=None
            Optional deterministic seed.

        Returns
        -------
        torch.Tensor
            Tensor of sampled normalized points.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        return torch.rand(n, self.dim, generator=generator, dtype=torch.double)


@dataclass(frozen=True)
class TrialResult:
    """One evaluated point in the hyperparameter search."""

    parameters: dict[str, Any]
    objective_value: float

    @property
    def score(self) -> float:
        """Return the maximization score derived from objective value.

        Returns
        -------
        float
            Negative objective value.
        """
        return -self.objective_value


@dataclass(frozen=True)
class BayesianOptimizationConfig:
    """Settings for BoTorch-driven hyperparameter optimization."""

    initial_trials: int = 5
    bo_trials: int = 10
    raw_samples: int = 128
    num_restarts: int = 8
    seed: int = 42

    def validate(self) -> None:
        if self.initial_trials <= 0:
            raise ValueError(f"initial_trials must be positive, got {self.initial_trials}")
        if self.bo_trials < 0:
            raise ValueError(f"bo_trials cannot be negative, got {self.bo_trials}")
        if self.raw_samples <= 0:
            raise ValueError(f"raw_samples must be positive, got {self.raw_samples}")
        if self.num_restarts <= 0:
            raise ValueError(f"num_restarts must be positive, got {self.num_restarts}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


class BayesianOptimizer:
    """Run BoTorch Bayesian optimization over a named search space."""

    def __init__(
        self,
        *,
        search_space: HyperparameterSearchSpace,
        objective: Callable[[dict[str, Any], int], float],
        config: BayesianOptimizationConfig,
    ) -> None:
        config.validate()
        self.search_space = search_space
        self.objective = objective
        self.config = config

    def run(self) -> list[TrialResult]:
        """Run initial random trials followed by BO-guided trials.

        Returns
        -------
        list[TrialResult]
            Evaluated trial results in execution order.
        """
        set_seed(self.config.seed)

        try:
            from botorch.acquisition import LogExpectedImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from botorch.optim import optimize_acqf
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise ImportError(
                "BayesianOptimizer requires botorch and gpytorch. "
                "Install them with: uv add botorch gpytorch"
            ) from exc

        train_x = self.search_space.sample(
            self.config.initial_trials,
            seed=self.config.seed,
        )
        results: list[TrialResult] = []
        train_y_values: list[list[float]] = []

        for row in train_x:
            result = self._evaluate(row, len(results))
            results.append(result)
            train_y_values.append([result.score])

        train_y = torch.tensor(train_y_values, dtype=torch.double)

        for _ in range(self.config.bo_trials):
            gp = SingleTaskGP(
                train_x,
                train_y,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            acquisition = LogExpectedImprovement(
                model=gp,
                best_f=train_y.max(),
            )
            next_x, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=self.search_space.bounds.to(dtype=torch.double),
                q=1,
                num_restarts=self.config.num_restarts,
                raw_samples=self.config.raw_samples,
            )

            result = self._evaluate(next_x.squeeze(0), len(results))
            results.append(result)
            train_x = torch.cat([train_x, next_x.detach().to(dtype=torch.double)], dim=0)
            train_y = torch.cat(
                [train_y, torch.tensor([[result.score]], dtype=torch.double)],
                dim=0,
            )

        return results

    def _evaluate(self, x: torch.Tensor, trial_index: int) -> TrialResult:
        parameters = self.search_space.decode(x)
        objective_value = self.objective(parameters, trial_index)
        return TrialResult(
            parameters=parameters,
            objective_value=objective_value,
        )


if __name__ == "__main__":
    search_space = HyperparameterSearchSpace(
        dimensions=(
            ChoiceDimension("num_layers", (1, 2, 3, 4)),
            ChoiceDimension("hidden_dim", (32, 64, 128, 256)),
            LogRangeDimension("learning_rate", 1e-4, 3e-3),
        )
    )

    print("Example 1: decode one BoTorch tensor into named parameters")
    x = torch.tensor(
        [0.2, 0.4, 0.5],
        dtype=torch.double,
    )
    print(search_space.decode(x))

    print("\nExample 2: run a tiny synthetic optimization")

    def objective(parameters: dict[str, Any], trial_index: int) -> float:
        hidden_penalty = abs(parameters["hidden_dim"] - 64) / 64
        layer_penalty = abs(parameters["num_layers"] - 2)
        lr_penalty = abs(math.log10(parameters["learning_rate"]) - math.log10(1e-3))
        return float(hidden_penalty + layer_penalty + lr_penalty)

    optimizer = BayesianOptimizer(
        search_space=search_space,
        objective=objective,
        config=BayesianOptimizationConfig(
            initial_trials=3,
            bo_trials=2,
            raw_samples=16,
            num_restarts=2,
        ),
    )
    results = optimizer.run()
    best = min(results, key=lambda result: result.objective_value)
    print(f"evaluated trials: {len(results)}")
    print(f"best objective: {best.objective_value:.4f}")
    print(best.parameters)
