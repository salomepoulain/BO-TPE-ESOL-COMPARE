"""Randomness utilities for reproducible experiment execution.

These helpers centralize seed handling for Python, NumPy, and PyTorch so all
training/search entry points can share the same reproducibility behavior.
"""

from __future__ import annotations

import random

import numpy as np
import torch


DEFAULT_SEED = 42
_DEFAULT_SEED: int | None = None


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Parameters
    ----------
    seed : int, default=DEFAULT_SEED
        Non-negative seed value.
    """
    global _DEFAULT_SEED

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    _DEFAULT_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_seed() -> int | None:
    """Return the most recently configured default seed.

    Returns
    -------
    int | None
        Last seed passed to :func:`set_seed`, or ``None`` if unset.
    """
    return _DEFAULT_SEED
