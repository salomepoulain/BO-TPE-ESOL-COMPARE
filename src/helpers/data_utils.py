"""Typed dataset helpers for graph data scripts."""

from __future__ import annotations

from typing import Protocol, cast

import torch
from torch_geometric.datasets import MoleculeNet


class GraphWithFeatures(Protocol):
    """Protocol for graph samples exposing node-feature matrix `x`."""

    x: torch.Tensor


def dataset_input_dim(dataset: MoleculeNet) -> int:
    """Return node-feature width from the first dataset sample.

    Parameters
    ----------
    dataset : MoleculeNet
        Graph dataset with samples exposing `x`.

    Returns
    -------
    int
        Feature dimension (`x.shape[1]`).
    """
    sample = cast(GraphWithFeatures, dataset[0])
    return int(sample.x.shape[1])
