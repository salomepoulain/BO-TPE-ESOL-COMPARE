"""Graph neural network components for molecular property prediction.

This module separates architecture configuration from model execution:

- ``LayerType``, ``ActivationType``, and ``PoolingType`` define the supported
  design choices as enums so search code can pass stable, serializable values.
- ``LayerConfig`` and ``ModelConfig`` validate dimensions and constraints early
  and build concrete ``torch.nn.Module`` objects only after checks pass.
- ``MolecularGNN`` stays minimal: stack message-passing layers, pool node
  embeddings to graph embeddings, then apply one linear prediction head.

The architecture is intentionally compact because this project is used for
hyperparameter search. Keeping a consistent backbone makes comparisons across
search methods clearer while still exposing high-impact knobs (depth, width,
operator family, activation, dropout, and pooling).

Running this file directly executes a small toy forward-pass example in the
``if __name__ == "__main__":`` block.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GatedGraphConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


# ── ENUMS ───────────────────────────────────────────────────────────────────

class LayerType(IntEnum):
    """Message-passing layer families supported by the project."""

    GCN = 1
    GRAPH_SAGE = 2
    GAT = 3
    GATED = 4
    GIN = 5

    def build_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a convolution module.

        Parameters
        ----------
        in_channels : int
            Input channel count.
        out_channels : int
            Output channel count.

        Returns
        -------
        nn.Module
            Message-passing module instance.
        """
        match self:
            case LayerType.GCN:
                return GCNConv(in_channels, out_channels)
            case LayerType.GRAPH_SAGE:
                return SAGEConv(in_channels, out_channels)
            case LayerType.GAT:
                return GATConv(in_channels, out_channels, heads=1, concat=False)
            case LayerType.GATED:
                return GatedGraphConv(out_channels, num_layers=1)
            case LayerType.GIN:
                # Xu et al. 2019: GIN uses a 2-layer MLP as its inner transform.
                return GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels),
                    )
                )


class ActivationType(IntEnum):
    """Activation functions used between GNN layers."""

    RELU = 1
    TANH = 2
    GELU = 3
    ELU = 4

    @property
    def function(self) -> Any:
        """Return the activation function.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Element-wise activation callable.
        """
        match self:
            case ActivationType.RELU:
                return F.relu
            case ActivationType.TANH:
                return F.tanh
            case ActivationType.GELU:
                return F.gelu
            case ActivationType.ELU:
                return F.elu


class PoolingType(IntEnum):
    """Graph-level pooling choices after node embeddings are computed."""

    MEAN = 1
    MAX = 2
    MEAN_MAX = 3
    ADD = 4

    def graph_channels(self, node_channels: int) -> int:
        """Return graph embedding width after pooling.

        Parameters
        ----------
        node_channels : int
            Node embedding width before pooling.

        Returns
        -------
        int
            Graph embedding width after pooling.
        """
        match self:
            case PoolingType.MEAN | PoolingType.MAX | PoolingType.ADD:
                return node_channels
            case PoolingType.MEAN_MAX:
                return node_channels * 2

    def pool(self, node_embeddings: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into graph embeddings.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node-level embeddings.
        batch_index : torch.Tensor
            Graph index for each node.

        Returns
        -------
        torch.Tensor
            Graph-level embeddings.
        """
        match self:
            case PoolingType.MEAN:
                return global_mean_pool(node_embeddings, batch_index)
            case PoolingType.MAX:
                return global_max_pool(node_embeddings, batch_index)
            case PoolingType.ADD:
                return global_add_pool(node_embeddings, batch_index)
            case PoolingType.MEAN_MAX:
                return torch.cat(
                    [
                        global_max_pool(node_embeddings, batch_index),
                        global_mean_pool(node_embeddings, batch_index),
                    ],
                    dim=1,
                )



# ── CONFIGS ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LayerConfig:
    """Settings for one hidden message-passing layer."""

    layer_type: LayerType
    output_dim: int
    activation: ActivationType
    dropout: float = 0.0

    def build(self, input_dim: int) -> HiddenLayer:
        """Build this hidden layer once input width is known.

        Parameters
        ----------
        input_dim : int
            Previous layer output width.

        Returns
        -------
        HiddenLayer
            Instantiated hidden layer.
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"layer output_dim must be positive, got {self.output_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.layer_type == LayerType.GATED and input_dim > self.output_dim:
            raise ValueError(
                "GATED layers require input_dim <= output_dim, "
                f"got input_dim={input_dim}, output_dim={self.output_dim}"
            )

        return HiddenLayer(
            self.layer_type,
            input_dim,
            self.output_dim,
            self.activation,
            self.dropout,
        )


@dataclass(frozen=True)
class ModelConfig:
    """Settings for a MolecularGNN."""

    input_dim: int
    layers: list[LayerConfig]
    pooling: PoolingType
    output_dim: int = 1

    def build(self) -> MolecularGNN:
        """Build a model and validate implied dimensions.

        Returns
        -------
        MolecularGNN
            Configured model instance.
        """
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if not self.layers:
            raise ValueError("ModelConfig needs at least one hidden layer")

        current_dim = self.input_dim
        hidden_layers: list[HiddenLayer] = []
        for index, layer_config in enumerate(self.layers, start=1):
            layer = layer_config.build(current_dim)
            hidden_layers.append(layer)
            current_dim = layer_config.output_dim

            if current_dim <= 0:
                raise ValueError(
                    f"layer {index} produced invalid output_dim={current_dim}"
                )

        output_head = OutputHead(
            pre_pooling_dim=current_dim,
            pooling=self.pooling,
            out_channels=self.output_dim,
        )

        expected_graph_dim = self.pooling.graph_channels(current_dim)
        if output_head.linear.in_features != expected_graph_dim:
            raise RuntimeError(
                "OutputHead input dimension mismatch: "
                f"expected {expected_graph_dim}, got {output_head.linear.in_features}"
            )
        if output_head.linear.out_features != self.output_dim:
            raise RuntimeError(
                "OutputHead output dimension mismatch: "
                f"expected {self.output_dim}, got {output_head.linear.out_features}"
            )

        return MolecularGNN(
            layers=hidden_layers,
            output_head=output_head,
        )


# ── LAYERS ──────────────────────────────────────────────────────────────────

class HiddenLayer(nn.Module):  # type: ignore[misc]
    """One message-passing layer: conv op + activation.

    Wraps a convolution and applies activation after.
    Gated layers handle their own activation internally (GRU).
    """

    def __init__(
        self,
        layer_type: LayerType,
        in_channels: int,
        out_channels: int,
        activation: ActivationType,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = layer_type.build_conv(in_channels, out_channels)
        self.activation_fn = activation.function
        self.dropout = nn.Dropout(p=dropout)
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply convolution → dropout → activation.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Graph connectivity.

        Returns
        -------
        torch.Tensor
            Updated node embeddings.
        """
        x = self.conv(x, edge_index)
        x = self.dropout(x)

        if self.layer_type != LayerType.GATED:
            x = self.activation_fn(x)

        return x


class OutputHead(nn.Module):  # type: ignore[misc]
    """Pool node embeddings and predict from the graph embedding."""

    def __init__(
        self,
        pre_pooling_dim: int,
        pooling: PoolingType,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.linear = nn.Linear(self.pooling.graph_channels(pre_pooling_dim), out_channels)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embedding = cast(torch.Tensor, self.pooling.pool(node_embeddings, batch_index))
        predictions = self.linear(graph_embedding)
        return predictions, graph_embedding



# ── MODEL ───────────────────────────────────────────────────────────────────

class MolecularGNN(nn.Module):  # type: ignore[misc]
    """Configurable graph-level GNN for molecular property prediction.

    Built from a list of HiddenLayers. Pools to graph-level, then predicts.
    """

    def __init__(self, 
        layers: list[HiddenLayer],
        output_head: OutputHead,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output_head = output_head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a full forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``[N, F]``.
        edge_index : torch.Tensor
            Edge indices ``[2, E]``.
        batch_index : torch.Tensor
            Graph assignment per node ``[N]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predictions and pooled graph embeddings.
        """
        # Message-passing loop
        for layer in self.layers:
            x = layer(x, edge_index)

        return cast(tuple[torch.Tensor, torch.Tensor], self.output_head(x, batch_index))


if __name__ == "__main__":
    model_config = ModelConfig(
        input_dim=9,
        layers=[
            LayerConfig(LayerType.GCN, output_dim=64, activation=ActivationType.RELU),
            LayerConfig(LayerType.GCN, output_dim=64, activation=ActivationType.RELU),
            LayerConfig(LayerType.GCN, output_dim=64, activation=ActivationType.RELU),
        ],
        pooling=PoolingType.MEAN_MAX, # 3
    )
    model = model_config.build()

    # Toy forward pass: 2 graphs, 5 nodes, 9 features
    x = torch.randn(5, 9)
    edge_index = torch.randint(0, 5, (2, 8))
    batch_index = torch.tensor([0, 0, 0, 1, 1])

    predictions, embeddings = model(x, edge_index, batch_index)
    print(f"predictions: {predictions.shape}")   # [2, 1]
    print(f"embeddings:  {embeddings.shape}")    # [2, 128]
