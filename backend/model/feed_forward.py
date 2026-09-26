"""Position-wise feed-forward network for Transformer blocks."""

from __future__ import annotations

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """Apply Linear -> GELU -> Dropout -> Linear independently per position."""

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if hidden_dimension is None:
            hidden_dimension = 4 * embedding_dimension
        if hidden_dimension < 1:
            raise ValueError("hidden_dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.input_projection = nn.Linear(embedding_dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dimension, embedding_dimension)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, embedding_dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError("embedding dimension does not match feed-forward network")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must contain floating-point values")
        hidden = self.input_projection(embeddings)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.output_projection(hidden)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
