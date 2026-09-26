"""Sinusoidal positional encoding for Transformer inputs."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sine/cosine position vectors to token embeddings."""

    def __init__(self, embedding_dimension: int, max_sequence_length: int = 256) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dimension)
        )
        encoding = torch.zeros(max_sequence_length, embedding_dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        cosine_columns = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(positions * frequencies[:cosine_columns])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, embedding_dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError("embedding dimension does not match positional encoding")
        if embeddings.shape[1] > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must contain floating-point values")
        return embeddings + self.encoding[:, : embeddings.shape[1]].to(dtype=embeddings.dtype)
