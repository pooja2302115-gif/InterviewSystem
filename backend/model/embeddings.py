"""Token embedding layers for the educational Transformer."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Map token IDs to trainable dense vectors."""

    def __init__(self, vocabulary_size: int, embedding_dimension: int, padding_idx: int = 0) -> None:
        super().__init__()
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if padding_idx < 0 or padding_idx >= vocabulary_size:
            raise ValueError("padding_idx must be within the vocabulary")

        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_idx,
        )
        with torch.no_grad():
            self.embedding.weight[padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token IDs")
        return self.embedding(input_ids)

    @property
    def parameter_count(self) -> int:
        return self.vocabulary_size * self.embedding_dimension
