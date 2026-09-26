"""Multi-head causal self-attention implemented with basic tensor operations."""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Apply scaled dot-product causal attention independently across heads."""

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        max_sequence_length: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if embedding_dimension % num_heads != 0:
            raise ValueError("embedding_dimension must be divisible by num_heads")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads
        self.max_sequence_length = max_sequence_length
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=True)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(embeddings, attention_mask)
        batch_size, sequence_length, _ = embeddings.shape
        queries = self._split_heads(self.query_projection(embeddings), batch_size, sequence_length)
        keys = self._split_heads(self.key_projection(embeddings), batch_size, sequence_length)
        values = self._split_heads(self.value_projection(embeddings), batch_size, sequence_length)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dimension)
        allowed = self.causal_mask[:sequence_length, :sequence_length]
        allowed = allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, values)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dimension,
        )
        output = self.output_projection(attended)
        if return_attention:
            return output, attention_weights
        return output

    def _split_heads(
        self,
        projected: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return projected.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def _validate_inputs(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, embedding_dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError("embedding dimension does not match attention")
        if embeddings.shape[1] > self.max_sequence_length:
            raise ValueError("sequence length exceeds max_sequence_length")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must contain floating-point values")
        if attention_mask is not None:
            expected_shape = embeddings.shape[:2]
            if attention_mask.shape != expected_shape:
                raise ValueError("attention_mask must have shape (batch, sequence)")
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must be boolean")
