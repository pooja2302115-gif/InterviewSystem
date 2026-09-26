"""Single-head scaled dot-product causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Compute self-attention while preventing access to future positions."""

    def __init__(
        self,
        embedding_dimension: int,
        max_sequence_length: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.embedding_dimension = embedding_dimension
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
        queries = self.query_projection(embeddings).unsqueeze(1)
        keys = self.key_projection(embeddings).unsqueeze(1)
        values = self.value_projection(embeddings).unsqueeze(1)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.embedding_dimension)
        allowed = self.causal_mask[:sequence_length, :sequence_length]
        allowed = allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, values).squeeze(1)
        output = self.output_projection(attended)
        if return_attention:
            return output, attention_weights.squeeze(1)
        return output

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
