"""Pre-layer-normalized Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from backend.model.feed_forward import FeedForwardNetwork
from backend.model.multi_head_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Combine normalization, multi-head attention, feed-forward, and residual paths."""

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        feed_forward_dimension: int | None = None,
        max_sequence_length: int = 256,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")

        self.embedding_dimension = embedding_dimension
        self.layer_norm1 = nn.LayerNorm(embedding_dimension, eps=layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(embedding_dimension, eps=layer_norm_epsilon)
        self.attention = MultiHeadSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )
        self.feed_forward = FeedForwardNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=feed_forward_dimension,
            dropout=dropout,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(embeddings)
        normalized = self.layer_norm1(embeddings)
        attention_result = self.attention(
            normalized,
            attention_mask,
            return_attention=return_attention,
        )
        if return_attention:
            attention_output, attention_weights = attention_result
        else:
            attention_output = attention_result
        residual = embeddings + attention_output
        feed_forward_output = self.feed_forward(self.layer_norm2(residual))
        output = residual + feed_forward_output
        if return_attention:
            return output, attention_weights
        return output

    def _validate_inputs(self, embeddings: torch.Tensor) -> None:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, sequence, embedding_dimension)")
        if embeddings.shape[-1] != self.embedding_dimension:
            raise ValueError("embedding dimension does not match Transformer block")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must contain floating-point values")
