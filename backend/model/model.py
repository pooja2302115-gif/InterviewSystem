"""Complete educational Transformer language model."""

from __future__ import annotations

import torch
from torch import nn

from backend.model.embeddings import TokenEmbedding
from backend.model.positional_encoding import SinusoidalPositionalEncoding
from backend.model.transformer_block import TransformerBlock


class InterviewLLM(nn.Module):
    """Predict the next token with a configurable decoder-only Transformer."""

    def __init__(
        self,
        vocabulary_size: int,
        context_length: int = 256,
        embedding_dimension: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        feed_forward_dimension: int | None = None,
        dropout: float = 0.1,
        padding_idx: int = 0,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if padding_idx < 0 or padding_idx >= vocabulary_size:
            raise ValueError("padding_idx must be within the vocabulary")

        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.padding_idx = padding_idx
        self.token_embedding = TokenEmbedding(vocabulary_size, embedding_dimension, padding_idx)
        self.position_encoding = SinusoidalPositionalEncoding(embedding_dimension, context_length)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    max_sequence_length=context_length,
                    dropout=dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dimension, eps=layer_norm_epsilon)
        self.language_model_head = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        self._validate_input_ids(input_ids, attention_mask)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.padding_idx)

        hidden = self.token_embedding(input_ids)
        hidden = self.position_encoding(hidden)
        attention_maps: list[torch.Tensor] = []
        for block in self.transformer_blocks:
            block_result = block(
                hidden,
                attention_mask,
                return_attention=return_attention,
            )
            if return_attention:
                hidden, attention_weights = block_result
                attention_maps.append(attention_weights)
            else:
                hidden = block_result

        hidden = self.final_layer_norm(hidden)
        logits = self.language_model_head(hidden)
        if return_attention:
            return logits, attention_maps
        return logits

    def _validate_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("sequence length exceeds context_length")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token IDs")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have shape (batch, sequence)")
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must be boolean")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
