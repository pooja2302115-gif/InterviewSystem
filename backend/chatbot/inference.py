"""Autoregressive inference for the custom InterviewLLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer
from backend.training.evaluate import load_checkpoint_model


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int | None = None


class InferenceEngine:
    """Generate text from a saved custom model checkpoint."""

    def __init__(
        self,
        model: InterviewLLM,
        tokenizer: InterviewTokenizer,
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocabulary_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> "InferenceEngine":
        resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint_model(checkpoint_path, device=resolved_device)
        tokenizer = InterviewTokenizer.load(vocabulary_path)
        if model.vocabulary_size != tokenizer.vocabulary_size:
            raise ValueError("checkpoint vocabulary size does not match tokenizer vocabulary")
        return cls(model, tokenizer, device=resolved_device)

    def generate_token_ids(
        self,
        prompt: str,
        *,
        config: GenerationConfig | None = None,
    ) -> list[int]:
        settings = config or GenerationConfig()
        self._validate_generation_config(settings)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated = [self.tokenizer.bos_id, *prompt_ids]
        if len(generated) > self.model.context_length:
            generated = [self.tokenizer.bos_id, *prompt_ids[-(self.model.context_length - 1) :]]
        prompt_length = len(generated)

        with torch.no_grad():
            for _ in range(settings.max_new_tokens):
                input_ids = torch.tensor([generated[-self.model.context_length :]], dtype=torch.long, device=self.device)
                attention_mask = input_ids.ne(self.tokenizer.pad_id)
                logits = self.model(input_ids, attention_mask)[0, -1]
                next_token = self._select_token(logits, settings)
                generated.append(next_token)
                if next_token == self.tokenizer.eos_id:
                    break
        return generated[prompt_length:]

    def generate_text(
        self,
        prompt: str,
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        generated_ids = self.generate_token_ids(prompt, config=config)
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate(self, prompt: str) -> str:
        return self.generate_text(prompt)

    def _select_token(self, logits: torch.Tensor, config: GenerationConfig) -> int:
        if config.temperature == 0.0:
            return int(torch.argmax(logits).item())
        scaled_logits = logits / config.temperature
        if config.top_k is not None:
            values, indices = torch.topk(scaled_logits, config.top_k)
            probabilities = torch.softmax(values, dim=-1)
            selected = torch.multinomial(probabilities, num_samples=1)
            return int(indices[selected].item())
        probabilities = torch.softmax(scaled_logits, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    @staticmethod
    def _validate_generation_config(config: GenerationConfig) -> None:
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if config.temperature < 0.0:
            raise ValueError("temperature cannot be negative")
        if config.top_k is not None and config.top_k < 1:
            raise ValueError("top_k must be positive")


class CallableGenerator:
    """Small adapter useful for injecting generation into conversation tests."""

    def __init__(self, engine: InferenceEngine, config: GenerationConfig | None = None) -> None:
        self.engine = engine
        self.config = config

    def generate(self, prompt: str) -> str:
        return self.engine.generate_text(prompt, config=self.config)
