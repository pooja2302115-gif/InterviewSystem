"""PyTorch datasets for causal language-model training sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from backend.model.tokenizer import InterviewTokenizer


class CausalTextDataset(Dataset):
    """Turn processed JSONL records into fixed-length shifted input/target pairs."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: InterviewTokenizer,
        context_length: int,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples: list[dict[str, torch.Tensor | str]] = []
        self._build_examples()

    def _build_examples(self) -> None:
        for line_number, line in enumerate(self.data_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record: Any = json.loads(line)
            text = record.get("text")
            record_id = record.get("id", f"line_{line_number}")
            if not isinstance(text, str):
                raise ValueError(f"{self.data_path}:{line_number}: missing text field")
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            self._add_record_windows(record_id, token_ids)

    def _add_record_windows(self, record_id: str, token_ids: list[int]) -> None:
        window_size = self.context_length + 1
        for start in range(0, len(token_ids) - 1, self.context_length):
            window = token_ids[start : start + window_size]
            if len(window) < 2:
                continue
            input_ids = window[:-1]
            target_ids = window[1:]
            valid_length = len(input_ids)
            input_ids = self._pad(input_ids)
            target_ids = self._pad(target_ids)
            attention_mask = [1] * valid_length + [0] * (self.context_length - valid_length)
            loss_mask = [1] * valid_length + [0] * (self.context_length - valid_length)
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "target_ids": torch.tensor(target_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
                    "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
                    "record_id": record_id,
                }
            )

    def _pad(self, values: list[int]) -> list[int]:
        return values + [self.tokenizer.pad_id] * (self.context_length - len(values))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.examples[index]


def create_dataloader(
    data_path: str | Path,
    tokenizer: InterviewTokenizer,
    context_length: int,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = CausalTextDataset(data_path, tokenizer, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
