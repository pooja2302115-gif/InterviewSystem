"""Dataset and DataLoader for response-focused supervised fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from backend.model.tokenizer import InterviewTokenizer


class InstructionDataset(Dataset):
    """Create causal examples whose loss is applied only to response tokens."""

    def __init__(self, data_path: str | Path, tokenizer: InterviewTokenizer, context_length: int) -> None:
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
            instruction = record.get("instruction")
            response = record.get("response")
            record_id = record.get("id", f"line_{line_number}")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{self.data_path}:{line_number}: instruction must be non-empty text")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"{self.data_path}:{line_number}: response must be non-empty text")
            self._add_example(str(record_id), instruction, response)

    def _add_example(self, record_id: str, instruction: str, response: str) -> None:
        prompt = f"Instruction: {instruction.strip()}\nResponse:"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(f"{prompt} {response.strip()}", add_special_tokens=True)
        if len(full_ids) > self.context_length + 1:
            full_ids = full_ids[: self.context_length + 1]
            full_ids[-1] = self.tokenizer.eos_id
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        valid_length = len(input_ids)
        response_start = len(prompt_ids)
        input_ids = self._pad(input_ids)
        target_ids = self._pad(target_ids)
        attention_mask = [1] * valid_length + [0] * (self.context_length - valid_length)
        loss_mask = [int(index >= response_start) for index in range(valid_length)]
        loss_mask.extend([0] * (self.context_length - valid_length))
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


def create_instruction_dataloader(
    data_path: str | Path,
    tokenizer: InterviewTokenizer,
    context_length: int,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = InstructionDataset(data_path, tokenizer, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
