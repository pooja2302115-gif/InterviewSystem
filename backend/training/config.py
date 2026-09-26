"""Training configuration for the educational language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    checkpoint_directory: Path = Path("checkpoints")
    device: str = "auto"
    seed: int = 42

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
