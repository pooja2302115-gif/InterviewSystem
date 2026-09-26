"""Supervised fine-tune an InterviewLLM on instruction-response data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from backend.model.tokenizer import InterviewTokenizer
from backend.training.config import TrainingConfig
from backend.training.evaluate import load_checkpoint_model
from backend.training.instruction_dataset import create_instruction_dataloader
from backend.training.train import train_model


def fine_tune_model(
    checkpoint_path: str | Path,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    *,
    pad_id: int,
    config: TrainingConfig,
) -> list[dict]:
    device = config.resolved_device()
    model, _ = load_checkpoint_model(checkpoint_path, device=device)
    return train_model(
        model,
        train_loader,
        validation_loader,
        pad_id=pad_id,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--train-data", type=Path, default=Path("backend/data/instruction_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("backend/data/instruction_validation.jsonl"))
    parser.add_argument("--vocabulary", type=Path, default=Path("backend/data/processed/vocab.json"))
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints/instruction"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tokenizer = InterviewTokenizer.load(args.vocabulary)
    model, _ = load_checkpoint_model(args.base_checkpoint, device=device)
    train_loader = create_instruction_dataloader(
        args.train_data,
        tokenizer,
        model.context_length,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = create_instruction_dataloader(
        args.validation_data,
        tokenizer,
        model.context_length,
        args.batch_size,
        shuffle=False,
    )
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_directory=args.checkpoint_directory,
        device=args.device,
    )
    history = train_model(
        model,
        train_loader,
        validation_loader,
        pad_id=tokenizer.pad_id,
        config=config,
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
