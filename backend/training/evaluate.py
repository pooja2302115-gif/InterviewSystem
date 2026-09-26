"""Evaluate a saved InterviewLLM checkpoint on held-out data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer
from backend.training.dataset import create_dataloader
from backend.training.train import evaluate_model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[InterviewLLM, dict]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain model_config and model_state_dict")
    model = InterviewLLM(**model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_loader: torch.utils.data.DataLoader,
    *,
    pad_id: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    metrics = evaluate_model(model, test_loader, pad_id=pad_id, device=device)
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--test-data", type=Path, default=Path("backend/data/processed/test.jsonl"))
    parser.add_argument("--vocabulary", type=Path, default=Path("backend/data/processed/vocab.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    tokenizer = InterviewTokenizer.load(args.vocabulary)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device=device)
    test_loader = create_dataloader(
        args.test_data,
        tokenizer,
        model.context_length,
        args.batch_size,
        shuffle=False,
    )
    metrics = evaluate_model(model, test_loader, pad_id=tokenizer.pad_id, device=device)
    report = {"checkpoint": str(args.checkpoint), "epoch": int(checkpoint.get("epoch", 0)), **metrics}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
