"""Train and validate the educational InterviewLLM."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer
from backend.training.config import TrainingConfig
from backend.training.dataset import create_dataloader


def run_epoch(
    model: InterviewLLM,
    data_loader: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        valid_mask = loss_mask & target_ids.ne(pad_id)
        valid_tokens = int(valid_mask.sum().item())
        if valid_tokens == 0:
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(input_ids, attention_mask)
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            ).reshape_as(target_ids)
            loss = token_losses[valid_mask].mean()
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        total_loss += float(token_losses[valid_mask].sum().item())
        total_correct += int((predictions[valid_mask] == target_ids[valid_mask]).sum().item())
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("data_loader produced no valid target tokens")
    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "perplexity": math.exp(min(average_loss, 20.0)),
        "accuracy": total_correct / total_tokens,
        "tokens": float(total_tokens),
    }


def evaluate_model(
    model: InterviewLLM,
    data_loader: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        return run_epoch(model, data_loader, pad_id=pad_id, device=device)


def model_configuration(model: InterviewLLM) -> dict[str, int]:
    first_block = model.transformer_blocks[0]
    return {
        "vocabulary_size": model.vocabulary_size,
        "context_length": model.context_length,
        "embedding_dimension": model.embedding_dimension,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "feed_forward_dimension": first_block.feed_forward.hidden_dimension,
        "padding_idx": model.padding_idx,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: InterviewLLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    training_config: TrainingConfig,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_configuration(model),
        "training_config": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "gradient_clip_norm": training_config.gradient_clip_norm,
            "seed": training_config.seed,
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    torch.save(payload, destination)


def train_model(
    model: InterviewLLM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    pad_id: int,
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = config.resolved_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            pad_id=pad_id,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            pad_id=pad_id,
            device=device,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        save_checkpoint(
            config.checkpoint_directory / f"model_epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            training_config=config,
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                config.checkpoint_directory / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                training_config=config,
            )
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"validation_loss={validation_metrics['loss']:.4f} | "
            f"learning_rate={record['learning_rate']:.6g}"
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=Path("backend/data/processed/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("backend/data/processed/validation.jsonl"))
    parser.add_argument("--vocabulary", type=Path, default=Path("backend/data/processed/vocab.json"))
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--embedding-dimension", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feed-forward-dimension", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tokenizer = InterviewTokenizer.load(args.vocabulary)
    train_loader = create_dataloader(
        args.train_data,
        tokenizer,
        args.context_length,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        args.validation_data,
        tokenizer,
        args.context_length,
        args.batch_size,
        shuffle=False,
    )
    model = InterviewLLM(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=args.context_length,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feed_forward_dimension=args.feed_forward_dimension,
        padding_idx=tokenizer.pad_id,
    )
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_directory=args.checkpoint_directory,
        device=args.device,
    )
    history = train_model(model, train_loader, validation_loader, pad_id=tokenizer.pad_id, config=config)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
