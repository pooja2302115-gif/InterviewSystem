"""Build a tokenizer vocabulary from the processed training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from backend.model.tokenizer import InterviewTokenizer, SPECIAL_TOKENS
except ModuleNotFoundError:
    from tokenizer import InterviewTokenizer, SPECIAL_TOKENS


def load_training_texts(path: Path) -> list[str]:
    texts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("text"), str):
            raise ValueError(f"{path}:{line_number}: missing processed text field")
        texts.append(record["text"])
    if not texts:
        raise ValueError(f"{path}: no training texts found")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("backend/data/processed/train.jsonl"),
        help="Processed training JSONL file; validation and test data are intentionally excluded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/data/processed/vocab.json"),
        help="Path for the generated vocabulary JSON file.",
    )
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocabulary-size", type=int, default=None)
    args = parser.parse_args()

    texts = load_training_texts(args.input)
    tokenizer = InterviewTokenizer.build(
        texts,
        min_frequency=args.min_frequency,
        max_vocabulary_size=args.max_vocabulary_size,
    )
    tokenizer.save(args.output)
    print(
        json.dumps(
            {
                "training_records": len(texts),
                "vocabulary_size": tokenizer.vocabulary_size,
                "output": str(args.output),
                "special_tokens": list(SPECIAL_TOKENS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
