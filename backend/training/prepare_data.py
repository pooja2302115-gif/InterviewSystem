"""Clean seed JSONL files and render records for tokenizer development."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "id",
    "category",
    "topic",
    "question",
    "answer",
    "difficulty",
    "metadata",
}
SPLITS = ("train", "validation", "test")
TEXT_FIELDS = ("category", "topic", "question", "answer", "example", "language")
DIFFICULTIES = {"beginner", "intermediate", "advanced"}
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(value: str) -> str:
    """Collapse horizontal whitespace without changing line structure."""
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in value.splitlines()]
    return "\n".join(lines).strip()


def normalize_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    cleaned = {normalize_text(str(tag)).lower() for tag in tags if str(tag).strip()}
    return sorted(cleaned)


def normalize_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    cleaned = dict(record)
    for field in TEXT_FIELDS:
        if field in cleaned and isinstance(cleaned[field], str):
            cleaned[field] = normalize_text(cleaned[field])

    cleaned["id"] = normalize_text(str(cleaned["id"]))
    cleaned["category"] = normalize_text(str(cleaned["category"])).lower()
    cleaned["topic"] = normalize_text(str(cleaned["topic"]))
    cleaned["difficulty"] = normalize_text(str(cleaned["difficulty"])).lower()

    metadata = dict(cleaned.get("metadata", {}))
    metadata["tags"] = normalize_tags(metadata.get("tags"))
    metadata["source"] = normalize_text(str(metadata.get("source", "seed"))).lower()
    cleaned["metadata"] = metadata
    cleaned["source_split"] = split
    cleaned["text"] = render_text(cleaned)
    return cleaned


def render_text(record: dict[str, Any]) -> str:
    """Render one record into a stable prompt/answer text template."""
    sections = [
        f"Category: {record['category']}",
        f"Topic: {record['topic']}",
        f"Difficulty: {record['difficulty']}",
        f"Question: {record['question']}",
        f"Answer: {record['answer']}",
    ]
    if record.get("example"):
        sections.append(f"Example: {record['example']}")
    if record.get("code"):
        language = record.get("language", "text")
        sections.append(f"Code ({language}):\n{record['code'].rstrip()}")
    if record.get("complexity"):
        complexity = record["complexity"]
        if isinstance(complexity, dict):
            details = "; ".join(f"{key}: {value}" for key, value in sorted(complexity.items()))
        else:
            details = str(complexity)
        sections.append(f"Complexity: {details}")
    return "\n".join(sections)


def validate_record(record: Any, path: Path, line_number: int) -> list[str]:
    location = f"{path}:{line_number}"
    if not isinstance(record, dict):
        return [f"{location}: record must be a JSON object"]

    errors = []
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        errors.append(f"{location}: missing fields {sorted(missing)}")
    if not isinstance(record.get("metadata"), dict):
        errors.append(f"{location}: metadata must be an object")
    for field in ("id", "category", "topic", "question", "answer", "difficulty"):
        if field in record and not isinstance(record[field], str):
            errors.append(f"{location}: {field} must be a string")
    if record.get("difficulty", "") not in DIFFICULTIES:
        errors.append(f"{location}: difficulty must be one of {sorted(DIFFICULTIES)}")
    return errors


def load_and_clean(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    duplicate_ids: list[str] = []
    duplicate_content: list[str] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    counts = Counter()

    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        cleaned_records: list[dict[str, Any]] = []
        if not input_path.exists():
            errors.append(f"{input_path}: file does not exist")
            continue

        for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{input_path}:{line_number}: invalid JSON ({exc.msg})")
                counts["malformed"] += 1
                continue

            record_errors = validate_record(record, input_path, line_number)
            if record_errors:
                errors.extend(record_errors)
                counts["invalid"] += 1
                continue

            cleaned = normalize_record(record, split)
            content_key = cleaned["text"]
            if cleaned["id"] in seen_ids:
                duplicate_ids.append(f"{input_path}:{line_number}: {cleaned['id']}")
                counts["duplicate_ids"] += 1
                continue
            if content_key in seen_content:
                duplicate_content.append(f"{input_path}:{line_number}: {cleaned['id']}")
                counts["duplicate_content"] += 1
                continue

            seen_ids.add(cleaned["id"])
            seen_content.add(content_key)
            cleaned_records.append(cleaned)
            counts["cleaned"] += 1

        with output_path.open("w", encoding="utf-8") as output_file:
            for record in cleaned_records:
                output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "counts": dict(counts),
        "duplicate_ids": duplicate_ids,
        "duplicate_content": duplicate_content,
        "errors": errors,
    }
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("backend/data"),
        help="Directory containing train.jsonl, validation.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/data/processed"),
        help="Directory where cleaned JSONL files and the report are written.",
    )
    args = parser.parse_args()
    report = load_and_clean(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
