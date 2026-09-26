"""A small educational tokenizer implemented without pretrained components."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
TOKEN_PATTERN = re.compile(r"\n|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
NO_SPACE_BEFORE = frozenset(",.!?;:%)]}>")
NO_SPACE_AFTER = frozenset("([{<")


class InterviewTokenizer:
    """Frequency-based word and symbol tokenizer for the educational corpus."""

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id = dict(token_to_id or {})
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        self._validate_vocabulary()

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def _validate_vocabulary(self) -> None:
        if any(token not in self.token_to_id for token in SPECIAL_TOKENS):
            raise ValueError(f"Vocabulary must contain {SPECIAL_TOKENS}")
        expected_ids = set(range(len(self.token_to_id)))
        if set(self.id_to_token) != expected_ids:
            raise ValueError("Vocabulary IDs must be contiguous integers starting at zero")

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_frequency: int = 1,
        max_vocabulary_size: int | None = None,
    ) -> "InterviewTokenizer":
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        frequencies = Counter(token for text in texts for token in cls.tokenize(text))
        vocabulary = [
            token
            for token, frequency in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
            if frequency >= min_frequency and token not in SPECIAL_TOKENS
        ]
        if max_vocabulary_size is not None:
            available_tokens = max_vocabulary_size - len(SPECIAL_TOKENS)
            if available_tokens < 0:
                raise ValueError("max_vocabulary_size must fit all special tokens")
            vocabulary = vocabulary[:available_tokens]
        token_to_id = {token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)}
        token_to_id.update(
            {token: token_id for token_id, token in enumerate(vocabulary, start=len(SPECIAL_TOKENS))}
        )
        return cls(token_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into words, numbers, symbols, and explicit newline tokens."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return TOKEN_PATTERN.findall(text)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ["<BOS>", *tokens, "<EOS>"]
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length must be positive")
            if len(tokens) > max_length:
                if not truncation:
                    raise ValueError("Encoded sequence exceeds max_length; set truncation=True")
                tokens = tokens[:max_length]
                if add_special_tokens:
                    tokens[-1] = "<EOS>"
            if padding and len(tokens) < max_length:
                tokens.extend(["<PAD>"] * (max_length - len(tokens)))
        elif padding:
            raise ValueError("padding=True requires max_length")
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return self._detokenize(tokens)

    @staticmethod
    def _detokenize(tokens: Sequence[str]) -> str:
        output = ""
        for token in tokens:
            if token == "\n":
                output = output.rstrip() + "\n"
            elif not output or output.endswith("\n"):
                output += token
            elif token in NO_SPACE_BEFORE or output[-1] in NO_SPACE_AFTER:
                output += token
            else:
                output += " " + token
        return output.strip()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"token_to_id": self.token_to_id}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "InterviewTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("token_to_id"), dict):
            raise ValueError("Tokenizer file must contain a token_to_id object")
        token_to_id = {str(token): int(token_id) for token, token_id in payload["token_to_id"].items()}
        return cls(token_to_id)
