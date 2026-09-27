"""High-confidence lookup over reviewed interview examples before model generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9+#.]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "about", "can", "do", "does", "for", "give", "how",
    "i", "in", "is", "it", "me", "of", "on", "please", "should", "tell", "that",
    "the", "this", "to", "use", "what", "when", "where", "which", "who", "why",
    "with", "explain", "describe", "define", "interview", "question", "simple", "terms",
}


class QuestionBankRetriever:
    """Find a clearly matching training example; return None when confidence is low."""

    def __init__(self, data_directory: str | Path | None = None, *, minimum_score: float = 0.34) -> None:
        self.data_directory = Path(data_directory) if data_directory else Path(__file__).resolve().parents[1] / "data"
        self.minimum_score = minimum_score
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        source_files = (
            "python_fundamentals.jsonl",
            "coding_questions.jsonl",
            "cross_subject_questions.jsonl",
            "full_stack_developer_questions.jsonl",
            "train.jsonl",
        )
        for filename in source_files:
            path = self.data_directory / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        return records

    def answer(self, question: str) -> str | None:
        query = re.sub(r"^\s*\[[^]]+\]\s*", "", question).strip()
        generated_list = self._generate_question_list(query)
        if generated_list is not None:
            return generated_list
        query_tokens = self._tokens(query)
        if not query_tokens:
            return None
        wants_code = any(word in query.lower() for word in ("code", "implement", "write", "solution"))

        best_record = None
        best_score = 0.0
        for record in self.records:
            searchable = " ".join(
                str(record.get(field, ""))
                for field in ("topic", "question", "instruction")
            )
            candidate_tokens = self._tokens(searchable)
            if not candidate_tokens:
                continue
            overlap = len(query_tokens & candidate_tokens)
            if not overlap:
                continue
            recall = overlap / len(query_tokens)
            precision = overlap / len(candidate_tokens)
            score = 2 * precision * recall / (precision + recall)
            if wants_code and record.get("code"):
                score += 0.5
            if score > best_score:
                best_record = record
                best_score = score

        if best_record is None or best_score < self.minimum_score:
            return None
        return self._format_answer(best_record, query)

    def _generate_question_list(self, query: str) -> str | None:
        normalized = query.casefold()
        if not re.search(r"\b(?:generate|enerate|create|list|give|prepare)\b", normalized):
            return None
        count_match = re.search(r"\b(\d{1,2})\s+questions?\b", normalized)
        if not count_match:
            return None
        requested_count = int(count_match.group(1))
        if requested_count < 1:
            return None

        role_records = [
            record
            for record in self.records
            if record.get("job_role", "").casefold() == "full stack developer"
        ]
        if not role_records:
            return None
        requested_skills = [
            skill
            for skill in ("python", "html", "css")
            if re.search(rf"\b{skill}\b", normalized)
        ]
        if requested_skills:
            role_records = [
                record for record in role_records
                if any(skill.casefold() in {item.casefold() for item in record.get("skills", [])} for skill in requested_skills)
            ]
        if "full stack" not in normalized and "full-stack" not in normalized:
            return None

        selected = role_records[:requested_count]
        if len(selected) < requested_count:
            return None
        return "\n".join(f"{index}. {record['question']}" for index, record in enumerate(selected, start=1))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP_WORDS}

    def _format_answer(self, record: dict[str, Any], query: str) -> str:
        query_lower = query.lower()
        question_type = str(record.get("question_type", "")).lower()
        if question_type == "coding" and any(word in query_lower for word in ("code", "implement", "write", "solution")):
            return self._format_coding_answer(record)

        seniority = next((level for level in ("senior", "junior", "fresher") if level in query_lower), None)
        follow_ups = record.get("interview_questions", [])
        if any(term in query_lower for term in ("follow-up", "follow up", "ask me", "interview question")) and isinstance(follow_ups, list):
            candidate = next(
                (item for item in follow_ups if isinstance(item, dict) and (seniority is None or item.get("seniority") == seniority)),
                None,
            )
            if candidate:
                return f"Question: {candidate.get('question', '')}\nAnswer guide: {candidate.get('answer', '')}".strip()

        if any(term in query_lower for term in ("formal", "textbook", "technical definition")) and record.get("book_definition"):
            return str(record["book_definition"])
        if any(term in query_lower for term in ("alternate", "another way", "secondary")) and record.get("secondary"):
            return str(record["secondary"])
        if record.get("simple_definition"):
            return str(record["simple_definition"])
        return str(record.get("answer", ""))

    @staticmethod
    def _format_coding_answer(record: dict[str, Any]) -> str:
        parts = []
        if record.get("explanation"):
            parts.append(str(record["explanation"]))
        elif record.get("answer"):
            parts.append(str(record["answer"]))
        if record.get("code"):
            parts.append(f"```{record.get('language', '')}\n{record['code']}\n```")
        if record.get("expected_output"):
            parts.append(f"Example output: {record['expected_output']}")
        if record.get("time_complexity"):
            parts.append(f"Time complexity: {record['time_complexity']}")
        if record.get("space_complexity"):
            parts.append(f"Space complexity: {record['space_complexity']}")
        return "\n\n".join(parts)
