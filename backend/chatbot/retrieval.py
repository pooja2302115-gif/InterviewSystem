"""High-confidence lookup over reviewed interview examples before model generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.chatbot.interview_dataset import InterviewDataset

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
        dataset_path = self.data_directory / "cs_interview_dataset.jsonl"
        role_path = self.data_directory / "cs_interview_job_roles.json"
        self.interview_dataset = (
            InterviewDataset(dataset_path, role_path)
            if dataset_path.is_file() and role_path.is_file()
            else None
        )

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
        structured_answer = self._answer_from_interview_dataset(query)
        if structured_answer is not None:
            return structured_answer
        generated_list = self._generate_question_list(query)
        if generated_list is not None:
            return generated_list
        query_tokens = self._tokens(query)
        if not query_tokens:
            return None
        wants_code = any(word in query.lower() for word in ("code", "implement", "write", "solution"))

        best_record = None
        best_score = 0.0
        candidate_records = self.records
        if wants_code:
            candidate_records = [
                record for record in self.records
                if str(record.get("question_type", "")).lower() == "coding" or record.get("code")
            ]
            if not candidate_records:
                candidate_records = self.records

        for record in candidate_records:
            field_weights = {
                "topic": 3.0,
                "question": 2.5,
                "instruction": 2.0,
                "book_definition": 2.0,
                "simple_definition": 1.8,
                "secondary": 1.5,
                "answer": 1.2,
            }
            total_overlap = 0
            exact_phrase_bonus = 0
            for field, weight in field_weights.items():
                value = str(record.get(field, ""))
                if not value:
                    continue
                tokens = self._tokens(value)
                overlap = len(query_tokens & tokens)
                if overlap:
                    total_overlap += overlap * weight
                for token in query_tokens:
                    pattern = rf"\b{re.escape(token)}s?\b"
                    if re.search(pattern, value.lower()):
                        exact_phrase_bonus += 3.0

            if total_overlap == 0 and exact_phrase_bonus == 0:
                continue
            score = (total_overlap / max(1, len(query_tokens))) + exact_phrase_bonus
            if wants_code:
                if str(record.get("question_type", "")).lower() == "coding":
                    score += 4.0
                if record.get("code"):
                    score += 2.0
            if score > best_score:
                best_record = record
                best_score = score

        should_allow_short_match = len(query_tokens) <= 2 and best_record is not None and best_score > 0.0
        if best_record is None:
            return None
        if best_score < self.minimum_score and not should_allow_short_match:
            return None

        if self._is_definition_or_comparison_query(query) or should_allow_short_match:
            return self._format_structured_definition(best_record, query)
        return self._format_answer(best_record, query)

    def _answer_from_interview_dataset(self, query: str) -> str | None:
        if self.interview_dataset is None:
            return None
        normalized = query.casefold()
        comparison_requested = any(
            marker in normalized
            for marker in ("compare", "comparison", "difference", "diff ", " vs ", "versus", "which is better", "which is best")
        )
        pair_topics = (
            (("binary", "linear"), "binary search vs linear search"),
            (("list", "tuple"), "python list vs tuple"),
            (("queue", "stack"), "queue vs stack for ticket entry"),
        )
        if comparison_requested:
            query_words = set(self._tokens(normalized))
            for aliases, topic in pair_topics:
                if all(any(alias in word or word in alias for word in query_words) for alias in aliases):
                    record = next(
                        (
                            item for item in self.interview_dataset.records
                            if item["topic"].casefold() == topic.casefold()
                            and item["question_type"] == "comparison"
                            and item["difficulty"] == "intermediate"
                        ),
                        None,
                    )
                    if record is not None:
                        return record["reference_answer"]

        evaluation_requested = any(
            marker in normalized
            for marker in ("evaluate", "score my answer", "grade my answer", "rate my answer")
        )
        if evaluation_requested:
            record = self._find_evaluation_record(normalized)
            if record is not None:
                result = self.interview_dataset.evaluate_answer(record["id"], query)
                score = f"{result['score']:g}"
                strengths = "\n".join(f"- {point}" for point in result["strengths"]) or "- No rubric points matched."
                missing = "\n".join(f"- {point}" for point in result["missing_points"]) or "- None"
                mistakes = "\n".join(
                    f"- {item['claim']}: {item['correction']}" for item in result["mistakes"]
                ) or "- No specific mistakes detected."
                improvement = result["improvement"]
                return (
                    f"I would rate this **{score}/10** "
                    f"({'strong' if result['score'] >= 8 else 'partially correct'} interview answer).\n\n"
                    f"Relevance: {result['relevance']}.\n\n"
                    f"### Why it is good\n\n{strengths}\n\n"
                    f"### Mistakes\n\n{mistakes}\n\n"
                    f"### Missing points\n\n{missing}\n\n"
                    f"### Small improvement\n\n{improvement}\n\n"
                    f"### Better interview answer\n\n> {result['improved_answer']}"
                )

        if re.search(r"\blists?\b", normalized) and not re.search(r"\btuples?\b", normalized):
            record = next(
                (
                    item for item in self.interview_dataset.records
                    if item["topic"] == "Python lists"
                    and item["question_type"] == "definition"
                    and item["difficulty"] == "easy"
                ),
                None,
            )
            if record is not None and any(
                marker in normalized for marker in ("what is", "define", "explain", "list")
            ):
                return (
                    f"## {record['topic']}\n\n"
                    f"**1. Proper definition**\n{record['book_definition']}\n\n"
                    f"**2. Simple explanation in easy English**\n{record['simple_answer']}\n\n"
                    f"**3. Real-world or practical example**\n{record['example']}"
                )
        return None

    def _find_evaluation_record(self, query: str) -> dict[str, Any] | None:
        if self.interview_dataset is None:
            return None
        answer_tokens = set(self._tokens(query))
        candidates = [
            record for record in self.interview_dataset.records
            if record["question_type"] == "definition"
        ]
        ranked = []
        for record in candidates:
            topic_tokens = set(self._tokens(record["topic"]))
            question_tokens = set(self._tokens(record["question"]))
            score = len(answer_tokens & topic_tokens) * 4 + len(answer_tokens & question_tokens)
            if score:
                difficulty_bonus = {"easy": 0.3, "intermediate": 0.2, "advanced": 0.1}.get(record["difficulty"], 0.0)
                ranked.append((score + difficulty_bonus, record))
        if not ranked:
            return None
        return max(ranked, key=lambda item: item[0])[1]

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
    def _is_definition_or_comparison_query(query: str) -> bool:
        normalized = query.lower()
        keywords = (
            "what is",
            "define",
            "explain",
            "difference between",
            "compare",
            "pros and cons",
            "advantages and disadvantages",
            "when to use",
            "why use",
            "simple definition",
            "easy explanation",
        )
        return any(keyword in normalized for keyword in keywords)

    def _format_structured_definition(self, record: dict[str, Any], query: str) -> str:
        topic = str(record.get("topic") or record.get("question") or "Concept")
        definition = (
            record.get("book_definition")
            or record.get("simple_definition")
            or record.get("answer")
            or "A concept used in Computer Science and software engineering to solve a common problem."
        )
        simple = record.get("simple_definition") or record.get("secondary") or definition
        example = record.get("example") or record.get("expected_output") or "A practical example is when a developer uses this approach in a real application to reduce duplication or improve efficiency."
        return "\n".join(
            [
            f"## {topic}",
            f"\n**1. Proper definition**\n{definition}",
            f"\n**2. Simple explanation in easy English**\n{simple}",
            f"\n**3. Real-world or practical example**\n{example}",
            ]
        )

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
