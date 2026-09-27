# Structured Question Bank Schema

The source question bank is JSONL, one question object per line. Preserve the distinctions between formal, beginner-friendly, and alternate explanations; do not concatenate them into one indistinguishable answer.

## Theory record

```json
{
  "id": "python_list",
  "subject": "programming",
  "topic": "Python lists",
  "question": "What is a list in Python?",
  "book_definition": "A list is a mutable, ordered sequence of objects.",
  "simple_definition": "A list stores multiple values that can be changed.",
  "secondary": "A list is an ordered and changeable collection.",
  "question_type": "theory",
  "seniority": "fresher"
}
```

## Coding record

Coding questions may additionally include:

- `language`: `python`, `java`, `c`, `c++`, or `sql`.
- `explanation`: approach and important assumptions.
- `code`: runnable or clearly scoped code.
- `expected_output`: output for a stated example.
- `time_complexity` and `space_complexity`.
- `interview_questions`: follow-ups with `seniority`, `question`, and `answer`.

Use question types such as `theory`, `coding`, `debugging`, `complexity`, `system_design`, `behavioral`, and `resume`. Seniority labels are prompts for expected depth, not a claim that one answer is universally correct for every candidate.

The corpus generator converts each explanation style into a distinct instruction-response target. Coding records generate a separate solution target that includes explanation, code, output, and complexity. Structured fields are also preserved in the base record and rendered into processed training text by `prepare_data.py`.

## Data quality

Prefer reviewed, non-duplicated examples with explicit assumptions and verified code/output pairs. Do not fabricate complexity or benchmark figures. Split by topic or source group before creating paraphrases so related examples do not leak between training and evaluation. Remove personal resume details or obtain consent before using them.
