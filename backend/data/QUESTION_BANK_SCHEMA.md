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

## Computer Science interview dataset

`cs_interview_dataset.jsonl` is a separate, reproducibly generated interview bank covering 30 CS subjects and 17 job roles. Build it from the repository root with:

```bash
python backend/data/build_cs_interview_dataset.py
```

Each question record includes an ID, subject, topic, skills, eligible job roles, difficulty (`easy`, `intermediate`, or `advanced`), question type, and question text. Answer references include a simple answer, formal definition, example, rubric points, and an improved answer. Comparison records explicitly cover both options, differences, advantages, disadvantages, use cases, and a situation-dependent recommendation. Coding records include a sample solution, output, and complexity where applicable. The companion `cs_interview_job_roles.json` maps the 17 supported roles to skills and subjects.

The FastAPI service exposes `GET /interview/subjects`, `GET /interview/job-roles`, `POST /interview/questions`, and `POST /interview/evaluate`. Generated question responses omit reference answers. Evaluation uses order-independent concept coverage, curated synonyms, topic relevance, and known contradiction checks rather than requiring exact sentences or word order. It is local and deterministic, not a general natural-language-inference model; subtle paraphrases and domain-specific claims still need human review.

Selected short definition questions may use weighted rubrics that prioritize essential concepts and treat refinements as smaller additions. For example, a binary-search definition that states the sorted-input precondition and repeated halving can earn 9/10, with comparing against the middle element as the suggested refinement.

## Data quality

Prefer reviewed, non-duplicated examples with explicit assumptions and verified code/output pairs. Do not fabricate complexity or benchmark figures. Split by topic or source group before creating paraphrases so related examples do not leak between training and evaluation. Remove personal resume details or obtain consent before using them.
