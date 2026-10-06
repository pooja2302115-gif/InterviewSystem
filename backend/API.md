# Phase 16: FastAPI Backend

## Endpoints

### `GET /health`

Returns service status and whether a checkpoint-backed model is loaded:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /chat`

Request:

```json
{
  "message": "Explain linked lists.",
  "session_id": "optional-existing-id"
}
```

Response:

```json
{
  "response": "...",
  "session_id": "..."
}
```

When `session_id` is omitted, the server creates one. Subsequent requests with the same ID reuse bounded conversation history.

### `GET /sessions/{session_id}/messages`

Returns the current in-memory message history for debugging and the future analytics layer.

### `DELETE /sessions/{session_id}`

Clears an in-memory conversation session.

### Interview practice

The structured interview bank can be used without loading a model checkpoint:

- `GET /interview/subjects` lists the 30 supported subjects.
- `GET /interview/job-roles` lists the 17 roles and their skills/subjects.
- `POST /interview/questions` selects questions using `job_role`, `skills`, `subject`, `difficulty` (`easy`, `intermediate`, or `advanced`), and `count` (1–50).
- `POST /interview/evaluate` evaluates a candidate answer by `question_id`.

Example question request:

```json
{
  "job_role": "Python Developer",
  "skills": ["algorithms"],
  "subject": "Algorithms",
  "difficulty": "intermediate",
  "count": 3
}
```

Example evaluation request:

```json
{
  "question_id": "algorithms_definition_easy",
  "answer": "Binary search requires sorted input and halves the search interval."
}
```

Evaluation returns a 0–10 score, relevance, detected misconceptions, covered and missing concepts, feedback, and an improved reference answer. It uses order-independent concept coverage, curated synonyms, and contradiction checks rather than exact sentence matching. This local deterministic evaluator is not a general natural-language-inference model, so nuanced answers should still receive human review. Generate or rebuild the dataset with `python backend/data/build_cs_interview_dataset.py`; see [data/QUESTION_BANK_SCHEMA.md](data/QUESTION_BANK_SCHEMA.md) for its record format.

### `POST /resume/analyze`

Upload a `PDF`, `DOCX`, or `TXT` resume as multipart form data. The endpoint returns regex-extracted contact details, education, skills, projects, experience, certifications, years, and detected sections.

```bash
curl -F "file=@resume.txt" http://localhost:8000/resume/analyze
```

## Run locally

First create a trained or fine-tuned checkpoint. Then configure paths:

```bash
export INTERVIEW_CHECKPOINT=checkpoints/current_vocab_instruction/best_model.pt
export INTERVIEW_VOCABULARY=backend/data/processed/vocab.json
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The checkpoint and vocabulary must come from the same training run. To create a compatible fine-tuned checkpoint from a compatible base checkpoint:

```bash
python backend/training/fine_tune.py \
  --base-checkpoint checkpoints/best_model.pt \
  --vocabulary backend/data/processed/vocab.json \
  --checkpoint-directory checkpoints/current_vocab_instruction
```

If the base checkpoint's vocabulary size does not match the tokenizer, retrain the base model with that vocabulary before fine-tuning.

Open the generated OpenAPI documentation at `http://localhost:8000/docs`.

The app imports without a checkpoint and reports `model_loaded: false`; `/chat` returns HTTP 503 until model paths are configured. This makes development and health checks possible before training is complete.

## Scope

Sessions are currently in memory, so they disappear when the process restarts and are not suitable for multiple workers or production persistence. Authentication, rate limiting, durable storage, and request observability belong to later backend work.
