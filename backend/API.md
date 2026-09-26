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

### `POST /resume/analyze`

Upload a `PDF`, `DOCX`, or `TXT` resume as multipart form data. The endpoint returns regex-extracted contact details, education, skills, projects, experience, certifications, years, and detected sections.

```bash
curl -F "file=@resume.txt" http://localhost:8000/resume/analyze
```

## Run locally

First create a trained or fine-tuned checkpoint. Then configure paths:

```bash
export INTERVIEW_CHECKPOINT=checkpoints/instruction/best_model.pt
export INTERVIEW_VOCABULARY=backend/data/processed/vocab.json
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open the generated OpenAPI documentation at `http://localhost:8000/docs`.

The app imports without a checkpoint and reports `model_loaded: false`; `/chat` returns HTTP 503 until model paths are configured. This makes development and health checks possible before training is complete.

## Scope

Sessions are currently in memory, so they disappear when the process restarts and are not suitable for multiple workers or production persistence. Authentication, rate limiting, durable storage, and request observability belong to later backend work.
