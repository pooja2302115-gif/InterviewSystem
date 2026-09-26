"""FastAPI service for the interview preparation chatbot."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.chatbot.conversation import ConversationManager
from backend.chatbot.inference import InferenceEngine
from backend.resume.parser import extract_resume_file


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    response: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ChatService:
    def __init__(self, manager: ConversationManager | None = None) -> None:
        self.manager = manager

    @property
    def model_loaded(self) -> bool:
        return self.manager is not None

    def chat(self, request: ChatRequest) -> dict[str, str]:
        if self.manager is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Configure INTERVIEW_CHECKPOINT and INTERVIEW_VOCABULARY.",
            )
        try:
            return self.manager.chat(request.message, request.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(
    *,
    manager: ConversationManager | None = None,
    checkpoint_path: str | Path | None = None,
    vocabulary_path: str | Path | None = None,
    device: torch.device | None = None,
) -> FastAPI:
    service = ChatService(manager)
    if service.manager is None and checkpoint_path and vocabulary_path:
        engine = InferenceEngine.from_checkpoint(checkpoint_path, vocabulary_path, device=device)
        service.manager = ConversationManager(engine)

    app = FastAPI(title="AI Interview Preparation System", version="0.1.0")
    app.state.chat_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=service.model_loaded)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        result = service.chat(request)
        return ChatResponse(**result)

    @app.post("/resume/analyze")
    async def analyze_resume(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".pdf", ".docx"}:
            raise HTTPException(status_code=415, detail="Only PDF, DOCX, and TXT resumes are supported.")
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Resume file must be 5 MB or smaller.")
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                profile = extract_resume_file(temporary_file.name)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"filename": file.filename, "profile": profile}

    @app.get("/sessions/{session_id}/messages")
    def history(session_id: str) -> dict[str, Any]:
        if service.manager is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return {"session_id": session_id, "messages": service.manager.get_history(session_id)}

    @app.delete("/sessions/{session_id}", status_code=204)
    def clear_session(session_id: str) -> None:
        if service.manager is not None:
            service.manager.clear(session_id)

    return app


def _default_app() -> FastAPI:
    checkpoint = os.getenv("INTERVIEW_CHECKPOINT")
    vocabulary = os.getenv("INTERVIEW_VOCABULARY")
    if checkpoint and vocabulary:
        return create_app(checkpoint_path=checkpoint, vocabulary_path=vocabulary)
    return create_app()


app = _default_app()
