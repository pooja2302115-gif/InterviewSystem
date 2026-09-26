"""Session-aware conversation handling for interview practice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Conversation:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)


class ConversationManager:
    """Maintain bounded user/assistant history and build generation prompts."""

    def __init__(self, generator: Any, *, max_history_turns: int = 6) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        self.generator = generator
        self.max_history_turns = max_history_turns
        self.sessions: dict[str, Conversation] = {}

    def chat(self, message: str, session_id: str | None = None) -> dict[str, str]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty text")
        conversation = self._get_or_create(session_id)
        conversation.messages.append({"role": "user", "content": message.strip()})
        prompt = self.build_prompt(conversation)
        response = self.generator.generate(prompt).strip()
        if not response:
            response = "I could not generate a response. Please try again."
        conversation.messages.append({"role": "assistant", "content": response})
        self._trim(conversation)
        return {"response": response, "session_id": conversation.session_id}

    def build_prompt(self, conversation: Conversation) -> str:
        lines = [
            "System: You are an educational AI interview preparation assistant.",
            "Give accurate, concise explanations and ask useful follow-up questions.",
        ]
        for message in conversation.messages:
            role = message["role"].capitalize()
            lines.append(f"{role}: {message['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        conversation = self.sessions.get(session_id)
        if conversation is None:
            return []
        return list(conversation.messages)

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _get_or_create(self, session_id: str | None) -> Conversation:
        resolved_id = session_id or str(uuid4())
        if resolved_id not in self.sessions:
            self.sessions[resolved_id] = Conversation(session_id=resolved_id)
        return self.sessions[resolved_id]

    def _trim(self, conversation: Conversation) -> None:
        max_messages = self.max_history_turns * 2
        if len(conversation.messages) > max_messages:
            conversation.messages[:] = conversation.messages[-max_messages:]
