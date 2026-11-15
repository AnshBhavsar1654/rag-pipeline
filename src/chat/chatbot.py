"""
RAG Pipeline — Conversational Chatbot

Wraps the RAG pipeline with conversation memory for multi-turn dialogue.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.generation.generator import SOURCES_HEADER
from src.pipeline.rag_pipeline import Event, RAGPipeline


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str       # "user" or "assistant"
    content: str


def format_history(messages: list[ChatMessage], memory_window: int) -> str:
    """Format the last *memory_window* exchanges as a transcript for the LLM."""
    recent = messages[-memory_window * 2:] if memory_window > 0 else []
    lines = [
        f"{'Human' if msg.role == 'user' else 'Assistant'}: {msg.content}"
        for msg in recent
    ]
    return "\n".join(lines)


def _strip_sources(content: str) -> str:
    """Drop the appended source list; it's display-only, not conversation."""
    return content.split(f"\n\n{SOURCES_HEADER}", 1)[0].strip()


def messages_from_dicts(history: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert an openai-style ``[{role, content}]`` history (from the web UI)."""
    messages: list[ChatMessage] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append(ChatMessage(role=role, content=_strip_sources(content)))
    return messages


class RAGChatbot:
    """Conversational RAG chatbot with sliding-window memory."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.history: list[ChatMessage] = []
        self._memory_window = pipeline.config.chat.memory_window

    def chat(self, user_message: str) -> str:
        """Process a user message and return the formatted response (with sources)."""
        answer, documents = self.pipeline.query(
            question=user_message,
            chat_history=self._format_history(),
        )
        self._remember(user_message, answer)
        return self.pipeline.format_response(answer, documents)

    def events(self, user_message: str) -> Iterator[Event]:
        """Run the pipeline for *user_message*, yielding its events; remembers the answer."""
        for event in self.pipeline.run(user_message, self._format_history()):
            if event.kind == "done":
                self._remember(user_message, event.answer)
            yield event

    def _remember(self, user_message: str, answer: str) -> None:
        self.history.append(ChatMessage(role="user", content=user_message))
        self.history.append(ChatMessage(role="assistant", content=answer))
        max_messages = self._memory_window * 2  # Each exchange = 2 messages
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _format_history(self) -> str:
        return format_history(self.history, self._memory_window)

    def clear_history(self) -> None:
        self.history.clear()
