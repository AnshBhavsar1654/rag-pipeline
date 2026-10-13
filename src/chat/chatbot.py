"""
RAG Pipeline — Conversational Chatbot

Wraps the RAG pipeline with conversation memory for multi-turn dialogue.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.generation.generator import SOURCES_HEADER
from src.pipeline.rag_pipeline import RAGPipeline


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


def messages_from_gradio(history: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert Gradio's openai-style chat history into ChatMessages."""
    messages: list[ChatMessage] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        # Gradio may hand content back as a list of typed parts
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append(ChatMessage(role=role, content=_strip_sources(content)))
    return messages


class RAGChatbot:
    """Conversational RAG chatbot with sliding-window memory.

    Maintains conversation history and formats it for the LLM context.
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.history: list[ChatMessage] = []
        self._memory_window = pipeline.config.chat.memory_window

    def chat(self, user_message: str) -> str:
        """Process a user message and return an assistant response.

        Args:
            user_message: The user's question or message.

        Returns:
            The assistant's formatted response (with optional sources).
        """
        answer, documents = self.pipeline.query(
            question=user_message,
            chat_history=self._format_history(),
        )
        self._remember(user_message, answer)
        return self.pipeline.format_response(answer, documents)

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """Like :meth:`chat`, but yields the answer in pieces, then the sources."""
        stream, documents = self.pipeline.stream_query(
            question=user_message,
            chat_history=self._format_history(),
        )
        answer = ""
        for piece in stream:
            answer += piece
            yield piece
        self._remember(user_message, answer)

        sources = self.pipeline.format_response("", documents)
        if sources:
            yield sources

    def _remember(self, user_message: str, answer: str) -> None:
        self.history.append(ChatMessage(role="user", content=user_message))
        self.history.append(ChatMessage(role="assistant", content=answer))

        # Trim history to window size
        max_messages = self._memory_window * 2  # Each exchange = 2 messages
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _format_history(self) -> str:
        """Format recent conversation history as a string for the LLM."""
        return format_history(self.history, self._memory_window)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def get_history_for_gradio(self) -> list[dict[str, str]]:
        """Convert history to Gradio chatbot format.

        Returns list of dicts with 'role' and 'content' keys.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history
        ]
