"""
RAG Pipeline — Conversational Chatbot

Wraps the RAG pipeline with conversation memory for multi-turn dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.rag_pipeline import RAGPipeline


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str       # "user" or "assistant"
    content: str


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
        # Format recent chat history for context
        chat_history = self._format_history()

        # Run RAG pipeline
        answer, documents = self.pipeline.query(
            question=user_message,
            chat_history=chat_history,
        )

        # Format response with sources
        formatted = self.pipeline.format_response(answer, documents)

        # Update history
        self.history.append(ChatMessage(role="user", content=user_message))
        self.history.append(ChatMessage(role="assistant", content=answer))

        # Trim history to window size
        max_messages = self._memory_window * 2  # Each exchange = 2 messages
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        return formatted

    def _format_history(self) -> str:
        """Format recent conversation history as a string for the LLM."""
        if not self.history:
            return ""

        lines: list[str] = []
        for msg in self.history:
            prefix = "Human" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")

        return "\n".join(lines)

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
