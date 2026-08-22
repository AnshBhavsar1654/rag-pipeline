"""
RAG Pipeline — Base Retriever Interface

Defines the abstract protocol for all retrieval strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseRAGRetriever(ABC):
    """Abstract base class for RAG retrieval strategies."""

    @abstractmethod
    def retrieve(self, question: str) -> list[Document]:
        """Retrieve relevant documents for a given question."""
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name of the retrieval strategy."""
        ...
