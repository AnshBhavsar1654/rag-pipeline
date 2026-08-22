"""
RAG Pipeline — Embedding Model Factory

Provides pluggable embedding models: OpenAI, Google, HuggingFace sentence-transformers.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.config import EmbeddingConfig


def create_embeddings(config: EmbeddingConfig) -> Embeddings:
    """Create an embedding model instance based on configuration."""

    if config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=config.model)
        print(f"[EMB] Embeddings: OpenAI ({config.model})")

    elif config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=config.model)
        print(f"[EMB] Embeddings: Google ({config.model})")

    elif config.provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=config.model)
        print(f"[EMB] Embeddings: HuggingFace ({config.model})")

    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    return embeddings
