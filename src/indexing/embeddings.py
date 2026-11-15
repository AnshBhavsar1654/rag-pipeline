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
        from langchain_huggingface import HuggingFaceEmbeddings

        encode_kwargs = {"normalize_embeddings": True}
        query_encode_kwargs = dict(encode_kwargs)
        if config.query_instruction:
            query_encode_kwargs["prompt"] = config.query_instruction

        embeddings = HuggingFaceEmbeddings(
            model_name=config.model,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
        )
        print(f"[EMB] Embeddings: HuggingFace ({config.model})")

    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    return embeddings


def max_input_tokens(embeddings: Embeddings) -> int | None:
    """Best-effort lookup of how many tokens the model embeds before truncating."""
    client = getattr(embeddings, "_client", None)
    return getattr(client, "max_seq_length", None)
