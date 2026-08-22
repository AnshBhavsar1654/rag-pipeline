"""
RAG Pipeline — Configuration Loader

Pydantic-based configuration management with YAML parsing and validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DocumentsConfig(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["./sample_docs/"])


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_tiktoken: bool = True
    tiktoken_encoding: str = "cl100k_base"


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "google", "huggingface"] = "google"
    model: str = "models/gemini-embedding-001"


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma"] = "chroma"
    persist_directory: str = "./vectorstore_db"
    collection_name: str = "rag_collection"


class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "google", "ollama"] = "google"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    base_url: Optional[str] = None


class MultiQueryConfig(BaseModel):
    num_queries: int = 4


class RAGFusionConfig(BaseModel):
    num_queries: int = 4
    rrf_k: int = 60


class HyDEConfig(BaseModel):
    hypothesis_prompt: str = (
        "Please write a detailed passage to answer the question:\n"
        "Question: {question}\nPassage:"
    )


class RetrievalConfig(BaseModel):
    strategy: Literal["simple", "multi_query", "rag_fusion", "hyde"] = "rag_fusion"
    top_k: int = 5
    multi_query: MultiQueryConfig = Field(default_factory=MultiQueryConfig)
    rag_fusion: RAGFusionConfig = Field(default_factory=RAGFusionConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)


class RerankingConfig(BaseModel):
    enabled: bool = False
    provider: Literal["cross_encoder", "cohere"] = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 3


class ChatConfig(BaseModel):
    system_prompt: str = (
        "You are a helpful AI assistant that answers questions based on the "
        "provided context. If you don't know the answer or the context doesn't "
        "contain relevant information, say so honestly. Always be concise and accurate."
    )
    memory_window: int = 10
    show_sources: bool = True


class UIConfig(BaseModel):
    title: str = "[DOCS] RAG Chatbot"
    description: str = "Ask questions about your documents"
    theme: str = "soft"
    share: bool = False
    port: int = 7860


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

class RAGConfig(BaseModel):
    """Root configuration for the RAG pipeline."""

    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_SEARCH_PATHS = [
    Path("config/config.yaml"),
    Path("config.yaml"),
]


def load_config(config_path: str | Path | None = None) -> RAGConfig:
    """Load and validate the RAG pipeline configuration.

    Resolution order:
    1. Explicit *config_path* argument
    2. ``RAG_CONFIG`` environment variable
    3. ``config/config.yaml`` (project default)
    4. ``config.yaml`` (fallback)
    """

    path: Path | None = None

    if config_path is not None:
        path = Path(config_path)
    elif env := os.environ.get("RAG_CONFIG"):
        path = Path(env)
    else:
        for candidate in _CONFIG_SEARCH_PATHS:
            if candidate.exists():
                path = candidate
                break

    if path is None or not path.exists():
        print("[WARN]  No config file found — using defaults.")
        return RAGConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    config = RAGConfig(**raw)
    print(f"[OK] Config loaded from {path}")
    return config
