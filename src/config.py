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
    # Measured in tokens when use_tiktoken is true, characters otherwise.
    # Keep chunks within the embedding model's max sequence length, or the
    # tail of every chunk is silently truncated before it is embedded.
    chunk_size: int = 400
    chunk_overlap: int = 60
    use_tiktoken: bool = True
    tiktoken_encoding: str = "cl100k_base"
    # "semantic" breaks chunks where the meaning shifts between sentences;
    # "recursive" splits on paragraphs/lines/sentences by length only.
    method: Literal["recursive", "semantic"] = "semantic"
    semantic_breakpoint_percentile: float = 90


class RaptorConfig(BaseModel):
    enabled: bool = True
    cluster_size: int = 6       # Average chunks per cluster summary
    max_levels: int = 3


class IndexingConfig(BaseModel):
    # none: chunks only | parent_document: search small chunks, answer with
    # their larger parent | summary: parent_document + an LLM summary per parent
    multi_representation: Literal["none", "parent_document", "summary"] = "parent_document"
    parent_chunk_size: int = 1000
    docstore_path: str = "./data/docstore.sqlite"
    raptor: RaptorConfig = Field(default_factory=RaptorConfig)


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "google", "huggingface"] = "huggingface"
    model: str = "BAAI/bge-small-en-v1.5"
    # Prefix prepended to search queries (not documents). Some models, like
    # the BGE family, retrieve noticeably better with an instruction prefix.
    query_instruction: str = ""

    @property
    def signature(self) -> str:
        """Identifies the vector space; stored alongside the index."""
        return f"{self.provider}:{self.model}"


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma"] = "chroma"
    persist_directory: str = "./vectorstore_db"
    collection_name: str = "rag_collection"


class RelationalDBConfig(BaseModel):
    """SQLite database holding every extracted table (Text-to-SQL target)."""
    enabled: bool = True
    path: str = "./data/relational.sqlite"
    max_rows: int = 50          # Rows of a SQL result passed to the LLM


class GraphDBConfig(BaseModel):
    """Knowledge graph built by LLM triple extraction (Text-to-Cypher target)."""
    enabled: bool = True
    path: str = "./data/graph.json"
    extraction_batch_size: int = 3   # Parent chunks per extraction call
    max_results: int = 50


class LLMConfig(BaseModel):
    provider: Literal["openai", "groq", "google", "ollama"] = "google"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    max_retries: int = 3
    base_url: Optional[str] = None
    # Optional second model used when the primary errors (outage, rate limit).
    fallback: Optional[LLMConfig] = None


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


class HybridConfig(BaseModel):
    enabled: bool = True
    fetch_k: int = 20       # Candidates pulled from each of dense and BM25
    rrf_k: int = 60


class CRAGConfig(BaseModel):
    """Corrective RAG: grade documents, re-retrieve when too few are relevant."""
    enabled: bool = True
    min_relevant: int = 1       # Fewer relevant documents than this triggers re-retrieval
    web_search: bool = True     # Also search the web (DuckDuckGo) when re-retrieving
    web_results: int = 4


class RetrievalConfig(BaseModel):
    strategy: Literal[
        "simple", "multi_query", "rag_fusion", "hyde", "step_back", "decomposition"
    ] = "rag_fusion"
    top_k: int = 5
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    multi_query: MultiQueryConfig = Field(default_factory=MultiQueryConfig)
    rag_fusion: RAGFusionConfig = Field(default_factory=RAGFusionConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    crag: CRAGConfig = Field(default_factory=CRAGConfig)


class PromptRoute(BaseModel):
    description: str        # What questions this prompt suits (gets embedded)
    prompt: str = ""        # Extra instructions appended to the system prompt


def _default_prompt_routes() -> dict[str, PromptRoute]:
    return {
        "general": PromptRoute(
            description="A general question about the documents, asking for a fact or detail.",
        ),
        "explain": PromptRoute(
            description="Explain a concept, how something works, or why something happens.",
            prompt="Explain step by step, starting from the basics, and define any jargon.",
        ),
        "compare": PromptRoute(
            description="Compare or contrast things, differences, pros and cons, X versus Y.",
            prompt="Structure the answer as a comparison; use a Markdown table when it helps.",
        ),
        "numbers": PromptRoute(
            description="Numbers, statistics, amounts, totals, prices, dates or rankings.",
            prompt="Lead with the exact figures and units, and show any calculation you make.",
        ),
        "summarize": PromptRoute(
            description="Summarize, give an overview, or list the key points of a document or topic.",
            prompt="Give a concise overview as bullet points of the key points.",
        ),
    }


class RoutingConfig(BaseModel):
    # Logical: the LLM picks the data store(s) - vectorstore, relational DB, graph DB.
    logical: bool = True
    # Semantic: embed the question and pick the most similar prompt below.
    semantic: bool = True
    prompts: dict[str, PromptRoute] = Field(default_factory=_default_prompt_routes)


class QueryConstructionConfig(BaseModel):
    text_to_sql: bool = True     # Relational DB
    text_to_cypher: bool = True  # Graph DB
    self_query: bool = True      # Vectorstore: metadata filters from the question


class RerankingConfig(BaseModel):
    enabled: bool = False
    provider: Literal["cross_encoder", "cohere", "rankgpt", "colbert"] = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20    # How many retrieved chunks the reranker scores
    top_k: int = 5          # How many survive to the LLM


class SelfRAGConfig(BaseModel):
    """Check each answer for grounding and usefulness; retry when it fails."""
    enabled: bool = True
    max_retries: int = 1        # Extra generate / re-retrieve rounds


class GenerationConfig(BaseModel):
    self_rag: SelfRAGConfig = Field(default_factory=SelfRAGConfig)


class ExtractionConfig(BaseModel):
    enabled: bool = True
    ocr_enabled: bool = True
    table_extraction: bool = True
    ocr_language: str = "eng"
    min_table_rows: int = 2
    detect_scanned_pages: bool = True


class ChatConfig(BaseModel):
    system_prompt: str = (
        "You are a helpful AI assistant that answers questions based on the "
        "provided context. If you don't know the answer or the context doesn't "
        "contain relevant information, say so honestly. Always be concise and accurate."
    )
    memory_window: int = 10
    show_sources: bool = True
    # Rewrite follow-ups ("what about its cost?") into standalone questions
    # before retrieval, so the search sees the full intent.
    rewrite_followups: bool = True


class UIConfig(BaseModel):
    title: str = "RAG Chatbot"
    description: str = "Ask questions about your documents"
    host: str = "127.0.0.1"
    port: int = 7860
    upload_dir: str = "./data/uploads"


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

class RAGConfig(BaseModel):
    """Root configuration for the RAG pipeline."""

    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    relational_db: RelationalDBConfig = Field(default_factory=RelationalDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    query_construction: QueryConstructionConfig = Field(default_factory=QueryConstructionConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
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
