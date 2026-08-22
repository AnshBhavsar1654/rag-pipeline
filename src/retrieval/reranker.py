"""
RAG Pipeline — Re-ranking Layer

Optional post-retrieval re-ranking using cross-encoder models or Cohere Rerank.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import RerankingConfig


class Reranker:
    """Re-rank retrieved documents using a cross-encoder or Cohere."""

    def __init__(self, config: RerankingConfig):
        self._config = config
        self._compressor = None

        if not config.enabled:
            return

        if config.provider == "cross_encoder":
            self._init_cross_encoder()
        elif config.provider == "cohere":
            self._init_cohere()
        else:
            raise ValueError(f"Unsupported reranker provider: {config.provider}")

    def _init_cross_encoder(self) -> None:
        """Initialize a cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._config.model)
            print(f"[RANK] Reranker: CrossEncoder ({self._config.model})")
        except ImportError:
            print("[WARN]  sentence-transformers not installed. Disabling re-ranking.")
            self._config.enabled = False

    def _init_cohere(self) -> None:
        """Initialize Cohere re-ranking."""
        try:
            from langchain.retrievers.document_compressors import CohereRerank
            self._compressor = CohereRerank(model=self._config.model)
            print(f"[RANK] Reranker: Cohere ({self._config.model})")
        except ImportError:
            print("[WARN]  langchain-cohere not installed. Disabling re-ranking.")
            self._config.enabled = False

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Re-rank documents and return top-k."""
        if not self._config.enabled or not documents:
            return documents

        top_k = self._config.top_k

        if self._config.provider == "cross_encoder":
            return self._rerank_cross_encoder(query, documents, top_k)
        elif self._config.provider == "cohere" and self._compressor:
            return self._rerank_cohere(query, documents, top_k)

        return documents[:top_k]

    def _rerank_cross_encoder(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """Re-rank using cross-encoder similarity scores."""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._cross_encoder.predict(pairs)

        # Sort by score descending
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in scored_docs[:top_k]]

    def _rerank_cohere(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """Re-rank using Cohere Rerank API."""
        reranked = self._compressor.compress_documents(documents, query)
        return list(reranked[:top_k])
