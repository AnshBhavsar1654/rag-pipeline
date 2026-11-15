"""
RAG Pipeline — Hybrid Search (Dense + BM25)

Dense embeddings capture meaning but miss exact tokens (names, codes, rare
terms); BM25 is the reverse. Running both and fusing the rankings with RRF
recovers results either one alone would drop. This is the base search every
retrieval strategy calls for each of its queries.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.retrieval.utils import reciprocal_rank_fusion


_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HybridRetriever:
    """Dense vector search fused with BM25 keyword search."""

    def __init__(
        self,
        vectorstore: VectorStore,
        documents: list[Document],
        k: int = 5,
        fetch_k: int = 20,
        rrf_k: int = 60,
        use_bm25: bool = True,
    ):
        self._vectorstore = vectorstore
        self._k = k
        self._fetch_k = max(fetch_k, k)
        self._rrf_k = rrf_k
        self._documents = documents
        self._bm25 = None

        if use_bm25 and documents:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(d.page_content) for d in documents])

    @property
    def uses_bm25(self) -> bool:
        return self._bm25 is not None

    @property
    def documents(self) -> list[Document]:
        return self._documents

    def _keyword_search(self, query: str, metadata_filter=None) -> list[Document]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        hits = [
            self._documents[i] for i in ranked
            if scores[i] > 0
            and (not metadata_filter or metadata_filter.matches(self._documents[i].metadata))
        ]
        return hits[: self._fetch_k]

    def invoke(self, query: str, metadata_filter=None) -> list[Document]:
        """Search; *metadata_filter* (a self-query MetadataFilter) narrows both halves."""
        where = metadata_filter.to_chroma() if metadata_filter else None
        if self._bm25 is None:
            return self._vectorstore.similarity_search(query, k=self._k, filter=where)

        dense = self._vectorstore.similarity_search(query, k=self._fetch_k, filter=where)
        sparse = self._keyword_search(query, metadata_filter)
        return reciprocal_rank_fusion([dense, sparse], k=self._rrf_k)[: self._k]
