"""
RAG Pipeline — Retrieval Helpers

Shared pieces used by several retrieval strategies.
"""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.documents import Document


class SearchRetriever(Protocol):
    """Anything with LangChain's ``invoke(query) -> list[Document]`` shape."""

    def invoke(self, query: str) -> list[Document]: ...


def doc_key(doc: Document) -> str:
    """Identity used to merge the same chunk returned by different searches."""
    return doc.id or doc.page_content


def reciprocal_rank_fusion(
    results: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Apply Reciprocal Rank Fusion across multiple ranked document lists.

    RRF score for each document = Σ 1 / (rank + k)
    where the sum is over all lists that contain the document.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            key = doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                fused_scores[key] = 0.0
            fused_scores[key] += 1.0 / (rank + k)

    sorted_keys = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
    return [doc_map[key] for key in sorted_keys]


_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))\s*")


def parse_query_list(text: str, limit: int | None = None) -> list[str]:
    """Parse an LLM's newline-separated list of queries.

    Models often number or bullet the list despite instructions, wrap items in
    quotes, or add a lead-in line ("Here are 4 queries:"); strip all of that.
    """
    queries: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip().strip('"').strip("'").strip()
        if not line or line.endswith(":"):
            continue
        if line not in queries:
            queries.append(line)
    return queries[:limit] if limit is not None else queries
