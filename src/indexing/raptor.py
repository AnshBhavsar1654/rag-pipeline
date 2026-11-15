"""
Indexing — Hierarchical indexing (RAPTOR)

Builds a tree of summaries on top of each document's chunks: cluster the
chunk embeddings, have the LLM summarize each cluster, then cluster and
summarize those summaries again, level by level. All summaries are added to
the vectorstore next to the chunks, so a question can match detail (a chunk)
or a theme spanning many chunks (a higher-level summary).

Trees are built per source file, so indexing a file only summarizes that
file, and its summaries are replaced together with its chunks.
"""

from __future__ import annotations

import math

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


RAPTOR_TYPE = "raptor_summary"

_CLUSTER_PROMPT = ChatPromptTemplate.from_template("""\
Here are related passages from a set of documents. Write a detailed summary \
of them that keeps the key facts, names and numbers, so it can stand in for \
the passages when someone searches for the topics they cover.

Passages:
{passages}

Summary:""")


def kmeans(vectors: np.ndarray, k: int, iterations: int = 25) -> np.ndarray:
    """Cosine k-means with farthest-point initialisation; returns a label per row."""
    x = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    centers = [x[0]]
    for _ in range(1, k):
        sims = np.max(x @ np.array(centers).T, axis=1)
        centers.append(x[int(np.argmin(sims))])
    centers = np.array(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        new_labels = np.argmax(x @ centers.T, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = x[labels == c]
            if len(members):
                center = members.mean(axis=0)
                centers[c] = center / (np.linalg.norm(center) + 1e-12)
    return labels


class RaptorBuilder:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        cluster_size: int = 6,
        max_levels: int = 3,
        max_chars: int = 12000,
    ):
        self._chain = _CLUSTER_PROMPT | llm | StrOutputParser()
        self._embeddings = embeddings
        self._cluster_size = max(2, cluster_size)
        self._max_levels = max_levels
        self._max_chars = max_chars

    def build(
        self,
        leaves: list[Document],
        vectors: list[list[float]] | None = None,
        source: str = "",
    ) -> list[Document]:
        """Return the summary nodes of every level above *leaves* (one source's chunks)."""
        if len(leaves) < 2:
            return []
        nodes = leaves
        matrix = np.array(vectors if vectors is not None
                          else self._embeddings.embed_documents([d.page_content for d in leaves]), dtype=float)
        summaries: list[Document] = []

        for level in range(1, self._max_levels + 1):
            k = max(1, math.ceil(len(nodes) / self._cluster_size))
            labels = kmeans(matrix, k) if k > 1 else np.zeros(len(nodes), dtype=int)
            level_nodes: list[Document] = []
            for c in range(k):
                members = [nodes[i] for i in np.flatnonzero(labels == c)]
                if not members:
                    continue
                text = "\n\n".join(m.page_content for m in members)[: self._max_chars]
                try:
                    summary = self._chain.invoke({"passages": text}).strip()
                except Exception as e:
                    print(f"[WARN]  RAPTOR summary failed at level {level} ({e}); stopping the tree here.")
                    return summaries + level_nodes
                if summary:
                    level_nodes.append(Document(
                        page_content=summary,
                        metadata={
                            "source": source, "type": RAPTOR_TYPE,
                            "raptor_level": level, "cluster_size": len(members),
                        },
                    ))
            summaries.extend(level_nodes)
            print(f"[RAPTOR] Level {level}: {len(nodes)} node(s) → {len(level_nodes)} summary(ies)")
            if len(level_nodes) <= 1:
                break
            nodes = level_nodes
            matrix = np.array(self._embeddings.embed_documents([d.page_content for d in nodes]), dtype=float)
        return summaries
