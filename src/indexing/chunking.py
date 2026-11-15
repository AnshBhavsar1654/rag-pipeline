"""
Indexing — Chunk optimization

Two splitters:

* ``recursive``: token-aware recursive splitting on paragraphs, lines and
  sentences up to ``chunk_size``.
* ``semantic`` (Semantic Splitter): embeds every sentence and starts a new
  chunk where the meaning shifts (the distance between neighbouring
  sentences is in the top percentile), so chunks follow topics instead of
  arbitrary lengths. Over-long topics are still capped at ``chunk_size``.

Tables are never split semantically: a table is one unit of meaning.
"""

from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig


def create_splitter(config: ChunkingConfig, chunk_size: int | None = None) -> RecursiveCharacterTextSplitter:
    """Create a recursive text splitter based on configuration."""
    size = chunk_size or config.chunk_size
    overlap = min(config.chunk_overlap, size // 2)

    if config.use_tiktoken:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=config.tiktoken_encoding,
            chunk_size=size,
            chunk_overlap=overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\s*\n|\n(?=\s*(?:#|[-*•]|\d+[.)])\s)")


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, blank lines, headings and list items."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def semantic_groups(sentences: list[str], embeddings: Embeddings, percentile: float) -> list[str]:
    """Join consecutive sentences, breaking where the meaning shifts most."""
    if len(sentences) < 3:
        return [" ".join(sentences)] if sentences else []
    # Embed each sentence with its neighbours for a steadier signal.
    windows = [" ".join(sentences[max(0, i - 1): i + 2]) for i in range(len(sentences))]
    vectors = np.array(embeddings.embed_documents(windows), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, percentile)

    groups, current = [], [sentences[0]]
    for sentence, distance in zip(sentences[1:], distances):
        if distance > threshold:
            groups.append(current)
            current = []
        current.append(sentence)
    groups.append(current)
    return ["\n".join(g) for g in groups]


def split_documents(
    documents: list[Document],
    config: ChunkingConfig,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """Split documents into chunks; metadata is preserved on every chunk."""
    splitter = create_splitter(config)
    semantic = config.method == "semantic" and embeddings is not None

    chunks: list[Document] = []
    for doc in documents:
        if not semantic or doc.metadata.get("type") == "table":
            chunks.extend(splitter.split_documents([doc]))
            continue
        groups = semantic_groups(
            split_sentences(doc.page_content), embeddings, config.semantic_breakpoint_percentile,
        )
        pieces = [Document(page_content=g, metadata=dict(doc.metadata)) for g in groups if g.strip()]
        # Cap each topic at chunk_size; short topics pass through unchanged.
        chunks.extend(splitter.split_documents(pieces))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    method = "semantic" if semantic else "recursive"
    print(f"[SPLIT]  Split {len(documents)} document(s) into {len(chunks)} chunks "
          f"({method}, max size={config.chunk_size}, overlap={config.chunk_overlap})")
    return chunks
