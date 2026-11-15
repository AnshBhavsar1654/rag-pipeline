"""
Indexing — Chunk optimization

Two splitters:

* ``recursive``: token-aware recursive splitting on paragraphs, lines and
  sentences up to ``chunk_size``.
* ``semantic`` (Semantic Splitter): starts a new chunk where the meaning
  shifts (the distance between neighbouring sentences is in the top
  percentile), so chunks follow topics instead of arbitrary lengths.
  Over-long topics are still capped at ``chunk_size``. Sentences are
  compared either by word statistics (``lexical``: TF-IDF, the TextTiling
  approach, near-instant) or by the embedding model (``embedding``: a little
  sharper, but it runs the model over the whole text a second time).

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
    """Split on sentence ends, blank lines, headings and list items, merging
    fragments (headings, short list items, PDF line debris) into units of at
    least ~100 characters so there is less to embed."""
    units: list[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = (piece or "").strip()
        if not piece:
            continue
        if units and len(units[-1]) < _MIN_UNIT_CHARS:
            units[-1] = f"{units[-1]}\n{piece}"
        else:
            units.append(piece)
    return units


_BOUNDARY_CHARS = 300
_MIN_UNIT_CHARS = 100


def _join(sentences: list[str]) -> str:
    return "\n".join(sentences)


_WORD = re.compile(r"[a-z0-9]{3,}")


def lexical_vectors(per_doc: list[list[str]]) -> list[np.ndarray]:
    """TF-IDF vectors for every sentence, one matrix per document.

    Document frequencies are counted over all sentences, so a word common
    everywhere ("the", the product name) does not register as a topic shift."""
    tokenized = [[_WORD.findall(s.lower()) for s in sents] for sents in per_doc]
    df: dict[str, int] = {}
    n = 0
    for doc in tokenized:
        for words in doc:
            n += 1
            for w in set(words):
                df[w] = df.get(w, 0) + 1
    out = []
    for doc in tokenized:
        vocab = {w: i for i, w in enumerate(sorted({w for words in doc for w in words}))}
        m = np.zeros((len(doc), max(len(vocab), 1)))
        for row, words in enumerate(doc):
            for w in words:
                m[row, vocab[w]] += 1.0
        if vocab:
            idf = np.log((1 + n) / (1 + np.array([df[w] for w in vocab])))  # vocab is insertion-ordered
            m *= idf
        out.append(m)
    return out


def _smooth(vectors: np.ndarray) -> np.ndarray:
    """Average each sentence vector with its neighbours for a steadier signal
    (the same effect as embedding 3-sentence windows, at a third of the cost)."""
    padded = np.vstack([vectors[:1], vectors, vectors[-1:]])
    smooth = padded[:-2] + padded[1:-1] + padded[2:]
    return smooth / (np.linalg.norm(smooth, axis=1, keepdims=True) + 1e-12)


def semantic_groups(
    sentences: list[str],
    vectors: np.ndarray,
    lengths: list[int],
    chunk_size: int,
    percentile: float,
) -> list[str]:
    """Pack consecutive sentences into chunks of up to *chunk_size*, cutting
    at topic shifts: a strong shift (above *percentile*) ends a chunk once it
    has some substance, and a full chunk is cut at its biggest shift."""
    if len(sentences) < 3:
        return [" ".join(sentences)] if sentences else []
    v = _smooth(vectors)
    distances = 1.0 - np.sum(v[:-1] * v[1:], axis=1)       # distances[i]: between i and i+1
    threshold = np.percentile(distances, percentile)
    min_size = chunk_size // 3

    groups: list[str] = []
    start, size = 0, lengths[0]
    i = 1
    while i < len(sentences):
        boundary = i - 1
        if distances[boundary] > threshold and size >= min_size:
            groups.append(_join(sentences[start:i]))
            start, size = i, 0
        elif size + lengths[i] > chunk_size and i > start:
            # Full: cut at the biggest shift that leaves a chunk of substance.
            candidates = [b for b in range(start, i - 1)
                          if sum(lengths[start: b + 1]) >= min_size]
            cut = max(candidates, key=lambda b: distances[b]) + 1 if candidates else i
            groups.append(_join(sentences[start:cut]))
            start, size = cut, sum(lengths[cut:i])
        size += lengths[i]
        i += 1
    groups.append(_join(sentences[start:]))
    return [g for g in groups if g.strip()]


def split_documents(
    documents: list[Document],
    config: ChunkingConfig,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """Split documents into chunks; metadata is preserved on every chunk."""
    splitter = create_splitter(config)
    semantic = config.method == "semantic" and embeddings is not None

    chunks: list[Document] = []
    if semantic:
        # Embed every sentence of every document in one batch: one model pass
        # instead of one per document.
        prose = [d for d in documents if d.metadata.get("type") != "table"]
        per_doc = [split_sentences(d.page_content) for d in prose]
        if config.semantic_boundaries == "embedding":
            flat = [s for sents in per_doc for s in sents]
            # The opening of a sentence is enough to tell a topic shift, and
            # truncating keeps long table rows / code lines from dominating cost.
            probe = [s[:_BOUNDARY_CHARS] for s in flat]
            vectors = np.array(embeddings.embed_documents(probe), dtype=float) if flat else np.zeros((0, 1))
            offsets = np.cumsum([0] + [len(x) for x in per_doc])
            matrices = [vectors[offsets[i]: offsets[i + 1]] for i in range(len(per_doc))]
        else:
            matrices = lexical_vectors(per_doc)
        length = splitter._length_function
        for doc, sents, matrix in zip(prose, per_doc, matrices):
            groups = semantic_groups(
                sents, matrix, [length(x) for x in sents],
                config.chunk_size, config.semantic_breakpoint_percentile,
            )
            pieces = [Document(page_content=g, metadata=dict(doc.metadata)) for g in groups]
            # A single over-long sentence is still capped at chunk_size.
            chunks.extend(splitter.split_documents(pieces))
        chunks.extend(splitter.split_documents([d for d in documents if d.metadata.get("type") == "table"]))
    else:
        chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    method = "semantic" if semantic else "recursive"
    print(f"[SPLIT]  Split {len(documents)} document(s) into {len(chunks)} chunks "
          f"({method}, max size={config.chunk_size}, overlap={config.chunk_overlap})")
    return chunks
