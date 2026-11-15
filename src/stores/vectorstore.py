"""
RAG Pipeline — Vector Store Abstraction

Manages Chroma vector store with disk persistence support.

The collection records which embedding model built it, so that an index
built with one model is never silently queried with another (vectors from
different models can share a dimension yet live in unrelated spaces).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import VectorStoreConfig


_EMBEDDING_KEY = "embedding_signature"


class VectorStoreNotReadyError(FileNotFoundError):
    """The index is missing, empty, or was built with a different embedding model."""


class EmbeddingMismatchError(VectorStoreNotReadyError):
    """The index holds chunks, but they were embedded by a different model."""


def chunk_id(doc: Document) -> str:
    """Stable ID for a chunk, so re-ingesting the same content upserts it."""
    source = Path(str(doc.metadata.get("source", ""))).name
    key = "\x1f".join([
        source,
        str(doc.metadata.get("type", "text")),
        str(doc.metadata.get("page", "")),
        doc.page_content,
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def add_documents(vectorstore: Chroma, documents: list[Document], progress=None, batch_size: int = 64) -> int:
    """Upsert chunks into the store and return how many unique chunks were written.

    Embeds in batches so *progress(done, total)* can report how far along it is."""
    unique: dict[str, Document] = {}
    for doc in documents:
        unique.setdefault(chunk_id(doc), doc)
    ids, docs = list(unique.keys()), list(unique.values())
    for start in range(0, len(docs), batch_size):
        vectorstore.add_documents(docs[start: start + batch_size], ids=ids[start: start + batch_size])
        if progress:
            progress(min(start + batch_size, len(docs)), len(docs))
    return len(unique)


def _open(
    config: VectorStoreConfig,
    embeddings: Embeddings,
    signature: str,
) -> Chroma:
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(Path(config.persist_directory)),
        collection_metadata={_EMBEDDING_KEY: signature, "hnsw:space": "cosine"},
    )


def delete_sources(vectorstore: Chroma, sources: set[str]) -> None:
    """Remove every chunk that came from one of *sources* (before re-adding them)."""
    if sources:
        vectorstore._collection.delete(where={"source": {"$in": sorted(sources)}})


def read_stored_documents(config: VectorStoreConfig) -> list[Document]:
    """Read chunk text and metadata straight from disk, without an embedding model."""
    store = Chroma(
        collection_name=config.collection_name,
        persist_directory=str(Path(config.persist_directory)),
        create_collection_if_not_exists=False,
    )
    return all_documents(store)


def backup_vectorstore(config: VectorStoreConfig) -> Path:
    """Copy the persist directory aside before a destructive rebuild."""
    import shutil
    from datetime import datetime

    persist_dir = Path(config.persist_directory)
    backup = persist_dir.with_name(
        f"{persist_dir.name}.backup-{datetime.now():%Y%m%d-%H%M%S}"
    )
    shutil.copytree(persist_dir, backup)
    return backup


def build_vectorstore(
    config: VectorStoreConfig,
    embeddings: Embeddings,
    signature: str,
    documents: list[Document],
) -> Chroma:
    """Create a fresh collection from *documents*, replacing any existing one."""
    persist_dir = Path(config.persist_directory)
    print(f"\n[SAVE] Building vector store with {len(documents)} chunks...")

    vectorstore = _open(config, embeddings, signature)
    vectorstore.reset_collection()
    written = add_documents(vectorstore, documents)

    print(f"[OK] Vector store persisted to: {persist_dir} ({written} unique chunks)")
    return vectorstore


def create_empty_vectorstore(
    config: VectorStoreConfig,
    embeddings: Embeddings,
    signature: str,
) -> Chroma:
    """Create (or reset to) an empty collection ready for incremental uploads."""
    vectorstore = _open(config, embeddings, signature)
    vectorstore.reset_collection()
    return vectorstore


def load_vectorstore(
    config: VectorStoreConfig,
    embeddings: Embeddings,
    signature: str,
) -> Chroma:
    """Load the persisted store, refusing one that is empty or built for another model."""
    persist_dir = Path(config.persist_directory)
    hint = "Run `python ingest.py` to build the index."

    if not persist_dir.exists():
        raise VectorStoreNotReadyError(f"Vector store not found at {persist_dir}. {hint}")

    vectorstore = _open(config, embeddings, signature)
    count = vectorstore._collection.count()
    if count == 0:
        raise VectorStoreNotReadyError(f"Vector store at {persist_dir} is empty. {hint}")

    stored = (vectorstore._collection.metadata or {}).get(_EMBEDDING_KEY)
    if stored != signature:
        raise EmbeddingMismatchError(
            f"Vector store at {persist_dir} was built with embedding model "
            f"'{stored or 'unknown'}', but the config uses '{signature}'. {hint}"
        )

    print(f"[OK] Loaded vector store from {persist_dir} ({count} vectors)")
    return vectorstore


def all_documents(vectorstore: Chroma) -> list[Document]:
    """Return every chunk in the store (used to build the keyword index)."""
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {}, id=doc_id)
        for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
