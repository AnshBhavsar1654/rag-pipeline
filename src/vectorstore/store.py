"""
RAG Pipeline — Vector Store Abstraction

Manages Chroma vector store with disk persistence support.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from src.config import VectorStoreConfig


def create_vectorstore(
    config: VectorStoreConfig,
    embeddings: Embeddings,
    documents: list[Document] | None = None,
) -> Chroma:
    """Create or load a Chroma vector store.

    If *documents* are provided, a new collection is created (replacing any
    existing one at the persist directory). Otherwise the existing persisted
    store is loaded.
    """
    persist_dir = Path(config.persist_directory)

    if documents is not None:
        # Build fresh vector store from documents
        print(f"\n[SAVE] Building vector store with {len(documents)} chunks...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=config.collection_name,
            persist_directory=str(persist_dir),
        )

        print(f"[OK] Vector store persisted to: {persist_dir}")
        return vectorstore

    # Load existing vector store
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. "
            "Run `python ingest.py` first to build the index."
        )

    vectorstore = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    count = vectorstore._collection.count()
    print(f"[OK] Loaded vector store from {persist_dir} ({count} vectors)")
    return vectorstore


def get_retriever(vectorstore: Chroma, top_k: int = 5):
    """Get a LangChain retriever from the vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
