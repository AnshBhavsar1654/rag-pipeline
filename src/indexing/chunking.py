"""
RAG Pipeline — Text Splitter / Chunking

Configurable text splitting with tiktoken-aware or character-based chunking.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig


def create_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Create a text splitter based on configuration."""

    if config.use_tiktoken:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=config.tiktoken_encoding,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    return splitter


def split_documents(
    documents: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    """Split documents into chunks according to configuration.

    Returns a list of Document chunks with preserved metadata.
    """
    splitter = create_splitter(config)
    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"[SPLIT]  Split {len(documents)} document(s) into {len(chunks)} chunks "
          f"(size={config.chunk_size}, overlap={config.chunk_overlap})")

    return chunks
