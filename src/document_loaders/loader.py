"""
RAG Pipeline — Universal Document Loader

Automatically detects file types and loads documents from:
- Files: PDF, TXT, MD, DOCX, CSV, HTML
- URLs: Web pages
- Directories: Recursively loads all supported files
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document

from src.config import DocumentsConfig


# Mapping of file extensions to LangChain loader classes (lazy imports)
_EXTENSION_LOADERS = {
    ".pdf": ("langchain_community.document_loaders", "PyPDFLoader"),
    ".txt": ("langchain_community.document_loaders", "TextLoader"),
    ".md": ("langchain_community.document_loaders", "TextLoader"),
    ".docx": ("langchain_community.document_loaders", "Docx2txtLoader"),
    ".csv": ("langchain_community.document_loaders", "CSVLoader"),
    ".html": ("langchain_community.document_loaders", "BSHTMLLoader"),
    ".htm": ("langchain_community.document_loaders", "BSHTMLLoader"),
}

SUPPORTED_EXTENSIONS = set(_EXTENSION_LOADERS.keys())


def _is_url(source: str) -> bool:
    """Check if a source string is a URL."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _load_from_url(url: str) -> list[Document]:
    """Load documents from a web URL."""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    print(f"  [FILE] Loaded {len(docs)} document(s) from URL: {url}")
    return docs


def _load_from_file(file_path: Path) -> list[Document]:
    """Load documents from a single file based on its extension."""
    ext = file_path.suffix.lower()

    if ext not in _EXTENSION_LOADERS:
        print(f"  [WARN]  Unsupported file type: {file_path} (skipping)")
        return []

    module_name, class_name = _EXTENSION_LOADERS[ext]

    # Lazy import the appropriate loader
    import importlib
    module = importlib.import_module(module_name)
    LoaderClass = getattr(module, class_name)

    try:
        # TextLoader needs encoding param for reliability
        if class_name == "TextLoader":
            loader = LoaderClass(str(file_path), encoding="utf-8")
        else:
            loader = LoaderClass(str(file_path))

        docs = loader.load()
        print(f"  [FILE] Loaded {len(docs)} document(s) from: {file_path.name}")
        return docs

    except Exception as e:
        print(f"  [ERROR] Error loading {file_path}: {e}")
        return []


def _load_from_directory(dir_path: Path) -> list[Document]:
    """Recursively load all supported files from a directory."""
    all_docs: list[Document] = []

    if not dir_path.exists():
        print(f"  [WARN]  Directory not found: {dir_path}")
        return all_docs

    for root, _, files in os.walk(dir_path):
        for file_name in sorted(files):
            file_path = Path(root) / file_name
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_docs.extend(_load_from_file(file_path))

    if not all_docs:
        print(f"  [WARN]  No supported files found in: {dir_path}")

    return all_docs


def load_documents(config: DocumentsConfig) -> list[Document]:
    """Load documents from all configured sources.

    Supports:
    - File paths (pdf, txt, md, docx, csv, html)
    - Web URLs
    - Directories (recursive)
    """
    all_docs: list[Document] = []

    print("\n[LOAD] Loading documents...")

    for source in config.sources:
        source = source.strip()

        if _is_url(source):
            all_docs.extend(_load_from_url(source))
        else:
            path = Path(source)
            if path.is_dir():
                all_docs.extend(_load_from_directory(path))
            elif path.is_file():
                all_docs.extend(_load_from_file(path))
            else:
                print(f"  [WARN]  Source not found: {source}")

    print(f"\n[OK] Total documents loaded: {len(all_docs)}")
    return all_docs
