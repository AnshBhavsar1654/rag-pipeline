"""
Indexing — Multi-representation indexing

Searching works best on small, focused units, while answering works best
with the surrounding context. So documents are stored twice:

* full-context **parent** chunks go to a document store (SQLite), and
* compact **retrieval units** go to the vectorstore, each pointing at its
  parent with ``parent_id``: the small child chunks (Parent Document), and
  in ``summary`` mode also an LLM summary of every parent.

At query time the matched units are swapped for their parents.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class DocStore:
    """Key-value store of parent documents, persisted in SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS parents "
            "(id TEXT PRIMARY KEY, source TEXT, content TEXT, metadata TEXT)"
        )
        return conn

    @contextmanager
    def _connect(self):
        conn = self._open()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def put(self, parents: dict[str, Document]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?)",
                [
                    (pid, str(d.metadata.get("source", "")), d.page_content,
                     json.dumps(d.metadata, default=str))
                    for pid, d in parents.items()
                ],
            )

    def get(self, ids: list[str]) -> dict[str, Document]:
        if not ids or not self.path.exists():
            return {}
        with self._connect() as conn:
            marks = ", ".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT id, content, metadata FROM parents WHERE id IN ({marks})", ids,
            ).fetchall()
        return {
            pid: Document(page_content=content, metadata=json.loads(meta), id=pid)
            for pid, content, meta in rows
        }

    def delete_sources(self, sources: set[str]) -> None:
        if sources and self.path.exists():
            with self._connect() as conn:
                marks = ", ".join("?" * len(sources))
                conn.execute(f"DELETE FROM parents WHERE source IN ({marks})", sorted(sources))

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()


def parent_id(doc: Document) -> str:
    key = "\x1f".join([
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", "")),
        str(doc.metadata.get("type", "")),
        doc.page_content,
    ])
    return "p_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""\
Summarize each numbered document below in 2-4 sentences that capture the \
facts someone might search for. Return a JSON array of strings, one summary \
per document, in the same order. Output ONLY the JSON array.

{documents}""")


class SummaryWriter:
    """Summarize parents in batches (one LLM call per batch)."""

    def __init__(self, llm: BaseChatModel, batch_size: int = 5):
        self._chain = _SUMMARY_PROMPT | llm | StrOutputParser()
        self._batch_size = batch_size

    def summarize(self, docs: list[Document]) -> list[str | None]:
        summaries: list[str | None] = []
        for i in range(0, len(docs), self._batch_size):
            batch = docs[i: i + self._batch_size]
            text = "\n\n".join(f"[{j + 1}]\n{d.page_content}" for j, d in enumerate(batch))
            try:
                reply = self._chain.invoke({"documents": text})
                match = re.search(r"\[.*\]", reply, re.S)
                items = json.loads(match.group(0)) if match else []
            except Exception as e:
                print(f"[WARN]  Summary batch failed ({e}); those parents keep only child chunks.")
                items = []
            items = [str(s).strip() if s else None for s in items][: len(batch)]
            summaries.extend(items + [None] * (len(batch) - len(items)))
        return summaries


def build_representations(
    documents: list[Document],
    mode: str,
    split_parents: Callable[[list[Document]], list[Document]],
    split_children: Callable[[list[Document]], list[Document]],
    summarizer: SummaryWriter | None = None,
) -> tuple[list[Document], dict[str, Document]]:
    """Return (documents for the vectorstore, parents for the docstore)."""
    if mode == "none":
        return split_children(documents), {}

    parents: dict[str, Document] = {}
    for parent in split_parents(documents):
        parent.metadata.pop("chunk_index", None)
        parents.setdefault(parent_id(parent), parent)

    # parent_id rides along in the metadata every child chunk inherits.
    for pid, parent in parents.items():
        parent.metadata["parent_id"] = pid
    units = split_children(list(parents.values()))

    if mode == "summary" and summarizer is not None:
        items = list(parents.items())
        summaries = summarizer.summarize([p for _, p in items])
        for (pid, parent), summary in zip(items, summaries):
            if summary:
                meta = {k: v for k, v in parent.metadata.items() if k in ("source", "page")}
                units.append(Document(
                    page_content=summary,
                    metadata={**meta, "type": "summary", "parent_id": pid},
                ))

    for i, unit in enumerate(units):
        unit.metadata["chunk_index"] = i
    print(f"[INDEX] Multi-representation ({mode}): {len(parents)} parent(s) in the docstore, "
          f"{len(units)} retrieval unit(s) for the vectorstore")
    return units, parents


def expand_to_parents(documents: list[Document], docstore: DocStore) -> list[Document]:
    """Swap retrieval units for their parent documents, keeping rank order."""
    ids = [d.metadata["parent_id"] for d in documents if d.metadata.get("parent_id")]
    parents = docstore.get(list(dict.fromkeys(ids)))
    seen: set[str] = set()
    expanded: list[Document] = []
    for doc in documents:
        pid = doc.metadata.get("parent_id")
        key = pid if pid in parents else (doc.id or doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        expanded.append(parents[pid] if pid in parents else doc)
    return expanded
