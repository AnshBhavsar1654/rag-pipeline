"""
Query Construction — Self-query retriever

Has the LLM turn restrictions in the question ("in the PDF report", "from
the tables", "on page 3") into a metadata filter on the vectorstore, so the
search only looks where the user asked. The filter only uses values that
actually exist in the index.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class MetadataFilter:
    """Field → allowed values; every field must match."""

    conditions: dict[str, list] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def to_chroma(self) -> dict | None:
        clauses = [
            {key: {"$in": values}} if len(values) > 1 else {key: values[0]}
            for key, values in self.conditions.items()
        ]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def matches(self, metadata: dict) -> bool:
        return all(metadata.get(k) in values for k, values in self.conditions.items())

    def describe(self) -> str:
        parts = []
        for key, values in self.conditions.items():
            shown = [Path(v).name if key == "source" else v for v in values]
            parts.append(f"{key} in {shown}")
        return " and ".join(parts)


@dataclass
class MetadataCatalog:
    """The metadata values present in the vectorstore."""

    sources: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    max_page: int = 0

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "MetadataCatalog":
        sources, types, max_page = set(), set(), 0
        for doc in documents:
            meta = doc.metadata
            if meta.get("source"):
                sources.add(str(meta["source"]))
            if meta.get("type"):
                types.add(str(meta["type"]))
            if isinstance(meta.get("page"), int):
                max_page = max(max_page, meta["page"])
        return cls(sorted(sources), sorted(types), max_page)

    @property
    def filterable(self) -> bool:
        return len(self.sources) > 1 or len(self.types) > 1 or self.max_page > 1


_SELF_QUERY_PROMPT = ChatPromptTemplate.from_template("""\
Decide whether the user's question restricts WHERE to search. Available metadata:
- source (file names): {sources}
- type (kind of passage): {types}
- page (integer page number, 1-{max_page}; 0 means no page numbers)

Return a JSON object with optional keys "source" (list of file names from \
the list), "type" (list of types from the list) and "page" (list of \
integers). Only include a key when the question explicitly asks for it, \
e.g. "in the pricing PDF", "from the tables", "on page 4". If there is no \
such restriction return {{}}. Output ONLY the JSON.

Question: {question}""")


class SelfQueryConstructor:
    def __init__(self, llm: BaseChatModel):
        self._chain = _SELF_QUERY_PROMPT | llm | StrOutputParser()

    def construct(self, question: str, catalog: MetadataCatalog) -> MetadataFilter:
        if not catalog.filterable:
            return MetadataFilter()
        try:
            reply = self._chain.invoke({
                "question": question,
                "sources": ", ".join(Path(s).name for s in catalog.sources),
                "types": ", ".join(catalog.types),
                "max_page": catalog.max_page,
            })
            match = re.search(r"\{.*\}", reply, re.S)
            data = json.loads(match.group(0)) if match else {}
        except Exception as e:
            print(f"[WARN]  Self-query failed ({e}); searching without a filter.")
            return MetadataFilter()

        return filter_from_json(data, catalog)


def filter_from_json(data: dict, catalog: MetadataCatalog) -> MetadataFilter:
    """Build a filter from a model's JSON, keeping only values that exist in the index."""
    if not isinstance(data, dict):
        return MetadataFilter()
    by_name = {Path(s).name.lower(): s for s in catalog.sources}
    conditions: dict[str, list] = {}
    sources = [by_name[str(s).lower()] for s in _as_list(data.get("source")) if str(s).lower() in by_name]
    if sources:
        conditions["source"] = sources
    types = [t for t in _as_list(data.get("type")) if t in catalog.types]
    if types:
        conditions["type"] = types
    pages = [int(p) for p in _as_list(data.get("page")) if str(p).isdigit() and 0 < int(p) <= catalog.max_page]
    if pages:
        conditions["page"] = pages
    return MetadataFilter(conditions)


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
