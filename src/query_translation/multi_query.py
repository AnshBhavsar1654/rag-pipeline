"""
Query Translation — Multi-Query

Rephrases the question several ways and unions the results of every
variant for broader recall.
"""

from __future__ import annotations

from itertools import zip_longest

from langchain_core.documents import Document

from src.query_translation.base import LLMQueryTranslator
from src.retrieval.utils import doc_key


class MultiQueryTranslator(LLMQueryTranslator):
    name = "Multi-Query"
    template = """\
You are an AI language model assistant. Your task is to generate {num_queries} \
different versions of the given user question to retrieve relevant documents \
from a vector database. By generating multiple perspectives on the user question, \
your goal is to help the user overcome some of the limitations of distance-based \
similarity search.

Output ONLY the alternative questions, one per line, with no numbering or preamble.

Original question: {question}"""

    def fuse(self, results: list[list[Document]]) -> list[Document]:
        # Interleave (best hit of each query first) so truncating to top-k
        # keeps every query's strongest match.
        seen: set[str] = set()
        unique: list[Document] = []
        for tier in zip_longest(*results):
            for doc in tier:
                if doc is not None and doc_key(doc) not in seen:
                    seen.add(doc_key(doc))
                    unique.append(doc)
        return unique
