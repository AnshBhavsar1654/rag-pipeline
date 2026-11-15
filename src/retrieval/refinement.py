"""
Retrieval — Refinement (CRAG relevance grading)

The LLM grades every retrieved document for relevance to the question in a
single call, and irrelevant ones are filtered out before generation, so the
answer is not distracted by near-misses.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_GRADE_PROMPT = ChatPromptTemplate.from_template("""\
You are a grader assessing the relevance of retrieved documents to a user \
question. A document is relevant if it contains keywords or meaning that \
help answer the question; it does not need to answer it fully.

Question: {question}

Documents:
{documents}

Reply with the numbers of the relevant documents, comma separated (e.g. \
1, 3), or "none" if no document is relevant. Output nothing else.""")


def parse_relevant(text: str, count: int) -> list[bool]:
    if re.search(r"\bnone\b", text.lower()) and not re.search(r"\d", text):
        return [False] * count
    chosen = {int(n) - 1 for n in re.findall(r"\d+", text)}
    return [i in chosen for i in range(count)]


class RelevanceGrader:
    def __init__(self, llm: BaseChatModel, max_words: int = 200):
        self._chain = _GRADE_PROMPT | llm | StrOutputParser()
        self._max_words = max_words

    def grade(self, question: str, documents: list[Document]) -> list[bool]:
        """One verdict per document; on failure every document is kept."""
        if not documents:
            return []
        listing = "\n\n".join(
            f"[{i}] {' '.join(d.page_content.split()[: self._max_words])}"
            for i, d in enumerate(documents, start=1)
        )
        try:
            reply = self._chain.invoke({"question": question, "documents": listing})
        except Exception as e:
            print(f"[WARN]  Relevance grading failed ({e}); keeping every document.")
            return [True] * len(documents)
        return parse_relevant(reply, len(documents))
