"""
Retrieval — Active retrieval (CRAG)

When grading leaves too few relevant documents, re-retrieve: rewrite the
question into a better search query, search the vectorstore again and,
optionally, a new data source, the web (DuckDuckGo, no API key needed).

The query rewriter is also what Self-RAG / RRR uses to re-retrieve after a
poor answer.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_REWRITE_PROMPT = ChatPromptTemplate.from_template("""\
You are a question re-writer that converts an input question into a better \
version optimized for search. Look at the input and reason about the \
underlying semantic intent / meaning. {hint}

Input question: {question}

Output ONLY the improved question.""")


class QueryRewriter:
    def __init__(self, llm: BaseChatModel):
        self._chain = _REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite(self, question: str, hint: str = "") -> str:
        try:
            rewritten = self._chain.invoke({"question": question, "hint": hint}).strip().strip('"')
        except Exception as e:
            print(f"[WARN]  Query rewrite failed ({e}); keeping the question.")
            return question
        return rewritten or question


def web_search(query: str, max_results: int = 4) -> list[Document]:
    """Search the web with DuckDuckGo; returns [] if unavailable or offline."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("[WARN]  ddgs not installed; skipping web search.")
        return []
    try:
        hits = DDGS().text(query, max_results=max_results) or []
    except Exception as e:
        print(f"[WARN]  Web search failed ({e}).")
        return []
    return [
        Document(
            page_content=f"{h.get('title', '')}\n{h.get('body', '')}".strip(),
            metadata={"source": h.get("href", "web"), "type": "web"},
        )
        for h in hits
        if h.get("body")
    ]
