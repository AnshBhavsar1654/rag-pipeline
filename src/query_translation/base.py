"""
Query Translation — shared interface

A translator turns one question into the list of search queries that the
rest of the pipeline (routing → query construction → stores) runs, and
says how to fuse the ranked lists those queries bring back.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.utils import parse_query_list, reciprocal_rank_fusion


class QueryTranslator:
    """No translation: search with the question as-is."""

    name = "Simple"

    def __init__(self, rrf_k: int = 60):
        self._rrf_k = rrf_k

    def translate(self, question: str) -> list[str]:
        """Return the queries to search with; the question itself always comes first."""
        return [question]

    def fuse(self, results: list[list[Document]]) -> list[Document]:
        """Merge the ranked lists returned for each query into one ranking."""
        return reciprocal_rank_fusion(results, k=self._rrf_k)

    def from_plan(self, question: str, generated: list[str]) -> list[str]:
        """Queries to search with when the query planner wrote *generated* for us."""
        if type(self) is QueryTranslator:
            return [question]
        return dedupe_queries([question, *generated])


class LLMQueryTranslator(QueryTranslator):
    """A translator whose extra queries come from one LLM prompt."""

    template: str = ""

    def __init__(self, llm: BaseChatModel, num_queries: int = 4, rrf_k: int = 60):
        super().__init__(rrf_k)
        self._num_queries = num_queries
        self._chain = ChatPromptTemplate.from_template(self.template) | llm | StrOutputParser()

    def generate(self, question: str) -> list[str]:
        text = self._chain.invoke({"question": question, "num_queries": self._num_queries})
        return parse_query_list(text, limit=self._num_queries)

    def translate(self, question: str) -> list[str]:
        try:
            extra = self.generate(question)
        except Exception as e:
            print(f"[WARN]  {self.name} query generation failed ({e}); using the original question only.")
            extra = []
        return dedupe_queries([question, *extra])


def dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique
