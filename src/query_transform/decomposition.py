"""
RAG Pipeline — Question Decomposition (Part 7)

Breaks a complex question into sub-questions that can each be searched in
isolation, retrieves for every sub-question, and fuses the results so the
final answer sees evidence for each part.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.base_retriever import BaseRAGRetriever
from src.retrieval.utils import (
    SearchRetriever,
    parse_query_list,
    reciprocal_rank_fusion,
    search_all,
)


_DECOMPOSITION_TEMPLATE = """\
You are a helpful assistant that generates multiple sub-questions related to \
an input question. The goal is to break down the input into a set of \
sub-problems / sub-questions that can be answered in isolation.

Generate up to {num_questions} sub-questions for: {question}

Output ONLY the sub-questions, one per line, with no numbering or preamble. \
If the question is already simple, output it unchanged."""


class DecompositionRetriever(BaseRAGRetriever):
    """Decompose a question into sub-questions and fuse their search results."""

    def __init__(
        self,
        retriever: SearchRetriever,
        llm: BaseChatModel,
        num_questions: int = 3,
        rrf_k: int = 60,
    ):
        self._retriever = retriever
        self._llm = llm
        self._num_questions = num_questions
        self._rrf_k = rrf_k

        self._decompose_chain = (
            ChatPromptTemplate.from_template(_DECOMPOSITION_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )

    @property
    def strategy_name(self) -> str:
        return "Question Decomposition"

    def retrieve(self, question: str) -> list[Document]:
        try:
            text = self._decompose_chain.invoke({
                "question": question,
                "num_questions": self._num_questions,
            })
            sub_questions = parse_query_list(text, limit=self._num_questions)
        except Exception as e:
            print(f"[WARN]  Decomposition failed ({e}); using the original question only.")
            sub_questions = []

        results = search_all(self._retriever, question, sub_questions)
        return reciprocal_rank_fusion(results, k=self._rrf_k)
