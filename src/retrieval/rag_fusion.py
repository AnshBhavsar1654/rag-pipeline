"""
RAG Pipeline — RAG-Fusion with Reciprocal Rank Fusion (Parts 6 & 15)

Generates multiple query variants, retrieves for each, then fuses results
using Reciprocal Rank Fusion (RRF) scoring for superior ranking.
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


_RAG_FUSION_TEMPLATE = """\
You are a helpful assistant that generates multiple search queries based on \
a single input query. Generate {num_queries} different search queries related to \
the following question. Provide diverse perspectives to maximize retrieval coverage.

Output ONLY the queries, one per line, with no numbering or preamble.

Original question: {question}"""


class RAGFusionRetriever(BaseRAGRetriever):
    """RAG-Fusion: multi-query + Reciprocal Rank Fusion re-ranking."""

    def __init__(
        self,
        retriever: SearchRetriever,
        llm: BaseChatModel,
        num_queries: int = 4,
        rrf_k: int = 60,
    ):
        self._retriever = retriever
        self._llm = llm
        self._num_queries = num_queries
        self._rrf_k = rrf_k

        self._query_chain = (
            ChatPromptTemplate.from_template(_RAG_FUSION_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )

    @property
    def strategy_name(self) -> str:
        return "RAG-Fusion (RRF)"

    def retrieve(self, question: str) -> list[Document]:
        try:
            text = self._query_chain.invoke({
                "question": question,
                "num_queries": self._num_queries,
            })
            queries = parse_query_list(text, limit=self._num_queries)
        except Exception as e:
            print(f"[WARN]  Query generation failed ({e}); using the original question only.")
            queries = []

        results = search_all(self._retriever, question, queries)
        return reciprocal_rank_fusion(results, k=self._rrf_k)
