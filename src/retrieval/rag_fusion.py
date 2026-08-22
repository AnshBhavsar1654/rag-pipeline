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
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval.base_retriever import BaseRAGRetriever


_RAG_FUSION_TEMPLATE = """\
You are a helpful assistant that generates multiple search queries based on \
a single input query. Generate {num_queries} different search queries related to \
the following question. Provide diverse perspectives to maximize retrieval coverage.

Provide the queries separated by newlines. Do NOT number them.

Original question: {question}"""


def reciprocal_rank_fusion(
    results: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Apply Reciprocal Rank Fusion across multiple ranked document lists.

    RRF score for each document = Σ 1 / (rank + k)
    where the sum is over all lists that contain the document.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
                fused_scores[key] = 0.0
            fused_scores[key] += 1.0 / (rank + k)

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys]


class RAGFusionRetriever(BaseRAGRetriever):
    """RAG-Fusion: multi-query + Reciprocal Rank Fusion re-ranking."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
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
            | (lambda text: [q.strip() for q in text.strip().split("\n") if q.strip()])
        )

    @property
    def strategy_name(self) -> str:
        return "RAG-Fusion (RRF)"

    def retrieve(self, question: str) -> list[Document]:
        # Generate query variants
        queries = self._query_chain.invoke({
            "question": question,
            "num_queries": self._num_queries,
        })

        # Always include the original question
        all_queries = [question] + queries[:self._num_queries]

        # Retrieve for each query
        all_results: list[list[Document]] = []
        for q in all_queries:
            docs = self._retriever.invoke(q)
            all_results.append(docs)

        # Apply Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(all_results, k=self._rrf_k)

        return fused
