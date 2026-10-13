"""
RAG Pipeline — Multi-Query Retrieval (Part 5)

Generates multiple query variants from the original question, retrieves
documents for each variant, and unions the results for broader recall.
"""

from __future__ import annotations

from itertools import zip_longest

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.base_retriever import BaseRAGRetriever
from src.retrieval.utils import SearchRetriever, doc_key, parse_query_list, search_all


_MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant. Your task is to generate {num_queries} \
different versions of the given user question to retrieve relevant documents \
from a vector database. By generating multiple perspectives on the user question, \
your goal is to help the user overcome some of the limitations of distance-based \
similarity search.

Output ONLY the alternative questions, one per line, with no numbering or preamble.

Original question: {question}"""


class MultiQueryRetriever(BaseRAGRetriever):
    """Generate multiple query variants, retrieve for each, and union results."""

    def __init__(
        self,
        retriever: SearchRetriever,
        llm: BaseChatModel,
        num_queries: int = 4,
    ):
        self._retriever = retriever
        self._llm = llm
        self._num_queries = num_queries

        self._query_chain = (
            ChatPromptTemplate.from_template(_MULTI_QUERY_TEMPLATE)
            | self._llm
            | StrOutputParser()
        )

    @property
    def strategy_name(self) -> str:
        return "Multi-Query"

    def retrieve(self, question: str) -> list[Document]:
        queries = self._generate_queries(question)
        results = search_all(self._retriever, question, queries)

        # Interleave the result lists (best hit of each query first) so that
        # truncating to top-k keeps every query's strongest match.
        seen: set[str] = set()
        unique_docs: list[Document] = []
        for tier in zip_longest(*results):
            for doc in tier:
                if doc is not None and doc_key(doc) not in seen:
                    seen.add(doc_key(doc))
                    unique_docs.append(doc)

        return unique_docs

    def _generate_queries(self, question: str) -> list[str]:
        try:
            text = self._query_chain.invoke({
                "question": question,
                "num_queries": self._num_queries,
            })
        except Exception as e:
            print(f"[WARN]  Query generation failed ({e}); using the original question only.")
            return []
        return parse_query_list(text, limit=self._num_queries)
