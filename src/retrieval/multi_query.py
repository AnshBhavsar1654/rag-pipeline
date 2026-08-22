"""
RAG Pipeline — Multi-Query Retrieval (Part 5)

Generates multiple query variants from the original question, retrieves
documents for each variant, and unions the results for broader recall.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval.base_retriever import BaseRAGRetriever


_MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant. Your task is to generate {num_queries} \
different versions of the given user question to retrieve relevant documents \
from a vector database. By generating multiple perspectives on the user question, \
your goal is to help the user overcome some of the limitations of distance-based \
similarity search.

Provide these alternative questions separated by newlines. Do NOT number them.

Original question: {question}"""


class MultiQueryRetriever(BaseRAGRetriever):
    """Generate multiple query variants, retrieve for each, and union results."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
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
            | (lambda text: [q.strip() for q in text.strip().split("\n") if q.strip()])
        )

    @property
    def strategy_name(self) -> str:
        return "Multi-Query"

    def retrieve(self, question: str) -> list[Document]:
        # Generate query variants
        queries = self._query_chain.invoke({
            "question": question,
            "num_queries": self._num_queries,
        })

        # Always include the original question
        all_queries = [question] + queries[:self._num_queries]

        # Retrieve for each query and deduplicate
        seen_contents: set[str] = set()
        unique_docs: list[Document] = []

        for q in all_queries:
            docs = self._retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)

        return unique_docs
