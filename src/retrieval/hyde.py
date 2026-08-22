"""
RAG Pipeline — HyDE Retrieval (Part 9)

Hypothetical Document Embeddings: generates a hypothetical answer to the
question, then uses that answer's embedding to retrieve similar real documents.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval.base_retriever import BaseRAGRetriever


_DEFAULT_HYDE_TEMPLATE = """\
Please write a detailed passage to answer the question.
Question: {question}
Passage:"""


class HyDERetriever(BaseRAGRetriever):
    """HyDE: generate a hypothetical document, then retrieve similar real docs."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
        hypothesis_prompt: str | None = None,
    ):
        self._retriever = retriever
        self._llm = llm

        template = hypothesis_prompt or _DEFAULT_HYDE_TEMPLATE
        self._hypothesis_chain = (
            ChatPromptTemplate.from_template(template)
            | self._llm
            | StrOutputParser()
        )

    @property
    def strategy_name(self) -> str:
        return "HyDE"

    def retrieve(self, question: str) -> list[Document]:
        # Generate hypothetical document
        hypothetical_doc = self._hypothesis_chain.invoke({"question": question})

        # Use the hypothetical document as the retrieval query
        docs = self._retriever.invoke(hypothetical_doc)

        return docs
