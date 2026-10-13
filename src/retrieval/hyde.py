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

from src.retrieval.base_retriever import BaseRAGRetriever
from src.retrieval.utils import SearchRetriever, reciprocal_rank_fusion


_DEFAULT_HYDE_TEMPLATE = """\
Please write a detailed passage to answer the question.
Question: {question}
Passage:"""


class HyDERetriever(BaseRAGRetriever):
    """HyDE: generate a hypothetical document, then retrieve similar real docs."""

    def __init__(
        self,
        retriever: SearchRetriever,
        llm: BaseChatModel,
        hypothesis_prompt: str | None = None,
        rrf_k: int = 60,
    ):
        self._retriever = retriever
        self._llm = llm
        self._rrf_k = rrf_k

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
        try:
            hypothetical_doc = self._hypothesis_chain.invoke({"question": question})
        except Exception as e:
            print(f"[WARN]  HyDE generation failed ({e}); using the original question only.")
            return self._retriever.invoke(question)

        # The hypothetical passage finds documents that *look like* an answer;
        # the question itself guards against a hallucinated passage drifting
        # off-topic. Fuse both rankings.
        return reciprocal_rank_fusion(
            [self._retriever.invoke(hypothetical_doc), self._retriever.invoke(question)],
            k=self._rrf_k,
        )
