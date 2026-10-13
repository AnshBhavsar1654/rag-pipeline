"""
RAG Pipeline — Step-Back Prompting (Part 8)

Generates a more generic "step-back" question to retrieve broader context,
then combines with direct retrieval for more comprehensive answers.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from src.retrieval.base_retriever import BaseRAGRetriever
from src.retrieval.utils import SearchRetriever, reciprocal_rank_fusion


# Few-shot examples for step-back prompting
_STEP_BACK_EXAMPLES = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "What can the members of The Police do?",
    },
    {
        "input": "Jan Sindel was born in what country?",
        "output": "What is Jan Sindel's personal history?",
    },
    {
        "input": "What are the specific algorithms used in task decomposition for LLM agents?",
        "output": "What is task decomposition in the context of LLM agents?",
    },
]


class StepBackRetriever(BaseRAGRetriever):
    """Retrieve for both the question and a more generic step-back question."""

    def __init__(
        self,
        retriever: SearchRetriever,
        llm: BaseChatModel,
        rrf_k: int = 60,
    ):
        self._retriever = retriever
        self._llm = llm
        self._rrf_k = rrf_k

        # Build few-shot prompt for step-back question generation
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=_STEP_BACK_EXAMPLES,
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert at world knowledge. Your task is to step back "
                "and paraphrase a question to a more generic step-back question, "
                "which is easier to answer. Here are a few examples:",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ])

        self._step_back_chain = prompt | self._llm | StrOutputParser()

    @property
    def strategy_name(self) -> str:
        return "Step-Back"

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve documents using both the original and step-back question,
        fusing the two rankings so specific matches stay ahead of background.
        """
        normal_docs = self._retriever.invoke(question)

        try:
            step_back_q = self._step_back_chain.invoke({"question": question}).strip()
        except Exception as e:
            print(f"[WARN]  Step-back generation failed ({e}); using the original question only.")
            return normal_docs

        step_back_docs = self._retriever.invoke(step_back_q)
        return reciprocal_rank_fusion([normal_docs, step_back_docs], k=self._rrf_k)
