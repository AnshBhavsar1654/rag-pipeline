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
from langchain_core.vectorstores import VectorStoreRetriever


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


class StepBackPrompter:
    """Generate a step-back question for broader retrieval context."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
    ):
        self._retriever = retriever
        self._llm = llm

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

    def retrieve_with_step_back(self, question: str) -> list[Document]:
        """Retrieve documents using both the original and step-back question.

        Returns a deduplicated list of documents from both retrieval passes.
        """
        # Generate step-back question
        step_back_q = self._step_back_chain.invoke({"question": question})

        # Retrieve for both questions
        normal_docs = self._retriever.invoke(question)
        step_back_docs = self._retriever.invoke(step_back_q)

        # Deduplicate
        seen: set[str] = set()
        combined: list[Document] = []
        for doc in normal_docs + step_back_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        return combined
