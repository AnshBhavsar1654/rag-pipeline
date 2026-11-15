"""
Query Translation — Step-Back Prompting

Adds a more generic "step-back" question so the search also brings back
background context, fused behind the specific matches.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from src.query_translation.base import QueryTranslator


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


class StepBackTranslator(QueryTranslator):
    """Search with both the question and a more generic step-back question."""

    name = "Step-Back"

    def __init__(self, llm: BaseChatModel, rrf_k: int = 60):
        super().__init__(rrf_k)
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
        self._chain = prompt | llm | StrOutputParser()

    def translate(self, question: str) -> list[str]:
        try:
            step_back = self._chain.invoke({"question": question}).strip()
        except Exception as e:
            print(f"[WARN]  Step-back generation failed ({e}); using the original question only.")
            return [question]
        if not step_back or step_back.lower() == question.strip().lower():
            return [question]
        return [question, step_back]
