"""
Query Translation — HyDE (pseudo-documents)

Writes a hypothetical passage that would answer the question and searches
with it as well: it looks more like the real answer than the question does.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.query_translation.base import QueryTranslator


_DEFAULT_HYDE_TEMPLATE = """\
Please write a detailed passage to answer the question.
Question: {question}
Passage:"""


class HyDETranslator(QueryTranslator):
    name = "HyDE"

    def __init__(self, llm: BaseChatModel, hypothesis_prompt: str | None = None, rrf_k: int = 60):
        super().__init__(rrf_k)
        template = hypothesis_prompt or _DEFAULT_HYDE_TEMPLATE
        self._chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()

    def translate(self, question: str) -> list[str]:
        # The question stays in the list so a hallucinated passage that
        # drifts off-topic can't take over the ranking on its own.
        try:
            passage = self._chain.invoke({"question": question}).strip()
        except Exception as e:
            print(f"[WARN]  HyDE generation failed ({e}); using the original question only.")
            return [question]
        return [question, passage] if passage else [question]
