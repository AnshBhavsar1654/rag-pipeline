"""
RAG Pipeline — Follow-up Question Rewriting

A follow-up like "what about its limitations?" retrieves poorly on its own:
the search never sees what "its" refers to. Rewrite it, using the
conversation so far, into a standalone question before retrieval.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_CONDENSE_TEMPLATE = """\
Given the conversation below and a follow-up question, rewrite the follow-up \
as a single standalone question that can be understood without the \
conversation. Resolve pronouns and references. If it is already standalone, \
return it unchanged. Output ONLY the question.

Conversation:
{chat_history}

Follow-up question: {question}

Standalone question:"""


class QuestionCondenser:
    """Turn a conversational follow-up into a standalone search question."""

    def __init__(self, llm: BaseChatModel):
        self._chain = (
            ChatPromptTemplate.from_template(_CONDENSE_TEMPLATE)
            | llm
            | StrOutputParser()
        )

    def condense(self, question: str, chat_history: str) -> str:
        if not chat_history.strip():
            return question
        try:
            rewritten = self._chain.invoke({
                "question": question,
                "chat_history": chat_history,
            }).strip().strip('"')
        except Exception as e:
            print(f"[WARN]  Follow-up rewrite failed ({e}); searching with the raw question.")
            return question
        return rewritten or question
