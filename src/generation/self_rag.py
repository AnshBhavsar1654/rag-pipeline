"""
Generation — Active retrieval (Self-RAG / RRR)

After an answer is generated, the LLM checks it twice in one call:

* is it grounded in the retrieved documents (no hallucination)?
* does it actually answer the question?

The pipeline uses the verdicts to regenerate (not grounded) or to rewrite
the question and re-retrieve (not useful), i.e. Rewrite-Retrieve-Read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.generation.generator import format_context


_CHECK_PROMPT = ChatPromptTemplate.from_template("""\
You are grading an answer produced by a retrieval-augmented assistant.

Documents:
{context}

Question: {question}

Answer: {answer}

1. grounded: is every factual claim in the answer supported by the \
documents? (An answer saying the documents do not contain the information \
counts as grounded.)
2. answers_question: does the answer resolve the question?

Return ONLY a JSON object: {{"grounded": "yes" or "no", "answers_question": "yes" or "no"}}""")


@dataclass
class AnswerCheck:
    grounded: bool = True
    answers_question: bool = True

    @property
    def ok(self) -> bool:
        return self.grounded and self.answers_question


def parse_check(text: str) -> AnswerCheck:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}

    def yes(key: str) -> bool:
        return str(data.get(key, "yes")).strip().lower().startswith(("y", "t"))

    return AnswerCheck(grounded=yes("grounded"), answers_question=yes("answers_question"))


class AnswerGrader:
    def __init__(self, llm: BaseChatModel):
        self._chain = _CHECK_PROMPT | llm | StrOutputParser()

    def check(self, question: str, documents: list[Document], answer: str) -> AnswerCheck:
        try:
            reply = self._chain.invoke({
                "context": format_context(documents), "question": question, "answer": answer,
            })
        except Exception as e:
            print(f"[WARN]  Self-RAG check failed ({e}); accepting the answer.")
            return AnswerCheck()
        return parse_check(reply)
