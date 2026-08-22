"""
RAG Pipeline — Question Decomposition (Part 7)

Breaks complex questions into sub-questions, answers each with RAG,
then synthesizes a final answer from all sub-answers.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever


_DECOMPOSITION_TEMPLATE = """\
You are a helpful assistant that generates multiple sub-questions related to \
an input question. The goal is to break down the input into a set of \
sub-problems / sub-questions that can be answered in isolation.

Generate 3 search queries related to: {question}

Output the queries separated by newlines. Do NOT number them."""


_SYNTHESIS_TEMPLATE = """\
Here is a set of Q+A pairs:

{context}

Use these to synthesize a comprehensive answer to the question: {question}"""


class QuestionDecomposer:
    """Decompose a complex question into sub-questions and answer each."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
    ):
        self._retriever = retriever
        self._llm = llm

        self._decompose_chain = (
            ChatPromptTemplate.from_template(_DECOMPOSITION_TEMPLATE)
            | self._llm
            | StrOutputParser()
            | (lambda text: [q.strip() for q in text.strip().split("\n") if q.strip()])
        )

        self._synthesis_prompt = ChatPromptTemplate.from_template(_SYNTHESIS_TEMPLATE)

    def decompose_and_answer(self, question: str) -> tuple[str, list[Document]]:
        """Decompose question, answer sub-questions, synthesize final answer.

        Returns (final_answer, all_retrieved_documents).
        """
        sub_questions = self._decompose_chain.invoke({"question": question})

        all_docs: list[Document] = []
        qa_pairs: list[str] = []

        for sub_q in sub_questions:
            # Retrieve for sub-question
            docs = self._retriever.invoke(sub_q)
            all_docs.extend(docs)

            # Answer sub-question
            context = "\n\n".join(doc.page_content for doc in docs)
            answer_chain = (
                ChatPromptTemplate.from_template(
                    "Answer the question based only on this context:\n\n"
                    "{context}\n\nQuestion: {question}"
                )
                | self._llm
                | StrOutputParser()
            )
            answer = answer_chain.invoke({"context": context, "question": sub_q})
            qa_pairs.append(f"Question: {sub_q}\nAnswer: {answer}")

        # Synthesize final answer
        qa_context = "\n\n".join(qa_pairs)
        final_answer = (
            self._synthesis_prompt | self._llm | StrOutputParser()
        ).invoke({"context": qa_context, "question": question})

        return final_answer, all_docs
