"""
Query Translation — Question Decomposition

Breaks a complex question into sub-questions that can each be searched in
isolation, so the answer sees evidence for every part.
"""

from __future__ import annotations

from src.query_translation.base import LLMQueryTranslator


class DecompositionTranslator(LLMQueryTranslator):
    name = "Question Decomposition"
    template = """\
You are a helpful assistant that generates multiple sub-questions related to \
an input question. The goal is to break down the input into a set of \
sub-problems / sub-questions that can be answered in isolation.

Generate up to {num_queries} sub-questions for: {question}

Output ONLY the sub-questions, one per line, with no numbering or preamble. \
If the question is already simple, output it unchanged."""
