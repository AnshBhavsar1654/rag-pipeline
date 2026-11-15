"""
Query Translation — RAG-Fusion

Generates query variants whose rankings are fused with Reciprocal Rank Fusion.
"""

from __future__ import annotations

from src.query_translation.base import LLMQueryTranslator


class RAGFusionTranslator(LLMQueryTranslator):
    name = "RAG-Fusion (RRF)"
    template = """\
You are a helpful assistant that generates multiple search queries based on \
a single input query. Generate {num_queries} different search queries related to \
the following question. Provide diverse perspectives to maximize retrieval coverage.

Output ONLY the queries, one per line, with no numbering or preamble.

Original question: {question}"""
