"""Query Translation: decompose or re-phrase the input question."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import RetrievalConfig
from src.query_translation.base import QueryTranslator
from src.query_translation.decomposition import DecompositionTranslator
from src.query_translation.hyde import HyDETranslator
from src.query_translation.multi_query import MultiQueryTranslator
from src.query_translation.rag_fusion import RAGFusionTranslator
from src.query_translation.step_back import StepBackTranslator


def create_translator(config: RetrievalConfig, llm: BaseChatModel) -> QueryTranslator:
    """Build the query translator named by ``retrieval.strategy``."""
    strategy = config.strategy
    if strategy == "simple":
        return QueryTranslator()
    if strategy == "multi_query":
        return MultiQueryTranslator(llm, num_queries=config.multi_query.num_queries)
    if strategy == "rag_fusion":
        return RAGFusionTranslator(
            llm, num_queries=config.rag_fusion.num_queries, rrf_k=config.rag_fusion.rrf_k,
        )
    if strategy == "hyde":
        return HyDETranslator(llm, hypothesis_prompt=config.hyde.hypothesis_prompt)
    if strategy == "step_back":
        return StepBackTranslator(llm)
    if strategy == "decomposition":
        return DecompositionTranslator(llm, num_queries=3)
    raise ValueError(f"Unknown retrieval strategy: {strategy}")
