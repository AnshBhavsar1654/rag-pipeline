"""
Retrieval — Ranking

Re-scores a candidate pool and keeps the best ``top_k``:

* ``cross_encoder``: a cross-encoder reads (question, passage) pairs (Re-Rank)
* ``cohere``: Cohere Rerank API
* ``rankgpt``: the LLM orders the passages listwise (RankGPT)
* ``colbert``: ColBERT-style late interaction, MaxSim over token embeddings
  of an encoder model (point ``model`` at a ColBERT or other BERT checkpoint)

(RAG-Fusion's reciprocal rank fusion happens earlier, when the results of
the translated queries are merged.)
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import RerankingConfig


_RANKGPT_PROMPT = ChatPromptTemplate.from_template("""\
I will provide you with {count} passages, each indicated by a number \
identifier []. Rank the passages based on their relevance to the search \
query: {question}

{passages}

Rank all {count} passages above in descending order of relevance. Output \
ONLY the ranking using identifiers, e.g. [2] > [1] > [3].""")


def parse_ranking(text: str, count: int) -> list[int]:
    """Indices (0-based) from a "[2] > [1]" style ranking; missing ones appended."""
    order: list[int] = []
    for match in re.findall(r"\[?(\d+)\]?", text):
        i = int(match) - 1
        if 0 <= i < count and i not in order:
            order.append(i)
    return order + [i for i in range(count) if i not in order]


class Reranker:
    """Re-rank retrieved documents."""

    def __init__(self, config: RerankingConfig, llm: BaseChatModel | None = None):
        self._config = config
        self._enabled = config.enabled
        self._compressor = None
        self._llm = llm

        if not config.enabled:
            return
        init = {
            "cross_encoder": self._init_cross_encoder,
            "cohere": self._init_cohere,
            "rankgpt": self._init_rankgpt,
            "colbert": self._init_colbert,
        }.get(config.provider)
        if init is None:
            raise ValueError(f"Unsupported reranker provider: {config.provider}")
        init()

    def _init_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._config.model)
            print(f"[RANK] Reranker: CrossEncoder ({self._config.model})")
        except ImportError:
            print("[WARN]  sentence-transformers not installed. Disabling re-ranking.")
            self._enabled = False

    def _init_cohere(self) -> None:
        try:
            from langchain_cohere import CohereRerank
            self._compressor = CohereRerank(model=self._config.model, top_n=self._config.top_k)
            print(f"[RANK] Reranker: Cohere ({self._config.model})")
        except ImportError:
            print("[WARN]  langchain-cohere not installed. Disabling re-ranking.")
            self._enabled = False

    def _init_rankgpt(self) -> None:
        if self._llm is None:
            print("[WARN]  RankGPT needs an LLM. Disabling re-ranking.")
            self._enabled = False
            return
        self._rankgpt = _RANKGPT_PROMPT | self._llm | StrOutputParser()
        print("[RANK] Reranker: RankGPT (listwise LLM ranking)")

    def _init_colbert(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._colbert = SentenceTransformer(self._config.model)
            print(f"[RANK] Reranker: ColBERT late interaction ({self._config.model})")
        except ImportError:
            print("[WARN]  sentence-transformers not installed. Disabling re-ranking.")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Re-rank documents and return the top_k."""
        if not self._enabled or not documents:
            return documents
        top_k = self._config.top_k
        provider = self._config.provider
        try:
            if provider == "cross_encoder":
                return self._rerank_cross_encoder(query, documents, top_k)
            if provider == "cohere" and self._compressor:
                return list(self._compressor.compress_documents(documents, query))[:top_k]
            if provider == "rankgpt":
                return self._rerank_rankgpt(query, documents, top_k)
            if provider == "colbert":
                return self._rerank_colbert(query, documents, top_k)
        except Exception as e:
            print(f"[WARN]  Re-ranking failed ({e}); keeping the retrieval order.")
        return documents[:top_k]

    def _rerank_cross_encoder(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def _rerank_rankgpt(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        passages = "\n".join(
            f"[{i}] {' '.join(doc.page_content.split()[:120])}"
            for i, doc in enumerate(documents, start=1)
        )
        reply = self._rankgpt.invoke({"question": query, "passages": passages, "count": len(documents)})
        return [documents[i] for i in parse_ranking(reply, len(documents))[:top_k]]

    def _rerank_colbert(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        import torch

        def tokens(texts: list[str]) -> list[torch.Tensor]:
            embs = self._colbert.encode(texts, output_value="token_embeddings", convert_to_numpy=False)
            return [torch.nn.functional.normalize(e.float(), dim=-1) for e in embs]

        (q,) = tokens([query])
        scores = [float((q @ d.T).max(dim=1).values.sum()) for d in tokens([d.page_content for d in documents])]
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
