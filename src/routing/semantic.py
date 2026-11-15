"""
Routing — Semantic routing

Embeds the question and picks the prompt whose description is most similar
(cosine similarity), e.g. a step-by-step explainer prompt for "how does X
work?" or a comparison prompt for "X vs Y". No LLM call is needed.
"""

from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings


class SemanticRouter:
    def __init__(self, embeddings: Embeddings, prompts: dict[str, tuple[str, str]]):
        """*prompts* maps a route name to (description, prompt text)."""
        self._embeddings = embeddings
        self._names = list(prompts)
        self._prompts = prompts
        self._matrix: np.ndarray | None = None

    def _description_vectors(self) -> np.ndarray:
        if self._matrix is None:
            vectors = np.array(self._embeddings.embed_documents(
                [self._prompts[n][0] for n in self._names]
            ), dtype=float)
            self._matrix = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        return self._matrix

    def route(self, question: str) -> tuple[str, str, float]:
        """Return (route name, prompt text, similarity) for *question*."""
        if not self._names:
            return "", "", 0.0
        q = np.array(self._embeddings.embed_query(question), dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        scores = self._description_vectors() @ q
        best = int(np.argmax(scores))
        name = self._names[best]
        return name, self._prompts[name][1], float(scores[best])
