"""Optional cross-encoder reranker for improving retrieval quality."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from backend.app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for retrieved chunks.

    Uses a cross-encoder model to rescore query-chunk pairs for improved
    relevance ranking. Can be disabled via config for latency-sensitive use.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
    ):
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model name.
            enabled: Whether reranking is enabled.
        """
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], float]:
        """Rerank retrieved chunks using cross-encoder scoring.

        Args:
            query: Query text.
            chunks: Retrieved chunks to rerank.

        Returns:
            Tuple of (reranked chunks, reranking latency in ms).
        """
        if not self.enabled or not chunks:
            return chunks, 0.0

        t0 = time.perf_counter()

        # Create query-chunk pairs
        pairs = [(query, chunk.chunk.text) for chunk in chunks]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Update scores and re-sort
        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked.append(
                RetrievedChunk(
                    chunk=chunk.chunk,
                    score=float(score),
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Reranked {len(chunks)} chunks in {latency_ms:.1f}ms")

        return reranked, latency_ms
