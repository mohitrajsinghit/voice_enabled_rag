"""Tests for retrieval module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.schemas import Chunk, RetrievedChunk


class TestRetriever:
    """Tests for the Retriever class."""

    def _make_retriever(self):
        """Create a retriever with mocked dependencies."""
        from backend.app.retrieval.retriever import Retriever

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)

        # Mock FAISS store
        mock_store = MagicMock()
        mock_store.search.return_value = [
            RetrievedChunk(
                chunk=Chunk(id="c1", text="Result 1", source_doc_id="d1", strategy="fixed", metadata={}),
                score=0.9,
            ),
            RetrievedChunk(
                chunk=Chunk(id="c2", text="Result 2", source_doc_id="d1", strategy="fixed", metadata={}),
                score=0.7,
            ),
        ]

        return Retriever(faiss_store=mock_store, embedder=mock_embedder), mock_embedder, mock_store

    def test_retrieve_returns_results(self):
        retriever, embedder, store = self._make_retriever()
        results, latencies = retriever.retrieve("test query", top_k=5)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        embedder.embed_query.assert_called_once_with("test query")
        store.search.assert_called_once()

    def test_retrieve_returns_latencies(self):
        retriever, _, _ = self._make_retriever()
        _, latencies = retriever.retrieve("test query")

        assert "embed_query_ms" in latencies
        assert "faiss_search_ms" in latencies
        assert "retrieval_total_ms" in latencies
        assert all(v >= 0 for v in latencies.values())

    def test_empty_query_returns_empty(self):
        retriever, _, _ = self._make_retriever()
        results, latencies = retriever.retrieve("")
        assert results == []

    def test_respects_top_k(self):
        retriever, _, store = self._make_retriever()
        retriever.retrieve("test", top_k=3)
        store.search.assert_called_once()
        call_args = store.search.call_args
        assert call_args[1].get("top_k", call_args[0][1] if len(call_args[0]) > 1 else 5) == 3


class TestReranker:
    """Tests for the optional reranker."""

    def test_disabled_reranker_passes_through(self):
        from backend.app.retrieval.reranker import Reranker

        reranker = Reranker(enabled=False)
        chunks = [
            RetrievedChunk(
                chunk=Chunk(id="c1", text="Text", source_doc_id="d1", strategy="fixed", metadata={}),
                score=0.5,
            )
        ]
        result, latency = reranker.rerank("query", chunks)
        assert result == chunks
        assert latency == 0.0

    def test_empty_chunks(self):
        from backend.app.retrieval.reranker import Reranker

        reranker = Reranker(enabled=True)
        result, latency = reranker.rerank("query", [])
        assert result == []
