"""Retriever: embed query + FAISS search — the <200ms critical path."""
from __future__ import annotations

import logging
import time

from backend.app.indexing.embedder import Embedder
from backend.app.indexing.faiss_store import FaissStore
from backend.app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """Query embedding + FAISS search pipeline.

    This is the performance-critical path that should complete in <200ms.
    No LLM calls happen here — just embed the query and search the index.
    """

    def __init__(self, faiss_store: FaissStore, embedder: Embedder):
        """Initialize the retriever.

        Args:
            faiss_store: Loaded FAISS vector store.
            embedder: Text embedder instance.
        """
        self.faiss_store = faiss_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query text string.
            top_k: Number of chunks to retrieve.

        Returns:
            Tuple of (retrieved chunks, latency dict with embed_query_ms and faiss_search_ms).
        """
        if not query or not query.strip():
            return [], {"embed_query_ms": 0, "faiss_search_ms": 0}

        # Embed query
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # FAISS search
        t1 = time.perf_counter()
        results = self.faiss_store.search(query_embedding, top_k=top_k)
        search_ms = (time.perf_counter() - t1) * 1000

        total_ms = embed_ms + search_ms
        latencies = {
            "embed_query_ms": round(embed_ms, 2),
            "faiss_search_ms": round(search_ms, 2),
            "retrieval_total_ms": round(total_ms, 2),
        }

        logger.info(
            f"Retrieved {len(results)} chunks for query in {total_ms:.1f}ms "
            f"(embed={embed_ms:.1f}ms, search={search_ms:.1f}ms)",
            extra={"stage": "retrieval", "latency_ms": total_ms},
        )

        return results, latencies
