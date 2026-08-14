"""FAISS vector store: load index from disk, search for similar chunks."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from backend.app.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class FaissStore:
    """FAISS vector store for chunk retrieval.

    Manages a FAISS index and its associated chunk metadata. Supports
    loading from disk and searching for similar chunks given a query embedding.
    """

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
    ):
        """Initialize the FAISS store.

        Args:
            index: Loaded FAISS index.
            chunks: List of Chunk objects matching the index vectors.
        """
        self.index = index
        self.chunks = chunks
        logger.info(f"FaissStore initialized with {index.ntotal} vectors and {len(chunks)} chunks")

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> FaissStore:
        """Load a FAISS store from disk.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the chunk metadata JSONL file.

        Returns:
            Loaded FaissStore instance.

        Raises:
            FileNotFoundError: If either file doesn't exist.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")

        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))

        logger.info(f"Loading chunk metadata from {metadata_path}")
        chunks = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    chunks.append(Chunk(**data))

        if index.ntotal != len(chunks):
            logger.warning(
                f"Index/metadata mismatch: {index.ntotal} vectors vs {len(chunks)} chunks. "
                f"Using min({index.ntotal}, {len(chunks)})."
            )

        return cls(index=index, chunks=chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search the FAISS index for similar chunks.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by similarity (descending).
        """
        if self.index.ntotal == 0:
            return []

        # Reshape for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            # FAISS inner product returns similarity directly for normalized vectors
            # For L2 index, convert distance to similarity
            score = float(dist)
            if score < 0:
                # L2 distance: convert to similarity (higher = better)
                score = 1.0 / (1.0 + abs(score))

            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=score,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal
