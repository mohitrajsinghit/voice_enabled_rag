"""Sentence-transformers embedding wrapper with batched encoding."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for embedding text.

    Supports batched encoding with progress tracking, and caches the
    model as a singleton to avoid repeated loading.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts in batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)


# Module-level cached embedder factory
_embedder_cache: dict[str, Embedder] = {}


def get_embedder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> Embedder:
    """Get a cached embedder instance.

    Args:
        model_name: Model identifier.

    Returns:
        Cached Embedder instance.
    """
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = Embedder(model_name)
    return _embedder_cache[model_name]
