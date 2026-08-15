"""Dual-provider embedding module supporting local sentence-transformers and LM Studio."""
from __future__ import annotations

import logging
from typing import Protocol, TYPE_CHECKING
import numpy as np

from backend.app.config import EmbeddingProvider, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbedder(Protocol):
    """Protocol defining the embedder interface."""

    @property
    def dimension(self) -> int:
        ...

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


class LocalEmbedder:
    """Wrapper around sentence-transformers for in-process embedding."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
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
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts in batches."""
        if not texts:
            return np.array([])

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            formatted_texts = [prefix + t if not t.startswith(prefix) else t for t in texts]
        else:
            formatted_texts = texts

        embeddings = self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        result = self.embed([text], show_progress=False, is_query=True)
        return result[0]


# Backwards compatibility alias
Embedder = LocalEmbedder


class LMStudioEmbedder:
    """Wrapper for LM Studio embeddings endpoint (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str = "text-embedding-qwen3-embedding-0.6b",
        base_url: str = "http://localhost:1234/v1",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension from LM Studio."""
        if self._dim is None:
            sample = self.embed_query("dimension probe")
            self._dim = len(sample)
            logger.info(f"LM Studio embedding dimension: {self._dim} (model: {self.model_name})")
        return self._dim

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts in batches using LM Studio /v1/embeddings."""
        if not texts:
            return np.array([])

        import httpx

        all_embeddings = []
        url = f"{self.base_url}/embeddings"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = httpx.post(
                    url,
                    json={"model": self.model_name, "input": batch},
                    timeout=60.0,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"LM Studio embedding error ({resp.status_code}): {resp.text}")

                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                batch_emb = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_emb)
            except Exception as e:
                logger.error(f"LM Studio embedding request failed: {e}")
                raise

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize vectors so inner product == cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed([text], batch_size=1, show_progress=False)[0]


# Module-level cached embedder factory
_embedder_cache: dict[str, BaseEmbedder] = {}


def get_embedder(
    model_name: str | None = None,
    provider: str | EmbeddingProvider | None = None,
    base_url: str | None = None,
) -> BaseEmbedder:
    """Get a cached embedder instance (Local or LM Studio).

    Args:
        model_name: Model identifier (optional, defaults to config).
        provider: "local" or "lmstudio" (optional, defaults to config).
        base_url: Base URL for LM Studio (optional, defaults to config).

    Returns:
        Embedder instance implementing BaseEmbedder.
    """
    settings = get_settings()
    prov = provider or settings.embedding_provider
    if isinstance(prov, str):
        prov = EmbeddingProvider(prov.lower())

    if prov == EmbeddingProvider.LMSTUDIO:
        m_name = model_name or settings.lmstudio_embedding_model
        b_url = base_url or settings.lmstudio_base_url
        cache_key = f"lmstudio:{b_url}:{m_name}"
        if cache_key not in _embedder_cache:
            _embedder_cache[cache_key] = LMStudioEmbedder(model_name=m_name, base_url=b_url)
        return _embedder_cache[cache_key]
    else:
        m_name = model_name or settings.embedding_model
        cache_key = f"local:{m_name}"
        if cache_key not in _embedder_cache:
            _embedder_cache[cache_key] = LocalEmbedder(model_name=m_name)
        return _embedder_cache[cache_key]
