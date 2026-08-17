"""Semantic chunking using embedding similarity to detect topic boundaries."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from backend.app.chunking.base import Chunker
from backend.app.schemas import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK with fallback."""
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback: split on period + space
        parts = text.replace("! ", ". ").replace("? ", ". ").split(". ")
        return [p.strip() + "." for p in parts if p.strip()]


class SemanticChunker(Chunker):
    """Split text at semantic boundaries detected via embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence with a sentence-transformer model
    3. Compute cosine similarity between adjacent sentence embeddings
    4. Identify boundary points where similarity drops below a threshold
       (using percentile-based cutoff of the similarity distribution)
    5. Group sentences between boundaries into chunks
    6. Merge chunks that are too small (below min_chunk_tokens)

    This produces chunks where each chunk is semantically coherent — sentences
    within a chunk discuss similar topics, and chunk boundaries align with
    topic shifts. Fundamentally different from fixed-size windowing.
    """

    name = "semantic"

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        embedder: Any | None = None,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_percentile: float = 25.0,
        min_chunk_tokens: int = 30,
        max_chunk_tokens: int = 512,
    ):
        self._model = model
        self._embedder = embedder
        self._model_name = model_name
        self.similarity_percentile = similarity_percentile
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model with GPU acceleration when available."""
        if self._model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from sentence_transformers import SentenceTransformer
            dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
            logger.info(f"Loading chunker embedding model on {device.upper()} ({dev_name}): {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    def _compute_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between adjacent embeddings."""
        if len(embeddings) < 2:
            return np.array([])

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        normalized = embeddings / norms

        # Cosine similarity between adjacent pairs
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
        return similarities

    def _find_boundaries(self, similarities: np.ndarray) -> list[int]:
        """Find boundary indices where similarity drops below threshold."""
        if len(similarities) == 0:
            return []

        # Compute threshold as a percentile of the similarity distribution
        threshold = np.percentile(similarities, self.similarity_percentile)

        # Boundaries are positions where similarity is below threshold
        boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        return boundaries

    def _merge_small_chunks(self, sentence_groups: list[list[str]]) -> list[list[str]]:
        """Merge chunks that are too small with their neighbors."""
        if not sentence_groups:
            return []

        merged = [sentence_groups[0]]
        for group in sentence_groups[1:]:
            prev_tokens = sum(len(s.split()) for s in merged[-1])
            curr_tokens = sum(len(s.split()) for s in group)

            if curr_tokens < self.min_chunk_tokens and prev_tokens < self.max_chunk_tokens:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text at semantic boundaries.

        Args:
            doc_id: Source document identifier.
            text: Document text to chunk.
            metadata: Optional metadata dict.

        Returns:
            List of semantically coherent Chunk objects.
        """
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Short text (1-2 sentences within max tokens) -> cohesive single chunk directly
        token_count = sum(len(s.split()) for s in sentences)
        if len(sentences) <= 2 and token_count <= self.max_chunk_tokens:
            return [
                Chunk(
                    id=f"{doc_id}_semantic_0",
                    text=" ".join(sentences),
                    source_doc_id=doc_id,
                    strategy=self.name,
                    metadata={**(metadata or {}), "sentence_count": len(sentences), "token_count": token_count},
                )
            ]

        # Embed all sentences (multi-sentence passages)
        if self._embedder is not None:
            embeddings = self._embedder.embed(sentences, show_progress=False)
        else:
            embeddings = self.model.encode(sentences, show_progress_bar=False)

        # Compute adjacent similarities
        similarities = self._compute_similarities(embeddings)

        # Find boundary points
        boundaries = self._find_boundaries(similarities)

        # Group sentences by boundaries
        groups: list[list[str]] = []
        prev_idx = 0
        for boundary_idx in boundaries:
            group = sentences[prev_idx:boundary_idx]
            if group:
                groups.append(group)
            prev_idx = boundary_idx
        # Add the last group
        remaining = sentences[prev_idx:]
        if remaining:
            groups.append(remaining)

        # Merge small chunks
        groups = self._merge_small_chunks(groups)

        # Build Chunk objects
        chunks = []
        meta = metadata or {}
        for i, group in enumerate(groups):
            chunk_text = " ".join(group)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_semantic_{i}",
                    text=chunk_text,
                    source_doc_id=doc_id,
                    strategy=self.name,
                    metadata={
                        **meta,
                        "sentence_count": len(group),
                        "token_count": len(chunk_text.split()),
                    },
                )
            )

        return chunks
