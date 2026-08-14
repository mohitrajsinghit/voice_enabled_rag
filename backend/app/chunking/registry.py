"""Chunker registry: factory for getting chunking strategies by name."""
from __future__ import annotations

from typing import Any

from backend.app.chunking.base import Chunker
from backend.app.chunking.fixed_size import FixedSizeChunker
from backend.app.chunking.recursive import RecursiveChunker
from backend.app.chunking.semantic import SemanticChunker
from backend.app.chunking.sentence_window import SentenceWindowChunker
from backend.app.chunking.metadata_aware import MetadataAwareChunker


# Registry of available chunking strategies
CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "semantic": SemanticChunker,
    "sentence_window": SentenceWindowChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(
    name: str,
    wrap_metadata: bool = True,
    **kwargs: Any,
) -> Chunker:
    """Get a chunker by name from the registry.

    Args:
        name: Strategy name (one of: fixed, semantic, sentence_window, recursive).
        wrap_metadata: Whether to wrap with MetadataAwareChunker (default True).
        **kwargs: Additional keyword arguments passed to the chunker constructor.

    Returns:
        Configured Chunker instance.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    if name not in CHUNKERS:
        available = ", ".join(sorted(CHUNKERS.keys()))
        raise ValueError(f"Unknown chunking strategy '{name}'. Available: {available}")

    chunker = CHUNKERS[name](**kwargs)

    if wrap_metadata:
        chunker = MetadataAwareChunker(chunker)

    return chunker


def list_strategies() -> list[str]:
    """List all available chunking strategy names."""
    return sorted(CHUNKERS.keys())
