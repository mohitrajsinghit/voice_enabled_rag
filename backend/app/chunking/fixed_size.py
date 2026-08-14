"""Fixed-size token window chunking with overlap."""
from __future__ import annotations

import uuid

from backend.app.chunking.base import Chunker
from backend.app.schemas import Chunk


class FixedSizeChunker(Chunker):
    """Split text into fixed-size token windows with configurable overlap.

    Uses whitespace tokenization for speed. Each chunk gets exactly
    `chunk_size` tokens (except possibly the last), with `overlap` tokens
    shared between consecutive chunks.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 256, overlap: int | None = None):
        """Initialize the fixed-size chunker.

        Args:
            chunk_size: Number of tokens per chunk.
            overlap: Number of overlapping tokens between consecutive chunks.
        """
        if overlap is None:
            overlap = 50 if chunk_size > 50 else max(0, chunk_size // 5)

        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into fixed-size token windows.

        Args:
            doc_id: Source document identifier.
            text: Document text to chunk.
            metadata: Optional metadata dict.

        Returns:
            List of Chunk objects with fixed token windows.
        """
        if not text or not text.strip():
            return []

        tokens = text.split()
        if not tokens:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        meta = metadata or {}

        for i in range(0, len(tokens), step):
            window_tokens = tokens[i : i + self.chunk_size]
            chunk_text = " ".join(window_tokens)

            if not chunk_text.strip():
                continue

            chunk_id = f"{doc_id}_fixed_{len(chunks)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_doc_id=doc_id,
                    strategy=self.name,
                    metadata={
                        **meta,
                        "token_start": i,
                        "token_end": i + len(window_tokens),
                        "token_count": len(window_tokens),
                    },
                )
            )

            # Stop if we've covered all tokens
            if i + self.chunk_size >= len(tokens):
                break

        return chunks
