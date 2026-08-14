"""Metadata-aware chunker wrapper that enriches chunks with standardized metadata."""
from __future__ import annotations

from backend.app.chunking.base import Chunker
from backend.app.schemas import Chunk


class MetadataAwareChunker(Chunker):
    """Decorator/wrapper that enriches any chunker's output with standardized metadata.

    This is NOT a separate splitting algorithm. It wraps any Chunker implementation
    and adds consistent metadata fields to every chunk:
    - passage_id: original document/passage identifier
    - language: source language of the text
    - token_count: number of whitespace tokens
    - chunk_strategy: which chunking strategy produced this chunk
    - source_doc_id: parent document ID

    Use this to ensure all chunks from any strategy have uniform metadata
    for downstream indexing and retrieval.
    """

    name = "metadata_aware"

    def __init__(
        self,
        inner_chunker: Chunker,
        default_language: str = "en",
    ):
        """Initialize the metadata-aware wrapper.

        Args:
            inner_chunker: The actual chunking strategy to wrap.
            default_language: Default language code if not specified in metadata.
        """
        self.inner_chunker = inner_chunker
        self.default_language = default_language
        # Expose the inner chunker's name
        self.name = inner_chunker.name

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk text using the inner strategy, then enrich metadata.

        Args:
            doc_id: Source document identifier.
            text: Document text to chunk.
            metadata: Optional metadata dict.

        Returns:
            List of Chunk objects with enriched metadata.
        """
        # Delegate to inner chunker
        chunks = self.inner_chunker.chunk(doc_id, text, metadata)

        # Enrich each chunk with standardized metadata
        enriched = []
        for chunk in chunks:
            enriched_meta = {
                **(metadata or {}),
                **chunk.metadata,
                "passage_id": doc_id,
                "language": (metadata or {}).get("language", self.default_language),
                "source_lang": (metadata or {}).get("source_lang", ""),
                "target_lang": (metadata or {}).get("target_lang", ""),
                "chunk_strategy": self.inner_chunker.name,
                "token_count": chunk.metadata.get("token_count", len(chunk.text.split())),
                "source_doc_id": chunk.source_doc_id,
            }

            enriched.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    source_doc_id=chunk.source_doc_id,
                    strategy=chunk.strategy,
                    metadata=enriched_meta,
                )
            )

        return enriched
