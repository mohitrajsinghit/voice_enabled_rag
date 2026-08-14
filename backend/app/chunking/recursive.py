"""Recursive text splitter: paragraph → sentence → word boundary splitting."""
from __future__ import annotations

from backend.app.chunking.base import Chunker
from backend.app.schemas import Chunk


class RecursiveChunker(Chunker):
    """Recursively split text trying progressively finer boundaries.

    Split hierarchy:
    1. Paragraph boundaries (double newline)
    2. Sentence boundaries (period + space, etc.)
    3. Word boundaries (whitespace)

    At each level, the algorithm tries to split into chunks that fit within
    `max_chunk_tokens`. If a segment is still too large, it recurses to the
    next finer boundary type. This produces chunks that respect natural
    document structure (paragraphs stay together if they fit).
    """

    name = "recursive"

    # Separators in order of preference: paragraph → sentence → clause → word
    SEPARATORS = [
        "\n\n",       # Paragraph boundary
        "\n",          # Line boundary
        ". ",          # Sentence boundary
        "? ",          # Question boundary
        "! ",          # Exclamation boundary
        "; ",          # Semicolon boundary
        ", ",          # Clause boundary
        " ",           # Word boundary
    ]

    def __init__(self, max_chunk_tokens: int = 256, min_chunk_tokens: int = 20, overlap_tokens: int = 0):
        """Initialize the recursive chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk.
            min_chunk_tokens: Minimum tokens per chunk; smaller chunks get merged.
            overlap_tokens: Number of overlapping tokens between chunks.
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text segments, each within max_chunk_tokens.
        """
        if not text.strip():
            return []

        token_count = len(text.split())

        # Base case: text fits in one chunk
        if token_count <= self.max_chunk_tokens:
            return [text.strip()] if text.strip() else []

        # Base case: no more separators — force split by tokens
        if not separators:
            tokens = text.split()
            segments = []
            for i in range(0, len(tokens), self.max_chunk_tokens):
                segment = " ".join(tokens[i : i + self.max_chunk_tokens])
                if segment.strip():
                    segments.append(segment.strip())
            return segments

        # Try the first separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        parts = text.split(separator)

        # Re-attach separator to each part (except the last) to preserve it
        if separator != " ":
            restored_parts = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    restored_parts.append(part + separator.rstrip())
                else:
                    restored_parts.append(part)
            parts = restored_parts

        # If separator didn't actually split, try next separator
        if len(parts) <= 1:
            return self._split_text(text, remaining_separators)

        # Greedily merge parts into chunks that fit
        segments = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = len(part.split())

            if current_tokens + part_tokens <= self.max_chunk_tokens:
                current_parts.append(part)
                current_tokens += part_tokens
            else:
                # Flush current accumulation
                if current_parts:
                    merged = (" " if separator == " " else " ").join(current_parts)
                    if merged.strip():
                        segments.append(merged.strip())

                # If this single part is too large, recurse with finer separators
                if part_tokens > self.max_chunk_tokens:
                    sub_segments = self._split_text(part, remaining_separators)
                    segments.extend(sub_segments)
                    current_parts = []
                    current_tokens = 0
                else:
                    current_parts = [part]
                    current_tokens = part_tokens

        # Flush remaining
        if current_parts:
            merged = (" " if separator == " " else " ").join(current_parts)
            if merged.strip():
                segments.append(merged.strip())

        return segments

    def _merge_small_chunks(self, segments: list[str]) -> list[str]:
        """Merge segments that are below min_chunk_tokens."""
        if not segments:
            return []

        merged = [segments[0]]
        for segment in segments[1:]:
            prev_tokens = len(merged[-1].split())
            curr_tokens = len(segment.split())

            if curr_tokens < self.min_chunk_tokens and prev_tokens + curr_tokens <= self.max_chunk_tokens:
                merged[-1] = merged[-1] + " " + segment
            else:
                merged.append(segment)

        return merged

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Recursively split text respecting natural boundaries.

        Args:
            doc_id: Source document identifier.
            text: Document text to chunk.
            metadata: Optional metadata dict.

        Returns:
            List of Chunk objects split at natural boundaries.
        """
        if not text or not text.strip():
            return []

        # Recursively split
        segments = self._split_text(text, self.SEPARATORS)

        # Merge small chunks
        segments = self._merge_small_chunks(segments)

        # Build Chunk objects
        chunks = []
        meta = metadata or {}

        for i, segment in enumerate(segments):
            if not segment.strip():
                continue

            chunks.append(
                Chunk(
                    id=f"{doc_id}_recursive_{i}",
                    text=segment.strip(),
                    source_doc_id=doc_id,
                    strategy=self.name,
                    metadata={
                        **meta,
                        "chunk_index": i,
                        "token_count": len(segment.split()),
                    },
                )
            )

        return chunks
