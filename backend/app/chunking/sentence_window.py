"""Sentence-window chunking: index atomic sentences, retrieve contextual windows."""
from __future__ import annotations

import json

from backend.app.chunking.base import Chunker
from backend.app.schemas import Chunk


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
        parts = text.replace("! ", ". ").replace("? ", ". ").split(". ")
        return [p.strip() + "." for p in parts if p.strip()]


class SentenceWindowChunker(Chunker):
    """Index individual sentences but store context for window expansion at retrieval.

    This strategy is fundamentally different from both fixed-size and semantic:
    - The indexed unit is a **single sentence** (for precise matching)
    - Each chunk stores the full sentence list and its position index
    - At retrieval time, the system expands to a window of ±N surrounding
      sentences to provide context

    This gives the best of both worlds: precise sentence-level retrieval
    relevance with paragraph-level context in the final output.

    The `window_size` parameter controls how many sentences before and after
    the matched sentence are included when expanding.
    """

    name = "sentence_window"

    def __init__(self, window_size: int = 2):
        """Initialize the sentence-window chunker.

        Args:
            window_size: Number of sentences before/after to include in the
                retrieval window. Default 2 means ±2 sentences (5 total).
        """
        self.window_size = window_size

    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into individual sentences with window metadata.

        Each chunk contains:
        - text: the single sentence (what gets embedded/indexed)
        - metadata.sentence_index: position in the sentence list
        - metadata.all_sentences: JSON-encoded list of all sentences (for window expansion)
        - metadata.window_size: configured window size
        - metadata.window_text: pre-computed expanded window text

        Args:
            doc_id: Source document identifier.
            text: Document text to chunk.
            metadata: Optional metadata dict.

        Returns:
            List of single-sentence Chunk objects with window metadata.
        """
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks = []
        meta = metadata or {}

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Pre-compute the expanded window text
            window_start = max(0, i - self.window_size)
            window_end = min(len(sentences), i + self.window_size + 1)
            window_text = " ".join(sentences[window_start:window_end])

            chunks.append(
                Chunk(
                    id=f"{doc_id}_sw_{i}",
                    text=sentence,  # Indexed unit: single sentence
                    source_doc_id=doc_id,
                    strategy=self.name,
                    metadata={
                        **meta,
                        "sentence_index": i,
                        "total_sentences": len(sentences),
                        "window_size": self.window_size,
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_text": window_text,  # Expanded context
                        "token_count": len(sentence.split()),
                        "window_token_count": len(window_text.split()),
                    },
                )
            )

        return chunks

    @staticmethod
    def expand_window(chunk: Chunk) -> str:
        """Expand a retrieved sentence chunk to its full context window.

        Args:
            chunk: A retrieved Chunk from sentence_window strategy.

        Returns:
            The expanded window text with surrounding sentences.
        """
        return chunk.metadata.get("window_text", chunk.text)
