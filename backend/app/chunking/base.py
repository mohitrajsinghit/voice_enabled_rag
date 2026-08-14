"""Abstract base class for all chunking strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.schemas import Chunk


class Chunker(ABC):
    """Base class for document chunking strategies.

    Every strategy must implement the `chunk` method which takes a document
    and returns a list of typed Chunk objects.
    """

    name: str = "base"

    @abstractmethod
    def chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            doc_id: Unique identifier for the source document.
            text: Full text content of the document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        ...
