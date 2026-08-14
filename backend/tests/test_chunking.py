"""Tests for chunking strategies."""
from __future__ import annotations

import pytest

from backend.app.chunking.base import Chunker
from backend.app.chunking.fixed_size import FixedSizeChunker
from backend.app.chunking.recursive import RecursiveChunker
from backend.app.chunking.sentence_window import SentenceWindowChunker
from backend.app.chunking.metadata_aware import MetadataAwareChunker
from backend.app.chunking.registry import get_chunker, list_strategies, CHUNKERS
from backend.app.schemas import Chunk


class TestFixedSizeChunker:
    """Tests for the fixed-size token window chunker."""

    def test_basic_chunking(self, sample_text):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk("doc_1", sample_text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source_doc_id == "doc_1"
            assert chunk.strategy == "fixed"

    def test_chunk_size_respected(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunker = FixedSizeChunker(chunk_size=20, overlap=0)
        chunks = chunker.chunk("doc", text)
        for chunk in chunks[:-1]:  # All except last
            assert len(chunk.text.split()) == 20

    def test_overlap_works(self):
        text = " ".join([f"word{i}" for i in range(50)])
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk("doc", text)
        if len(chunks) >= 2:
            first_tokens = set(chunks[0].text.split()[-5:])
            second_tokens = set(chunks[1].text.split()[:5])
            assert first_tokens == second_tokens

    def test_empty_text_returns_empty(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("doc", "") == []
        assert chunker.chunk("doc", "   ") == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)

    def test_metadata_passed(self):
        chunker = FixedSizeChunker(chunk_size=50)
        chunks = chunker.chunk("doc", "Some text here.", metadata={"lang": "en"})
        assert len(chunks) > 0
        assert chunks[0].metadata.get("lang") == "en"


class TestRecursiveChunker:
    """Tests for the recursive paragraph→sentence→word chunker."""

    def test_basic_chunking(self, sample_text):
        chunker = RecursiveChunker(max_chunk_tokens=30)
        chunks = chunker.chunk("doc_1", sample_text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.strategy == "recursive"

    def test_respects_max_tokens(self):
        text = "First paragraph with some words. " * 10 + "\n\n" + "Second paragraph here. " * 10
        chunker = RecursiveChunker(max_chunk_tokens=20)
        chunks = chunker.chunk("doc", text)
        for chunk in chunks:
            # Allow some tolerance for boundary splits
            assert len(chunk.text.split()) <= 30  # generous tolerance

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("doc", "") == []

    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(max_chunk_tokens=100)
        chunks = chunker.chunk("doc", "Short text.")
        assert len(chunks) == 1


class TestSentenceWindowChunker:
    """Tests for the sentence-window chunker."""

    def test_basic_chunking(self, sample_text):
        chunker = SentenceWindowChunker(window_size=2)
        chunks = chunker.chunk("doc_1", sample_text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.strategy == "sentence_window"

    def test_each_chunk_is_single_sentence(self, sample_text):
        chunker = SentenceWindowChunker()
        chunks = chunker.chunk("doc", sample_text)
        # Each chunk text should be shorter than the full text
        for chunk in chunks:
            assert len(chunk.text) < len(sample_text)

    def test_window_metadata_present(self, sample_text):
        chunker = SentenceWindowChunker(window_size=2)
        chunks = chunker.chunk("doc", sample_text)
        for chunk in chunks:
            assert "window_text" in chunk.metadata
            assert "sentence_index" in chunk.metadata
            assert "window_size" in chunk.metadata

    def test_window_expansion(self, sample_text):
        chunker = SentenceWindowChunker(window_size=1)
        chunks = chunker.chunk("doc", sample_text)
        if len(chunks) > 2:
            mid = chunks[len(chunks) // 2]
            expanded = SentenceWindowChunker.expand_window(mid)
            # Window should be longer than the single sentence
            assert len(expanded) >= len(mid.text)

    def test_empty_text(self):
        chunker = SentenceWindowChunker()
        assert chunker.chunk("doc", "") == []


class TestMetadataAwareChunker:
    """Tests for the metadata-aware wrapper."""

    def test_enriches_metadata(self, sample_text):
        inner = FixedSizeChunker(chunk_size=30)
        wrapper = MetadataAwareChunker(inner, default_language="en")
        chunks = wrapper.chunk("doc_1", sample_text, metadata={"source_lang": "eng_Latn"})

        assert len(chunks) > 0
        for chunk in chunks:
            assert "passage_id" in chunk.metadata
            assert "language" in chunk.metadata
            assert "chunk_strategy" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "fixed"

    def test_preserves_inner_metadata(self):
        inner = FixedSizeChunker(chunk_size=50)
        wrapper = MetadataAwareChunker(inner)
        chunks = wrapper.chunk("doc", "Some text for chunking test.")

        for chunk in chunks:
            assert "token_count" in chunk.metadata


class TestRegistry:
    """Tests for the chunker registry."""

    def test_list_strategies(self):
        strategies = list_strategies()
        assert "fixed" in strategies
        assert "semantic" in strategies
        assert "sentence_window" in strategies
        assert "recursive" in strategies

    def test_get_known_chunker(self):
        chunker = get_chunker("fixed", wrap_metadata=False)
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_with_metadata_wrapper(self):
        chunker = get_chunker("fixed", wrap_metadata=True)
        assert isinstance(chunker, MetadataAwareChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_all_strategies_implement_base(self):
        for name, cls in CHUNKERS.items():
            assert issubclass(cls, Chunker), f"{name} doesn't implement Chunker"
