"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.schemas import Chunk, RetrievedChunk, TranscriptResult


@pytest.fixture
def sample_text():
    """A sample paragraph for testing chunking."""
    return (
        "The Taj Mahal is a white marble mausoleum in Agra, India. "
        "It was built by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal. "
        "The construction began in 1632 and was completed in 1653. "
        "It is considered one of the most beautiful buildings in the world. "
        "The Taj Mahal is a UNESCO World Heritage Site and attracts millions of visitors each year. "
        "The main dome is 73 meters high and is surrounded by four minarets. "
        "The gardens around the Taj Mahal follow the Persian garden design. "
        "The interior contains the cenotaphs of Shah Jahan and Mumtaz Mahal."
    )


@pytest.fixture
def sample_chunks():
    """A list of sample chunks for testing."""
    return [
        Chunk(
            id="doc_1_0",
            text="The Taj Mahal is a white marble mausoleum in Agra, India.",
            source_doc_id="doc_1",
            strategy="fixed",
            metadata={"token_count": 12},
        ),
        Chunk(
            id="doc_1_1",
            text="It was built by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal.",
            source_doc_id="doc_1",
            strategy="fixed",
            metadata={"token_count": 15},
        ),
        Chunk(
            id="doc_2_0",
            text="Python is a programming language known for its simplicity.",
            source_doc_id="doc_2",
            strategy="fixed",
            metadata={"token_count": 10},
        ),
    ]


@pytest.fixture
def sample_retrieved_chunks(sample_chunks):
    """Sample retrieved chunks with scores."""
    return [
        RetrievedChunk(chunk=sample_chunks[0], score=0.85),
        RetrievedChunk(chunk=sample_chunks[1], score=0.72),
    ]


@pytest.fixture
def sample_transcript():
    """A sample transcript result."""
    return TranscriptResult(
        text="What is the Taj Mahal?",
        language="en",
        confidence=0.95,
        latency_ms=150.0,
    )
