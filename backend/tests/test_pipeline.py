"""Integration test for the full pipeline with mocked STT/LLM."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.app.schemas import (
    Chunk, GuardrailVerdict, PipelineContext, PipelineResponse,
    RetrievedChunk, TranscriptResult,
)


@pytest.fixture
def mock_stt_client():
    """Mock Sarvam STT client."""
    client = MagicMock()
    client.transcribe_or_passthrough = AsyncMock(return_value=TranscriptResult(
        text="What is the Taj Mahal?",
        language="en",
        confidence=0.95,
        latency_ms=0.0,
    ))
    return client


@pytest.fixture
def mock_input_filter():
    """Mock input filter that passes everything."""
    filter_ = MagicMock()
    filter_.check.return_value = GuardrailVerdict(passed=True, category="ok")
    return filter_


@pytest.fixture
def mock_retriever():
    """Mock retriever returning relevant chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = (
        [
            RetrievedChunk(
                chunk=Chunk(
                    id="c1",
                    text="The Taj Mahal is a white marble mausoleum in Agra, India.",
                    source_doc_id="d1",
                    strategy="semantic",
                    metadata={},
                ),
                score=0.85,
            ),
            RetrievedChunk(
                chunk=Chunk(
                    id="c2",
                    text="It was built by Shah Jahan in memory of Mumtaz Mahal.",
                    source_doc_id="d1",
                    strategy="semantic",
                    metadata={},
                ),
                score=0.72,
            ),
        ],
        {"embed_query_ms": 5.0, "faiss_search_ms": 1.0, "retrieval_total_ms": 6.0},
    )
    return retriever


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.generate.return_value = (
        "The Taj Mahal is a white marble mausoleum in Agra, India, "
        "built by Mughal emperor Shah Jahan [Source 1][Source 2].",
        500.0,
    )
    client.parse_json_response.return_value = {
        "verdict": "supported",
        "reason": "All claims verified in sources",
    }
    return client


@pytest.fixture
def mock_policy(mock_llm_client):
    """Mock guardrail policy."""
    from backend.app.guardrails.policy import GuardrailPolicy
    policy = MagicMock(spec=GuardrailPolicy)
    policy.check_retrieval_quality.return_value = GuardrailVerdict(passed=True, category="ok")
    policy.check_grounding_with_retry.return_value = (
        "The Taj Mahal is a white marble mausoleum [Source 1].",
        GuardrailVerdict(passed=True, reason="Grounded", category="ok"),
        200.0,
    )
    return policy


@pytest.mark.asyncio
async def test_happy_path_text_query(
    mock_stt_client, mock_input_filter, mock_retriever, mock_llm_client, mock_policy
):
    """Test successful pipeline: text → retrieval → generation → grounded answer."""
    from backend.app.harness.pipeline import VoiceRAGPipeline

    pipeline = VoiceRAGPipeline(
        stt_client=mock_stt_client,
        input_filter=mock_input_filter,
        retriever=mock_retriever,
        llm_client=mock_llm_client,
        policy=mock_policy,
    )

    response = await pipeline.run(text_input="What is the Taj Mahal?")

    assert isinstance(response, PipelineResponse)
    assert response.status == "answered"
    assert response.transcript == "What is the Taj Mahal?"
    assert response.answer is not None
    assert len(response.sources) > 0
    assert response.guardrail.passed
    assert "end_to_end_ms" in response.latencies


@pytest.mark.asyncio
async def test_off_topic_query_refused(
    mock_stt_client, mock_retriever, mock_llm_client, mock_policy
):
    """Test that off-topic queries are refused at the input guardrail."""
    from backend.app.harness.pipeline import VoiceRAGPipeline

    # Input filter rejects
    rejecting_filter = MagicMock()
    rejecting_filter.check.return_value = GuardrailVerdict(
        passed=False,
        reason="Query is off-topic",
        category="off_topic",
    )

    pipeline = VoiceRAGPipeline(
        stt_client=mock_stt_client,
        input_filter=rejecting_filter,
        retriever=mock_retriever,
        llm_client=mock_llm_client,
        policy=mock_policy,
    )

    response = await pipeline.run(text_input="What is the weather today?")

    assert response.status == "refused"
    assert response.answer is None
    # Retriever should NOT have been called (short-circuited)
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_ungrounded_answer_refused(
    mock_stt_client, mock_input_filter, mock_retriever, mock_llm_client
):
    """Test that ungrounded answers are refused after retry."""
    from backend.app.harness.pipeline import VoiceRAGPipeline

    # Policy says grounding failed
    failing_policy = MagicMock()
    failing_policy.check_retrieval_quality.return_value = GuardrailVerdict(passed=True, category="ok")
    failing_policy.check_grounding_with_retry.return_value = (
        "",
        GuardrailVerdict(
            passed=False,
            reason="Answer not grounded",
            category="ungrounded",
        ),
        300.0,
    )

    pipeline = VoiceRAGPipeline(
        stt_client=mock_stt_client,
        input_filter=mock_input_filter,
        retriever=mock_retriever,
        llm_client=mock_llm_client,
        policy=failing_policy,
    )

    response = await pipeline.run(text_input="What is the Taj Mahal?")

    assert response.status == "refused"
    assert response.guardrail.category == "ungrounded"


@pytest.mark.asyncio
async def test_pipeline_never_raises(mock_stt_client, mock_input_filter, mock_llm_client, mock_policy):
    """Test that the pipeline always returns a response, never raises."""
    from backend.app.harness.pipeline import VoiceRAGPipeline

    # Retriever that crashes
    crashing_retriever = MagicMock()
    crashing_retriever.retrieve.side_effect = RuntimeError("FAISS index corrupted!")

    pipeline = VoiceRAGPipeline(
        stt_client=mock_stt_client,
        input_filter=mock_input_filter,
        retriever=crashing_retriever,
        llm_client=mock_llm_client,
        policy=mock_policy,
    )

    # Should NOT raise
    response = await pipeline.run(text_input="test query")
    assert isinstance(response, PipelineResponse)
    assert response.status == "error"
