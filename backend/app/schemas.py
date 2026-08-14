"""Pydantic models for every pipeline stage I/O."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptResult(BaseModel):
    """Result from STT transcription."""
    text: str = Field(description="Transcribed text")
    language: str = Field(default="en", description="Detected language code")
    confidence: float = Field(default=1.0, description="Transcription confidence score")
    latency_ms: float = Field(default=0.0, description="STT latency in milliseconds")


class Chunk(BaseModel):
    """A single text chunk from a document."""
    id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    source_doc_id: str = Field(description="Source document identifier")
    strategy: str = Field(description="Chunking strategy used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with its similarity score."""
    chunk: Chunk = Field(description="The retrieved chunk")
    score: float = Field(description="Similarity score (higher = more relevant)")


class GuardrailVerdict(BaseModel):
    """Result of a guardrail check."""
    passed: bool = Field(description="Whether the guardrail passed")
    reason: str | None = Field(default=None, description="Reason for the verdict")
    category: Literal["ok", "off_topic", "unsafe", "ungrounded"] = Field(
        default="ok", description="Verdict category"
    )


class PipelineResponse(BaseModel):
    """Complete response from the RAG pipeline."""
    transcript: str = Field(default="", description="Transcribed query text")
    answer: str | None = Field(default=None, description="Generated answer")
    sources: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved source chunks")
    guardrail: GuardrailVerdict = Field(
        default_factory=lambda: GuardrailVerdict(passed=True, category="ok"),
        description="Guardrail check result",
    )
    latencies: dict[str, float] = Field(default_factory=dict, description="Per-stage latencies in ms")
    status: Literal["answered", "refused", "error"] = Field(
        default="answered", description="Pipeline outcome status"
    )


class PipelineContext(BaseModel):
    """Mutable context passed between pipeline stages."""
    # Input
    audio_bytes: bytes | None = Field(default=None, description="Raw audio input", exclude=True)
    text_input: str | None = Field(default=None, description="Direct text input (bypass STT)")

    # Stage outputs
    transcript: TranscriptResult | None = None
    input_guardrail: GuardrailVerdict | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    generated_answer: str | None = None
    grounding_verdict: GuardrailVerdict | None = None

    # Metadata
    latencies: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    should_stop: bool = False
    status: Literal["answered", "refused", "error"] = "answered"

    model_config = ConfigDict(arbitrary_types_allowed=True)
