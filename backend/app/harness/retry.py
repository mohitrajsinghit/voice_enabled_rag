"""Shared retry configurations and custom exception types."""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    stage: str = "unknown"


class STTError(PipelineError):
    """Error during speech-to-text transcription."""
    stage = "transcribe"


class RetrievalError(PipelineError):
    """Error during retrieval (embedding + FAISS search)."""
    stage = "retrieve"


class GenerationError(PipelineError):
    """Error during LLM answer generation."""
    stage = "generate"


class GuardrailError(PipelineError):
    """Error during guardrail checks."""
    stage = "guardrail"


# Shared retry configurations
def api_retry(func):
    """Retry decorator for external API calls (STT, LLM).

    Retries up to 3 times with exponential backoff: 1s, 2s, 4s.
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((STTError, GenerationError)),
        reraise=True,
    )(func)


def retrieval_retry(func):
    """Retry decorator for retrieval operations.

    Retries up to 2 times with exponential backoff.
    """
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(RetrievalError),
        reraise=True,
    )(func)
