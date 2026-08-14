"""Pipeline state machine: typed stages with retries, tracing, and error recovery.

This is the harness — NOT a single function that calls STT then LLM.
It's a staged pipeline where each stage is a typed, independent unit with
its own error handling, retry policy, and timing instrumentation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from backend.app.config import get_settings
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.prompts import ANSWER_GENERATION_PROMPT, format_context
from backend.app.guardrails.input_filter import InputFilter
from backend.app.guardrails.policy import GuardrailPolicy
from backend.app.harness.retry import (
    STTError, RetrievalError, GenerationError, GuardrailError,
)
from backend.app.harness.tracing import TraceCollector
from backend.app.retrieval.retriever import Retriever
from backend.app.schemas import (
    GuardrailVerdict, PipelineContext, PipelineResponse, TranscriptResult,
)
from backend.app.stt.sarvam_client import SarvamClient

logger = logging.getLogger(__name__)


# ─── Pipeline Stage ABC ─────────────────────────────────────────────────

class PipelineStage(ABC):
    """Abstract pipeline stage with typed run method."""

    name: str = "base"

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage, updating the pipeline context.

        Args:
            ctx: Current pipeline context.

        Returns:
            Updated pipeline context.
        """
        ...


# ─── Stage Implementations ──────────────────────────────────────────────

class TranscribeStage(PipelineStage):
    """Stage 1: Transcribe audio to text using Sarvam STT."""

    name = "transcribe"

    def __init__(self, stt_client: SarvamClient):
        self.stt_client = stt_client

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            result = await self.stt_client.transcribe_or_passthrough(
                audio_bytes=ctx.audio_bytes,
                text_input=ctx.text_input,
            )
            ctx.transcript = result
            return ctx
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            ctx.error = f"Speech transcription failed: {e}"
            ctx.status = "error"
            ctx.should_stop = True
            return ctx


class InputGuardrailStage(PipelineStage):
    """Stage 2: Check input for off-topic/unsafe content."""

    name = "input_guardrail"

    def __init__(self, input_filter: InputFilter):
        self.input_filter = input_filter

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.transcript or not ctx.transcript.text:
            ctx.error = "No transcript available for guardrail check"
            ctx.status = "error"
            ctx.should_stop = True
            return ctx

        try:
            verdict = self.input_filter.check(ctx.transcript.text)
            ctx.input_guardrail = verdict

            if not verdict.passed:
                ctx.should_stop = True
                ctx.status = "refused"
                logger.info(f"Input guardrail rejected: {verdict.reason}")

            return ctx
        except Exception as e:
            logger.error(f"Input guardrail error: {e}")
            # Fail open — don't block on guardrail errors
            ctx.input_guardrail = GuardrailVerdict(passed=True, category="ok")
            return ctx


class RetrieveStage(PipelineStage):
    """Stage 3: Embed query + FAISS search for relevant chunks."""

    name = "retrieve"

    def __init__(self, retriever: Retriever, policy: GuardrailPolicy):
        self.retriever = retriever
        self.policy = policy

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.transcript or not ctx.transcript.text:
            ctx.error = "No transcript for retrieval"
            ctx.status = "error"
            ctx.should_stop = True
            return ctx

        try:
            settings = get_settings()
            chunks, latencies = self.retriever.retrieve(
                ctx.transcript.text,
                top_k=settings.top_k,
            )
            ctx.retrieved_chunks = chunks
            ctx.latencies.update(latencies)

            # Check retrieval quality
            quality_verdict = self.policy.check_retrieval_quality(chunks)
            if not quality_verdict.passed:
                ctx.input_guardrail = quality_verdict
                ctx.should_stop = True
                ctx.status = "refused"
                logger.info(f"Retrieval quality check failed: {quality_verdict.reason}")

            return ctx
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            ctx.error = f"Retrieval failed: {e}"
            ctx.status = "error"
            ctx.should_stop = True
            return ctx


class GenerateStage(PipelineStage):
    """Stage 4: Generate answer using LLM with retrieved context."""

    name = "generate"

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.retrieved_chunks:
            ctx.error = "No chunks available for generation"
            ctx.status = "error"
            ctx.should_stop = True
            return ctx

        try:
            context = format_context(ctx.retrieved_chunks)
            prompt = ANSWER_GENERATION_PROMPT.format(
                context=context,
                query=ctx.transcript.text,
            )

            answer, latency_ms = self.llm_client.generate(
                system_prompt="You are a helpful RAG assistant. Answer questions using only the provided context.",
                user_prompt=prompt,
            )
            ctx.generated_answer = answer
            ctx.latencies["generation_ms"] = round(latency_ms, 2)

            return ctx
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            ctx.error = f"Answer generation failed: {e}"
            ctx.generated_answer = f"⚠️ Could not generate an answer: {e}. Please check your LLM provider configuration."
            ctx.status = "error"
            ctx.should_stop = True
            return ctx


class GroundingGuardrailStage(PipelineStage):
    """Stage 5: Verify generated answer is grounded in sources."""

    name = "grounding_guardrail"

    def __init__(self, policy: GuardrailPolicy):
        self.policy = policy

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.generated_answer or not ctx.retrieved_chunks:
            return ctx

        try:
            final_answer, verdict, latency_ms = self.policy.check_grounding_with_retry(
                answer=ctx.generated_answer,
                query=ctx.transcript.text if ctx.transcript else "",
                chunks=ctx.retrieved_chunks,
            )
            ctx.grounding_verdict = verdict
            ctx.latencies["grounding_check_ms"] = round(latency_ms, 2)

            if not verdict.passed:
                ctx.generated_answer = None
                ctx.should_stop = True
                ctx.status = "refused"
                logger.info(f"Grounding check rejected: {verdict.reason}")
            elif final_answer != ctx.generated_answer:
                # Answer was regenerated with strict prompt
                ctx.generated_answer = final_answer

            return ctx
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            # Fail open
            ctx.grounding_verdict = GuardrailVerdict(
                passed=True,
                reason=f"Grounding check error (fail-open): {e}",
                category="ok",
            )
            return ctx


# ─── Pipeline Orchestrator ──────────────────────────────────────────────

class VoiceRAGPipeline:
    """The main pipeline harness: orchestrates all stages with tracing.

    Stages: Transcribe → InputGuardrail → Retrieve → Generate → GroundingGuardrail

    Each stage:
    - Has its own typed error handling (never raises to caller)
    - Is wrapped with Timer for latency tracking
    - Can short-circuit the pipeline via ctx.should_stop
    - Produces a typed PipelineResponse

    The pipeline always returns a PipelineResponse, never raises.
    """

    def __init__(
        self,
        stt_client: SarvamClient,
        input_filter: InputFilter,
        retriever: Retriever,
        llm_client: LLMClient,
        policy: GuardrailPolicy,
    ):
        """Initialize the pipeline with all component dependencies.

        Args:
            stt_client: Sarvam STT client.
            input_filter: Input guardrail filter.
            retriever: FAISS-based retriever.
            llm_client: LLM client for generation.
            policy: Guardrail policy orchestrator.
        """
        self.stages: list[PipelineStage] = [
            TranscribeStage(stt_client),
            InputGuardrailStage(input_filter),
            RetrieveStage(retriever, policy),
            GenerateStage(llm_client),
            GroundingGuardrailStage(policy),
        ]

    async def run(
        self,
        audio_bytes: bytes | None = None,
        text_input: str | None = None,
    ) -> PipelineResponse:
        """Execute the full pipeline.

        Args:
            audio_bytes: Raw audio input (for STT).
            text_input: Direct text input (bypass STT).

        Returns:
            PipelineResponse — always returns, never raises.
        """
        trace = TraceCollector()
        ctx = PipelineContext(
            audio_bytes=audio_bytes,
            text_input=text_input,
        )

        for stage in self.stages:
            if ctx.should_stop:
                logger.info(f"Pipeline short-circuited at stage '{stage.name}' — status={ctx.status}")
                break

            try:
                with trace.trace(f"{stage.name}_ms") as timer:
                    ctx = await stage.run(ctx)
            except Exception as e:
                # Catch-all: no stage should ever crash the pipeline
                logger.error(f"Unhandled error in stage '{stage.name}': {e}", exc_info=True)
                ctx.error = f"Internal error in {stage.name}: {e}"
                ctx.status = "error"
                break

        # Merge traced latencies with stage-reported latencies
        latencies = {**ctx.latencies, **trace.get_latencies()}
        trace.log_summary()

        # Determine the guardrail verdict to include in the response
        guardrail = ctx.grounding_verdict or ctx.input_guardrail or GuardrailVerdict(passed=True, category="ok")

        return PipelineResponse(
            transcript=ctx.transcript.text if ctx.transcript else "",
            answer=ctx.generated_answer,
            sources=ctx.retrieved_chunks,
            guardrail=guardrail,
            latencies=latencies,
            status=ctx.status,
        )
