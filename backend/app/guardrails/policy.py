"""Guardrail policy orchestrator: decides reject / hedge / answer."""
from __future__ import annotations

import logging

from backend.app.config import get_settings
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.prompts import STRICT_GENERATION_PROMPT, format_context
from backend.app.guardrails.grounding_check import GroundingChecker
from backend.app.schemas import GuardrailVerdict, RetrievedChunk

logger = logging.getLogger(__name__)


class GuardrailPolicy:
    """Orchestrates guardrail decisions across the pipeline.

    Decision order (per spec section 3.4):
    1. input_filter — reject before retrieval if off-topic or unsafe
    2. retrieval score check — refuse if max score < threshold
    3. generate answer
    4. grounding_check — regenerate once with strict prompt if ungrounded, then refuse

    This class handles steps 2 and 4. Step 1 is handled separately in
    InputGuardrailStage. Step 3 is handled by GenerateStage.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_score_threshold: float | None = None,
    ):
        """Initialize the guardrail policy.

        Args:
            llm_client: LLM client for grounding checks and regeneration.
            retrieval_score_threshold: Min retrieval score to consider results valid.
        """
        settings = get_settings()
        self.llm_client = llm_client
        self.grounding_checker = GroundingChecker(llm_client)
        self.retrieval_score_threshold = retrieval_score_threshold or settings.retrieval_score_threshold

    def check_retrieval_quality(
        self,
        chunks: list[RetrievedChunk],
    ) -> GuardrailVerdict:
        """Check if retrieval results meet quality threshold.

        Args:
            chunks: Retrieved chunks with scores.

        Returns:
            GuardrailVerdict — fails if max score is below threshold.
        """
        if not chunks:
            return GuardrailVerdict(
                passed=False,
                reason="No relevant passages found in the knowledge base",
                category="off_topic",
            )

        max_score = max(c.score for c in chunks)
        if max_score < self.retrieval_score_threshold:
            return GuardrailVerdict(
                passed=False,
                reason=f"Low retrieval confidence (max_score={max_score:.3f} < threshold={self.retrieval_score_threshold})",
                category="off_topic",
            )

        return GuardrailVerdict(passed=True, category="ok")

    def check_grounding_with_retry(
        self,
        answer: str,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, GuardrailVerdict, float]:
        """Check grounding, retry with strict prompt if ungrounded.

        Args:
            answer: Generated answer to verify.
            query: Original query text.
            chunks: Source chunks.

        Returns:
            Tuple of (final_answer, verdict, total_grounding_latency_ms).
        """
        # First grounding check
        verdict, latency = self.grounding_checker.check(answer, chunks)

        if verdict.passed:
            return answer, verdict, latency

        # Failed — try once more with strict prompt
        logger.warning(f"Grounding check failed: {verdict.reason}. Retrying with strict prompt.")

        context = format_context(chunks)
        strict_prompt = STRICT_GENERATION_PROMPT.format(
            context=context,
            query=query,
        )

        try:
            new_answer, gen_latency = self.llm_client.generate(
                system_prompt="You are a strictly grounded QA assistant. Only use information from the provided context.",
                user_prompt=strict_prompt,
                max_tokens=1024,
                temperature=0.1,
            )

            # Check grounding of the new answer
            new_verdict, check_latency = self.grounding_checker.check(new_answer, chunks)
            total_latency = latency + gen_latency + check_latency

            if new_verdict.passed:
                logger.info("Strict regeneration passed grounding check")
                return new_answer, new_verdict, total_latency

            # Still ungrounded — refuse
            logger.warning("Strict regeneration still ungrounded — refusing")
            return "", GuardrailVerdict(
                passed=False,
                reason="Answer could not be grounded in sources after retry",
                category="ungrounded",
            ), total_latency

        except Exception as e:
            logger.error(f"Strict regeneration failed: {e}")
            return "", GuardrailVerdict(
                passed=False,
                reason=f"Grounding retry failed: {e}",
                category="ungrounded",
            ), latency
