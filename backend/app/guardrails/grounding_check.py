"""Grounding check: verify generated answers are supported by retrieved context."""
from __future__ import annotations

import logging

from backend.app.generation.llm_client import LLMClient
from backend.app.generation.prompts import GROUNDING_CHECK_PROMPT, format_context
from backend.app.schemas import GuardrailVerdict, RetrievedChunk

logger = logging.getLogger(__name__)


class GroundingChecker:
    """Post-generation grounding verification.

    Uses LLM-as-judge to determine whether a generated answer is
    supported by the retrieved source passages. Returns a verdict
    of supported/partially_supported/not_supported.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize the grounding checker.

        Args:
            llm_client: LLM client for running the grounding judge.
        """
        self.llm_client = llm_client

    def check(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[GuardrailVerdict, float]:
        """Check if an answer is grounded in the retrieved chunks.

        Args:
            answer: Generated answer to verify.
            chunks: Source chunks the answer should be grounded in.

        Returns:
            Tuple of (GuardrailVerdict, latency_ms).
        """
        if not answer or not chunks:
            return GuardrailVerdict(
                passed=False,
                reason="No answer or no sources to verify against",
                category="ungrounded",
            ), 0.0

        # Format context
        context = format_context(chunks)

        # Build grounding check prompt
        prompt = GROUNDING_CHECK_PROMPT.format(
            context=context,
            answer=answer,
        )

        try:
            response_text, latency_ms = self.llm_client.generate(
                system_prompt="You are a grounding verification judge. Respond only with valid JSON.",
                user_prompt=prompt,
                max_tokens=512,
                temperature=0.1,
            )

            # Parse the verdict
            result = self.llm_client.parse_json_response(response_text)

            verdict = result.get("verdict", "not_supported")
            reason = result.get("reason", "")
            unsupported = result.get("unsupported_claims", [])

            if verdict == "supported":
                return GuardrailVerdict(
                    passed=True,
                    reason=reason or "Answer is fully supported by sources",
                    category="ok",
                ), latency_ms

            elif verdict == "partially_supported":
                # Partially supported — we'll let it pass with a note
                return GuardrailVerdict(
                    passed=True,
                    reason=f"Partially supported: {reason}. Unsupported: {unsupported}",
                    category="ok",
                ), latency_ms

            else:  # not_supported
                return GuardrailVerdict(
                    passed=False,
                    reason=f"Answer not grounded in sources: {reason}",
                    category="ungrounded",
                ), latency_ms

        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            # On error, fail open (allow the answer) but log the issue
            return GuardrailVerdict(
                passed=True,
                reason=f"Grounding check error (fail-open): {e}",
                category="ok",
            ), 0.0
