"""Input guardrail: off-topic detection + safety filter."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from backend.app.indexing.embedder import Embedder
from backend.app.schemas import GuardrailVerdict

logger = logging.getLogger(__name__)


# Patterns indicating prompt injection or unsafe content
UNSAFE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"bypass\s+(safety|content|filter)",
    r"system\s+prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",
    r"act\s+as\s+(if|a)",
    r"disregard\s+(all|any|the)\s+(previous|safety|instructions)",
    r"<\s*script\s*>",
    r"eval\s*\(",
    r"exec\s*\(",
]

COMPILED_UNSAFE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


class InputFilter:
    """Pre-retrieval input guardrail.

    Two checks:
    1. Off-topic detection: compare query embedding to corpus centroid
       via cosine similarity. Low similarity = off-topic.
    2. Safety filter: regex patterns for prompt injection and unsafe content.
    """

    def __init__(
        self,
        embedder: Embedder,
        centroid_path: str | Path | None = None,
        centroid: np.ndarray | None = None,
        off_topic_threshold: float = 0.25,
    ):
        """Initialize the input filter.

        Args:
            embedder: Text embedder for query encoding.
            centroid_path: Path to pre-computed corpus centroid .npy file.
            centroid: Pre-loaded corpus centroid array.
            off_topic_threshold: Cosine similarity threshold below which
                queries are considered off-topic.
        """
        self.embedder = embedder
        self.off_topic_threshold = off_topic_threshold

        if centroid is not None:
            self._centroid = centroid
        elif centroid_path and Path(centroid_path).exists():
            self._centroid = np.load(str(centroid_path))
            logger.info(f"Loaded corpus centroid from {centroid_path}")
        else:
            self._centroid = None
            logger.warning("No corpus centroid available — off-topic detection disabled")

    def check_safety(self, text: str) -> GuardrailVerdict:
        """Check input text for prompt injection and unsafe patterns.

        Args:
            text: Input text to check.

        Returns:
            GuardrailVerdict with category="unsafe" if patterns detected.
        """
        for pattern in COMPILED_UNSAFE:
            if pattern.search(text):
                match = pattern.pattern
                logger.warning(f"Safety filter triggered: pattern='{match}', text='{text[:100]}'")
                return GuardrailVerdict(
                    passed=False,
                    reason=f"Input contains potentially unsafe content (pattern: {match})",
                    category="unsafe",
                )

        return GuardrailVerdict(passed=True, category="ok")

    def check_off_topic(
        self, text: str, query_embedding: np.ndarray | None = None
    ) -> tuple[GuardrailVerdict, np.ndarray | None]:
        """Check if query is off-topic relative to the corpus.

        Args:
            text: Query text to check.
            query_embedding: Pre-computed query vector (optional).

        Returns:
            Tuple of (GuardrailVerdict, query_embedding).
        """
        if self._centroid is None or self.off_topic_threshold <= 0.0:
            # Off-topic check disabled by config or missing centroid
            return GuardrailVerdict(passed=True, category="ok"), query_embedding

        # Embed query if not provided
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(text)

        # Compute cosine similarity with corpus centroid
        similarity = float(np.dot(query_embedding, self._centroid))

        logger.info(
            f"Off-topic check: similarity={similarity:.4f}, threshold={self.off_topic_threshold}",
            extra={"stage": "input_guardrail"},
        )

        if similarity < self.off_topic_threshold:
            return (
                GuardrailVerdict(
                    passed=False,
                    reason=f"Query appears off-topic (similarity={similarity:.3f} < threshold={self.off_topic_threshold})",
                    category="off_topic",
                ),
                query_embedding,
            )

        return GuardrailVerdict(passed=True, category="ok"), query_embedding

    def check(
        self, text: str, query_embedding: np.ndarray | None = None
    ) -> tuple[GuardrailVerdict, np.ndarray | None]:
        """Run all input checks: safety first, then off-topic.

        Args:
            text: Input text to check.
            query_embedding: Pre-computed query vector (optional).

        Returns:
            Tuple of (GuardrailVerdict, query_embedding).
        """
        # Safety check first (cheaper regex)
        safety = self.check_safety(text)
        if not safety.passed:
            return safety, query_embedding

        # Off-topic check
        return self.check_off_topic(text, query_embedding=query_embedding)
