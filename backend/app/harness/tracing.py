"""Timer context manager and trace collector for per-stage latency capture."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("retrieval") as t:
            result = do_retrieval()
        print(f"Took {t.elapsed_ms:.1f}ms")
    """

    def __init__(self, name: str = ""):
        """Initialize timer.

        Args:
            name: Name of the timed operation.
        """
        self.name = name
        self._start: float = 0
        self._end: float = 0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._end > 0:
            return (self._end - self._start) * 1000
        elif self._start > 0:
            return (time.perf_counter() - self._start) * 1000
        return 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._end = time.perf_counter()
        logger.debug(
            f"Timer [{self.name}]: {self.elapsed_ms:.2f}ms",
            extra={"stage": self.name, "latency_ms": self.elapsed_ms},
        )


class TraceCollector:
    """Collects per-stage timing data for a pipeline run.

    Thread-safe accumulation of stage latencies, outputs
    a structured dict for the PipelineResponse.
    """

    def __init__(self):
        self._timings: dict[str, float] = {}
        self._start_time: float = time.perf_counter()

    def record(self, stage: str, latency_ms: float) -> None:
        """Record a stage timing.

        Args:
            stage: Stage name.
            latency_ms: Latency in milliseconds.
        """
        self._timings[stage] = round(latency_ms, 2)

    def record_sub_timings(self, timings: dict[str, float]) -> None:
        """Record multiple sub-stage timings at once.

        Args:
            timings: Dict of stage_name -> latency_ms.
        """
        for stage, ms in timings.items():
            self._timings[stage] = round(ms, 2)

    @contextmanager
    def trace(self, stage: str) -> Generator[Timer, None, None]:
        """Context manager that times a stage and records it.

        Args:
            stage: Stage name.

        Yields:
            Timer instance.
        """
        timer = Timer(stage)
        try:
            timer.__enter__()
            yield timer
        finally:
            timer.__exit__(None, None, None)
            self.record(stage, timer.elapsed_ms)

    @property
    def total_ms(self) -> float:
        """Total elapsed time since collector creation."""
        return (time.perf_counter() - self._start_time) * 1000

    def get_latencies(self) -> dict[str, float]:
        """Get all recorded latencies plus the total.

        Returns:
            Dict of stage -> latency_ms, including 'end_to_end_ms'.
        """
        result = dict(self._timings)
        result["end_to_end_ms"] = round(self.total_ms, 2)
        return result

    def log_summary(self) -> None:
        """Log a summary of all recorded timings."""
        latencies = self.get_latencies()
        parts = [f"{k}={v:.1f}ms" for k, v in latencies.items()]
        logger.info(f"Pipeline trace: {', '.join(parts)}")
