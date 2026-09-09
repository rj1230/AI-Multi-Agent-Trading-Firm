"""
Generic retry-then-fallback for any validation-gated call — the pattern
NewsAgent already used inline (Phase 4), pulled out here so RiskAgent's
"insufficient data" path and NewsAgent's guardrails check (Phase 8) share
one implementation instead of three copies of the same retry-once logic.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_then_fallback(
    attempt: Callable[[], T | None],
    fallback: Callable[[str], T],
    max_retries: int = 1,
    label: str = "operation",
) -> T:
    """Calls attempt() up to (max_retries + 1) times. attempt() should
    return None (or raise) on a failed/invalid result, a valid T on
    success. Returns fallback(reason) if every attempt fails."""
    last_reason = "unknown failure"
    for i in range(max_retries + 1):
        try:
            result = attempt()
        except Exception as e:
            last_reason = str(e)
            logger.warning(
                "%s attempt %d/%d failed: %s", label, i + 1, max_retries + 1, e
            )
            continue
        if result is not None:
            return result
        last_reason = "validation returned None"
        logger.warning(
            "%s attempt %d/%d: %s", label, i + 1, max_retries + 1, last_reason
        )
    return fallback(f"{label} failed after {max_retries + 1} attempt(s): {last_reason}")
