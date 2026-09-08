"""
SignalMerger: combines NewsAgent's and ChartAgent's independent signals into
one directional call. Deliberately NOT an LLM node -- see the architecture
doc, section 2. This is plain, deterministic, reproducible code.

Integration note: if `Signal` already exists in your schemas.py (it should,
since chart_agent/news_agent already produce {direction, confidence,
rationale}), delete the local definition below and import theirs instead --
this module only needs the three fields, not a specific class identity.

Merge rule, in one sentence: if News and Chart agree on direction, the
merged confidence is the average of the two; if they disagree, the merged
signal defaults to neutral/hold with zero confidence, because a
disagreement between an LLM read and a technical read is itself the
information -- averaging opposite-signed conviction would hide that.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["bullish", "bearish", "neutral"]


class Signal(BaseModel):
    """Shape shared by NewsAgent and ChartAgent outputs."""
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class MergedSignal(BaseModel):
    direction: Direction
    combined_confidence: float = Field(ge=0.0, le=1.0)
    agreement: bool
    rationale: str


def merge_signals(news_signal: Signal, chart_signal: Signal) -> MergedSignal:
    """
    Deterministic merge rule (see module docstring for the one-sentence
    version). No LLM call, no randomness -- same inputs always produce the
    same output, which is what makes this step free and backtest-safe.
    """
    if news_signal.direction == chart_signal.direction:
        return MergedSignal(
            direction=news_signal.direction,
            combined_confidence=round(
                (news_signal.confidence + chart_signal.confidence) / 2, 4
            ),
            agreement=True,
            rationale=(
                f"News and Chart agree on {news_signal.direction} "
                f"(news: {news_signal.rationale}; chart: {chart_signal.rationale})."
            ),
        )

    # Direction mismatch -> default to hold. Neither agent's confidence is
    # trustworthy enough on its own to override an outright contradiction.
    return MergedSignal(
        direction="neutral",
        combined_confidence=0.0,
        agreement=False,
        rationale=(
            f"Disagreement: News says {news_signal.direction} "
            f"({news_signal.confidence}), Chart says {chart_signal.direction} "
            f"({chart_signal.confidence}). Defaulting to hold."
        ),
    )
