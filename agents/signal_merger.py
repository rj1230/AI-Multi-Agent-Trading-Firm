"""
Deterministic SignalMerger.

Combines NewsAgent and ChartAgent signals into one directional signal.

Important semantic rule:
    neutral = "no opinion / insufficient evidence"
    neutral is NOT the same thing as "opposite direction".

Therefore:
    bullish + bullish -> bullish
    bearish + bearish -> bearish

    bullish + neutral -> bullish, reduced confidence
    neutral + bullish -> bullish, reduced confidence

    bearish + neutral -> bearish, reduced confidence
    neutral + bearish -> bearish, reduced confidence

    bullish + bearish -> neutral
    bearish + bullish -> neutral

    neutral + neutral -> neutral
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Direction = Literal["bullish", "bearish", "neutral"]


class Signal(BaseModel):
    """Canonical signal shape."""

    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)


class MergedSignal(BaseModel):
    """Output of the deterministic signal merger."""

    direction: Direction
    combined_confidence: float = Field(ge=0.0, le=1.0)
    agreement: bool
    rationale: str


def merge_signals(
    news_signal: Signal,
    chart_signal: Signal,
) -> MergedSignal:
    """
    Deterministically merge NewsAgent and ChartAgent signals.

    Neutral means an agent has no directional opinion.

    A neutral signal therefore does not contradict the other agent.
    Actual bullish-vs-bearish disagreement is treated as a veto.

    No LLM call.
    No randomness.
    Backtest-safe and reproducible.
    """

    news_direction = news_signal.direction
    chart_direction = chart_signal.direction

    # ---------------------------------------------------------
    # Case 1: Both agents are neutral
    # ---------------------------------------------------------
    if news_direction == "neutral" and chart_direction == "neutral":
        return MergedSignal(
            direction="neutral",
            combined_confidence=0.0,
            agreement=False,
            rationale=(
                "Both News and Chart are neutral. "
                "There is insufficient directional evidence."
            ),
        )

    # ---------------------------------------------------------
    # Case 2: Both agents agree on bullish/bearish
    # ---------------------------------------------------------
    if news_direction == chart_direction and news_direction in ("bullish", "bearish"):
        confidence = round(
            (news_signal.confidence + chart_signal.confidence) / 2,
            4,
        )

        return MergedSignal(
            direction=news_direction,
            combined_confidence=confidence,
            agreement=True,
            rationale=(
                f"News and Chart agree on {news_direction}. "
                f"News confidence={news_signal.confidence:.3f}; "
                f"Chart confidence={chart_signal.confidence:.3f}. "
                f"Combined confidence={confidence:.3f}."
            ),
        )

    # ---------------------------------------------------------
    # Case 3: News is neutral, Chart has direction
    #
    # Neutral = no opinion, NOT contradiction.
    # Reduce confidence because only one source is directional.
    # ---------------------------------------------------------
    if news_direction == "neutral" and chart_direction in (
        "bullish",
        "bearish",
    ):
        confidence = round(chart_signal.confidence * 0.75, 4)

        return MergedSignal(
            direction=chart_direction,
            combined_confidence=confidence,
            agreement=False,
            rationale=(
                f"News is neutral while Chart is {chart_direction}. "
                f"Treating News as no-opinion rather than contradiction. "
                f"Chart confidence={chart_signal.confidence:.3f}; "
                f"reduced combined confidence={confidence:.3f}."
            ),
        )

    # ---------------------------------------------------------
    # Case 4: Chart is neutral, News has direction
    #
    # Same principle as above.
    # ---------------------------------------------------------
    if chart_direction == "neutral" and news_direction in (
        "bullish",
        "bearish",
    ):
        confidence = round(news_signal.confidence * 0.75, 4)

        return MergedSignal(
            direction=news_direction,
            combined_confidence=confidence,
            agreement=False,
            rationale=(
                f"Chart is neutral while News is {news_direction}. "
                f"Treating Chart as no-opinion rather than contradiction. "
                f"News confidence={news_signal.confidence:.3f}; "
                f"reduced combined confidence={confidence:.3f}."
            ),
        )

    # ---------------------------------------------------------
    # Case 5: Genuine directional disagreement
    #
    # bullish vs bearish
    # bearish vs bullish
    #
    # This remains a veto.
    # ---------------------------------------------------------
    return MergedSignal(
        direction="neutral",
        combined_confidence=0.0,
        agreement=False,
        rationale=(
            f"Directional disagreement: News says {news_direction} "
            f"(confidence={news_signal.confidence:.3f}), while Chart says "
            f"{chart_direction} (confidence={chart_signal.confidence:.3f}). "
            "Defaulting to neutral."
        ),
    )
