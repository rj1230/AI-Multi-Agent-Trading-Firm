"""
Shared state for the trading graph.

Nodes communicate through TradingState.

The state is the contract between:
    NewsAgent
    ChartAgent
    SignalMerger
    RiskAgent
    ExecutionAgent
    HoldNode

The state remains inspectable, replayable, and suitable for
multi-ticker orchestration.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AgentLogEntry(BaseModel):
    """One structured event emitted by a graph node."""

    node: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Signal(BaseModel):
    """
    Canonical directional signal.

    neutral means the agent has no directional opinion.
    """

    direction: Literal["bullish", "bearish", "neutral"]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str = Field(
        min_length=1,
        max_length=500,
    )


class RiskDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TradingState(BaseModel):
    """
    Shared state for one ticker at one point in time.

    TradingState is intentionally explicit so that every important
    transition remains inspectable and replayable.
    """

    # ------------------------------------------------------------------
    # Identity / replay metadata
    # ------------------------------------------------------------------

    ticker: str

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    run_id: str | None = None
    tick_id: int | None = None

    # ------------------------------------------------------------------
    # Market-analysis signals
    # ------------------------------------------------------------------

    news_signal: Signal | None = None
    chart_signal: Signal | None = None
    merged_signal: Signal | None = None

    merge_agreement: bool | None = None

    # ------------------------------------------------------------------
    # News provenance
    # ------------------------------------------------------------------

    # Structured availability from the actual NewsAgent fetch.
    news_availability: str | None = None

    # Number of eligible articles returned for this exact simulated time.
    news_article_count: int = 0

    # Source that produced the NewsResult.
    news_source: str | None = None

    # Timestamp associated with the NewsResult.
    news_as_of: datetime | None = None

    # ------------------------------------------------------------------
    # Risk layer
    # ------------------------------------------------------------------

    risk_decision: RiskDecision = RiskDecision.PENDING

    risk_notes: list[str] = Field(default_factory=list)

    portfolio_snapshot: dict = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    atr: float | None = None

    entry_price: float | None = None

    sector: str | None = None

    proposed_shares: float = 0.0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    # None:
    #     Execution has not been attempted.
    #
    # True:
    #     ExecutionAgent confirmed that the broker order succeeded.
    #
    # False:
    #     ExecutionAgent was reached but the broker execution failed.
    #
    # Keeping this separate from risk_decision is important:
    #
    #     Risk approved != trade executed
    #
    # This field is consumed by tick_runner.py so failed broker
    # executions are never mislabeled as "executed".
    execution_success: bool | None = None

    execution_notes: str | None = None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    agent_logs: Annotated[
        list[AgentLogEntry],
        operator.add,
    ] = Field(default_factory=list)
