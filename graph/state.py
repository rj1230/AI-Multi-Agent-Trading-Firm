"""
Shared state for the trading graph.

Design rule from the architecture doc:
nodes communicate ONLY through this object, never directly with each other.

The state is the contract between:
    NewsAgent
    ChartAgent
    SignalMerger
    RiskAgent
    ExecutionAgent
    HoldNode

The state is deliberately structured so that the graph remains:
    - inspectable
    - replayable
    - deterministic where required
    - easy to extend for multi-ticker trading
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AgentLogEntry(BaseModel):
    """One structured event emitted by a graph node."""

    node: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Signal(BaseModel):
    """
    Canonical directional signal.

    A Signal is always valid once it exists in TradingState.

    `neutral` means the agent has no directional opinion.
    A missing signal is represented by `Signal | None` at the state level.
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

    Every graph node reads from and writes to this object rather than
    communicating directly with another node.
    """

    # ------------------------------------------------------------------
    # Identity / replay metadata
    # ------------------------------------------------------------------

    ticker: str

    # Important for historical replay/backtesting.
    # In live mode this represents the current decision timestamp.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional identifiers for future multi-ticker/concurrent runs.
    run_id: str | None = None
    tick_id: int | None = None

    # ------------------------------------------------------------------
    # Market-analysis signals
    # ------------------------------------------------------------------

    news_signal: Signal | None = None
    chart_signal: Signal | None = None
    merged_signal: Signal | None = None

    # SignalMerger produces richer metadata than graph.state.Signal.
    # Preserve the actual agreement state instead of reconstructing it.
    merge_agreement: bool | None = None

    # ------------------------------------------------------------------
    # Risk layer
    # ------------------------------------------------------------------

    risk_decision: RiskDecision = RiskDecision.PENDING

    risk_notes: list[str] = Field(default_factory=list)

    portfolio_snapshot: dict = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    # ATR used by RiskAgent for stop/risk sizing.
    atr: float | None = None

    # Price at which the proposed trade would enter.
    entry_price: float | None = None

    # Sector used for sector concentration checks.
    sector: str | None = None

    # Position size proposed by RiskAgent.
    proposed_shares: float = 0.0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    execution_notes: str | None = None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    # operator.add allows LangGraph node outputs to append logs rather
    # than overwrite logs from previous nodes.
    agent_logs: Annotated[
        list[AgentLogEntry],
        operator.add,
    ] = Field(default_factory=list)
