"""
Shared state for the trading graph.

Design rule from the architecture doc: nodes communicate ONLY through this
object, never directly with each other. That's what makes each node
independently inspectable, replayable, and swappable — the contract is
"what's in the state," not "what this specific agent expects to receive."

Phase 5/6 update: portfolio_snapshot is now the real PortfolioSnapshot
(portfolio/state.py), not a placeholder dict — RiskAgent and the
coordinator both read it as a typed object (.equity, .positions,
.correlation_matrix), not dict keys. proposed_shares carries RiskAgent's
sizing decision forward to ExecutionAgent without re-deriving it.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, ConfigDict

from portfolio.state import PortfolioSnapshot


class AgentLogEntry(BaseModel):
    """One line in the audit trail. Every node appends exactly one of
    these per run — this is what makes a trade's full reasoning chain
    reconstructable later (see the doc's tracing/observability goal)."""

    node: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Signal(BaseModel):
    """Shape shared by News/Chart/Merged signals: {direction, confidence,
    rationale}. Note: when this holds a merged result, `confidence` carries
    MergedSignal.combined_confidence — the `agreement` flag itself is not
    persisted here, only logged (see signal_merger_node). Add an
    `agreement: Optional[bool]` field here if the dashboard ends up needing
    it downstream, not just in the log line."""

    direction: Optional[str] = None  # "bullish" | "bearish" | "neutral"
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class RiskDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TradingState(BaseModel):
    """The one object every node reads from and writes to."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # PortfolioSnapshot carries a DataFrame

    ticker: str

    news_signal: Optional[Signal] = None
    chart_signal: Optional[Signal] = None
    merged_signal: Optional[Signal] = None

    risk_decision: RiskDecision = RiskDecision.PENDING
    risk_notes: list[str] = Field(default_factory=list)
    proposed_shares: float = 0.0

    portfolio_snapshot: Optional[PortfolioSnapshot] = None

    # Annotated with operator.add so LangGraph concatenates writes instead
    # of last-value-wins — required because NewsAgent and ChartAgent write
    # to this field concurrently in the same parallel-fan-out step.
    agent_logs: Annotated[list[AgentLogEntry], operator.add] = Field(
        default_factory=list
    )
