"""
Shared state for the trading graph.

Design rule from the architecture doc: nodes communicate ONLY through this
object, never directly with each other. That's what makes each node
independently inspectable, replayable, and swappable — the contract is
"what's in the state," not "what this specific agent expects to receive."

This is intentionally Phase-3-minimal: just enough fields for stub nodes
to prove the graph's shape is correct. Phase 4/5/6 will extend it (real
signal payloads, RiskConfig, order confirmations) without changing this
file's role as the single source of truth for what flows through the graph.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class AgentLogEntry(BaseModel):
    """One line in the audit trail. Every node appends exactly one of
    these per run — this is what makes a trade's full reasoning chain
    reconstructable later (see the doc's tracing/observability goal)."""

    node: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Signal(BaseModel):
    """Placeholder shape for News/Chart/Merged signals. Phase 4/5 will
    flesh this out to match the {direction, confidence, rationale} shape
    from the architecture doc — kept minimal here since Phase 3 only
    needs to prove data flows, not what the data means yet."""

    direction: Optional[str] = None  # "bullish" | "bearish" | "neutral"
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class RiskDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TradingState(BaseModel):
    """The one object every node reads from and writes to."""

    ticker: str

    news_signal: Optional[Signal] = None
    chart_signal: Optional[Signal] = None
    merged_signal: Optional[Signal] = None

    risk_decision: RiskDecision = RiskDecision.PENDING
    risk_notes: list[str] = Field(default_factory=list)

    portfolio_snapshot: dict = Field(default_factory=dict)

    # Annotated with operator.add so LangGraph concatenates writes instead
    # of last-value-wins — required because NewsAgent and ChartAgent write
    # to this field concurrently in the same parallel-fan-out step.
    agent_logs: Annotated[list[AgentLogEntry], operator.add] = Field(
        default_factory=list
    )
