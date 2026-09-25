"""
Deterministic trading decision and execution trace.

TradeTrace is an observability/audit structure. It does not make trading
decisions and does not introduce an additional agent.

The trace captures the important state transitions for a single ticker
during one orchestration tick:

    News / Chart
        -> SignalMerger
        -> Risk
        -> Coordinator
        -> Execution
        -> Accounting

The structure is intentionally framework-independent so it can later be
serialized to JSON, persisted to SQLite, displayed in Streamlit, or emitted
through an observability backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NewsTrace:
    """News-agent provenance for a trading decision."""

    direction: str | None = None
    confidence: float | None = None
    availability: str | None = None
    article_count: int | None = None
    source: str | None = None
    as_of: datetime | None = None


@dataclass
class ChartTrace:
    """Chart-agent output for a trading decision."""

    direction: str | None = None
    confidence: float | None = None


@dataclass
class SignalTrace:
    """Merged signal produced from upstream agent outputs."""

    direction: str | None = None
    confidence: float | None = None
    agreement: bool | None = None


@dataclass
class RiskTrace:
    """Risk-sizing and risk-gate information."""

    approved: bool | None = None
    raw_shares: float | None = None
    proposed_shares: float | None = None
    final_shares: float | None = None
    checks: list[str] = field(default_factory=list)


@dataclass
class CoordinatorTrace:
    """Portfolio coordinator decision."""

    approved: bool | None = None
    shares: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    """Broker/execution outcome."""

    attempted: bool = False
    success: bool | None = None
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    order_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AccountingTrace:
    """Portfolio accounting result after execution."""

    average_cost: float | None = None
    realized_pnl: float | None = None
    remaining_shares: float | None = None
    position_closed: bool | None = None


@dataclass
class TradeTrace:
    """
    Complete deterministic audit record for one ticker/tick.

    TradeTrace is observational only. It must never modify trading state or
    influence the decision pipeline.
    """

    run_id: str
    ticker: str
    timestamp: datetime
    simulated_date: datetime | None = None

    news: NewsTrace = field(default_factory=NewsTrace)
    chart: ChartTrace = field(default_factory=ChartTrace)
    signal: SignalTrace = field(default_factory=SignalTrace)
    risk: RiskTrace = field(default_factory=RiskTrace)
    coordinator: CoordinatorTrace = field(
        default_factory=CoordinatorTrace,
    )
    execution: ExecutionTrace = field(
        default_factory=ExecutionTrace,
    )
    accounting: AccountingTrace = field(
        default_factory=AccountingTrace,
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the trace into a JSON-friendly dictionary.

        Datetimes are converted to ISO-8601 strings recursively.
        """

        return _serialize(asdict(self))


def _serialize(value: Any) -> Any:
    """Recursively convert trace values into JSON-compatible values."""

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_serialize(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize(item) for item in value]

    return value
