"""
Tests for the deterministic TradeTrace observability model.
"""

from datetime import UTC, datetime

from telemetry.trade_trace import (
    AccountingTrace,
    ChartTrace,
    CoordinatorTrace,
    ExecutionTrace,
    NewsTrace,
    RiskTrace,
    SignalTrace,
    TradeTrace,
)


def test_trade_trace_serializes_nested_state():
    timestamp = datetime(
        2026,
        8,
        2,
        10,
        30,
        tzinfo=UTC,
    )

    trace = TradeTrace(
        run_id="test-run-001",
        ticker="AAPL",
        timestamp=timestamp,
        simulated_date=timestamp,
        news=NewsTrace(
            direction="bullish",
            confidence=0.90,
            availability="available",
            article_count=3,
            source="test_fixture",
            as_of=timestamp,
        ),
        chart=ChartTrace(
            direction="bullish",
            confidence=0.80,
        ),
        signal=SignalTrace(
            direction="bullish",
            confidence=0.85,
            agreement=True,
        ),
        risk=RiskTrace(
            approved=True,
            raw_shares=50.0,
            proposed_shares=50.0,
            final_shares=40.0,
            checks=["ticker_cap"],
        ),
        coordinator=CoordinatorTrace(
            approved=True,
            shares=40.0,
            notes=["position within portfolio limits"],
        ),
        execution=ExecutionTrace(
            attempted=True,
            success=True,
            side="buy",
            quantity=40.0,
            price=150.0,
            order_id="sim-001",
        ),
        accounting=AccountingTrace(
            average_cost=150.0,
            realized_pnl=0.0,
            remaining_shares=40.0,
            position_closed=False,
        ),
    )

    result = trace.to_dict()

    assert result["run_id"] == "test-run-001"
    assert result["ticker"] == "AAPL"

    assert result["timestamp"] == timestamp.isoformat()
    assert result["simulated_date"] == timestamp.isoformat()

    assert result["news"]["direction"] == "bullish"
    assert result["news"]["article_count"] == 3
    assert result["news"]["as_of"] == timestamp.isoformat()

    assert result["chart"]["confidence"] == 0.80

    assert result["signal"]["agreement"] is True

    assert result["risk"]["approved"] is True
    assert result["risk"]["final_shares"] == 40.0

    assert result["coordinator"]["approved"] is True

    assert result["execution"]["success"] is True
    assert result["execution"]["side"] == "buy"
    assert result["execution"]["quantity"] == 40.0
    assert result["execution"]["price"] == 150.0

    assert result["accounting"]["remaining_shares"] == 40.0
    assert result["accounting"]["position_closed"] is False


def test_trade_trace_defaults_are_safe():
    timestamp = datetime(
        2026,
        8,
        2,
        tzinfo=UTC,
    )

    trace = TradeTrace(
        run_id="test-run-002",
        ticker="MSFT",
        timestamp=timestamp,
    )

    result = trace.to_dict()

    assert result["run_id"] == "test-run-002"
    assert result["ticker"] == "MSFT"

    assert result["news"]["direction"] is None
    assert result["chart"]["direction"] is None
    assert result["signal"]["direction"] is None

    assert result["risk"]["approved"] is None
    assert result["coordinator"]["approved"] is None

    assert result["execution"]["attempted"] is False

    assert result["accounting"]["position_closed"] is None
