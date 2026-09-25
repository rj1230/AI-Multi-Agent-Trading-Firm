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


def test_trade_trace_schema_supports_full_tick_audit():
    timestamp = datetime(
        2026,
        8,
        2,
        10,
        30,
        tzinfo=UTC,
    )

    trace = TradeTrace(
        run_id="integration-test",
        ticker="AAPL",
        timestamp=timestamp,
        simulated_date=timestamp,
        news=NewsTrace(
            direction="bullish",
            confidence=0.90,
            availability="available",
            article_count=2,
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
            checks=["position_size", "ticker_cap"],
        ),
        coordinator=CoordinatorTrace(
            approved=True,
            shares=40.0,
            notes=["approved"],
        ),
        execution=ExecutionTrace(
            attempted=True,
            success=True,
            side="buy",
            quantity=40.0,
            price=150.0,
            order_id="sim-001",
            notes=["filled"],
        ),
        accounting=AccountingTrace(
            average_cost=150.0,
            realized_pnl=0.0,
            remaining_shares=40.0,
            position_closed=False,
        ),
        metadata={
            "outcome": "executed",
            "sector": "Technology",
        },
    )

    payload = trace.to_dict()

    assert payload["run_id"] == "integration-test"
    assert payload["ticker"] == "AAPL"

    assert payload["news"]["direction"] == "bullish"
    assert payload["chart"]["direction"] == "bullish"

    assert payload["signal"]["agreement"] is True

    assert payload["risk"]["approved"] is True
    assert payload["risk"]["final_shares"] == 40.0

    assert payload["coordinator"]["approved"] is True
    assert payload["coordinator"]["shares"] == 40.0

    assert payload["execution"]["success"] is True
    assert payload["execution"]["side"] == "buy"
    assert payload["execution"]["quantity"] == 40.0
    assert payload["execution"]["order_id"] == "sim-001"

    assert payload["accounting"]["remaining_shares"] == 40.0
    assert payload["accounting"]["position_closed"] is False

    assert payload["metadata"]["outcome"] == "executed"
