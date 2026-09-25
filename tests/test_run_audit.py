from datetime import UTC, datetime

import pytest

from storage.run_audit import RunAuditStore
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


def _make_trace(
    run_id: str,
    ticker: str,
    simulated_date: datetime,
) -> TradeTrace:
    """Construct a deterministic TradeTrace for tests."""
    return TradeTrace(
        run_id=run_id,
        ticker=ticker,
        timestamp=datetime(2026, 9, 25, 10, 0, tzinfo=UTC),
        simulated_date=simulated_date,
        news=NewsTrace(
            direction="bullish",
            confidence=0.91,
            availability="available",
            article_count=3,
            source="historical_cache",
            as_of=simulated_date,
        ),
        chart=ChartTrace(
            direction="bullish",
            confidence=0.88,
        ),
        signal=SignalTrace(
            direction="bullish",
            confidence=0.895,
            agreement=True,
        ),
        risk=RiskTrace(
            approved=True,
            raw_shares=41.6667,
            proposed_shares=41.6667,
            final_shares=41.6667,
            checks=["risk_limit_pass"],
        ),
        coordinator=CoordinatorTrace(
            approved=True,
            shares=41.6667,
            notes=["portfolio_limit_pass"],
        ),
        execution=ExecutionTrace(
            attempted=True,
            success=True,
            side="buy",
            quantity=41.6667,
            price=150.0,
            order_id="order-001",
        ),
        accounting=AccountingTrace(
            average_cost=150.0,
            realized_pnl=None,
            remaining_shares=41.6667,
            position_closed=False,
        ),
        metadata={
            "tick_id": f"{ticker}-2026-08-01",
            "outcome": "executed",
        },
    )


@pytest.fixture
def audit_store(tmp_path):
    """Provide a fresh RunAuditStore backed by a temporary SQLite DB."""
    db_path = tmp_path / "audit.db"
    return RunAuditStore(str(db_path))


def test_run_round_trip(audit_store):
    """Persist a run and one trace, then read them back."""
    started = datetime(2026, 9, 25, 9, 0, tzinfo=UTC)

    audit_store.start_run(
        "run-001",
        started_at=started,
        start_date="2026-08-01",
        end_date="2026-08-03",
        tickers=["AAPL", "MSFT"],
        starting_equity=100_000.0,
        metadata={"strategy": "v1"},
    )

    trace = _make_trace(
        "run-001",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )
    audit_store.record_trade_trace(trace)

    audit_store.finish_run(
        "run-001",
        finished_at=started,
        final_equity=100_500.0,
        realized_pnl=500.0,
        benchmark_return=0.004,
    )

    run = audit_store.get_run("run-001")
    assert run is not None
    assert run["run_id"] == "run-001"
    assert run["status"] == "completed"
    assert run["tickers"] == ["AAPL", "MSFT"]
    assert run["starting_equity"] == pytest.approx(100_000.0)
    assert run["final_equity"] == pytest.approx(100_500.0)
    assert run["realized_pnl"] == pytest.approx(500.0)
    assert run["trade_trace_count"] == 1
    assert run["metadata"] == {"strategy": "v1"}

    traces = audit_store.get_trade_traces("run-001")
    assert len(traces) == 1

    t = traces[0]
    assert t["ticker"] == "AAPL"
    assert t["signal"]["direction"] == "bullish"
    assert t["execution"]["order_id"] == "order-001"
    assert t["accounting"]["remaining_shares"] == pytest.approx(41.6667)


def test_duplicate_trace_is_idempotent(audit_store):
    """Recording the same trace twice should not create duplicates."""
    audit_store.start_run(
        "run-001",
        started_at="2026-09-25T09:00:00+00:00",
        tickers=["AAPL"],
        starting_equity=100_000.0,
    )

    trace = _make_trace(
        "run-001",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    audit_store.record_trade_trace(trace)
    audit_store.record_trade_trace(trace)

    traces = audit_store.get_trade_traces("run-001")
    assert len(traces) == 1

    run = audit_store.get_run("run-001")
    assert run["trade_trace_count"] == 1


def test_runs_are_isolated(audit_store):
    """Traces from different runs must not mix."""
    audit_store.start_run(
        "run-A",
        started_at="2026-09-25T09:00:00+00:00",
        tickers=["AAPL"],
        starting_equity=100_000.0,
    )

    audit_store.start_run(
        "run-B",
        started_at="2026-09-25T10:00:00+00:00",
        tickers=["MSFT"],
        starting_equity=200_000.0,
    )

    audit_store.record_trade_trace(
        _make_trace("run-A", "AAPL", datetime(2026, 8, 1, tzinfo=UTC))
    )
    audit_store.record_trade_trace(
        _make_trace("run-B", "MSFT", datetime(2026, 8, 1, tzinfo=UTC))
    )

    run_a_traces = audit_store.get_trade_traces("run-A")
    run_b_traces = audit_store.get_trade_traces("run-B")

    assert [t["ticker"] for t in run_a_traces] == ["AAPL"]
    assert [t["ticker"] for t in run_b_traces] == ["MSFT"]


def test_delete_run_cascades_trade_traces(audit_store):
    """Deleting a run must also delete its trade traces (ON DELETE CASCADE)."""
    audit_store.start_run(
        "run-delete",
        started_at="2026-09-25T09:00:00+00:00",
        tickers=["AAPL"],
        starting_equity=100_000.0,
    )

    audit_store.record_trade_trace(
        _make_trace("run-delete", "AAPL", datetime(2026, 8, 1, tzinfo=UTC))
    )

    audit_store.delete_run("run-delete")

    assert audit_store.get_run("run-delete") is None
    assert audit_store.get_trade_traces("run-delete") == []


def test_unknown_run_raises(audit_store):
    """Operations on unknown run_id should raise KeyError."""
    with pytest.raises(KeyError, match="Unknown run_id"):
        audit_store.finish_run(
            "nonexistent",
            finished_at="2026-09-25T10:00:00+00:00",
        )

    with pytest.raises(KeyError, match="Unknown run_id"):
        audit_store.delete_run("nonexistent")
