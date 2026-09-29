from dataclasses import replace
from datetime import UTC, datetime

import pytest

from storage.run_audit import RunAuditStore
from telemetry.trade_trace import (
    CoordinatorTrace,
    ExecutionTrace,
)
from tests.test_run_audit import _make_trace


@pytest.fixture
def audit_store(tmp_path):
    """Provide a fresh RunAuditStore backed by temporary SQLite."""
    return RunAuditStore(str(tmp_path / "audit.db"))


def _start_run(audit_store, run_id="run-audit"):
    audit_store.start_run(
        run_id,
        started_at="2026-09-25T09:00:00+00:00",
        start_date="2026-08-01",
        end_date="2026-08-03",
        tickers=["AAPL", "MSFT"],
        starting_equity=100_000.0,
        metadata={"strategy": "v1"},
    )


def test_reconstructs_complete_run_audit(audit_store):
    """A persisted run can be reconstructed end-to-end."""
    _start_run(audit_store)

    trace = _make_trace(
        "run-audit",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    audit_store.record_trade_trace(trace)

    audit_store.finish_run(
        "run-audit",
        finished_at="2026-09-25T10:00:00+00:00",
        final_equity=100_500.0,
        realized_pnl=500.0,
        benchmark_return=0.004,
    )

    audit = audit_store.get_run_audit("run-audit")

    assert audit["valid"] is True

    assert audit["run"]["run_id"] == "run-audit"
    assert audit["run"]["status"] == "completed"

    assert audit["summary"]["trace_count"] == 1
    assert audit["summary"]["successful_executions"] == 1
    assert audit["summary"]["normalized_trades"] == 1

    assert len(audit["traces"]) == 1
    assert len(audit["trades"]) == 1

    reconstructed = audit["traces"][0]

    assert reconstructed["ticker"] == "AAPL"
    assert reconstructed["execution_success"] is True
    assert reconstructed["trace"]["execution"]["side"] == "buy"

    normalized_trade = reconstructed["normalized_trades"][0]

    assert normalized_trade["ticker"] == "AAPL"
    assert normalized_trade["side"] == "BUY"
    assert normalized_trade["qty"] == pytest.approx(41.6667)
    assert normalized_trade["price"] == pytest.approx(150.0)
    assert normalized_trade["order_id"] == "order-001"


def test_reconstruction_validates_trace_count(audit_store):
    """Run metadata must agree with persisted TradeTrace count."""
    _start_run(audit_store)

    trace = _make_trace(
        "run-audit",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    audit_store.record_trade_trace(trace)

    audit = audit_store.get_run_audit("run-audit")

    assert audit["invariants"]["run_trace_count_matches"] is True
    assert audit["run"]["trade_trace_count"] == 1
    assert audit["summary"]["trace_count"] == 1


def test_successful_execution_reconciles_to_one_trade(audit_store):
    """Every successful execution must have exactly one normalized trade."""
    _start_run(audit_store)

    trace = _make_trace(
        "run-audit",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    audit_store.record_trade_trace(trace)

    audit = audit_store.get_run_audit("run-audit")

    assert audit["invariants"]["successful_execution_count_matches_trade_count"] is True

    assert audit["invariants"]["successful_traces_have_exactly_one_trade"] is True

    assert audit["diagnostics"]["invalid_trade_link_count"] == 0


def test_rejected_trace_has_no_normalized_trade(audit_store):
    """A non-executed trace must not produce a normalized trade."""
    _start_run(audit_store)

    base_trace = _make_trace(
        "run-audit",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    rejected_trace = replace(
        base_trace,
        coordinator=CoordinatorTrace(
            approved=False,
            shares=0.0,
            notes=["portfolio_limit_reject"],
        ),
        execution=ExecutionTrace(
            attempted=False,
            success=False,
            side=None,
            quantity=None,
            price=None,
            order_id=None,
        ),
        metadata={
            "tick_id": "AAPL-2026-08-01",
            "outcome": "ticker_capacity",
        },
    )

    audit_store.record_trade_trace(rejected_trace)

    audit = audit_store.get_run_audit("run-audit")

    assert audit["valid"] is True

    assert audit["summary"]["trace_count"] == 1
    assert audit["summary"]["successful_executions"] == 0
    assert audit["summary"]["normalized_trades"] == 0

    assert audit["invariants"]["rejected_traces_have_no_trade"] is True


def test_execution_and_normalized_trade_fields_match(audit_store):
    """Normalized trade fields must match the execution inside TradeTrace."""
    _start_run(audit_store)

    trace = _make_trace(
        "run-audit",
        "AAPL",
        datetime(2026, 8, 1, tzinfo=UTC),
    )

    audit_store.record_trade_trace(trace)

    audit = audit_store.get_run_audit("run-audit")

    assert audit["invariants"]["execution_fields_match_trade"] is True
    assert audit["diagnostics"]["execution_trade_mismatch_count"] == 0


def test_multi_ticker_reconstruction_isolated_by_run(audit_store):
    """A run reconstruction contains only that run's tickers and trades."""
    _start_run(audit_store, "run-A")

    audit_store.start_run(
        "run-B",
        started_at="2026-09-25T10:00:00+00:00",
        tickers=["TSLA"],
        starting_equity=200_000.0,
    )

    audit_store.record_trade_trace(
        _make_trace(
            "run-A",
            "AAPL",
            datetime(2026, 8, 1, tzinfo=UTC),
        )
    )

    audit_store.record_trade_trace(
        _make_trace(
            "run-A",
            "MSFT",
            datetime(2026, 8, 2, tzinfo=UTC),
        )
    )

    audit_store.record_trade_trace(
        _make_trace(
            "run-B",
            "TSLA",
            datetime(2026, 8, 1, tzinfo=UTC),
        )
    )

    audit_a = audit_store.get_run_audit("run-A")

    assert audit_a["valid"] is True
    assert audit_a["summary"]["trace_count"] == 2
    assert audit_a["summary"]["normalized_trades"] == 2

    assert {trace["ticker"] for trace in audit_a["traces"]} == {"AAPL", "MSFT"}

    assert {trade["ticker"] for trade in audit_a["trades"]} == {"AAPL", "MSFT"}


def test_unknown_run_audit_raises(audit_store):
    """Reconstruction of an unknown run must fail explicitly."""
    with pytest.raises(KeyError, match="Unknown run_id"):
        audit_store.get_run_audit("does-not-exist")
