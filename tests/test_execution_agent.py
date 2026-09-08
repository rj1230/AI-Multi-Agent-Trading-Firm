import pytest
from agents.execution_agent import run_execution_agent
from agents.risk_agent import RiskDecision, RiskCheckResult
from agents.signal_merger import MergedSignal
from broker.sim_broker import SimBroker
from broker.protocol import OrderResult


def _approved_decision(ticker="AAPL", shares=10.0):
    return RiskDecision(
        approved=True, ticker=ticker, proposed_shares=shares,
        checks=[RiskCheckResult(rule="circuit_breaker", passed=True, note="ok")],
    )


def _bullish_signal():
    return MergedSignal(direction="bullish", combined_confidence=0.7, agreement=True, rationale="test")


def test_approved_decision_executes_via_broker():
    broker = SimBroker(starting_cash=10_000.0, price_lookup=lambda t: 100.0)
    result = run_execution_agent(_approved_decision(), _bullish_signal(), broker)
    assert result.executed is True
    assert result.order.status == "filled"
    assert "AAPL" in broker.get_positions()


def test_bearish_signal_maps_to_sell_side():
    broker = SimBroker(starting_cash=10_000.0, price_lookup=lambda t: 100.0)
    broker.submit_order("AAPL", "buy", 10)  # something to sell
    decision = _approved_decision(shares=5.0)
    bearish = MergedSignal(direction="bearish", combined_confidence=0.6, agreement=True, rationale="x")
    result = run_execution_agent(decision, bearish, broker)
    assert result.order.side == "sell"


def test_unapproved_decision_is_never_sent_to_broker():
    class ExplodesIfCalled:
        def submit_order(self, *a, **kw):
            raise AssertionError("should never be called for an unapproved decision")

    rejected = RiskDecision(
        approved=False, ticker="AAPL", proposed_shares=0.0,
        checks=[RiskCheckResult(rule="circuit_breaker", passed=False, note="halted")],
    )
    result = run_execution_agent(rejected, _bullish_signal(), ExplodesIfCalled())
    assert result.executed is False
    assert "not approve" in result.notes.lower() or "not approve" in result.notes


class _FlakyThenOkBroker:
    """Fails the first submit_order call, succeeds on the second."""
    def __init__(self):
        self.calls = 0

    def submit_order(self, ticker, side, qty, client_order_id=None):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("simulated timeout")
        return OrderResult(order_id="X1", ticker=ticker, side=side, qty=qty, status="filled", filled_avg_price=100.0)

    def get_positions(self):
        return {}

    def get_account(self):
        raise NotImplementedError


def test_retries_once_after_transient_failure_then_succeeds():
    broker = _FlakyThenOkBroker()
    result = run_execution_agent(_approved_decision(), _bullish_signal(), broker, max_retries=1)
    assert result.executed is True
    assert broker.calls == 2


class _AlwaysRejectsBroker:
    def submit_order(self, ticker, side, qty, client_order_id=None):
        return OrderResult(order_id="X", ticker=ticker, side=side, qty=qty, status="rejected", raw={"reason": "no buying power"})

    def get_positions(self):
        return {}

    def get_account(self):
        raise NotImplementedError


def test_falls_back_to_hold_after_exhausting_retries():
    broker = _AlwaysRejectsBroker()
    result = run_execution_agent(_approved_decision(), _bullish_signal(), broker, max_retries=1)
    assert result.executed is False
    assert result.order is None
    assert "fall" in result.notes.lower()


def test_duplicate_client_order_id_prevents_double_submit_end_to_end():
    """Same approved decision run through ExecutionAgent twice (e.g. a
    retried graph node) must not double-fill, because both calls generate
    the same client_order_id."""
    broker = SimBroker(starting_cash=10_000.0, price_lookup=lambda t: 100.0)
    decision = _approved_decision()
    signal = _bullish_signal()
    run_execution_agent(decision, signal, broker)
    run_execution_agent(decision, signal, broker)
    assert broker.get_positions()["AAPL"].qty == 10  # not 20
