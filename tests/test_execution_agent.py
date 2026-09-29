from agents.execution_agent import run_execution_agent
from agents.risk_agent import RiskCheckResult, RiskDecision
from agents.signal_merger import MergedSignal
from broker.protocol import OrderResult
from broker.sim_broker import SimBroker


def _approved_decision(ticker="AAPL", shares=10.0):
    return RiskDecision(
        approved=True,
        ticker=ticker,
        proposed_shares=shares,
        checks=[RiskCheckResult(rule="circuit_breaker", passed=True, note="ok")],
    )


def _bullish_signal():
    return MergedSignal(
        direction="bullish",
        combined_confidence=0.7,
        agreement=True,
        rationale="test",
    )


def test_approved_decision_executes_via_broker():
    broker = SimBroker(starting_cash=10_000.0, price_lookup=lambda t: 100.0)

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
    )

    assert result.executed is True
    assert result.order is not None
    assert result.order.status == "filled"
    assert result.order.filled_qty == 10.0
    assert result.order.filled_qty == result.order.qty
    assert "AAPL" in broker.get_positions()


def test_bearish_signal_maps_to_sell_side():
    broker = SimBroker(starting_cash=10_000.0, price_lookup=lambda t: 100.0)
    broker.submit_order("AAPL", "buy", 10)

    decision = _approved_decision(shares=5.0)

    bearish = MergedSignal(
        direction="bearish",
        combined_confidence=0.6,
        agreement=True,
        rationale="x",
    )

    result = run_execution_agent(
        decision,
        bearish,
        broker,
    )

    assert result.executed is True
    assert result.order is not None
    assert result.order.status == "filled"
    assert result.order.side == "sell"
    assert result.order.filled_qty == 5.0


def test_unapproved_decision_is_never_sent_to_broker():
    class ExplodesIfCalled:
        def submit_order(self, *a, **kw):
            raise AssertionError("should never be called for an unapproved decision")

    rejected = RiskDecision(
        approved=False,
        ticker="AAPL",
        proposed_shares=0.0,
        checks=[
            RiskCheckResult(
                rule="circuit_breaker",
                passed=False,
                note="halted",
            )
        ],
    )

    result = run_execution_agent(
        rejected,
        _bullish_signal(),
        ExplodesIfCalled(),
    )

    assert result.executed is False
    assert result.order is None
    assert "not approve" in result.notes.lower()


class _FlakyThenOkBroker:
    """Fails the first submit_order call, succeeds on the second."""

    def __init__(self):
        self.calls = 0
        self.client_order_ids = []

    def submit_order(self, ticker, side, qty, client_order_id=None):
        self.calls += 1
        self.client_order_ids.append(client_order_id)

        if self.calls == 1:
            raise ConnectionError("simulated timeout")

        return OrderResult(
            order_id="X1",
            ticker=ticker,
            side=side,
            qty=qty,
            status="filled",
            filled_qty=qty,
            filled_avg_price=100.0,
        )

    def get_positions(self):
        return {}

    def get_account(self):
        raise NotImplementedError


def test_retries_once_after_transient_failure_then_succeeds():
    broker = _FlakyThenOkBroker()

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
        max_retries=1,
        run_id="run-1",
        tick_id=1,
    )

    assert result.executed is True
    assert result.order is not None
    assert result.order.status == "filled"
    assert result.order.filled_qty == result.order.qty
    assert broker.calls == 2

    # Retry must reuse the exact same logical client order identity.
    assert broker.client_order_ids[0] == broker.client_order_ids[1]


class _AlwaysRejectsBroker:
    def __init__(self):
        self.calls = 0

    def submit_order(self, ticker, side, qty, client_order_id=None):
        self.calls += 1

        return OrderResult(
            order_id="X",
            ticker=ticker,
            side=side,
            qty=qty,
            status="rejected",
            filled_qty=0.0,
            raw={"reason": "no buying power"},
        )

    def get_positions(self):
        return {}

    def get_account(self):
        raise NotImplementedError


def test_preserves_rejected_order_without_retry():
    broker = _AlwaysRejectsBroker()

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
        max_retries=1,
        run_id="run-1",
        tick_id=1,
    )

    assert result.executed is False
    assert result.order is not None
    assert result.order.status == "rejected"
    assert result.order.filled_qty == 0.0
    assert result.order.order_id == "X"
    assert broker.calls == 1
    assert "no buying power" in result.notes


class _LifecycleBroker:
    """Returns one deterministic broker lifecycle result."""

    def __init__(self, status, filled_qty=0.0):
        self.status = status
        self.filled_qty = filled_qty
        self.calls = 0

    def submit_order(self, ticker, side, qty, client_order_id=None):
        self.calls += 1

        return OrderResult(
            order_id=f"ORDER-{self.status}",
            ticker=ticker,
            side=side,
            qty=qty,
            status=self.status,
            filled_qty=self.filled_qty,
            filled_avg_price=100.0 if self.filled_qty > 0 else None,
            raw={"reason": "test"} if self.status == "rejected" else {},
        )

    def get_positions(self):
        return {}

    def get_account(self):
        raise NotImplementedError


def test_partial_fill_is_not_treated_as_full_execution():
    broker = _LifecycleBroker(
        status="partially_filled",
        filled_qty=4.0,
    )

    result = run_execution_agent(
        _approved_decision(shares=10.0),
        _bullish_signal(),
        broker,
    )

    assert result.executed is False
    assert result.order is not None
    assert result.order.status == "partially_filled"
    assert result.order.qty == 10.0
    assert result.order.filled_qty == 4.0
    assert result.order.filled_qty < result.order.qty
    assert broker.calls == 1


def test_pending_order_is_preserved_without_retry():
    broker = _LifecycleBroker(
        status="pending",
        filled_qty=0.0,
    )

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
        max_retries=1,
    )

    assert result.executed is False
    assert result.order is not None
    assert result.order.status == "pending"
    assert result.order.filled_qty == 0.0
    assert broker.calls == 1


def test_canceled_order_is_preserved_without_retry():
    broker = _LifecycleBroker(
        status="canceled",
        filled_qty=0.0,
    )

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
        max_retries=1,
    )

    assert result.executed is False
    assert result.order is not None
    assert result.order.status == "canceled"
    assert result.order.filled_qty == 0.0
    assert broker.calls == 1


def test_expired_order_is_preserved_without_retry():
    broker = _LifecycleBroker(
        status="expired",
        filled_qty=0.0,
    )

    result = run_execution_agent(
        _approved_decision(),
        _bullish_signal(),
        broker,
        max_retries=1,
    )

    assert result.executed is False
    assert result.order is not None
    assert result.order.status == "expired"
    assert result.order.filled_qty == 0.0
    assert broker.calls == 1


def test_same_logical_execution_is_idempotent():
    """Same run/tick execution must not double-fill."""

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda t: 100.0,
    )

    decision = _approved_decision()
    signal = _bullish_signal()

    run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-1",
        tick_id=1,
    )

    run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-1",
        tick_id=1,
    )

    assert broker.get_positions()["AAPL"].qty == 10
    assert broker.cash == 9_000.0


def test_different_ticks_allow_legitimate_separate_executions():
    """Different ticks represent distinct logical executions."""

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda t: 100.0,
    )

    decision = _approved_decision()
    signal = _bullish_signal()

    first = run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-1",
        tick_id=1,
    )

    second = run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-1",
        tick_id=2,
    )

    assert first.executed is True
    assert second.executed is True
    assert first.order is not None
    assert second.order is not None
    assert first.order.order_id != second.order.order_id
    assert broker.get_positions()["AAPL"].qty == 20
    assert broker.cash == 8_000.0


def test_different_runs_allow_legitimate_separate_executions():
    """Different runs must not share an idempotency key."""

    broker = SimBroker(
        starting_cash=10_000.0,
        price_lookup=lambda t: 100.0,
    )

    decision = _approved_decision()
    signal = _bullish_signal()

    first = run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-1",
        tick_id=1,
    )

    second = run_execution_agent(
        decision,
        signal,
        broker,
        run_id="run-2",
        tick_id=1,
    )

    assert first.executed is True
    assert second.executed is True
    assert first.order is not None
    assert second.order is not None
    assert first.order.order_id != second.order.order_id
    assert broker.get_positions()["AAPL"].qty == 20
    assert broker.cash == 8_000.0
