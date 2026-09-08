import pytest
from portfolio.circuit_breaker import CircuitBreaker


def test_no_trip_on_flat_day():
    breaker = CircuitBreaker(threshold_pct=-0.03)
    assert breaker.check(100_000.0, 100_000.0) is False


def test_no_trip_just_above_threshold():
    breaker = CircuitBreaker(threshold_pct=-0.03)
    # -2.99% should NOT trip a -3% breaker
    assert breaker.check(100_000.0, 97_010.0) is False


def test_trips_exactly_at_threshold():
    breaker = CircuitBreaker(threshold_pct=-0.03)
    assert breaker.check(100_000.0, 97_000.0) is True


def test_trips_below_threshold():
    breaker = CircuitBreaker(threshold_pct=-0.03)
    assert breaker.check(100_000.0, 90_000.0) is True


def test_daily_pnl_pct_computation():
    assert CircuitBreaker.daily_pnl_pct(100_000.0, 105_000.0) == pytest.approx(0.05)
    assert CircuitBreaker.daily_pnl_pct(100_000.0, 95_000.0) == pytest.approx(-0.05)


def test_zero_starting_equity_raises():
    breaker = CircuitBreaker()
    with pytest.raises(ValueError):
        breaker.check(0.0, 100.0)
