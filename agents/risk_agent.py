"""
Local RiskAgent: gates a merged signal against the fund's risk mandate.

Deliberately zero LLM involvement (architecture doc, section 2) -- every
rule below is a plain function so each one can independently produce a
risk_notes entry and be unit-tested in isolation. This is Phase 5 only:
the per-ticker view. PortfolioRiskCoordinator (portfolio/coordinator.py)
is the portfolio-wide layer that looks across all tickers at once.

Circuit breaker and correlation checks now call the real implementations
in portfolio/circuit_breaker.py and portfolio/correlation.py -- no more
fixture floats/dicts standing in for them. Caller is responsible for
building a fresh PortfolioSnapshot (portfolio/state.py) each tick, which
includes calling build_correlation_matrix() once before looping over
tickers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.risk_config import RiskConfig
from agents.signal_merger import MergedSignal
from portfolio.circuit_breaker import CircuitBreaker
from portfolio.correlation import check_correlation as _check_correlation
from portfolio.state import PortfolioSnapshot


@dataclass
class RiskCheckResult:
    rule: str
    passed: bool
    note: str


@dataclass
class RiskDecision:
    approved: bool
    ticker: str
    proposed_shares: float
    checks: list[RiskCheckResult] = field(default_factory=list)

    @property
    def risk_notes(self) -> list[str]:
        return [c.note for c in self.checks]


# --- individual rule functions -------------------------------------------------

def check_circuit_breaker(portfolio: PortfolioSnapshot, config: RiskConfig) -> RiskCheckResult:
    breaker = CircuitBreaker(threshold_pct=config.circuit_breaker.daily_drawdown_pct)
    tripped = breaker.check(portfolio.starting_equity, portfolio.equity)
    pnl_pct = CircuitBreaker.daily_pnl_pct(portfolio.starting_equity, portfolio.equity)
    return RiskCheckResult(
        rule="circuit_breaker",
        passed=not tripped,
        note=(
            f"Circuit breaker: today's P&L {pnl_pct:.2%} "
            f"{'<=' if tripped else '>'} threshold {config.circuit_breaker.daily_drawdown_pct:.2%} -> "
            f"{'HALTED' if tripped else 'ok'}."
        ),
    )


def check_concurrent_positions(portfolio: PortfolioSnapshot, ticker: str, config: RiskConfig) -> RiskCheckResult:
    open_count = len(portfolio.positions)
    already_held = ticker in portfolio.positions
    would_exceed = (not already_held) and (open_count + 1 > config.max_concurrent_positions)
    return RiskCheckResult(
        rule="concurrent_positions",
        passed=not would_exceed,
        note=(
            f"Concurrent positions: {open_count} open, max {config.max_concurrent_positions} -> "
            f"{'would exceed limit' if would_exceed else 'ok'}."
        ),
    )


def compute_position_size(
    equity: float, entry_price: float, atr: float, config: RiskConfig
) -> float:
    """1% equity risk, stop at 1.5x ATR from entry -> shares to buy."""
    if atr <= 0 or entry_price <= 0:
        return 0.0
    risk_dollars = equity * config.position_sizing.equity_risk_pct
    stop_distance = atr * config.position_sizing.atr_stop_multiple
    return round(risk_dollars / stop_distance, 4)


def check_per_ticker_cap(
    portfolio: PortfolioSnapshot, ticker: str, proposed_shares: float,
    entry_price: float, config: RiskConfig,
) -> RiskCheckResult:
    existing = portfolio.positions.get(ticker)
    existing_value = existing.market_value if existing else 0.0
    proposed_value = existing_value + proposed_shares * entry_price
    pct = proposed_value / portfolio.equity if portfolio.equity else 1.0
    passed = pct <= config.per_ticker_cap_pct
    return RiskCheckResult(
        rule="per_ticker_cap",
        passed=passed,
        note=(
            f"Per-ticker cap ({ticker}): would be {pct:.1%} of equity, "
            f"cap {config.per_ticker_cap_pct:.0%} -> {'ok' if passed else 'BREACH'}."
        ),
    )


def check_per_sector_cap(
    portfolio: PortfolioSnapshot, ticker: str, sector: str, proposed_shares: float,
    entry_price: float, config: RiskConfig,
) -> RiskCheckResult:
    sector_value = sum(
        p.market_value for t, p in portfolio.positions.items()
        if p.sector == sector and t != ticker
    )
    existing = portfolio.positions.get(ticker)
    existing_value = existing.market_value if existing else 0.0
    proposed_sector_value = sector_value + existing_value + proposed_shares * entry_price
    pct = proposed_sector_value / portfolio.equity if portfolio.equity else 1.0
    passed = pct <= config.per_sector_cap_pct
    return RiskCheckResult(
        rule="per_sector_cap",
        passed=passed,
        note=(
            f"Per-sector cap ({sector}): would be {pct:.1%} of equity, "
            f"cap {config.per_sector_cap_pct:.0%} -> {'ok' if passed else 'BREACH'}."
        ),
    )


def check_correlation(
    portfolio: PortfolioSnapshot, ticker: str, config: RiskConfig,
) -> RiskCheckResult:
    held_tickers = list(portfolio.positions.keys())
    result = _check_correlation(
        ticker, held_tickers, portfolio.correlation_matrix, config.correlation.max_correlation
    )
    passed = not result["flagged"]
    return RiskCheckResult(
        rule="correlation",
        passed=passed,
        note=(
            f"Correlation: ok, no existing holding above {config.correlation.max_correlation}."
            if passed else
            f"Correlation: {ticker} at {result['max_correlation']} vs {result['against']} "
            f"exceeds {config.correlation.max_correlation} -> reject/downsize."
        ),
    )


# --- orchestration ---------------------------------------------------------

def run_risk_agent(
    merged_signal: MergedSignal,
    ticker: str,
    sector: str,
    entry_price: float,
    atr: float,
    portfolio: PortfolioSnapshot,
    config: RiskConfig,
) -> RiskDecision:
    """Runs every rule independently and approves only if all pass."""
    checks: list[RiskCheckResult] = [check_circuit_breaker(portfolio, config)]

    if merged_signal.direction == "neutral":
        checks.append(RiskCheckResult(
            rule="signal_direction",
            passed=False,
            note="Merged signal is neutral -- no trade to evaluate.",
        ))
        return RiskDecision(approved=False, ticker=ticker, proposed_shares=0.0, checks=checks)

    proposed_shares = compute_position_size(portfolio.equity, entry_price, atr, config)

    checks.append(check_concurrent_positions(portfolio, ticker, config))
    checks.append(check_per_ticker_cap(portfolio, ticker, proposed_shares, entry_price, config))
    checks.append(check_per_sector_cap(portfolio, ticker, sector, proposed_shares, entry_price, config))
    checks.append(check_correlation(portfolio, ticker, config))

    approved = all(c.passed for c in checks)
    return RiskDecision(
        approved=approved,
        ticker=ticker,
        proposed_shares=proposed_shares if approved else 0.0,
        checks=checks,
    )
