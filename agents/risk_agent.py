"""
Local RiskAgent: gates a merged signal against the fund's risk mandate.

Deliberately zero LLM involvement (architecture doc, section 2) -- every
rule below is a plain function so each one can independently produce a
risk_notes entry and be unit-tested in isolation. This is Phase 5 only:
the per-ticker view. PortfolioRiskCoordinator (portfolio/coordinator.py)
is the portfolio-wide layer that looks across all tickers at once.

CHANGE (post-Phase-5 fix): position sizing and the per-ticker/per-sector
caps are two independent formulas -- nothing previously reconciled them,
so any risk-based size that happened to exceed a cap caused an outright
rejection instead of a resize. For most real price/ATR ratios (low-
volatility large caps especially), the risk-based share count structurally
exceeds the caps, which meant almost nothing ever executed.

Fix: compute_capped_size() clamps the risk-based share count down to
whatever room is left under BOTH caps before the trade is evaluated.
Rejection now only happens when there's no room left at all (already at
or over a cap), not whenever raw sizing happens to be large. The raw
(pre-cap) size is preserved in risk_notes for debugging -- previously it
was computed internally, used to test the cap, and then discarded, so a
0.0 proposed_shares told you nothing about what actually triggered the
breach.

Circuit breaker and correlation checks call the real implementations in
portfolio/circuit_breaker.py and portfolio/correlation.py. Caller is
responsible for building a fresh PortfolioSnapshot (portfolio/state.py)
each tick, which includes calling build_correlation_matrix() once before
looping over tickers.
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
    raw_shares: float = 0.0  # pre-cap size, kept for debugging/observability
    checks: list[RiskCheckResult] = field(default_factory=list)

    @property
    def risk_notes(self) -> list[str]:
        return [c.note for c in self.checks]


# --- individual rule functions -------------------------------------------------


def check_circuit_breaker(
    portfolio: PortfolioSnapshot, config: RiskConfig
) -> RiskCheckResult:
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


def check_concurrent_positions(
    portfolio: PortfolioSnapshot, ticker: str, config: RiskConfig
) -> RiskCheckResult:
    open_count = len(portfolio.positions)
    already_held = ticker in portfolio.positions
    would_exceed = (not already_held) and (
        open_count + 1 > config.max_concurrent_positions
    )
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
    """1% equity risk, stop at 1.5x ATR from entry -> raw (pre-cap) shares to buy."""
    if atr <= 0 or entry_price <= 0:
        return 0.0
    risk_dollars = equity * config.position_sizing.equity_risk_pct
    stop_distance = atr * config.position_sizing.atr_stop_multiple
    return round(risk_dollars / stop_distance, 4)


def compute_capped_size(
    risk_based_shares: float,
    entry_price: float,
    portfolio: PortfolioSnapshot,
    ticker: str,
    sector: str,
    config: RiskConfig,
) -> tuple[float, float, float]:
    """
    Clamp the risk-based share count down to whatever room is left under
    BOTH the per-ticker and per-sector caps. Returns
    (capped_shares, ticker_room_value, sector_room_value) so callers can
    log exactly why a size was reduced (or zeroed out).

    A negative room value means the cap is already breached by existing
    holdings alone -- capped_shares will be 0.0 in that case.
    """
    if entry_price <= 0:
        return 0.0, 0.0, 0.0

    existing = portfolio.positions.get(ticker)
    existing_value = existing.market_value if existing else 0.0

    ticker_room_value = config.per_ticker_cap_pct * portfolio.equity - existing_value

    sector_value = sum(
        p.market_value
        for t, p in portfolio.positions.items()
        if p.sector == sector and t != ticker
    )
    sector_room_value = config.per_sector_cap_pct * portfolio.equity - (
        sector_value + existing_value
    )

    room_value = min(ticker_room_value, sector_room_value)
    room_shares = max(0.0, room_value) / entry_price
    capped_shares = round(max(0.0, min(risk_based_shares, room_shares)), 4)
    return capped_shares, ticker_room_value, sector_room_value


def check_per_ticker_cap(
    portfolio: PortfolioSnapshot,
    ticker: str,
    proposed_shares: float,
    entry_price: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """
    Reports the resulting per-ticker exposure for the (already-capped)
    proposed_shares. With clamping upstream this should always pass;
    it only fails if there's zero room left (existing holdings alone
    already breach the cap).
    """
    existing = portfolio.positions.get(ticker)
    existing_value = existing.market_value if existing else 0.0
    proposed_value = existing_value + proposed_shares * entry_price
    pct = proposed_value / portfolio.equity if portfolio.equity else 1.0
    passed = pct <= config.per_ticker_cap_pct + 1e-9
    return RiskCheckResult(
        rule="per_ticker_cap",
        passed=passed,
        note=(
            f"Per-ticker cap ({ticker}): would be {pct:.1%} of equity, "
            f"cap {config.per_ticker_cap_pct:.0%} -> {'ok' if passed else 'BREACH (no room left)'}."
        ),
    )


def check_per_sector_cap(
    portfolio: PortfolioSnapshot,
    ticker: str,
    sector: str,
    proposed_shares: float,
    entry_price: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """
    Reports the resulting per-sector exposure for the (already-capped)
    proposed_shares. With clamping upstream this should always pass;
    it only fails if there's zero room left in the sector.
    """
    sector_value = sum(
        p.market_value
        for t, p in portfolio.positions.items()
        if p.sector == sector and t != ticker
    )
    existing = portfolio.positions.get(ticker)
    existing_value = existing.market_value if existing else 0.0
    proposed_sector_value = (
        sector_value + existing_value + proposed_shares * entry_price
    )
    pct = proposed_sector_value / portfolio.equity if portfolio.equity else 1.0
    passed = pct <= config.per_sector_cap_pct + 1e-9
    return RiskCheckResult(
        rule="per_sector_cap",
        passed=passed,
        note=(
            f"Per-sector cap ({sector}): would be {pct:.1%} of equity, "
            f"cap {config.per_sector_cap_pct:.0%} -> {'ok' if passed else 'BREACH (no room left)'}."
        ),
    )


def check_correlation(
    portfolio: PortfolioSnapshot,
    ticker: str,
    config: RiskConfig,
) -> RiskCheckResult:
    held_tickers = list(portfolio.positions.keys())
    result = _check_correlation(
        ticker,
        held_tickers,
        portfolio.correlation_matrix,
        config.correlation.max_correlation,
    )
    passed = not result["flagged"]
    return RiskCheckResult(
        rule="correlation",
        passed=passed,
        note=(
            f"Correlation: ok, no existing holding above {config.correlation.max_correlation}."
            if passed
            else f"Correlation: {ticker} at {result['max_correlation']} vs {result['against']} "
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
    """
    Runs every rule. Sizing now clamps to available cap room BEFORE the
    cap checks run, so a trade is only rejected outright when there's no
    room left at all -- not whenever the raw risk-based size happens to
    exceed a cap (which, for many real price/ATR ratios, is most of the
    time).
    """
    checks: list[RiskCheckResult] = [check_circuit_breaker(portfolio, config)]

    if merged_signal.direction == "neutral":
        checks.append(
            RiskCheckResult(
                rule="signal_direction",
                passed=False,
                note="Merged signal is neutral -- no trade to evaluate.",
            )
        )
        return RiskDecision(
            approved=False,
            ticker=ticker,
            proposed_shares=0.0,
            raw_shares=0.0,
            checks=checks,
        )

    raw_shares = compute_position_size(portfolio.equity, entry_price, atr, config)
    capped_shares, ticker_room_value, sector_room_value = compute_capped_size(
        raw_shares, entry_price, portfolio, ticker, sector, config
    )

    if capped_shares < raw_shares:
        checks.append(
            RiskCheckResult(
                rule="position_sizing",
                passed=True,
                note=(
                    f"Sizing: raw risk-based size {raw_shares:.2f} sh "
                    f"(${raw_shares * entry_price:,.0f}) resized to {capped_shares:.2f} sh "
                    f"(${capped_shares * entry_price:,.0f}) -- ticker room "
                    f"${ticker_room_value:,.0f}, sector room ${sector_room_value:,.0f}."
                ),
            )
        )
    else:
        checks.append(
            RiskCheckResult(
                rule="position_sizing",
                passed=True,
                note=f"Sizing: raw risk-based size {raw_shares:.2f} sh fits within caps, no resize needed.",
            )
        )

    checks.append(check_concurrent_positions(portfolio, ticker, config))
    checks.append(
        check_per_ticker_cap(portfolio, ticker, capped_shares, entry_price, config)
    )
    checks.append(
        check_per_sector_cap(
            portfolio, ticker, sector, capped_shares, entry_price, config
        )
    )
    checks.append(check_correlation(portfolio, ticker, config))

    # Reject only if there's truly no room (capped size rounds to ~0) or
    # another rule (circuit breaker, concurrency, correlation) failed.
    no_room = capped_shares <= 0.0001
    other_checks_pass = all(
        c.passed for c in checks if c.rule not in ("per_ticker_cap", "per_sector_cap")
    )
    approved = other_checks_pass and not no_room

    if no_room and other_checks_pass:
        checks.append(
            RiskCheckResult(
                rule="sizing_rejected",
                passed=False,
                note=f"No cap room remains for {ticker} in {sector} -- trade rejected, not resized.",
            )
        )

    return RiskDecision(
        approved=approved,
        ticker=ticker,
        proposed_shares=capped_shares if approved else 0.0,
        raw_shares=raw_shares,
        checks=checks,
    )
