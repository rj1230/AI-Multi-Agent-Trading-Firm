"""
PortfolioRiskCoordinator -- deterministic book-level risk gate.

Responsibilities:
- total exposure vs. equity
- sector concentration
- max concurrent positions
- correlation checks for new/increasing exposure
- circuit breaker
- long-only position availability for SELL orders

The coordinator receives proposals that already passed the per-ticker
RiskAgent.

V1 is long-only:
    bullish -> BUY
    bearish -> SELL existing long position only

A SELL cannot create a short position. If the portfolio owns fewer shares
than requested, the SELL quantity is clamped to the actual position size.
This guarantees that the book never creates a negative position while
allowing an oversized exit signal to fully close the existing long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from config.risk_config import RiskConfig
from portfolio.circuit_breaker import CircuitBreaker
from portfolio.correlation import check_correlation
from portfolio.state import PortfolioSnapshot, Position

TradeSide = Literal["buy", "sell"]


@dataclass
class TickerProposal:
    """
    One ticker's locally-approved trade.

    Comes from the ticker's local RiskAgent.

    side:
        buy  -> increase/open a long position
        sell -> reduce/close an existing long position

    Defaults to "buy" for backwards compatibility with existing tests and
    callers that predate explicit trade-side telemetry.
    """

    ticker: str
    sector: str
    entry_price: float
    proposed_shares: float
    combined_confidence: float
    side: TradeSide = "buy"


@dataclass
class CoordinatorDecision:
    ticker: str
    approved: bool
    shares: float
    notes: list[str] = field(default_factory=list)


def _position_value(
    position: Position | None,
) -> float:
    return position.market_value if position is not None else 0.0


def run_portfolio_coordinator(
    proposals: list[TickerProposal],
    portfolio: PortfolioSnapshot,
    config: RiskConfig,
) -> dict[str, CoordinatorDecision]:
    """
    Return {ticker: CoordinatorDecision} for every proposal.

    BUY proposals consume book risk budget.

    SELL proposals reduce an existing long position and therefore do not
    consume additional exposure budget.

    SELL quantities larger than the current long position are clamped to
    the actual held shares.

    V1 never permits a SELL to create a negative position or a short.
    """

    decisions: dict[str, CoordinatorDecision] = {}

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    breaker = CircuitBreaker(threshold_pct=config.circuit_breaker.daily_drawdown_pct)

    if breaker.check(
        portfolio.starting_equity,
        portfolio.equity,
    ):
        pnl = CircuitBreaker.daily_pnl_pct(
            portfolio.starting_equity,
            portfolio.equity,
        )

        for proposal in proposals:
            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker,
                approved=False,
                shares=0.0,
                notes=[f"Circuit breaker halted the book (today's P&L {pnl:.2%})."],
            )

        return decisions

    # ------------------------------------------------------------------
    # Highest-conviction proposals get first claim on shared risk budget.
    # ------------------------------------------------------------------

    ordered = sorted(
        proposals,
        key=lambda proposal: (
            -proposal.combined_confidence,
            proposal.ticker,
        ),
    )

    # Running state represents the portfolio after already-approved
    # proposals from this same tick have been applied.
    running_positions: dict[str, Position] = dict(portfolio.positions)

    running_held_tickers: list[str] = list(portfolio.positions.keys())

    # ------------------------------------------------------------------
    # Evaluate each proposal.
    # ------------------------------------------------------------------

    for proposal in ordered:
        notes: list[str] = []

        existing = running_positions.get(proposal.ticker)

        existing_shares = existing.shares if existing is not None else 0.0

        existing_value = _position_value(existing)

        # ==============================================================
        # SELL
        # ==============================================================

        if proposal.side == "sell":
            if existing is None or existing_shares <= 0:
                decisions[proposal.ticker] = CoordinatorDecision(
                    ticker=proposal.ticker,
                    approved=False,
                    shares=0.0,
                    notes=[
                        (
                            f"Long-only SELL rejected: no existing "
                            f"{proposal.ticker} position to sell."
                        )
                    ],
                )
                continue

            # A long-only SELL must never exceed the actual position.
            # RiskAgent sizing can legitimately produce a quantity larger
            # than the current holding, so clamp it at the book boundary.
            sell_shares = min(
                proposal.proposed_shares,
                existing_shares,
            )

            if sell_shares <= 1e-12:
                decisions[proposal.ticker] = CoordinatorDecision(
                    ticker=proposal.ticker,
                    approved=False,
                    shares=0.0,
                    notes=[
                        (
                            f"Long-only SELL rejected: no executable "
                            f"shares available for {proposal.ticker}."
                        )
                    ],
                )
                continue

            if proposal.proposed_shares > existing_shares:
                notes.append(
                    f"Long-only SELL clamped: requested "
                    f"{proposal.proposed_shares:.4f} shares but only "
                    f"{existing_shares:.4f} shares are held; "
                    f"selling {sell_shares:.4f}."
                )

            sell_value = sell_shares * proposal.entry_price

            remaining_shares = existing_shares - sell_shares

            remaining_value = max(
                0.0,
                existing_value - sell_value,
            )

            # Remove the ticker completely when the SELL closes the
            # position. Otherwise update the running position so later
            # proposals in this same tick see the reduced exposure.
            if remaining_shares <= 1e-12:
                running_positions.pop(
                    proposal.ticker,
                    None,
                )

                if proposal.ticker in running_held_tickers:
                    running_held_tickers.remove(
                        proposal.ticker,
                    )
            else:
                running_positions[proposal.ticker] = Position(
                    ticker=proposal.ticker,
                    sector=existing.sector,
                    shares=remaining_shares,
                    market_value=remaining_value,
                )

            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker,
                approved=True,
                shares=sell_shares,
                notes=[
                    *notes,
                    (
                        f"Approved long-only SELL: selling "
                        f"{sell_shares:.4f} of "
                        f"{existing_shares:.4f} held shares."
                    ),
                ],
            )

            continue

        # ==============================================================
        # BUY
        # ==============================================================

        trade_value = proposal.proposed_shares * proposal.entry_price

        new_ticker_value = existing_value + trade_value

        ticker_pct = new_ticker_value / portfolio.equity if portfolio.equity else 1.0

        # Existing positions from the same sector, excluding the ticker
        # being evaluated because its new value is calculated separately.
        sector_value = sum(
            position.market_value
            for ticker, position in running_positions.items()
            if (position.sector == proposal.sector and ticker != proposal.ticker)
        )

        new_sector_value = sector_value + new_ticker_value

        sector_pct = new_sector_value / portfolio.equity if portfolio.equity else 1.0

        would_open_new_ticker = proposal.ticker not in running_positions

        new_open_count = len(running_positions) + (1 if would_open_new_ticker else 0)

        # Correlation matters when adding/increasing exposure.
        corr_result = check_correlation(
            proposal.ticker,
            running_held_tickers,
            portfolio.correlation_matrix,
            config.correlation.max_correlation,
        )

        checks_passed = True

        if ticker_pct > config.per_ticker_cap_pct:
            checks_passed = False
            notes.append(
                f"Book-level per-ticker cap: would be "
                f"{ticker_pct:.1%} of equity, cap "
                f"{config.per_ticker_cap_pct:.0%}."
            )

        if sector_pct > config.per_sector_cap_pct:
            checks_passed = False
            notes.append(
                f"Book-level per-sector cap "
                f"({proposal.sector}): would be "
                f"{sector_pct:.1%} of equity "
                f"(including other trades approved this tick), "
                f"cap {config.per_sector_cap_pct:.0%}."
            )

        if new_open_count > config.max_concurrent_positions:
            checks_passed = False
            notes.append(
                f"Book-level concurrent positions: would be "
                f"{new_open_count} open, max "
                f"{config.max_concurrent_positions}."
            )

        if corr_result["flagged"]:
            checks_passed = False
            notes.append(
                f"Book-level correlation: {proposal.ticker} at "
                f"{corr_result['max_correlation']} vs "
                f"{corr_result['against']} exceeds "
                f"{config.correlation.max_correlation}."
            )

        if checks_passed:
            running_positions[proposal.ticker] = Position(
                ticker=proposal.ticker,
                sector=proposal.sector,
                shares=(existing_shares + proposal.proposed_shares),
                market_value=new_ticker_value,
            )

            if would_open_new_ticker:
                running_held_tickers.append(proposal.ticker)

            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker,
                approved=True,
                shares=proposal.proposed_shares,
                notes=["Approved at book level."],
            )

        else:
            notes.append(
                "Rejected at book level -- higher-conviction "
                "trades approved this tick already used the "
                "available risk budget."
            )

            decisions[proposal.ticker] = CoordinatorDecision(
                ticker=proposal.ticker,
                approved=False,
                shares=0.0,
                notes=notes,
            )

    return decisions
