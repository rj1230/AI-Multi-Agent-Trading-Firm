"""
Real node implementations for the trading graph.

Graph flow:

    NewsAgent ───────┐
                     │
                     ▼
                SignalMerger
                     │
    ChartAgent ──────┘
                     │
                     ▼
                 RiskAgent
                  /      \
                 /        \
           APPROVED      REJECTED
              |             |
              v             v
       ExecutionAgent    HoldNode

Design principles:

1. Nodes communicate through TradingState.
2. SignalMerger is deterministic and contains no LLM call.
3. ChartAgent produces directional information only.
4. RiskAgent owns ATR, entry-price and position sizing logic.
5. ExecutionAgent only executes trades that passed RiskAgent.
6. HoldNode records blocked trades.
7. Broker implementation is hidden behind the Broker protocol.
8. PortfolioLedger is the local portfolio source of truth.
"""

from __future__ import annotations

from typing import Optional

from agents.chart_agent import compute_atr, run_chart_agent
from agents.execution_agent import run_execution_agent
from agents.news_agent import run_news_agent
from agents.risk_agent import RiskDecision as RiskAgentDecision
from agents.risk_agent import run_risk_agent
from agents.signal_merger import (
    MergedSignal,
    Signal as MergerSignal,
    merge_signals,
)

from broker.alpaca_broker import AlpacaBroker
from broker.protocol import Broker
from broker.sim_broker import SimBroker

from config.risk_config import load_risk_config
from config.sectors import get_sector
from config.settings import MODE, PAPER_STARTING_EQUITY

from data_sources import fetch_ohlcv

from graph.state import (
    AgentLogEntry,
    RiskDecision,
    Signal,
    TradingState,
)

from portfolio.ledger import PortfolioLedger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Enough history for ATR calculation.
ATR_LOOKBACK_DAYS = 30


# ---------------------------------------------------------------------------
# Lazy process-level singletons
# ---------------------------------------------------------------------------
#
# LangGraph nodes receive TradingState, not arbitrary infrastructure objects.
# These lazy singletons therefore provide:
#
#   - one loaded risk mandate
#   - one portfolio ledger
#   - one broker
#
# per running process.
#
# This is intentionally kept compatible with the current architecture.
# A future multi-account/service architecture can replace this with explicit
# dependency injection.
# ---------------------------------------------------------------------------

_risk_config_singleton = None
_ledger_singleton: Optional[PortfolioLedger] = None
_broker_singleton: Optional[Broker] = None


def _risk_config():
    """
    Load the validated YAML/Pydantic risk configuration once per process.
    """
    global _risk_config_singleton

    if _risk_config_singleton is None:
        _risk_config_singleton = load_risk_config()

    return _risk_config_singleton


def _ledger() -> PortfolioLedger:
    """
    Return the process-level portfolio ledger.

    The ledger is shared by all graph nodes in this process so RiskAgent
    and ExecutionAgent operate on the same portfolio state.
    """
    global _ledger_singleton

    if _ledger_singleton is None:
        _ledger_singleton = PortfolioLedger(starting_equity=PAPER_STARTING_EQUITY)

    return _ledger_singleton


def _price_lookup(ticker: str) -> float:
    """
    Return the latest available close for a ticker.

    Used by SimBroker when it needs a current price.

    IMPORTANT:
    This function is appropriate for the current/live simulation path.
    Historical backtesting should eventually use a timestamp-aware price
    lookup so that future prices can never leak into earlier ticks.
    """
    series = fetch_ohlcv(
        ticker,
        lookback_days=1,
    )

    if series.is_empty or series.latest is None:
        raise ValueError(f"No price data available for {ticker}")

    return series.latest.close


def _broker() -> Broker:
    """
    Return the process-level broker.

    LIVE:
        AlpacaBroker

    BACKTEST/PAPER:
        SimBroker seeded from the current ledger equity.

    ExecutionAgent depends only on the Broker protocol, not on a concrete
    broker implementation.
    """
    global _broker_singleton

    if _broker_singleton is None:
        if MODE == "live":
            _broker_singleton = AlpacaBroker()

        else:
            starting_cash = _ledger().snapshot().equity

            _broker_singleton = SimBroker(
                starting_cash=starting_cash,
                price_lookup=_price_lookup,
            )

    return _broker_singleton


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log(
    node: str,
    message: str,
) -> list[AgentLogEntry]:
    """
    Create one structured graph log entry.

    Returning a list works with TradingState.agent_logs using operator.add,
    allowing LangGraph to append logs from each node.
    """
    return [
        AgentLogEntry(
            node=node,
            message=message,
        )
    ]


# ---------------------------------------------------------------------------
# Signal conversion helpers
# ---------------------------------------------------------------------------


def _to_merger_signal(
    signal: Signal | None,
    label: str,
) -> MergerSignal:
    """
    Convert graph.state.Signal into signal_merger.Signal.

    The graph state uses a strict Signal contract, so once a Signal exists,
    direction/confidence/rationale are guaranteed to be valid.

    Missing signals fail safe to neutral.
    """

    if signal is None:
        return MergerSignal(
            direction="neutral",
            confidence=0.0,
            rationale=f"{label} signal missing",
        )

    return MergerSignal(
        direction=signal.direction,
        confidence=signal.confidence,
        rationale=signal.rationale,
    )


def _merged_signal_for_downstream(
    state: TradingState,
) -> MergedSignal:
    """
    Reconstruct the richer SignalMerger MergedSignal from TradingState.

    TradingState intentionally stores the common lightweight Signal plus
    explicit merge_agreement metadata.

    IMPORTANT:
    Do NOT hardcode agreement=True here.

    The improved SignalMerger distinguishes:

        bullish + bullish -> agreement=True
        bearish + bearish -> agreement=True

        bullish + neutral -> agreement=False
        neutral + bullish -> agreement=False

        bullish + bearish -> agreement=False
        bearish + bullish -> agreement=False

    The actual value produced by SignalMerger is therefore preserved in
    state.merge_agreement.
    """

    merged = state.merged_signal

    if merged is None:
        return MergedSignal(
            direction="neutral",
            combined_confidence=0.0,
            agreement=False,
            rationale="Missing merged signal.",
        )

    return MergedSignal(
        direction=merged.direction,
        combined_confidence=merged.confidence,
        agreement=state.merge_agreement is True,
        rationale=merged.rationale,
    )


# ---------------------------------------------------------------------------
# NewsAgent node
# ---------------------------------------------------------------------------


def news_agent_node(
    state: TradingState,
) -> dict:
    """
    Run NewsAgent for the current ticker.

    NewsAgent is responsible for interpreting market/news information and
    producing a structured directional Signal.
    """

    signal = run_news_agent(state.ticker)

    return {
        "news_signal": signal,
        "agent_logs": _log(
            "NewsAgent",
            (f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}"),
        ),
    }


# ---------------------------------------------------------------------------
# ChartAgent node
# ---------------------------------------------------------------------------


def chart_agent_node(
    state: TradingState,
) -> dict:
    """
    Run the deterministic ChartAgent.

    ChartAgent provides directional technical evidence.

    ATR is deliberately NOT placed in Signal. ATR is a risk/sizing input
    and is computed separately inside risk_agent_node().
    """

    signal = run_chart_agent(state.ticker)

    return {
        "chart_signal": signal,
        "agent_logs": _log(
            "ChartAgent",
            (f"{signal.direction} (conf={signal.confidence:.2f}): {signal.rationale}"),
        ),
    }


# ---------------------------------------------------------------------------
# SignalMerger node
# ---------------------------------------------------------------------------


def signal_merger_node(
    state: TradingState,
) -> dict:
    """
    Deterministically combine NewsAgent and ChartAgent signals.

    SignalMerger contains no LLM call.

    Important semantics:

        neutral = no opinion

    Therefore neutral does NOT automatically contradict a directional
    signal.

    Example:

        News neutral + Chart bullish
            -> bullish with reduced confidence

        News bullish + Chart bearish
            -> neutral because this is genuine disagreement
    """

    news = _to_merger_signal(
        state.news_signal,
        "News",
    )

    chart = _to_merger_signal(
        state.chart_signal,
        "Chart",
    )

    merged = merge_signals(
        news,
        chart,
    )

    merged_signal = Signal(
        direction=merged.direction,
        confidence=merged.combined_confidence,
        rationale=merged.rationale,
    )

    return {
        "merged_signal": merged_signal,
        # Preserve the actual merger metadata.
        "merge_agreement": merged.agreement,
        "agent_logs": _log(
            "SignalMerger",
            (
                f"news={news.direction} "
                f"(conf={news.confidence:.2f}); "
                f"chart={chart.direction} "
                f"(conf={chart.confidence:.2f}); "
                f"merged={merged.direction} "
                f"(conf={merged.combined_confidence:.2f}, "
                f"agreement={merged.agreement}): "
                f"{merged.rationale}"
            ),
        ),
    }


# ---------------------------------------------------------------------------
# RiskAgent node
# ---------------------------------------------------------------------------


def risk_agent_node(
    state: TradingState,
) -> dict:
    """
    Run deterministic portfolio/risk checks.

    Responsibilities here:

        1. Load validated risk mandate.
        2. Read current portfolio.
        3. Determine ticker sector.
        4. Fetch OHLCV history.
        5. Compute ATR.
        6. Determine entry price.
        7. Pass merged directional signal + sizing inputs to RiskAgent.
        8. Store the resulting risk decision and proposed position size.

    RiskAgent remains the gatekeeper between signal generation and execution.
    """

    config = _risk_config()

    portfolio = _ledger().snapshot()

    sector = get_sector(state.ticker)

    # ------------------------------------------------------------------
    # Fetch enough price history for ATR.
    # ------------------------------------------------------------------

    series = fetch_ohlcv(
        state.ticker,
        lookback_days=ATR_LOOKBACK_DAYS,
    )

    atr = compute_atr(series.bars) if not series.is_empty else None

    entry_price = (
        series.latest.close
        if (not series.is_empty and series.latest is not None)
        else None
    )

    # ------------------------------------------------------------------
    # Fail safely if we cannot calculate sizing inputs.
    # ------------------------------------------------------------------

    if atr is None or entry_price is None:
        return {
            "risk_decision": RiskDecision.REJECTED,
            "risk_notes": [
                ("Cannot size trade: insufficient OHLCV history for ATR/entry price.")
            ],
            "atr": atr,
            "entry_price": entry_price,
            "sector": sector,
            "proposed_shares": 0.0,
            "agent_logs": _log(
                "RiskAgent",
                ("rejected: insufficient price history for sizing"),
            ),
        }

    # ------------------------------------------------------------------
    # Retrieve the merged directional signal.
    # ------------------------------------------------------------------

    merged = _merged_signal_for_downstream(state)

    # ------------------------------------------------------------------
    # Run the deterministic risk engine.
    # ------------------------------------------------------------------

    decision = run_risk_agent(
        merged,
        state.ticker,
        sector,
        entry_price,
        atr,
        portfolio,
        config,
    )

    risk_status = "approved" if decision.approved else "rejected"

    risk_message = (
        f"merged={merged.direction} "
        f"conf={merged.combined_confidence:.3f} "
        f"agreement={merged.agreement}; "
        f"entry={entry_price:.2f}; "
        f"ATR={atr:.4f}; "
        f"{risk_status}: "
        f"{'; '.join(decision.risk_notes)}"
    )

    return {
        "risk_decision": (
            RiskDecision.APPROVED if decision.approved else RiskDecision.REJECTED
        ),
        "risk_notes": decision.risk_notes,
        "atr": atr,
        "entry_price": entry_price,
        "sector": sector,
        "proposed_shares": decision.proposed_shares,
        "agent_logs": _log(
            "RiskAgent",
            risk_message,
        ),
    }


# ---------------------------------------------------------------------------
# ExecutionAgent node
# ---------------------------------------------------------------------------


def execution_agent_node(
    state: TradingState,
) -> dict:
    """
    Execute a trade that has already passed RiskAgent.

    This node should only be reachable through route_after_risk() when
    TradingState.risk_decision == APPROVED.
    """

    broker = _broker()

    merged = _merged_signal_for_downstream(state)

    # The graph routing itself is the gate.
    #
    # RiskAgent has already approved this proposal before execution is
    # reachable. We therefore pass the approved decision downstream.
    risk_decision = RiskAgentDecision(
        approved=True,
        ticker=state.ticker,
        proposed_shares=state.proposed_shares,
        checks=[],
    )

    result = run_execution_agent(
        risk_decision,
        merged,
        broker,
    )

    # ------------------------------------------------------------------
    # Synchronize the local portfolio ledger with the fill.
    # ------------------------------------------------------------------

    if result.executed and result.order is not None:
        fill_price = result.order.filled_avg_price or state.entry_price or 0.0

        fill_sector = state.sector or get_sector(state.ticker)

        _ledger().record_fill(
            ticker=state.ticker,
            side=result.order.side,
            qty=result.order.qty,
            price=fill_price,
            sector=fill_sector,
        )

    return {
        "execution_notes": result.notes,
        "agent_logs": _log(
            "ExecutionAgent",
            result.notes,
        ),
    }


# ---------------------------------------------------------------------------
# Hold node
# ---------------------------------------------------------------------------


def hold_node(
    state: TradingState,
) -> dict:
    """
    Record a trade that did not pass the risk gate.
    """

    reason = "; ".join(state.risk_notes) if state.risk_notes else "held"

    return {
        "execution_notes": reason,
        "agent_logs": _log(
            "HoldNode",
            (f"holding {state.ticker}, reason={state.risk_notes}"),
        ),
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_risk(
    state: TradingState,
) -> str:
    """
    Route approved trades to execution.

    Everything else goes to HoldNode.

    This is the graph-level execution gate.
    """

    if state.risk_decision == RiskDecision.APPROVED:
        return "execution_agent"

    return "hold_node"


# ---------------------------------------------------------------------------
# Public infrastructure accessors
# ---------------------------------------------------------------------------
#
# These are useful for the future multi-ticker orchestrator and portfolio
# coordinator so they can access the SAME risk config, ledger and broker
# instances used by the individual graph nodes.
# ---------------------------------------------------------------------------


def get_risk_config():
    """
    Return the shared validated risk configuration.
    """
    return _risk_config()


def get_ledger() -> PortfolioLedger:
    """
    Return the shared portfolio ledger.
    """
    return _ledger()


def get_broker() -> Broker:
    """
    Return the shared broker.
    """
    return _broker()
