from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from pathlib import Path

import streamlit as st

# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEDGER_PATH = Path(
    os.getenv(
        "LEDGER_PATH",
        "portfolio/ledger.json",
    )
)

BACKTEST_RESULTS_DIR = Path(
    os.getenv(
        "BACKTEST_RESULTS_DIR",
        "backtest/results",
    )
)

AUDIT_DB_PATH = Path(
    os.getenv(
        "TRADING_FIRM_DB_PATH",
        "storage/trading_firm.db",
    )
)

TRADING_MODE = os.getenv(
    "TRADING_MODE",
    "backtest",
).lower()


# ============================================================
# LOGFIRE
# ============================================================

LOGFIRE_OK = False

try:
    import logfire

    token = os.getenv("LOGFIRE_TOKEN")

    if token:
        logfire.configure(token=token)
        LOGFIRE_OK = True

except Exception:  # noqa: BLE001
    LOGFIRE_OK = False


def trace_context():
    """Return Logfire span context when telemetry is configured."""
    if LOGFIRE_OK:
        return logfire.span("Trading Command Center operation")

    return nullcontext()


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="AI Multi-Agent Trading Firm",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================


def display_html(markup: str) -> None:
    native = getattr(st, "html", None)

    if callable(native):
        native(markup)
    else:
        st.markdown(
            markup,
            unsafe_allow_html=True,
        )


def esc(value) -> str:
    return html.escape(
        str(value),
        quote=True,
    )


def fmt_money(
    value,
    default="—",
) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return default


def fmt_pct(
    value,
    default="—",
) -> str:
    if value is None:
        return default

    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return default


def fmt_num(
    value,
    decimals=3,
    default="—",
) -> str:
    if value is None:
        return default

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def fmt_shares(
    value,
    default="—",
) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return default


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_value(
    value,
    default=None,
):
    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return value

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def first_value(
    data: dict,
    *keys,
    default=None,
):
    for key in keys:
        value = data.get(key)

        if value is not None:
            return value

    return default


def outcome_label(
    outcome: str | None,
) -> str:
    return {
        "executed": "EXECUTED",
        "held_local_reject": "LOCAL RISK BLOCK",
        "held_book_reject": "BOOK RISK BLOCK",
        "held_execution_reject": "BROKER EXECUTION FAILED",
        "neutral_signal": "NEUTRAL SIGNAL",
        "ticker_capacity": "TICKER CAPACITY BLOCK",
        "sector_capacity": "SECTOR CAPACITY BLOCK",
        "book_risk": "BOOK RISK BLOCK",
    }.get(
        outcome or "",
        str(outcome or "UNKNOWN").upper(),
    )


def outcome_class(
    outcome: str | None,
) -> str:
    if outcome == "executed":
        return "success"

    if outcome == "held_execution_reject":
        return "danger"

    if outcome in {
        "held_local_reject",
        "held_book_reject",
        "ticker_capacity",
        "sector_capacity",
        "book_risk",
    }:
        return "warning"

    return "neutral"


def direction_class(
    direction: str | None,
) -> str:
    if direction == "bullish":
        return "bull"

    if direction == "bearish":
        return "bear"

    return "neutral"


def serialize_object(value):
    """
    Convert dataclasses and nested objects into something Streamlit
    can display safely.

    This is presentation-only. It does not modify the underlying
    TradeTrace or runtime objects.
    """
    if value is None:
        return None

    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)

    if isinstance(value, dict):
        return {str(key): serialize_object(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [serialize_object(item) for item in value]

    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:  # noqa: BLE001
            return value

    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): serialize_object(item) for key, item in vars(value).items()
            }
        except Exception:  # noqa: BLE001
            return str(value)

    return value


def result_to_dict(result):
    """Normalize TickResult for display and session-state inspection."""
    return serialize_object(result)


# ============================================================
# OBSERVABILITY HELPERS
# ============================================================


def safe_int(value, default=0):
    """Convert a value to int without raising on malformed telemetry."""
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return int(value)

        return int(float(value))

    except (TypeError, ValueError):
        return default


def bool_or_none(value):
    """Normalize common telemetry boolean representations."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
            "approved",
            "success",
            "filled",
            "executed",
            "mutated",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
            "rejected",
            "failed",
            "blocked",
            "held",
            "unchanged",
        }:
            return False

    return None


def execution_success_from_summary(summary):
    """Return the canonical execution-success state from UI telemetry."""
    value = summary.get("execution_success")

    normalized = bool_or_none(value)

    if normalized is not None:
        return normalized

    return None


def coordinator_approval_from_summary(summary):
    """Return explicit coordinator approval when available."""
    value = summary.get("coordinator_approved")

    normalized = bool_or_none(value)

    if normalized is not None:
        return normalized

    outcome = str(summary.get("outcome") or "").strip().lower()

    if outcome == "executed":
        return True

    if outcome in {
        "held_local_reject",
        "held_book_reject",
    }:
        return False

    return None


def accounting_mutation_from_summary(summary):
    """
    Determine whether portfolio accounting actually mutated.

    Explicit accounting telemetry takes precedence. Otherwise a
    successful execution is treated as the compatibility fallback,
    because successful fills mutate the portfolio ledger.
    """
    for key in (
        "accounting_mutated",
        "portfolio_mutated",
        "ledger_mutated",
        "accounting_mutation",
    ):
        if key in summary:
            normalized = bool_or_none(summary.get(key))

            if normalized is not None:
                return normalized

    execution_success = execution_success_from_summary(summary)

    if execution_success is True:
        return True

    if execution_success is False:
        return False

    return None


def build_run_observability(run_data):
    """
    Build presentation-only observability metrics for one backtest run.

    This function does not modify strategy, risk, execution, accounting,
    or persisted audit state.
    """
    if not isinstance(run_data, dict):
        run_data = {}

    tick_log = run_data.get("tick_log") or []

    if not isinstance(tick_log, list):
        tick_log = []

    diagnostics = run_data.get("diagnostics") or {}

    if not isinstance(diagnostics, dict):
        diagnostics = {}

    traces = run_data.get("trade_traces") or []

    if not isinstance(traces, list):
        traces = []

    trades = run_data.get("trades") or []

    if not isinstance(trades, list):
        trades = []

    closed_trades_raw = run_data.get(
        "closed_trades",
        [],
    )

    if isinstance(
        closed_trades_raw,
        (list, tuple, dict),
    ):
        closed_trades = len(closed_trades_raw)
    else:
        closed_trades = safe_int(
            closed_trades_raw,
        )

    # --------------------------------------------------------
    # Trace count
    # --------------------------------------------------------

    trace_count = len(traces)

    if trace_count == 0:
        persisted_trace_count = diagnostics.get("trade_trace_count")

        if persisted_trace_count is None:
            persisted_trace_count = run_data.get("trade_trace_count")

        trace_count = safe_int(
            persisted_trace_count,
            default=0,
        )

    ticks = len(tick_log)

    trace_coverage = trace_count / ticks if ticks > 0 else None

    if trace_coverage is not None:
        trace_coverage = min(
            max(trace_coverage, 0.0),
            1.0,
        )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_approvals = safe_int(
        diagnostics.get(
            "risk_approvals",
            0,
        )
    )

    risk_rejections = safe_int(
        diagnostics.get(
            "risk_rejections",
            0,
        )
    )

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    execution_attempts = safe_int(
        diagnostics.get(
            "execution_attempts",
            0,
        )
    )

    execution_failures = safe_int(
        diagnostics.get(
            "execution_failures",
            0,
        )
    )

    execution_successes = safe_int(
        diagnostics.get(
            "execution_successes",
            max(
                execution_attempts - execution_failures,
                0,
            ),
        )
    )

    # --------------------------------------------------------
    # Coordinator
    # --------------------------------------------------------

    coordinator_approvals = 0

    coordinator_rejections = 0

    coordinator_observed = False

    for entry in tick_log:
        if not isinstance(entry, dict):
            continue

        summary = paper_trade_result_summary(entry)

        approval = coordinator_approval_from_summary(summary)

        if approval is True:
            coordinator_approvals += 1
            coordinator_observed = True

        elif approval is False:
            coordinator_rejections += 1
            coordinator_observed = True

    # Prefer persisted diagnostics when explicitly available.
    diagnostic_coordinator_approvals = diagnostics.get("coordinator_approvals")

    if diagnostic_coordinator_approvals is not None:
        coordinator_approvals = safe_int(diagnostic_coordinator_approvals)

    diagnostic_coordinator_rejections = diagnostics.get("coordinator_rejections")

    if diagnostic_coordinator_rejections is not None:
        coordinator_rejections = safe_int(diagnostic_coordinator_rejections)

    # --------------------------------------------------------
    # Normalized trades
    # --------------------------------------------------------

    normalized_trades = len(trades)

    diagnostic_trade_count = diagnostics.get("normalized_trades")

    if diagnostic_trade_count is not None:
        normalized_trades = safe_int(diagnostic_trade_count)

    # --------------------------------------------------------
    # Return canonical presentation metrics
    # --------------------------------------------------------

    return {
        "ticks": ticks,
        "traces": trace_count,
        "trace_coverage": trace_coverage,
        "risk_approvals": risk_approvals,
        "risk_rejections": risk_rejections,
        "coordinator_approvals": coordinator_approvals,
        "coordinator_rejections": coordinator_rejections,
        "execution_attempts": execution_attempts,
        "execution_successes": execution_successes,
        "execution_failures": execution_failures,
        "normalized_trades": normalized_trades,
        "closed_trades": closed_trades,
        "coordinator_observed": coordinator_observed,
    }


# ============================================================
# AGENTIC PAPER TRADE
# ============================================================


def run_agentic_paper_trade(
    ticker: str,
    simulated_date: date,
):
    """
    Run one ticker through the real top-level trading orchestrator.

    The UI intentionally calls run_tick() instead of directly invoking
    ExecutionAgent or SimBroker. This keeps the recruiter/demo path
    identical to the production orchestration path.
    """
    from orchestrator.tick_runner import run_tick

    run_id = f"ui_{ticker}_{simulated_date.isoformat()}_{uuid.uuid4().hex[:8]}"

    simulated_datetime = datetime.combine(
        simulated_date,
        datetime_time.min,
    )

    with trace_context():
        results = asyncio.run(
            run_tick(
                [ticker],
                run_id=run_id,
                simulated_date=simulated_datetime,
            )
        )

    result = results.get(ticker)

    if result is None:
        raise RuntimeError(f"run_tick() completed but returned no result for {ticker}.")

    return run_id, result


def run_historical_lifecycle_demo(
    ticker: str,
    start_date: date,
    end_date: date,
):
    """
    Run a short historical replay through the canonical backtest pipeline.

    This is a recruiter/demo presentation layer over run_backtest().
    It does not implement a second trading path.
    """
    from backtest.runner import run_backtest

    run_id = (
        f"lifecycle_{ticker}_{start_date.isoformat()}_"
        f"{end_date.isoformat()}_{uuid.uuid4().hex[:8]}"
    )

    with trace_context():
        result = run_backtest(
            tickers=[ticker],
            start_date=str(start_date),
            end_date=str(end_date),
            tick_delay_seconds=0,
            run_id=run_id,
        )

    return run_id, result


def paper_trade_result_summary(result):
    """Extract presentation fields from TickResult."""
    return {
        "outcome": getattr(result, "outcome", None),
        "notes": getattr(result, "notes", ""),
        "shares": getattr(result, "shares", None),
        "price": getattr(result, "price", None),
        "signal_direction": getattr(result, "signal_direction", None),
        "signal_confidence": getattr(result, "signal_confidence", None),
        "merge_agreement": getattr(result, "merge_agreement", None),
        "atr": getattr(result, "atr", None),
        "entry_price": getattr(result, "entry_price", None),
        "sector": getattr(result, "sector", None),
        "proposed_shares": getattr(result, "proposed_shares", None),
        "risk_decision": getattr(result, "risk_decision", None),
        "risk_notes": getattr(result, "risk_notes", None) or [],
        "execution_attempted": getattr(result, "execution_attempted", False),
        "execution_success": getattr(result, "execution_success", None),
        "execution_side": getattr(result, "execution_side", None),
        "average_cost": getattr(result, "average_cost", None),
        "realized_pnl": getattr(result, "realized_pnl", None),
        "remaining_shares": getattr(result, "remaining_shares", None),
        "position_closed": getattr(result, "position_closed", False),
        "news_availability": getattr(result, "news_availability", None),
        "news_article_count": getattr(result, "news_article_count", 0),
        "news_source": getattr(result, "news_source", None),
        "news_direction": getattr(result, "news_direction", None),
        "news_confidence": getattr(result, "news_confidence", None),
        "news_as_of": getattr(result, "news_as_of", None),
        "chart_direction": getattr(result, "chart_direction", None),
        "chart_confidence": getattr(result, "chart_confidence", None),
        "coordinator_approved": getattr(
            result,
            "coordinator_approved",
            getattr(result, "outcome", None) == "executed",
        ),
    }


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>

:root {
    --bg: #0b0e13;
    --panel: #12161d;
    --panel2: #171c24;
    --panel3: #1c222c;
    --border: #29313d;

    --text: #edf1f7;
    --muted: #8b95a7;
    --muted2: #626d7e;

    --accent: #5bc0be;
    --accent-soft: rgba(91,192,190,.13);

    --green: #6ee7b7;
    --green-soft: rgba(110,231,183,.12);

    --red: #f0707a;
    --red-soft: rgba(240,112,122,.12);

    --yellow: #e0a95c;
    --yellow-soft: rgba(224,169,92,.12);

    --blue: #82aaff;
    --blue-soft: rgba(130,170,255,.12);
}

html,
body,
[class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.hero {
    padding: 8px 0 18px;
}

.hero-kicker {
    color: var(--accent);
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 5px 0 7px;
    font-size: 31px;
    letter-spacing: -.03em;
}

.hero p {
    max-width: 900px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.65;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 8px;
    padding: 6px 11px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--panel);
    color: var(--muted);
    font-family: monospace;
    font-size: 10px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
}

.arch-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 7px;
    margin: 15px 0 20px;
}

.arch-node {
    position: relative;
    padding: 12px 8px;
    min-height: 74px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--panel);
    text-align: center;
}

.arch-node::after {
    content: "→";
    position: absolute;
    right: -10px;
    top: 25px;
    color: var(--muted2);
    z-index: 2;
}

.arch-node:last-child::after {
    display: none;
}

.arch-node .number {
    color: var(--accent);
    font-family: monospace;
    font-size: 10px;
}

.arch-node strong {
    display: block;
    margin-top: 4px;
    font-size: 11px;
}

.arch-node small {
    display: block;
    margin-top: 4px;
    color: var(--muted2);
    font-size: 9px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin: 10px 0 20px;
}

.kpi-grid.five {
    grid-template-columns: repeat(5, 1fr);
}

.kpi-grid.four {
    grid-template-columns: repeat(4, 1fr);
}

.kpi {
    padding: 13px 14px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

.kpi .label {
    color: var(--muted2);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}

.kpi .value {
    margin-top: 5px;
    color: var(--text);
    font-family: monospace;
    font-size: 17px;
    font-weight: 700;
}

.kpi .value.green {
    color: var(--green);
}

.kpi .value.red {
    color: var(--red);
}

.kpi .value.yellow {
    color: var(--yellow);
}

.agent-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 9px;
    margin: 10px 0 15px;
}

.agent-card {
    padding: 13px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

.agent-card .agent-name {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.agent-card .agent-value {
    margin-top: 7px;
    font-family: monospace;
    font-size: 18px;
    font-weight: 800;
}

.agent-card .agent-detail {
    margin-top: 5px;
    color: var(--muted2);
    font-size: 10px;
    line-height: 1.5;
}

.bull {
    color: var(--green);
}

.bear {
    color: var(--red);
}

.neutral {
    color: var(--muted);
}

.decision-card {
    padding: 16px;
    margin: 8px 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}

.decision-card.success {
    border-left: 4px solid var(--green);
}

.decision-card.warning {
    border-left: 4px solid var(--yellow);
}

.decision-card.danger {
    border-left: 4px solid var(--red);
}

.decision-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.decision-title {
    font-family: monospace;
    font-size: 16px;
    font-weight: 800;
}

.decision-date {
    color: var(--muted2);
    font-family: monospace;
    font-size: 10px;
}

.badge {
    display: inline-flex;
    padding: 4px 9px;
    border-radius: 999px;
    font-family: monospace;
    font-size: 9px;
    font-weight: 800;
}

.badge.success {
    color: var(--green);
    background: var(--green-soft);
}

.badge.warning {
    color: var(--yellow);
    background: var(--yellow-soft);
}

.badge.danger {
    color: var(--red);
    background: var(--red-soft);
}

.badge.neutral {
    color: var(--muted);
    background: var(--panel2);
}

.pipeline {
    display: flex;
    align-items: stretch;
    gap: 6px;
    margin: 13px 0;
}

.pipeline-step {
    flex: 1;
    padding: 9px 7px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel2);
    text-align: center;
}

.pipeline-step.active {
    border-color: var(--accent);
    background: var(--accent-soft);
}

.pipeline-step.state-pass {
    border-color: var(--green);
    background: var(--green-soft);
}

.pipeline-step.state-block {
    border-color: var(--red);
    background: var(--red-soft);
}

.pipeline-step.state-skip {
    border-color: var(--border);
    background: var(--panel2);
    opacity: .55;
}

.pipeline-step .index {
    color: var(--accent);
    font-family: monospace;
    font-size: 9px;
}

.pipeline-step strong {
    display: block;
    margin-top: 4px;
    font-size: 10px;
}

.pipeline-step small {
    display: block;
    margin-top: 3px;
    color: var(--muted2);
    font-size: 8px;
}

.reason-box {
    padding: 11px 13px;
    margin-top: 8px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel2);
    color: var(--muted);
    font-size: 11px;
    line-height: 1.6;
}

.reason-box strong {
    color: var(--text);
}

.decision-headline {
    padding: 18px 20px;
    margin: 14px 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
}

.decision-headline.success {
    border-left: 4px solid var(--green);
}

.decision-headline.warning {
    border-left: 4px solid var(--yellow);
}

.decision-headline.danger {
    border-left: 4px solid var(--red);
}

.decision-headline-label {
    color: var(--muted2);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.decision-headline-title {
    margin-top: 6px;
    font-family: monospace;
    font-size: 22px;
    font-weight: 800;
}

.decision-headline-text {
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}

.lineage-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px 18px;
    margin: 4px 0 16px;
    font-family: monospace;
    font-size: 11px;
}

.lineage-item .symbol {
    font-weight: 800;
    margin-right: 3px;
}

.lineage-item.success {
    color: var(--green);
}

.lineage-item.danger {
    color: var(--red);
}

.lineage-item.neutral {
    color: var(--muted2);
}

.risk-matrix {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 10px 0;
}

.risk-check {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel2);
    font-family: monospace;
    font-size: 11px;
}

.risk-check .symbol {
    font-weight: 800;
}

.risk-check.success {
    border-color: var(--green);
    color: var(--green);
}

.risk-check.danger {
    border-color: var(--red);
    color: var(--red);
}

.risk-check.neutral {
    color: var(--muted);
}

.temporal-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin: 10px 0;
}

.temporal-item {
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel2);
}

.temporal-item .label {
    color: var(--muted2);
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.temporal-item .value {
    margin-top: 4px;
    font-family: monospace;
    font-size: 11px;
    color: var(--text);
}

.demo-banner {
    padding: 14px 16px;
    margin: 10px 0 16px;
    border: 1px solid var(--accent);
    border-radius: 10px;
    background: var(--accent-soft);
}

.demo-banner-title {
    color: var(--accent);
    font-family: monospace;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .05em;
}

.demo-banner-text {
    margin-top: 5px;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.55;
}

.paper-stage {
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

.paper-stage .stage-number {
    color: var(--accent);
    font-family: monospace;
    font-size: 10px;
}

.paper-stage .stage-title {
    margin-top: 4px;
    font-size: 12px;
    font-weight: 750;
}

.paper-stage .stage-description {
    margin-top: 4px;
    color: var(--muted2);
    font-size: 10px;
}

.accounting-highlight {
    padding: 15px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

.accounting-highlight .title {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.accounting-highlight .value {
    margin-top: 6px;
    font-family: monospace;
    font-size: 19px;
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background: #10141a;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] button {
    border-color: var(--border) !important;
}

.section-title {
    margin-top: 18px;
    margin-bottom: 5px;
    color: var(--text);
    font-size: 16px;
    font-weight: 750;
}

.section-subtitle {
    margin-bottom: 12px;
    color: var(--muted);
    font-size: 11px;
}

.audit-ok {
    color: var(--green);
}

.audit-warning {
    color: var(--yellow);
}

.audit-danger {
    color: var(--red);
}

@media (max-width: 1100px) {
    .arch-grid {
        grid-template-columns: repeat(4, 1fr);
    }

    .arch-node::after {
        display: none;
    }

    .kpi-grid {
        grid-template-columns: repeat(3, 1fr);
    }

    .kpi-grid.five {
        grid-template-columns: repeat(3, 1fr);
    }

    .kpi-grid.four {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 900px) {
    .pipeline {
        overflow-x: auto;
    }

    .pipeline-step {
        min-width: 125px;
    }

    .risk-matrix {
        grid-template-columns: repeat(2, 1fr);
    }

    .temporal-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# JSON DATA LOADERS
# ============================================================


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def load_json_file(
    path_str: str,
    mtime: float,
):
    path = Path(path_str)

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def list_backtest_runs(
    dir_str: str,
):
    directory = Path(dir_str)

    if not directory.exists():
        return []

    files = sorted(
        directory.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return [f.name for f in files]


def load_ledger():
    if not LEDGER_PATH.exists():
        return None

    return load_json_file(
        str(LEDGER_PATH),
        LEDGER_PATH.stat().st_mtime,
    )


def load_run(
    filename: str,
):
    path = BACKTEST_RESULTS_DIR / filename

    if not path.exists():
        return None

    return load_json_file(
        str(path),
        path.stat().st_mtime,
    )


# ============================================================
# SQLITE AUDIT
# ============================================================


def _audit_connection():
    """
    Open the audit database read-only.

    Streamlit must never mutate the persistent audit database.
    """
    if not AUDIT_DB_PATH.exists():
        return None

    try:
        connection = sqlite3.connect(
            f"file:{AUDIT_DB_PATH.resolve()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error:
        return None


def _sqlite_rows(
    query: str,
    params=(),
):
    connection = _audit_connection()

    if connection is None:
        return []

    try:
        rows = connection.execute(
            query,
            params,
        ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error:
        return []

    finally:
        connection.close()


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def audit_table_exists(
    table_name: str,
) -> bool:
    rows = _sqlite_rows(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table_name,),
    )

    return bool(rows)


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def audit_table_columns(
    table_name: str,
):
    if not audit_table_exists(table_name):
        return []

    rows = _sqlite_rows(f"PRAGMA table_info({table_name})")

    return [str(row["name"]) for row in rows if row.get("name")]


def _safe_select_columns(
    table_name: str,
    preferred_columns: list[str],
):
    """
    Select only columns that actually exist.

    This prevents the UI from becoming coupled to one exact
    storage schema revision.
    """
    available = set(audit_table_columns(table_name))

    selected = [column for column in preferred_columns if column in available]

    if not selected:
        return None

    return ", ".join(f'"{column}"' for column in selected)


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def list_audit_runs(
    db_path_str: str,
):
    if not Path(db_path_str).exists():
        return []

    columns = _safe_select_columns(
        "runs",
        [
            "run_id",
            "status",
            "started_at",
            "finished_at",
            "start_date",
            "end_date",
            "tickers",
            "starting_equity",
            "final_equity",
            "realized_pnl",
            "benchmark_return",
            "trade_trace_count",
            "metadata",
        ],
    )

    if not columns:
        return []

    rows = _sqlite_rows(
        f"""
        SELECT {columns}
        FROM runs
        ORDER BY
            COALESCE(
                finished_at,
                started_at
            ) DESC
        """
    )

    return rows


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def load_audit_run(
    db_path_str: str,
    run_id: str,
):
    if not Path(db_path_str).exists():
        return None

    run_columns = _safe_select_columns(
        "runs",
        [
            "run_id",
            "status",
            "started_at",
            "finished_at",
            "start_date",
            "end_date",
            "tickers",
            "starting_equity",
            "final_equity",
            "realized_pnl",
            "benchmark_return",
            "trade_trace_count",
            "metadata",
        ],
    )

    if not run_columns:
        return None

    runs = _sqlite_rows(
        f"""
        SELECT {run_columns}
        FROM runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    )

    if not runs:
        return None

    traces = []

    if audit_table_exists("trade_traces"):
        traces = _sqlite_rows(
            """
            SELECT *
            FROM trade_traces
            WHERE run_id = ?
            ORDER BY
                COALESCE(
                    simulated_date,
                    created_at
                ),
                ticker,
                id
            """,
            (run_id,),
        )

    return {
        "run": runs[0],
        "traces": traces,
    }


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def audit_table_counts(
    db_path_str: str,
):
    counts = {}

    if not Path(db_path_str).exists():
        return counts

    for table in [
        "runs",
        "trade_traces",
        "trades",
        "agent_logs",
        "portfolio_snapshots",
    ]:
        if not audit_table_exists(table):
            continue

        rows = _sqlite_rows(
            f"""
            SELECT COUNT(*) AS count
            FROM {table}
            """
        )

        counts[table] = rows[0]["count"] if rows else 0

    return counts


# ============================================================
# AUDIT NORMALIZATION
# ============================================================


def normalize_audit_trace(
    trace: dict,
) -> dict:
    """
    Normalize persistent TradeTrace fields for the UI.

    Storage may evolve while the semantic TradeTrace contract
    remains stable.
    """

    metadata = parse_json_value(
        trace.get("metadata"),
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    risk_checks = parse_json_value(
        first_value(
            trace,
            "risk_checks",
            "checks",
            default=[],
        ),
        [],
    )

    if not isinstance(risk_checks, list):
        risk_checks = []

    outcome = first_value(
        trace,
        "outcome",
        default=None,
    )

    return {
        **trace,
        "metadata": metadata,
        "risk_checks": risk_checks,
        "ticker": first_value(
            trace,
            "ticker",
            default="—",
        ),
        "tick_id": first_value(
            trace,
            "tick_id",
            default="—",
        ),
        "simulated_date": first_value(
            trace,
            "simulated_date",
            "date",
            default="—",
        ),
        "outcome": outcome or "unknown",
        "signal_direction": first_value(
            trace,
            "signal_direction",
            "direction",
            default="neutral",
        ),
        "signal_confidence": first_value(
            trace,
            "signal_confidence",
            "confidence",
        ),
        "merge_agreement": first_value(
            trace,
            "merge_agreement",
            "agreement",
        ),
        "risk_approved": first_value(
            trace,
            "risk_approved",
            "risk_decision",
        ),
        "risk_decision": first_value(
            trace,
            "risk_decision",
            "risk_approved",
        ),
        "raw_shares": first_value(
            trace,
            "raw_shares",
        ),
        "proposed_shares": first_value(
            trace,
            "proposed_shares",
        ),
        "final_shares": first_value(
            trace,
            "final_shares",
            "coordinator_shares",
        ),
        "coordinator_approved": first_value(
            trace,
            "coordinator_approved",
        ),
        "execution_side": first_value(
            trace,
            "execution_side",
        ),
        "execution_attempted": first_value(
            trace,
            "execution_attempted",
            default=False,
        ),
        "execution_success": first_value(
            trace,
            "execution_success",
        ),
        "execution_quantity": first_value(
            trace,
            "execution_quantity",
            "quantity",
        ),
        "execution_price": first_value(
            trace,
            "execution_price",
            "price",
        ),
        "order_id": first_value(
            trace,
            "order_id",
        ),
        "average_cost": first_value(
            trace,
            "average_cost",
        ),
        "realized_pnl": first_value(
            trace,
            "realized_pnl",
        ),
        "remaining_shares": first_value(
            trace,
            "remaining_shares",
        ),
        "position_closed": first_value(
            trace,
            "position_closed",
        ),
    }


def parse_run_metadata(
    run: dict,
):
    metadata = parse_json_value(
        run.get("metadata"),
        {},
    )

    return metadata if isinstance(metadata, dict) else {}


# ============================================================
# LEDGER
# ============================================================


def normalize_ledger(
    raw,
):
    if not raw:
        return {
            "equity": None,
            "starting_equity": None,
            "cash": None,
            "positions": {},
        }

    positions = raw.get("positions") or {}

    equity = raw.get("equity")

    if equity is None:
        cash = safe_float(raw.get("cash"))

        market_value = 0.0

        for position in positions.values():
            if isinstance(position, dict):
                market_value += safe_float(position.get("market_value")) or 0.0

        if cash is not None:
            equity = cash + market_value

    return {
        "equity": equity,
        "starting_equity": (
            raw.get("starting_equity") or raw.get("session_starting_equity")
        ),
        "cash": raw.get("cash"),
        "positions": positions,
    }


# ============================================================
# NEWS QUALITY
# ============================================================

NEWS_LABELS = {
    "available": "Available",
    "unavailable": "Unavailable",
    "error": "Error",
}


def normalize_news_status(value):
    if value is None:
        return None

    if isinstance(value, dict):
        value = value.get("availability")

    if hasattr(value, "value"):
        value = value.value

    value = str(value).lower().strip()

    return value if value in NEWS_LABELS else None


def extract_news_quality(run_data):
    raw = run_data.get("news_availability")

    result = {
        "by_ticker": {},
        "available": 0,
        "unavailable": 0,
        "error": 0,
        "total": 0,
        "coverage": None,
    }

    if not raw:
        return result

    by_ticker = {}

    if isinstance(raw, dict):
        candidate = raw.get("by_ticker")

        if candidate is None:
            candidate = raw.get("tickers")

        if isinstance(candidate, dict):
            for ticker, value in candidate.items():
                status = normalize_news_status(value)

                if status:
                    by_ticker[str(ticker).upper()] = status

        else:
            reserved = {
                "summary",
                "by_ticker",
                "tickers",
                "available",
                "unavailable",
                "error",
                "total",
                "coverage",
                "coverage_pct",
            }

            for ticker, value in raw.items():
                if ticker in reserved:
                    continue

                status = normalize_news_status(value)

                if status:
                    by_ticker[str(ticker).upper()] = status

    result["by_ticker"] = by_ticker

    result["available"] = sum(value == "available" for value in by_ticker.values())

    result["unavailable"] = sum(value == "unavailable" for value in by_ticker.values())

    result["error"] = sum(value == "error" for value in by_ticker.values())

    result["total"] = len(by_ticker)

    if isinstance(raw, dict):
        summary = raw.get("summary")

        source = summary if isinstance(summary, dict) else raw

        for key in [
            "available",
            "unavailable",
            "error",
            "total",
        ]:
            if key in source:
                try:
                    result[key] = int(source[key] or 0)
                except (TypeError, ValueError):
                    pass

        coverage = source.get("coverage")

        if coverage is None:
            coverage = source.get("coverage_pct")

        if coverage is not None:
            try:
                coverage = float(coverage)

                if coverage > 1:
                    coverage /= 100

                result["coverage"] = coverage

            except (TypeError, ValueError):
                pass

    if result["coverage"] is None and result["total"] > 0:
        result["coverage"] = result["available"] / result["total"]

    return result


# ============================================================
# BACKTEST TELEMETRY (presentation-only)
# ============================================================

DECISION_REASONS = [
    ("executed", "Executed", "green"),
    ("neutral_signal", "Neutral signal", ""),
    ("ticker_capacity", "Ticker capacity", "yellow"),
    ("sector_capacity", "Sector capacity", "yellow"),
    ("book_risk", "Book risk", "yellow"),
]

DECISION_REASON_LABELS = {key: label for key, label, _ in DECISION_REASONS}


def extract_decision_reasons(run_data: dict) -> dict:
    """
    Read decision-reason counts from the backtest JSON.

    Looks for a pre-aggregated dict first (diagnostics or top level),
    then falls back to counting a per-tick reason field. Purely
    diagnostic: it never feeds back into strategy behavior.
    """
    diagnostics = run_data.get("diagnostics") or {}

    for key in ("decision_reasons", "decision_reason_counts"):
        for source in (diagnostics, run_data):
            value = source.get(key)

            if isinstance(value, dict) and value:
                return {str(k): int(v or 0) for k, v in value.items()}

    counts: dict = {}

    for entry in run_data.get("tick_log") or []:
        reason = first_value(
            entry,
            "decision_reason",
            "reason_category",
            "decision_category",
        )

        if reason:
            counts[str(reason)] = counts.get(str(reason), 0) + 1

    return counts


def render_decision_reasons(reasons: dict, total_ticks: int) -> None:
    if not reasons:
        st.info(
            "This run has no decision-reason telemetry. "
            "Re-run the backtest with the latest runner."
        )
        return

    known = [key for key, _, _ in DECISION_REASONS]

    ordered = [k for k in known if k in reasons] + [
        k for k in reasons if k not in known
    ]

    reason_total = sum(reasons.values())
    denominator = reason_total or 1

    color_by_key = {key: color for key, _, color in DECISION_REASONS}

    display_html(
        '<div class="kpi-grid five">'
        + "".join(
            f"""
            <div class="kpi">
                <div class="label">{
                esc(DECISION_REASON_LABELS.get(k, k.replace("_", " ").title()))
            }</div>
                <div class="value {color_by_key.get(k, "")}">{esc(reasons[k])}</div>
            </div>
            """
            for k in ordered
        )
        + "</div>"
    )

    st.dataframe(
        [
            {
                "Reason": DECISION_REASON_LABELS.get(k, k.replace("_", " ").title()),
                "Count": reasons[k],
                "Share": reasons[k] / denominator,
            }
            for k in ordered
        ]
        + [
            {
                "Reason": "Total",
                "Count": reason_total,
                "Share": 1.0,
            }
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Share": st.column_config.ProgressColumn(
                "Share",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
        },
    )

    if total_ticks and reason_total != total_ticks:
        st.warning(
            f"Reason counts sum to {reason_total}, but the run has "
            f"{total_ticks} ticks. Some ticks are missing a decision reason."
        )


def trace_coverage(run_data: dict, run_stem: str) -> dict:
    """Ticks in the result JSON vs TradeTraces persisted in SQLite."""
    tick_log = run_data.get("tick_log") or []

    ticks = len(tick_log)

    tickers = len({x.get("ticker") for x in tick_log if x.get("ticker")})

    traces = None

    if AUDIT_DB_PATH.exists() and audit_table_exists("trade_traces"):
        audit = load_audit_run(str(AUDIT_DB_PATH), run_stem)

        if audit:
            traces = len(audit["traces"])

    coverage = (traces / ticks) if (traces is not None and ticks) else None

    return {
        "ticks": ticks,
        "traces": traces,
        "coverage": coverage,
        "tickers": tickers,
    }


# ============================================================
# PIPELINE
# ============================================================

PIPELINE = [
    ("1", "NewsAgent", "Market/news"),
    ("2", "ChartAgent", "Technicals"),
    ("3", "SignalMerger", "Deterministic"),
    ("4", "RiskAgent", "Position sizing"),
    ("5", "Portfolio", "Book-level risk"),
    ("6", "Execution", "Broker / Hold"),
]


def render_architecture():
    nodes = []

    for index, name, description in PIPELINE:
        nodes.append(
            f"""
            <div class="arch-node">
                <div class="number">{esc(index)}</div>
                <strong>{esc(name)}</strong>
                <small>{esc(description)}</small>
            </div>
            """
        )

    display_html(
        f"""
        <div class="arch-grid">
            {"".join(nodes)}
        </div>
        """
    )


def render_pipeline():
    nodes = []

    for index, name, description in PIPELINE:
        nodes.append(
            f"""
            <div class="pipeline-step active">
                <div class="index">{esc(index)}</div>
                <strong>{esc(name)}</strong>
                <small>{esc(description)}</small>
            </div>
            """
        )

    display_html(
        f"""
        <div class="pipeline">
            {"".join(nodes)}
        </div>
        """
    )


def render_paper_trade_pipeline(
    result,
):
    """
    Render the actual agentic execution path.

    The result object is used only for presentation. The UI does not
    recompute strategy decisions.
    """

    outcome = getattr(
        result,
        "outcome",
        None,
    )

    execution_success = getattr(
        result,
        "execution_success",
        None,
    )

    coordinator_approved = outcome == "executed"

    # Each stage carries a state of "pass" (completed cleanly),
    # "block" (this stage is where the pipeline was stopped), or
    # "skip" (never reached / not attempted because an earlier
    # stage already blocked). This mirrors the semantics already
    # present on TickResult - it does not add new decision logic.
    stages = [
        (
            "1",
            "NewsAgent",
            "Evidence collected",
            "pass",
        ),
        (
            "2",
            "ChartAgent",
            "Technical evidence",
            "pass",
        ),
        (
            "3",
            "SignalMerger",
            "Deterministic merge",
            "pass",
        ),
        (
            "4",
            "RiskAgent",
            "Local risk gate",
            "pass",
        ),
        (
            "5",
            "Portfolio",
            ("Approved" if coordinator_approved else "Book risk block"),
            ("pass" if coordinator_approved else "block"),
        ),
        (
            "6",
            "Execution",
            (
                "Filled"
                if execution_success is True
                else "Rejected"
                if execution_success is False
                else "Skipped upstream"
            ),
            (
                "pass"
                if execution_success is True
                else "block"
                if execution_success is False
                else "skip"
            ),
        ),
    ]

    nodes = []

    for index, name, description, state in stages:
        nodes.append(
            f"""
            <div class="pipeline-step state-{state}">
                <div class="index">{esc(index)}</div>
                <strong>{esc(name)}</strong>
                <small>{esc(description)}</small>
            </div>
            """
        )

    display_html(
        f"""
        <div class="pipeline">
            {"".join(nodes)}
        </div>
        """
    )


# ============================================================
# DECISION RENDERING
# ============================================================


def render_agent_card(
    name,
    direction,
    confidence,
    detail,
):
    cls = direction_class(direction)

    display_html(
        f"""
        <div class="agent-card">
            <div class="agent-name">
                {esc(name)}
            </div>

            <div class="agent-value {cls}">
                {esc(direction or "—").upper()}
            </div>

            <div class="agent-detail">
                Confidence:
                <b>{fmt_num(confidence, 3)}</b>
                <br/>
                {esc(detail)}
            </div>
        </div>
        """
    )


def render_decision(entry):
    outcome = entry.get("outcome")
    ticker = entry.get("ticker", "—")
    date_value = entry.get("date", "—")

    signal_direction = entry.get("signal_direction")

    confidence = entry.get(
        "signal_confidence",
        0,
    )

    merge_agreement = entry.get("merge_agreement")

    risk_decision = entry.get("risk_decision")

    execution_attempted = entry.get(
        "execution_attempted",
        False,
    )

    execution_success = entry.get("execution_success")

    risk_notes = entry.get("risk_notes") or []

    atr = entry.get("atr")
    entry_price = entry.get("entry_price")
    proposed_shares = entry.get("proposed_shares")

    news_availability = entry.get("news_availability")

    news_count = entry.get(
        "news_article_count",
        0,
    )

    news_source = entry.get("news_source")

    css_class = outcome_class(outcome)

    display_html(
        f"""
        <div class="decision-card {css_class}">
            <div class="decision-head">
                <div>
                    <div class="decision-title">
                        {esc(ticker)}
                    </div>
                    <div class="decision-date">
                        {esc(date_value)}
                    </div>
                </div>

                <span class="badge {css_class}">
                    {esc(outcome_label(outcome))}
                </span>
            </div>
        </div>
        """
    )

    render_pipeline()

    st.markdown("#### Agent decisions")

    c1, c2, c3 = st.columns(3)

    with c1:
        render_agent_card(
            "NewsAgent",
            entry.get(
                "news_direction",
                "neutral",
            ),
            entry.get(
                "news_confidence",
                0,
            ),
            (f"{news_count} article(s) · {news_availability or 'unknown'}"),
        )

    with c2:
        render_agent_card(
            "ChartAgent",
            entry.get(
                "chart_direction",
                signal_direction,
            ),
            entry.get(
                "chart_confidence",
                confidence,
            ),
            "RSI / SMA / volume technical evidence",
        )

    with c3:
        render_agent_card(
            "SignalMerger",
            signal_direction,
            confidence,
            (
                "Agreement: "
                + (
                    "YES"
                    if merge_agreement is True
                    else "NO"
                    if merge_agreement is False
                    else "NOT REPORTED"
                )
            ),
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### RiskAgent")

        display_html(
            f"""
            <div class="reason-box">
                <strong>Risk decision:</strong>
                {esc(risk_decision or "—")}
                <br/>
                <strong>Entry:</strong>
                {fmt_money(entry_price)}
                <br/>
                <strong>ATR:</strong>
                {fmt_num(atr, 4)}
                <br/>
                <strong>Proposed shares:</strong>
                {fmt_shares(proposed_shares)}
            </div>
            """
        )

        if risk_notes:
            with st.expander(f"Risk reasoning ({len(risk_notes)} checks)"):
                for note in risk_notes:
                    if "BREACH" in str(note).upper():
                        st.error(note)
                    elif "REJECT" in str(note).upper():
                        st.warning(note)
                    else:
                        st.caption("✓ " + str(note))

    with c2:
        st.markdown("#### ExecutionAgent")

        if execution_attempted:
            if execution_success is True:
                st.success("Broker execution succeeded.")
            elif execution_success is False:
                st.error("Broker execution was attempted but did not succeed.")
            else:
                st.warning("Execution was attempted, but the result is not reported.")
        else:
            st.info("Execution was not attempted. The trade was blocked upstream.")

        display_html(
            f"""
            <div class="reason-box">
                <strong>Attempted:</strong>
                {esc(execution_attempted)}
                <br/>
                <strong>Success:</strong>
                {esc(execution_success)}
                <br/>
                <strong>News source:</strong>
                {esc(news_source or "—")}
            </div>
            """
        )

    st.markdown("#### Why did the system make this decision?")

    display_html(
        f"""
        <div class="reason-box">
            <strong>Final outcome:</strong>
            {esc(outcome_label(outcome))}
            <br/><br/>
            {esc(entry.get("notes", ""))}
        </div>
        """
    )


# ============================================================
# PAPER TRADE RENDERING
# ============================================================

# Best-effort keyword map used only to *label* risk_notes strings
# that RiskAgent / PortfolioRiskCoordinator already produced. This
# never re-derives or overrides the underlying risk decision - it
# only groups existing notes under the named guardrail they most
# likely came from, matching the documented risk rule set (circuit
# breaker, position/ticker/sector caps, correlation, book-level
# risk).
RISK_CHECK_PATTERNS = [
    ("Circuit breaker", ("circuit breaker", "daily equity", "daily loss")),
    ("Position limit", ("position limit", "max position", "concurrent position")),
    (
        "Ticker cap",
        (
            "ticker cap",
            "per-ticker",
            "per ticker",
            "ticker-level",
            "ticker exposure",
        ),
    ),
    (
        "Sector cap",
        ("sector cap", "sector exposure", "sector limit", "sector-level"),
    ),
    ("Correlation", ("correlation",)),
    (
        "Book-level risk",
        (
            "book-level",
            "book level",
            "book risk",
            "portfolio risk",
            "portfolio-level",
            "coordinator",
        ),
    ),
    ("Drawdown", ("drawdown",)),
]

RISK_BLOCK_KEYWORDS = (
    "breach",
    "block",
    "reject",
    "exceed",
    "fail",
    "denied",
)

RISK_PASS_KEYWORDS = (
    "ok",
    "pass",
    "approved",
    "within",
    "clear",
)


def classify_risk_notes(risk_notes):
    """
    Best-effort classification of raw risk_notes strings into named
    checks with a PASS / BLOCK / INFO status, purely for display.

    Presentation-only: it never changes, recomputes, or overrides
    the risk decision already made upstream.
    """
    checks = []
    unmatched = []

    for note in risk_notes or []:
        text = str(note)
        lowered = text.lower()

        label = None

        for check_name, keywords in RISK_CHECK_PATTERNS:
            if any(keyword in lowered for keyword in keywords):
                label = check_name
                break

        if any(keyword in lowered for keyword in RISK_BLOCK_KEYWORDS):
            status = "block"
        elif any(keyword in lowered for keyword in RISK_PASS_KEYWORDS):
            status = "pass"
        else:
            status = "info"

        if label:
            checks.append((label, status, text))
        else:
            unmatched.append((status, text))

    return checks, unmatched


def render_risk_matrix(risk_notes):
    """Structured PASS / BLOCK view of the guardrails RiskAgent and
    PortfolioRiskCoordinator already evaluated."""
    checks, unmatched = classify_risk_notes(risk_notes)

    if not checks and not unmatched:
        st.caption("No risk notes were returned.")
        return

    if checks:
        badge_class = {
            "pass": "success",
            "block": "danger",
            "info": "neutral",
        }

        symbol = {
            "pass": "✓",
            "block": "✕",
            "info": "○",
        }

        cells = [
            f"""
            <div class="risk-check {badge_class[status]}" title="{esc(text)}">
                <span class="symbol">{symbol[status]}</span>
                <span class="name">{esc(label)}</span>
            </div>
            """
            for label, status, text in checks
        ]

        display_html(
            f"""
            <div class="risk-matrix">
                {"".join(cells)}
            </div>
            """
        )

    if unmatched:
        with st.expander(f"Additional risk notes ({len(unmatched)})"):
            for status, text in unmatched:
                if status == "block":
                    st.error(text)
                elif status == "pass":
                    st.caption("✓ " + text)
                else:
                    st.caption(text)


def render_decision_headline(outcome, notes):
    """
    A single, unambiguous plain-language headline for the final
    decision, shown ahead of any supporting detail.
    """
    css_class = outcome_class(outcome)

    headline = {
        "executed": "TRADE EXECUTED",
        "held_local_reject": "HOLD — LOCAL RISK BLOCK",
        "held_book_reject": "HOLD — BOOK RISK BLOCK",
        "held_execution_reject": "HOLD — BROKER EXECUTION FAILED",
        "neutral_signal": "HOLD — NEUTRAL SIGNAL",
        "ticker_capacity": "HOLD — TICKER CAPACITY BLOCK",
        "sector_capacity": "HOLD — SECTOR CAPACITY BLOCK",
        "book_risk": "HOLD — BOOK RISK BLOCK",
    }.get(
        outcome or "",
        outcome_label(outcome),
    )

    display_html(
        f"""
        <div class="decision-headline {css_class}">
            <div class="decision-headline-label">
                Final decision
            </div>
            <div class="decision-headline-title">
                {esc(headline)}
            </div>
            <div class="decision-headline-text">
                {esc(notes or "No additional notes were returned.")}
            </div>
        </div>
        """
    )


def render_decision_lineage(summary):
    """
    A compact status row over the full decision lineage: evidence →
    signal → local risk → book risk → execution → accounting
    mutation. Complements render_paper_trade_pipeline's per-agent
    view with a single scannable line.
    """
    coordinator_approved = summary.get("coordinator_approved")

    execution_success = summary.get("execution_success")

    position_mutated = execution_success is True

    stages = [
        ("Evidence", "pass"),
        ("Signal", "pass"),
        ("Local risk", "pass"),
        ("Book risk", "pass" if coordinator_approved else "block"),
        (
            "Execution",
            (
                "pass"
                if execution_success is True
                else "block"
                if execution_success is False
                else "skip"
            ),
        ),
        ("Accounting mutation", "pass" if position_mutated else "skip"),
    ]

    symbols = {"pass": "✓", "block": "✕", "skip": "○"}
    classes = {"pass": "success", "block": "danger", "skip": "neutral"}

    items = [
        f"""
        <div class="lineage-item {classes[state]}">
            <span class="symbol">{symbols[state]}</span>{esc(label)}
        </div>
        """
        for label, state in stages
    ]

    display_html(
        f"""
        <div class="lineage-row">
            {"".join(items)}
        </div>
        """
    )


def render_paper_evidence_cards(summary):
    """NewsAgent / ChartAgent / SignalMerger evidence, matching the
    Decision Explorer's agent cards so both views read the same
    way."""
    c1, c2, c3 = st.columns(3)

    with c1:
        render_agent_card(
            "NewsAgent",
            summary.get("news_direction") or "neutral",
            summary.get("news_confidence"),
            (
                f"{summary.get('news_article_count', 0)} article(s) · "
                f"{summary.get('news_availability') or 'unknown'}"
            ),
        )

    with c2:
        render_agent_card(
            "ChartAgent",
            summary.get("chart_direction") or summary.get("signal_direction"),
            summary.get("chart_confidence") or summary.get("signal_confidence"),
            f"ATR {fmt_num(summary.get('atr'), 3)}",
        )

    with c3:
        merge_agreement = summary.get("merge_agreement")

        render_agent_card(
            "SignalMerger",
            summary.get("signal_direction"),
            summary.get("signal_confidence"),
            (
                "Agreement: "
                + (
                    "YES"
                    if merge_agreement is True
                    else "NO"
                    if merge_agreement is False
                    else "NOT REPORTED"
                )
            ),
        )


def render_temporal_integrity(simulated_date, summary):
    """
    Make the distinction between simulation date, news as-of date,
    and wall-clock audit timestamp explicit, so the no-lookahead
    guarantee is visible rather than implied.
    """
    news_as_of = summary.get("news_as_of") or simulated_date.isoformat()

    audit_timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = [
        ("Simulation date", simulated_date.isoformat()),
        ("News as-of", str(news_as_of)),
        ("Replay source", "historical_replay"),
        ("Audit timestamp", audit_timestamp),
        ("Future data", "BLOCKED / NOT USED"),
    ]

    display_html(
        '<div class="temporal-grid">'
        + "".join(
            f"""
            <div class="temporal-item">
                <div class="label">{esc(label)}</div>
                <div class="value">{esc(value)}</div>
            </div>
            """
            for label, value in rows
        )
        + "</div>"
    )


def render_paper_trade_result(
    run_id: str,
    ticker: str,
    simulated_date: date,
    result,
):
    summary = paper_trade_result_summary(result)

    outcome = summary["outcome"]
    css_class = outcome_class(outcome)

    st.markdown("### Orchestration status")

    display_html(
        f"""
        <div class="decision-card {css_class}">
            <div class="decision-head">
                <div>
                    <div class="decision-title">
                        {esc(ticker)}
                    </div>
                    <div class="decision-date">
                        Simulation date:
                        {esc(simulated_date.isoformat())}
                        · Run <code>{esc(run_id)}</code>
                    </div>
                </div>

                <span class="badge {css_class}">
                    {esc(outcome_label(outcome))}
                </span>
            </div>
        </div>
        """
    )

    render_paper_trade_pipeline(result)

    render_decision_lineage(summary)

    render_decision_headline(outcome, summary["notes"])

    st.markdown("#### Agent evidence")

    render_paper_evidence_cards(summary)

    st.markdown("#### Risk decision")

    display_html(
        f"""
        <div class="reason-box">
            <strong>Risk decision:</strong>
            {esc(summary["risk_decision"] or "—")}
            <br/>
            <strong>Entry:</strong>
            {fmt_money(summary["entry_price"])}
            <br/>
            <strong>ATR:</strong>
            {fmt_num(summary["atr"], 4)}
            <br/>
            <strong>Proposed shares:</strong>
            {fmt_shares(summary["proposed_shares"])}
        </div>
        """
    )

    render_risk_matrix(summary["risk_notes"])

    st.markdown("#### Decision summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Signal",
            str(summary["signal_direction"] or "—").upper(),
        )

    with c2:
        confidence = safe_float(summary["signal_confidence"])

        st.metric(
            "Confidence",
            (f"{confidence:.3f}" if confidence is not None else "—"),
        )

    with c3:
        st.metric(
            "Risk",
            str(summary["risk_decision"] or "—").upper(),
        )

    with c4:
        st.metric(
            "Execution",
            (
                "FILLED"
                if summary["execution_success"] is True
                else "REJECTED"
                if summary["execution_success"] is False
                else "NOT ATTEMPTED"
            ),
        )

    st.markdown("#### Execution")

    if summary["execution_attempted"]:
        if summary["execution_success"] is True:
            st.success("Broker execution succeeded.")
        elif summary["execution_success"] is False:
            st.error("Broker execution was attempted but did not succeed.")
        else:
            st.warning("Execution was attempted, but the result is not reported.")
    else:
        blocked_at = {
            "held_local_reject": "the local risk gate (RiskAgent)",
            "held_book_reject": (
                "the portfolio-level risk gate (PortfolioRiskCoordinator)"
            ),
        }.get(outcome, "an earlier stage in the pipeline")

        st.info(
            "Execution was not attempted. The trade was blocked upstream at "
            f"{blocked_at}, so no order was sent to the broker and portfolio "
            "state is unchanged."
        )

    execution_rows = [
        {
            "Field": "Attempted",
            "Value": str(summary["execution_attempted"]),
        },
        {
            "Field": "Success",
            "Value": str(summary["execution_success"]),
        },
        {
            "Field": "Side",
            "Value": (summary["execution_side"] or "—"),
        },
        {
            "Field": "Quantity",
            "Value": fmt_shares(summary["shares"]),
        },
        {
            "Field": "Price",
            "Value": fmt_money(summary["price"]),
        },
        {
            "Field": "Entry price",
            "Value": fmt_money(summary["entry_price"]),
        },
        {
            "Field": "Proposed shares",
            "Value": fmt_shares(summary["proposed_shares"]),
        },
        {
            "Field": "Sector",
            "Value": summary["sector"] or "—",
        },
    ]

    st.dataframe(
        execution_rows,
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Temporal integrity")

    render_temporal_integrity(simulated_date, summary)

    st.markdown("#### Portfolio accounting")

    pnl = safe_float(summary["realized_pnl"])

    remaining = safe_float(summary["remaining_shares"])

    avg_cost = safe_float(summary["average_cost"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Average cost",
            fmt_money(avg_cost),
        )

    with c2:
        st.metric(
            "Realized P&L",
            fmt_money(pnl),
        )

    with c3:
        st.metric(
            "Remaining shares",
            fmt_shares(remaining),
        )

    with c4:
        st.metric(
            "Position closed",
            "YES" if summary["position_closed"] else "NO",
        )

    if summary["position_closed"]:
        display_html(
            """
            <div class="accounting-highlight">
                <div class="title">
                    Lifecycle state
                </div>
                <div class="value">
                    POSITION CLOSED
                </div>
            </div>
            """
        )
    elif remaining is not None and remaining > 0:
        display_html(
            """
            <div class="accounting-highlight">
                <div class="title">
                    Lifecycle state
                </div>
                <div class="value">
                    POSITION OPEN
                </div>
            </div>
            """
        )

    else:
        display_html(
            """
            <div class="accounting-highlight">
                <div class="title">
                    Lifecycle state
                </div>
                <div class="value">
                    NO POSITION
                </div>
            </div>
            """
        )

        st.caption(
            "No position exists for this ticker. This is not the same as a "
            "position having been closed — none was ever opened."
        )

    st.markdown("#### Additional context")

    display_html(
        f"""
        <div class="reason-box">
            <strong>News availability:</strong>
            {esc(summary["news_availability"] or "—")}
            <br/>
            <strong>News articles:</strong>
            {esc(summary["news_article_count"])}
            <br/>
            <strong>News source:</strong>
            {esc(summary["news_source"] or "—")}
        </div>
        """
    )

    st.markdown("#### Audit identity")

    audit_rows = [
        {
            "Field": "Run ID",
            "Value": run_id,
        },
        {
            "Field": "Ticker",
            "Value": ticker,
        },
        {
            "Field": "Simulated date",
            "Value": simulated_date.isoformat(),
        },
        {
            "Field": "Outcome",
            "Value": outcome_label(outcome),
        },
    ]

    st.dataframe(
        audit_rows,
        width="stretch",
        hide_index=True,
    )

    trade_trace = getattr(
        result,
        "trade_trace",
        None,
    )

    if trade_trace is not None:
        st.markdown("#### TradeTrace")

        trace_dict = serialize_object(trade_trace)

        with st.expander(
            "Inspect full TradeTrace",
            expanded=True,
        ):
            st.json(trace_dict)

    else:
        st.info("This TickResult did not expose a TradeTrace object.")


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

if "selected_run" not in st.session_state:
    st.session_state.selected_run = None

if "paper_trade_result" not in st.session_state:
    st.session_state.paper_trade_result = None

if "paper_trade_run_id" not in st.session_state:
    st.session_state.paper_trade_run_id = None

if "paper_trade_ticker" not in st.session_state:
    st.session_state.paper_trade_ticker = None

if "paper_trade_date" not in st.session_state:
    st.session_state.paper_trade_date = None


# ============================================================
# INITIAL DATA
# ============================================================

raw_ledger = load_ledger()

ledger = normalize_ledger(raw_ledger)

backtest_runs = list_backtest_runs(str(BACKTEST_RESULTS_DIR))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        ## 📈 Trading Firm
        **AI Multi-Agent Command Center**
        """
    )

    st.caption("Understand every decision from signal → risk → execution.")

    st.divider()

    st.markdown("### System")

    st.write(f"**Mode:** `{TRADING_MODE.upper()}`")

    st.write(f"**Backtests:** `{len(backtest_runs)}`")

    st.write(f"**Audit DB:** `{'FOUND' if AUDIT_DB_PATH.exists() else 'NOT FOUND'}`")

    st.write(f"**Ledger:** `{'FOUND' if raw_ledger else 'NOT FOUND'}`")

    st.divider()

    if st.button(
        "🔄 Refresh",
        width="stretch",
    ):
        load_json_file.clear()
        list_backtest_runs.clear()
        list_audit_runs.clear()
        load_audit_run.clear()
        audit_table_exists.clear()
        audit_table_columns.clear()
        audit_table_counts.clear()

        st.rerun()

    st.markdown("### Run backtest")

    bt_tickers = st.text_input(
        "Tickers",
        value="AAPL,MSFT,GOOGL,JPM",
    )

    bt_start = st.date_input("Start date")

    bt_end = st.date_input("End date")

    bt_delay = st.number_input(
        "Tick delay",
        min_value=0.0,
        value=1.5,
        step=0.5,
    )

    bt_run_id = st.text_input(
        "Run ID",
        value="",
        help=("Leave blank to generate a unique audit run ID."),
    )

    if st.button(
        "▶ Run backtest",
        width="stretch",
    ):
        tickers = [
            ticker.strip().upper() for ticker in bt_tickers.split(",") if ticker.strip()
        ]

        if not tickers:
            st.error("Enter at least one ticker.")
            st.stop()

        run_id = bt_run_id.strip() or (f"bt_{bt_start}_{bt_end}_{uuid.uuid4().hex[:8]}")

        try:
            with st.spinner("Running multi-agent backtest..."):
                from backtest.runner import (
                    _serialize_result,
                    run_backtest,
                )

                with trace_context():
                    result = run_backtest(
                        tickers=tickers,
                        start_date=str(bt_start),
                        end_date=str(bt_end),
                        tick_delay_seconds=bt_delay,
                        run_id=run_id,
                    )

                BACKTEST_RESULTS_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path = BACKTEST_RESULTS_DIR / f"{run_id}.json"

                path.write_text(
                    json.dumps(
                        _serialize_result(result),
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            st.session_state.selected_run = f"{run_id}.json"

            list_backtest_runs.clear()
            list_audit_runs.clear()
            load_audit_run.clear()
            audit_table_counts.clear()

            st.success(f"Completed {len(result.tick_log)} ticks for `{run_id}`.")

            st.rerun()

        except Exception as exc:  # noqa: BLE001
            if LOGFIRE_OK:
                logfire.exception(
                    "Backtest failed",
                    error_type=type(exc).__name__,
                )

            st.error(f"Backtest failed: {exc}")

    st.divider()

    st.caption("LangGraph · Risk-gated execution · SimBroker / Alpaca · SQLite audit")


# ============================================================
# HERO
# ============================================================

display_html(
    f"""
    <div class="hero">

        <div class="hero-kicker">
            Multi-agent decision intelligence
        </div>

        <h1>
            AI Multi-Agent Trading Firm
        </h1>

        <p>
            Trace how the trading system moves from market
            information to broker execution. NewsAgent and
            ChartAgent generate independent evidence, SignalMerger
            combines it deterministically, RiskAgent performs
            position sizing and local risk checks, Portfolio Risk
            evaluates the whole book, and only then can
            ExecutionAgent submit a trade.
        </p>

        <div class="status-pill">
            <span class="status-dot"></span>
            {esc(TRADING_MODE.upper())} MODE
        </div>

    </div>
    """
)


# ============================================================
# TABS
# ============================================================

(
    tab_dashboard,
    tab_decisions,
    tab_backtest,
    tab_paper,
    tab_audit,
    tab_data,
) = st.tabs(
    [
        "🧠 Command Center",
        "🔎 Decision Explorer",
        "📊 Backtest Lab",
        "🎯 Agentic Paper Trade",
        "🧾 Run Audit",
        "🗄️ Data & System",
    ]
)


# ============================================================
# COMMAND CENTER
# ============================================================

with tab_dashboard:
    st.markdown(
        '<div class="section-title">How the firm makes a trade</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Every ticker follows the same risk-gated pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    render_architecture()

    if ledger:
        st.markdown(
            '<div class="section-title">Live portfolio</div>',
            unsafe_allow_html=True,
        )

        equity = ledger.get("equity")

        starting = ledger.get("starting_equity")

        if equity is not None and starting not in (
            None,
            0,
        ):
            portfolio_return = float(equity) / float(starting) - 1
        else:
            portfolio_return = None

        kpis = [
            (
                "Equity",
                fmt_money(equity),
                "",
            ),
            (
                "Cash",
                fmt_money(ledger.get("cash")),
                "",
            ),
            (
                "Portfolio return",
                fmt_pct(portfolio_return),
                (
                    "green"
                    if (portfolio_return is not None and portfolio_return >= 0)
                    else "red"
                ),
            ),
            (
                "Positions",
                len(
                    ledger.get(
                        "positions",
                        {},
                    )
                ),
                "",
            ),
        ]

        display_html(
            '<div class="kpi-grid">'
            + "".join(
                f"""
                <div class="kpi">
                    <div class="label">
                        {esc(label)}
                    </div>
                    <div class="value {cls}">
                        {esc(value)}
                    </div>
                </div>
                """
                for label, value, cls in kpis
            )
            + "</div>"
        )

        positions = ledger.get(
            "positions",
            {},
        )

        if positions:
            st.markdown("#### Current positions")

            rows = []

            for ticker, position in positions.items():
                if not isinstance(
                    position,
                    dict,
                ):
                    continue

                rows.append(
                    {
                        "Ticker": ticker,
                        "Sector": position.get(
                            "sector",
                            "—",
                        ),
                        "Shares": fmt_shares(position.get("shares")),
                        "Market value": fmt_money(position.get("market_value")),
                    }
                )

            st.dataframe(
                rows,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info("No open positions.")

    else:
        st.info(
            "No live ledger found. "
            "Use the Backtest Lab or Agentic Paper Trade "
            "to inspect runtime behavior."
        )


# ============================================================
# DECISION EXPLORER
# ============================================================

# ============================================================
# DECISION EXPLORER
# ============================================================

with tab_decisions:
    if not backtest_runs:
        st.info("No backtest results found.")

    else:
        default_run = (
            st.session_state.selected_run
            if (st.session_state.selected_run in backtest_runs)
            else backtest_runs[0]
        )

        selected_run = st.selectbox(
            "Backtest run",
            backtest_runs,
            index=backtest_runs.index(default_run),
            key="decision_run_selector",
        )

        run_data = load_run(selected_run)

        if not run_data:
            st.error("Unable to load selected run.")

        else:
            tick_log = run_data.get(
                "tick_log",
                [],
            )

            if not tick_log:
                st.warning("This run does not contain tick telemetry.")

            else:
                st.markdown(f"### `{selected_run}`")

                observability = build_run_observability(
                    run_data,
                )

                # ------------------------------------------------
                # RUN SUMMARY
                # ------------------------------------------------

                st.markdown("#### Run decision summary")

                summary_kpis = [
                    (
                        "Ticks",
                        observability["ticks"],
                        "",
                    ),
                    (
                        "TradeTraces",
                        observability["traces"],
                        "",
                    ),
                    (
                        "Trace coverage",
                        (
                            f"{observability['trace_coverage']:.0%}"
                            if observability["trace_coverage"] is not None
                            else "—"
                        ),
                        (
                            "green"
                            if observability["trace_coverage"] == 1
                            else "yellow"
                            if observability["trace_coverage"] is not None
                            else ""
                        ),
                    ),
                    (
                        "Execution success",
                        observability["execution_successes"],
                        ("green" if observability["execution_successes"] else ""),
                    ),
                ]

                display_html(
                    '<div class="kpi-grid four">'
                    + "".join(
                        f"""
                        <div class="kpi">
                            <div class="label">
                                {esc(label)}
                            </div>
                            <div class="value {cls}">
                                {esc(value)}
                            </div>
                        </div>
                        """
                        for label, value, cls in summary_kpis
                    )
                    + "</div>"
                )

                # ------------------------------------------------
                # FILTERS
                # ------------------------------------------------

                st.markdown("#### Decision filters")

                tickers = sorted(
                    {
                        str(entry.get("ticker"))
                        for entry in tick_log
                        if entry.get("ticker")
                    }
                )

                outcomes = sorted(
                    {
                        str(entry.get("outcome"))
                        for entry in tick_log
                        if entry.get("outcome")
                    }
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    ticker_filter = st.selectbox(
                        "Ticker",
                        ["All"] + tickers,
                        key="decision_ticker_filter",
                    )

                with c2:
                    outcome_filter = st.selectbox(
                        "Outcome",
                        ["All"] + outcomes,
                        key="decision_outcome_filter",
                        format_func=outcome_label,
                    )

                with c3:
                    show_latest = st.number_input(
                        "Events to show",
                        min_value=1,
                        max_value=500,
                        value=20,
                        key="decision_events_limit",
                    )

                filtered = [
                    entry
                    for entry in reversed(tick_log)
                    if (ticker_filter == "All" or entry.get("ticker") == ticker_filter)
                    and (
                        outcome_filter == "All"
                        or entry.get("outcome") == outcome_filter
                    )
                ]

                st.caption(f"{len(filtered)} matching decision(s)")

                visible = filtered[: int(show_latest)]

                if not visible:
                    st.info("No decisions match the selected filters.")

                else:
                    # ------------------------------------------------
                    # DECISION TABLE
                    # ------------------------------------------------

                    st.markdown("#### Decision timeline")

                    decision_rows = []

                    for entry in visible:
                        summary = paper_trade_result_summary(
                            entry,
                        )

                        execution_success = execution_success_from_summary(
                            summary,
                        )

                        coordinator_approved = coordinator_approval_from_summary(
                            summary,
                        )

                        accounting_mutation = accounting_mutation_from_summary(
                            summary,
                        )

                        decision_rows.append(
                            {
                                "Date": entry.get(
                                    "date",
                                    entry.get(
                                        "simulated_date",
                                        "—",
                                    ),
                                ),
                                "Ticker": entry.get(
                                    "ticker",
                                    "—",
                                ),
                                "Signal": str(
                                    summary.get(
                                        "signal_direction",
                                        "—",
                                    )
                                    or "—"
                                ).upper(),
                                "Confidence": fmt_num(
                                    summary.get(
                                        "signal_confidence",
                                    ),
                                    3,
                                ),
                                "Risk": str(
                                    summary.get(
                                        "risk_decision",
                                        "—",
                                    )
                                    or "—"
                                ).upper(),
                                "Coordinator": (
                                    "APPROVED"
                                    if coordinator_approved is True
                                    else "REJECTED"
                                    if coordinator_approved is False
                                    else "—"
                                ),
                                "Execution": (
                                    "FILLED"
                                    if execution_success is True
                                    else "FAILED"
                                    if execution_success is False
                                    else "NOT ATTEMPTED"
                                ),
                                "Accounting": (
                                    "MUTATED"
                                    if accounting_mutation is True
                                    else "UNCHANGED"
                                    if accounting_mutation is False
                                    else "—"
                                ),
                                "Outcome": outcome_label(
                                    entry.get("outcome"),
                                ),
                            }
                        )

                    st.dataframe(
                        decision_rows,
                        width="stretch",
                        hide_index=True,
                    )

                    # ------------------------------------------------
                    # TRACE INSPECTOR
                    # ------------------------------------------------

                    st.markdown("#### Decision inspector")

                    selected_index = st.selectbox(
                        "Inspect decision",
                        range(len(visible)),
                        format_func=lambda i: (
                            f"{visible[i].get('date', visible[i].get('simulated_date', '—'))}"
                            f" · {visible[i].get('ticker', '—')}"
                            f" · {outcome_label(visible[i].get('outcome'))}"
                        ),
                        key="decision_trace_selector",
                    )

                    render_decision(
                        visible[selected_index],
                    )

                    # ------------------------------------------------
                    # DECISION DISTRIBUTION
                    # ------------------------------------------------

                    st.markdown("#### Decision distribution")

                    reason_counts = extract_decision_reasons(
                        {
                            **run_data,
                            "tick_log": visible,
                        }
                    )

                    render_decision_reasons(
                        reason_counts,
                        total_ticks=len(visible),
                    )

                    # ------------------------------------------------
                    # OBSERVABILITY DETAILS
                    # ------------------------------------------------

                    with st.expander(
                        "Run observability",
                        expanded=False,
                    ):
                        observability_rows = [
                            {
                                "Metric": "Ticks",
                                "Value": str(observability["ticks"]),
                            },
                            {
                                "Metric": "TradeTraces",
                                "Value": str(observability["traces"]),
                            },
                            {
                                "Metric": "Trace coverage",
                                "Value": (
                                    f"{observability['trace_coverage']:.0%}"
                                    if observability["trace_coverage"] is not None
                                    else "—"
                                ),
                            },
                            {
                                "Metric": "Risk approvals",
                                "Value": str(observability["risk_approvals"]),
                            },
                            {
                                "Metric": "Coordinator approvals",
                                "Value": str(observability["coordinator_approvals"]),
                            },
                            {
                                "Metric": "Execution attempts",
                                "Value": str(observability["execution_attempts"]),
                            },
                            {
                                "Metric": "Execution successes",
                                "Value": str(observability["execution_successes"]),
                            },
                            {
                                "Metric": "Execution failures",
                                "Value": str(observability["execution_failures"]),
                            },
                            {
                                "Metric": "Normalized trades",
                                "Value": str(observability["normalized_trades"]),
                            },
                            {
                                "Metric": "Closed trades",
                                "Value": str(observability["closed_trades"]),
                            },
                        ]

                        st.dataframe(
                            observability_rows,
                            width="stretch",
                            hide_index=True,
                        )

# ============================================================
# BACKTEST LAB
# ============================================================

with tab_backtest:
    if not backtest_runs:
        st.info("No backtest results found.")

    else:
        default_run = (
            st.session_state.selected_run
            if (st.session_state.selected_run in backtest_runs)
            else backtest_runs[0]
        )

        selected_run = st.selectbox(
            "Select run",
            backtest_runs,
            index=backtest_runs.index(default_run),
            key="backtest_run_selector",
        )

        run_data = load_run(selected_run)

        if not run_data:
            st.error("Unable to load selected run.")

        else:
            st.markdown(f"### Backtest: `{selected_run}`")

            diagnostics = run_data.get(
                "diagnostics",
                {},
            )

            total_return = run_data.get("total_return")

            benchmark_return = run_data.get("buy_hold_return")

            excess_return = run_data.get("excess_return_vs_buy_and_hold")

            max_drawdown = run_data.get("max_drawdown")

            sharpe = run_data.get("sharpe_ratio")

            trades = run_data.get(
                "trades",
                [],
            )

            kpis = [
                (
                    "Strategy return",
                    fmt_pct(total_return),
                    (
                        "green"
                        if (
                            safe_float(total_return) is not None
                            and safe_float(total_return) >= 0
                        )
                        else "red"
                    ),
                ),
                (
                    "Buy & hold",
                    fmt_pct(benchmark_return),
                    "",
                ),
                (
                    "Excess return",
                    fmt_pct(excess_return),
                    (
                        "green"
                        if (
                            safe_float(excess_return) is not None
                            and safe_float(excess_return) >= 0
                        )
                        else "red"
                    ),
                ),
                (
                    "Sharpe",
                    fmt_num(
                        sharpe,
                        2,
                    ),
                    "",
                ),
                (
                    "Max drawdown",
                    fmt_pct(max_drawdown),
                    "red",
                ),
                (
                    "Executed",
                    len(trades),
                    "",
                ),
            ]

            display_html(
                '<div class="kpi-grid">'
                + "".join(
                    f"""
                    <div class="kpi">
                        <div class="label">
                            {esc(label)}
                        </div>
                        <div class="value {cls}">
                            {esc(value)}
                        </div>
                    </div>
                    """
                    for label, value, cls in kpis
                )
                + "</div>"
            )

            tick_log = run_data.get(
                "tick_log",
                [],
            )

            # ------------------------------------------------
            # DECISION OUTCOMES
            # ------------------------------------------------

            st.markdown("#### Decision outcomes")

            st.caption(
                "Why each tick ended the way it did. Diagnostic telemetry "
                "only; these counts do not influence strategy behavior."
            )

            render_decision_reasons(
                extract_decision_reasons(run_data),
                total_ticks=len(tick_log),
            )

            # ------------------------------------------------
            # EXECUTION
            # ------------------------------------------------

            st.markdown("#### Execution")

            execution_attempts = diagnostics.get(
                "execution_attempts",
                0,
            )

            execution_failures = diagnostics.get(
                "execution_failures",
                0,
            )

            execution_successes = diagnostics.get(
                "execution_successes",
                max(
                    (execution_attempts or 0) - (execution_failures or 0),
                    0,
                ),
            )

            closed_trades_raw = run_data.get(
                "closed_trades",
                [],
            )

            # closed_trades may be stored as a list of round-trips or
            # as a pre-computed integer count, depending on runner version.
            if isinstance(closed_trades_raw, (list, tuple, dict)):
                closed_trades_count = len(closed_trades_raw)
            else:
                closed_trades_count = int(safe_float(closed_trades_raw) or 0)

            exec_kpis = [
                (
                    "Attempts",
                    execution_attempts,
                    "",
                ),
                (
                    "Successful",
                    execution_successes,
                    "green" if execution_successes else "",
                ),
                (
                    "Failures",
                    execution_failures,
                    "red" if execution_failures else "green",
                ),
                (
                    "Execution events",
                    len(trades),
                    "",
                ),
                (
                    "Closed round-trips",
                    closed_trades_count,
                    "",
                ),
                (
                    "Risk approvals",
                    diagnostics.get(
                        "risk_approvals",
                        0,
                    ),
                    "",
                ),
            ]

            display_html(
                '<div class="kpi-grid">'
                + "".join(
                    f"""
                    <div class="kpi">
                        <div class="label">
                            {esc(label)}
                        </div>
                        <div class="value {cls}">
                            {esc(value)}
                        </div>
                    </div>
                    """
                    for label, value, cls in exec_kpis
                )
                + "</div>"
            )

            st.caption(
                "Execution events are individual fills (including partial "
                "sells). Closed round-trips only count positions that were "
                "fully closed, so the two numbers can legitimately differ."
            )

            # ------------------------------------------------
            # TRACE COVERAGE
            # ------------------------------------------------

            st.markdown("#### Trace coverage")

            coverage_info = trace_coverage(
                run_data,
                Path(selected_run).stem,
            )

            coverage_value = coverage_info["coverage"]

            coverage_kpis = [
                (
                    "Ticks",
                    coverage_info["ticks"],
                    "",
                ),
                (
                    "TradeTraces",
                    (
                        coverage_info["traces"]
                        if coverage_info["traces"] is not None
                        else "—"
                    ),
                    "",
                ),
                (
                    "Coverage",
                    (f"{coverage_value:.0%}" if coverage_value is not None else "—"),
                    (
                        "green"
                        if coverage_value == 1
                        else "yellow"
                        if coverage_value is not None
                        else ""
                    ),
                ),
                (
                    "Tickers",
                    coverage_info["tickers"],
                    "",
                ),
            ]

            display_html(
                '<div class="kpi-grid four">'
                + "".join(
                    f"""
                    <div class="kpi">
                        <div class="label">
                            {esc(label)}
                        </div>
                        <div class="value {cls}">
                            {esc(value)}
                        </div>
                    </div>
                    """
                    for label, value, cls in coverage_kpis
                )
                + "</div>"
            )

            if coverage_info["traces"] is None:
                st.caption(
                    "TradeTraces not found in the SQLite audit DB for this "
                    "run (the run may predate persistence)."
                )

            with st.expander("Raw outcome counts (per tick)"):
                raw_outcome_counts = {}

                for entry in tick_log:
                    outcome_key = outcome_label(entry.get("outcome"))

                    raw_outcome_counts[outcome_key] = (
                        raw_outcome_counts.get(
                            outcome_key,
                            0,
                        )
                        + 1
                    )

                if raw_outcome_counts:
                    st.bar_chart(raw_outcome_counts)
                else:
                    st.caption("No tick telemetry recorded.")

            equity_curve = run_data.get(
                "equity_curve",
                [],
            )

            benchmark_curve = run_data.get(
                "buy_hold_curve",
                [],
            )

            if equity_curve:
                st.markdown("#### Strategy equity")

                st.line_chart({"Strategy": [x[1] for x in equity_curve]})

                if benchmark_curve:
                    st.markdown("#### Strategy vs buy & hold")

                    n = min(
                        len(equity_curve),
                        len(benchmark_curve),
                    )

                    st.line_chart(
                        {
                            "Strategy": [x[1] for x in equity_curve[:n]],
                            "Buy & hold": [x[1] for x in benchmark_curve[:n]],
                        }
                    )

            st.markdown("#### Executed trades")

            if trades:
                rows = [
                    {
                        "Date": trade.get("date"),
                        "Ticker": trade.get("ticker"),
                        "Shares": fmt_shares(trade.get("shares")),
                        "Price": fmt_money(trade.get("price")),
                        "Notes": trade.get(
                            "notes",
                            "",
                        ),
                    }
                    for trade in trades
                ]

                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.info("No executed trades.")


# ============================================================
# AGENTIC PAPER TRADE
# ============================================================

with tab_paper:
    st.markdown("### 🎯 Agentic Paper Trade")

    st.caption(
        "Recruiter/demo control plane: one ticker is sent through "
        "the real top-level orchestration path. The UI does not "
        "directly call the broker or alter strategy decisions."
    )

    display_html(
        """
        <div class="demo-banner">
            <div class="demo-banner-title">
                LIVE ORCHESTRATION DEMONSTRATION
            </div>

            <div class="demo-banner-text">
                Select a historical simulation date and ticker, then
                run the complete agentic decision pipeline:
                NewsAgent → ChartAgent → SignalMerger → RiskAgent →
                PortfolioRiskCoordinator → ExecutionAgent → Broker.
                The resulting accounting and TradeTrace are displayed
                below.
            </div>
        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        paper_ticker = st.selectbox(
            "Ticker",
            [
                "AAPL",
                "MSFT",
                "GOOGL",
                "JPM",
            ],
            key="paper_ticker_selector",
        )

    with c2:
        paper_date = st.date_input(
            "Simulation date",
            value=date(
                2024,
                6,
                6,
            ),
            key="paper_date_selector",
            help=(
                "Use a date supported by the historical replay "
                "data available to your trading runtime."
            ),
        )

    st.markdown("#### Execution path")

    render_architecture()

    st.markdown("#### Run the agentic trade")

    st.caption(
        "This control invokes orchestrator.tick_runner.run_tick(). "
        "It does not submit an order directly from Streamlit."
    )

    if st.button(
        "🚀 Run Agentic Paper Trade",
        width="stretch",
        type="primary",
    ):
        try:
            with st.spinner(
                f"Running {paper_ticker} through the multi-agent trading pipeline..."
            ):
                (
                    paper_run_id,
                    paper_result,
                ) = run_agentic_paper_trade(
                    ticker=paper_ticker,
                    simulated_date=paper_date,
                )

            st.session_state.paper_trade_run_id = paper_run_id

            st.session_state.paper_trade_result = paper_result

            st.session_state.paper_trade_ticker = paper_ticker

            st.session_state.paper_trade_date = paper_date

            st.success(f"Agentic run completed: `{paper_run_id}`")

        except Exception as exc:  # noqa: BLE001
            if LOGFIRE_OK:
                logfire.exception(
                    "Agentic paper trade failed",
                    error_type=type(exc).__name__,
                )

            st.error("Agentic paper trade failed.")

            with st.expander("Technical error"):
                st.exception(exc)

    paper_result = st.session_state.paper_trade_result

    if paper_result is not None:
        st.divider()

        render_paper_trade_result(
            run_id=(st.session_state.paper_trade_run_id or "—"),
            ticker=(st.session_state.paper_trade_ticker or paper_ticker),
            simulated_date=(st.session_state.paper_trade_date or paper_date),
            result=paper_result,
        )

        st.divider()

        st.markdown("#### What this demonstrates")

        demo_rows = [
            {
                "Architecture property": "Agent orchestration",
                "Demonstration": ("UI calls run_tick(), not the broker"),
            },
            {
                "Architecture property": "Independent evidence",
                "Demonstration": ("NewsAgent + ChartAgent"),
            },
            {
                "Architecture property": "Deterministic decision",
                "Demonstration": ("SignalMerger"),
            },
            {
                "Architecture property": "Local risk",
                "Demonstration": ("RiskAgent"),
            },
            {
                "Architecture property": "Book-level risk",
                "Demonstration": ("PortfolioRiskCoordinator"),
            },
            {
                "Architecture property": "Execution abstraction",
                "Demonstration": ("ExecutionAgent → Broker"),
            },
            {
                "Architecture property": "Accounting",
                "Demonstration": ("Average cost / realized P&L / position state"),
            },
            {
                "Architecture property": "Observability",
                "Demonstration": ("TradeTrace + run identity"),
            },
        ]

        st.dataframe(
            demo_rows,
            width="stretch",
            hide_index=True,
        )

# ============================================================
# RUN AUDIT
# ============================================================

with tab_audit:
    st.markdown("### Persistent Run Audit")

    st.caption(
        "SQLite-backed audit history for reproducible trading "
        "decisions, execution outcomes, portfolio accounting, "
        "and deterministic audit reconstruction."
    )

    audit_runs = list_audit_runs(str(AUDIT_DB_PATH))

    if not AUDIT_DB_PATH.exists():
        st.warning(
            "Audit database does not exist yet. "
            "Run a backtest to create the persistent audit database."
        )

    elif not audit_table_exists("runs"):
        st.error("SQLite database exists, but the `runs` table is missing.")

    elif not audit_runs:
        st.info("No persisted audit runs found.")

    else:
        run_ids = [str(run.get("run_id")) for run in audit_runs if run.get("run_id")]

        selected_result_stem = None

        if st.session_state.selected_run:
            selected_result_stem = Path(st.session_state.selected_run).stem

        default_index = (
            run_ids.index(selected_result_stem)
            if selected_result_stem in run_ids
            else 0
        )

        selected_audit_run = st.selectbox(
            "Audit run",
            run_ids,
            index=default_index,
            key="audit_run_selector",
        )

        audit_data = load_audit_run(
            str(AUDIT_DB_PATH),
            selected_audit_run,
        )

        if not audit_data:
            st.error("Unable to load selected audit run.")

        else:
            run = audit_data["run"]

            traces = [normalize_audit_trace(trace) for trace in audit_data["traces"]]

            status = run.get(
                "status",
                "unknown",
            )

            starting_equity = run.get("starting_equity")
            final_equity = run.get("final_equity")
            realized_pnl = run.get("realized_pnl")
            benchmark_return = run.get("benchmark_return")

            trace_count = run.get("trade_trace_count")

            if trace_count is None:
                trace_count = len(traces)

            # ------------------------------------------------
            # RUN SUMMARY
            # ------------------------------------------------

            summary = [
                (
                    "Status",
                    str(status).upper(),
                    (
                        "green"
                        if str(status).lower()
                        in {
                            "completed",
                            "success",
                            "finished",
                        }
                        else ""
                    ),
                ),
                (
                    "Starting equity",
                    fmt_money(starting_equity),
                    "",
                ),
                (
                    "Final equity",
                    fmt_money(final_equity),
                    "",
                ),
                (
                    "Realized P&L",
                    fmt_money(realized_pnl),
                    (
                        "green"
                        if (
                            safe_float(realized_pnl) is not None
                            and safe_float(realized_pnl) >= 0
                        )
                        else "red"
                    ),
                ),
                (
                    "Benchmark",
                    fmt_pct(benchmark_return),
                    "",
                ),
                (
                    "Trade traces",
                    trace_count,
                    "",
                ),
            ]

            display_html(
                '<div class="kpi-grid">'
                + "".join(
                    f"""
                    <div class="kpi">
                        <div class="label">
                            {esc(label)}
                        </div>
                        <div class="value {cls}">
                            {esc(value)}
                        </div>
                    </div>
                    """
                    for label, value, cls in summary
                )
                + "</div>"
            )
            # ------------------------------------------------
            # JSON ↔ SQLITE RECONCILIATION
            # ------------------------------------------------

            st.markdown("#### Persistence reconciliation")

            json_path = BACKTEST_RESULTS_DIR / f"{selected_audit_run}.json"

            json_run = None

            if json_path.exists():
                json_run = load_run(json_path.name)

            json_trace_count = len(json_run.get("tick_log", [])) if json_run else None

            json_diagnostics = json_run.get("diagnostics", {}) if json_run else {}

            json_diagnostic_trace_count = (
                json_diagnostics.get("trade_trace_count") if json_diagnostics else None
            )

            json_trace_coverage = (
                json_diagnostics.get("trace_coverage_complete")
                if json_diagnostics
                else None
            )

            sqlite_trace_count = len(traces)

            if json_run is None:
                reconciliation_status = "JSON RESULT NOT FOUND"
                reconciliation_class = "audit-warning"

            elif json_trace_count == sqlite_trace_count:
                reconciliation_status = "TRACE COUNTS MATCH"
                reconciliation_class = "audit-ok"

            else:
                reconciliation_status = "TRACE COUNTS DIFFER"
                reconciliation_class = "audit-warning"

            display_html(
                f"""
                <div class="reason-box">
                    <strong>Run:</strong>
                    {esc(selected_audit_run)}
                    <br/>
                    <strong>JSON result:</strong>
                    {esc("FOUND" if json_run else "NOT FOUND")}
                    <br/>
                    <strong>JSON tick records:</strong>
                    {esc(json_trace_count if json_trace_count is not None else "—")}
                    <br/>
                    <strong>SQLite TradeTrace records:</strong>
                    {esc(sqlite_trace_count)}
                    <br/>
                    <strong>JSON diagnostic traces:</strong>
                    {
                    esc(
                        json_diagnostic_trace_count
                        if json_diagnostic_trace_count is not None
                        else "—"
                    )
                }
                    <br/>
                    <strong>JSON trace coverage:</strong>
                    {
                    esc(
                        "COMPLETE"
                        if json_trace_coverage is True
                        else ("INCOMPLETE" if json_trace_coverage is False else "—")
                    )
                }
                    <br/>
                    <strong>Reconciliation:</strong>
                    <span class="{reconciliation_class}">
                        {esc(reconciliation_status)}
                    </span>
                </div>
                """
            )

            st.caption(
                "JSON reconciliation is optional. "
                "SQLite reconstruction below is the persistent "
                "audit source for this run."
            )

            # ------------------------------------------------
            # ORCHESTRATION DIAGNOSTICS
            # ------------------------------------------------

            st.markdown("#### Orchestration diagnostics")

            diagnostics = json_diagnostics

            if diagnostics:
                outcome_counts = diagnostics.get("outcome_counts") or {}

                decision_reason_counts = diagnostics.get("decision_reason_counts") or {}

                diagnostic_summary = [
                    (
                        "Ticks",
                        diagnostics.get("tick_count", 0),
                    ),
                    (
                        "Trade traces",
                        diagnostics.get("trade_trace_count", 0),
                    ),
                    (
                        "Trace tickers",
                        diagnostics.get("trade_trace_tickers", 0),
                    ),
                    (
                        "Risk approvals",
                        diagnostics.get("risk_approvals", 0),
                    ),
                    (
                        "Local rejections",
                        diagnostics.get("local_rejections", 0),
                    ),
                    (
                        "Coordinator approvals",
                        diagnostics.get("coordinator_approvals", 0),
                    ),
                    (
                        "Book rejections",
                        diagnostics.get("book_rejections", 0),
                    ),
                    (
                        "Execution attempts",
                        diagnostics.get("execution_attempts", 0),
                    ),
                    (
                        "Execution successes",
                        diagnostics.get("execution_successes", 0),
                    ),
                    (
                        "Execution failures",
                        diagnostics.get("execution_failures", 0),
                    ),
                    (
                        "Execution events",
                        diagnostics.get("execution_events", 0),
                    ),
                    (
                        "Closed trades",
                        diagnostics.get("closed_trades", 0),
                    ),
                ]

                display_html(
                    '<div class="kpi-grid">'
                    + "".join(
                        f"""
                        <div class="kpi">
                            <div class="label">
                                {esc(label)}
                            </div>
                            <div class="value">
                                {esc(value)}
                            </div>
                        </div>
                        """
                        for label, value in diagnostic_summary
                    )
                    + "</div>"
                )

                trace_coverage = diagnostics.get("trace_coverage_complete")

                if trace_coverage is True:
                    st.success(
                        "Trace coverage complete: every persisted "
                        "backtest tick has a TradeTrace."
                    )

                elif trace_coverage is False:
                    st.warning(
                        "Trace coverage incomplete: the number of "
                        "TradeTrace records does not match the tick count."
                    )

                d1, d2 = st.columns(2)

                with d1:
                    st.markdown("**Outcome distribution**")

                    outcome_rows = [
                        {
                            "Outcome": outcome_label(outcome),
                            "Count": count,
                        }
                        for outcome, count in sorted(outcome_counts.items())
                    ]

                    if outcome_rows:
                        st.table(outcome_rows)
                    else:
                        st.caption("No outcome diagnostics recorded.")

                with d2:
                    st.markdown("**Decision reasons**")

                    reason_rows = [
                        {
                            "Decision reason": str(reason),
                            "Count": count,
                        }
                        for reason, count in sorted(decision_reason_counts.items())
                    ]

                    if reason_rows:
                        st.table(reason_rows)
                    else:
                        st.caption("No decision-reason diagnostics recorded.")

            else:
                st.info(
                    "No orchestration diagnostics were recorded "
                    "in the persisted JSON result."
                )

            # ------------------------------------------------
            # PHASE-6 AUDIT RECONSTRUCTION
            # ------------------------------------------------
            # ------------------------------------------------
            # PHASE-6 AUDIT RECONSTRUCTION
            # ------------------------------------------------

            st.markdown("#### Audit reconstruction")

            try:
                from storage.run_audit import RunAuditStore

                reconstructed_audit = RunAuditStore().get_run_audit(selected_audit_run)

                reconstruction_valid = bool(reconstructed_audit.get("valid"))

                reconstruction_summary = reconstructed_audit.get("summary") or {}

                reconstruction_invariants = reconstructed_audit.get("invariants") or {}

                reconstruction_diagnostics = (
                    reconstructed_audit.get("diagnostics") or {}
                )

                reconstruction_status = "VALID" if reconstruction_valid else "INVALID"

                reconstruction_class = (
                    "audit-ok" if reconstruction_valid else "audit-warning"
                )

                # --------------------------------------------
                # RECONSTRUCTION STATUS
                # --------------------------------------------

                display_html(
                    f"""
                    <div class="reason-box">
                        <strong>SQLite reconstruction:</strong>
                        <span class="{reconstruction_class}">
                            {esc(reconstruction_status)}
                        </span>
                        <br/>
                        <strong>Source:</strong>
                        runs → trade_traces → trades
                        <br/>
                        <strong>Trace records:</strong>
                        {
                        esc(
                            reconstruction_summary.get(
                                "trace_count",
                                0,
                            )
                        )
                    }
                        <br/>
                        <strong>Successful executions:</strong>
                        {
                        esc(
                            reconstruction_summary.get(
                                "successful_executions",
                                0,
                            )
                        )
                    }
                        <br/>
                        <strong>Normalized trades:</strong>
                        {
                        esc(
                            reconstruction_summary.get(
                                "normalized_trades",
                                0,
                            )
                        )
                    }
                        <br/>
                        <strong>Rejected / unexecuted:</strong>
                        {
                        esc(
                            reconstruction_summary.get(
                                "rejected_or_unexecuted",
                                0,
                            )
                        )
                    }
                    </div>
                    """
                )

                # --------------------------------------------
                # INVARIANTS + DIAGNOSTICS
                # --------------------------------------------

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Reconstruction invariants**")

                    invariant_rows = [
                        {
                            "Invariant": key,
                            "Result": ("PASS" if value is True else "FAIL"),
                        }
                        for key, value in reconstruction_invariants.items()
                    ]

                    if invariant_rows:
                        st.table(invariant_rows)

                    else:
                        st.caption("No reconstruction invariants returned.")

                with c2:
                    st.markdown("**Reconstruction diagnostics**")

                    diagnostic_rows = [
                        {
                            "Metric": key,
                            "Value": str(value),
                        }
                        for key, value in reconstruction_diagnostics.items()
                    ]

                    if diagnostic_rows:
                        st.table(diagnostic_rows)

                    else:
                        st.caption("No reconstruction diagnostics.")

                # --------------------------------------------
                # NORMALIZED TRADE LEDGER
                # --------------------------------------------

                reconstructed_trades = reconstructed_audit.get("trades") or []

                if reconstructed_trades:
                    st.markdown("**Normalized trade ledger**")

                    trade_rows = [
                        {
                            "Ticker": trade.get(
                                "ticker",
                                "—",
                            ),
                            "Side": str(
                                trade.get(
                                    "side",
                                    "—",
                                )
                            ).upper(),
                            "Quantity": fmt_shares(trade.get("qty")),
                            "Price": fmt_money(trade.get("price")),
                            "Order ID": trade.get(
                                "order_id",
                                "—",
                            ),
                            "Status": trade.get(
                                "status",
                                "—",
                            ),
                            "Mode": trade.get(
                                "mode",
                                "—",
                            ),
                        }
                        for trade in reconstructed_trades
                    ]

                    st.table(trade_rows)

                else:
                    st.caption("No normalized trades were reconstructed for this run.")

                # --------------------------------------------
                # RAW RECONSTRUCTION
                # --------------------------------------------

                with st.expander("Raw reconstructed audit"):
                    st.json(reconstructed_audit)

            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                st.error("SQLite audit reconstruction failed: " + str(exc))
            # ------------------------------------------------
            # RUN METADATA
            # ------------------------------------------------

            with st.expander("Run metadata"):
                metadata = parse_run_metadata(run)

                if metadata:
                    st.json(metadata)
                else:
                    st.caption("No run metadata recorded.")

            # ------------------------------------------------
            # RUN LIFECYCLE
            # ------------------------------------------------

            st.markdown("#### Run lifecycle")

            lifecycle_rows = [
                {
                    "Field": "Run ID",
                    "Value": run.get(
                        "run_id",
                        "—",
                    ),
                },
                {
                    "Field": "Status",
                    "Value": run.get(
                        "status",
                        "—",
                    ),
                },
                {
                    "Field": "Started",
                    "Value": run.get(
                        "started_at",
                        "—",
                    ),
                },
                {
                    "Field": "Finished",
                    "Value": run.get(
                        "finished_at",
                        "—",
                    ),
                },
                {
                    "Field": "Start date",
                    "Value": run.get(
                        "start_date",
                        "—",
                    ),
                },
                {
                    "Field": "End date",
                    "Value": run.get(
                        "end_date",
                        "—",
                    ),
                },
                {
                    "Field": "Tickers",
                    "Value": run.get(
                        "tickers",
                        "—",
                    ),
                },
            ]

            st.table(lifecycle_rows)

            # ------------------------------------------------
            # TRACE FILTERS
            # ------------------------------------------------

            st.markdown("#### Decision traces")

            if not traces:
                st.info("No TradeTrace records were persisted for this run.")

            else:
                tickers = sorted(
                    {
                        str(trace.get("ticker"))
                        for trace in traces
                        if trace.get("ticker")
                    }
                )

                outcomes = sorted(
                    {
                        str(trace.get("outcome"))
                        for trace in traces
                        if trace.get("outcome")
                    }
                )

                c1, c2 = st.columns(2)

                with c1:
                    trace_ticker = st.selectbox(
                        "Ticker",
                        ["All"] + tickers,
                        key="audit_ticker_filter",
                    )

                with c2:
                    trace_outcome = st.selectbox(
                        "Outcome",
                        ["All"] + outcomes,
                        key="audit_outcome_filter",
                        format_func=outcome_label,
                    )

                filtered_traces = [
                    trace
                    for trace in traces
                    if (trace_ticker == "All" or trace.get("ticker") == trace_ticker)
                    and (
                        trace_outcome == "All" or trace.get("outcome") == trace_outcome
                    )
                ]

                st.caption(f"{len(filtered_traces)} persisted trace(s)")

                rows = [
                    {
                        "Ticker": trace.get(
                            "ticker",
                            "—",
                        ),
                        "Tick": trace.get(
                            "tick_id",
                            "—",
                        ),
                        "Date": trace.get(
                            "simulated_date",
                            "—",
                        ),
                        "Signal": str(
                            trace.get(
                                "signal_direction",
                                "—",
                            )
                            or "—"
                        ).upper(),
                        "Confidence": fmt_num(
                            trace.get("signal_confidence"),
                            3,
                        ),
                        "Risk": str(
                            trace.get(
                                "risk_decision",
                                "—",
                            )
                        ),
                        "Shares": fmt_shares(trace.get("final_shares")),
                        "Execution": (trace.get("execution_side") or "—"),
                        "Outcome": outcome_label(trace.get("outcome")),
                    }
                    for trace in filtered_traces
                ]

                st.table(rows)

                if filtered_traces:
                    selected_trace_index = st.selectbox(
                        "Inspect persisted trace",
                        range(len(filtered_traces)),
                        format_func=lambda i: (
                            f"{filtered_traces[i].get('ticker', '—')} · "
                            f"{filtered_traces[i].get('simulated_date', '—')} · "
                            f"{outcome_label(filtered_traces[i].get('outcome'))}"
                        ),
                        key="audit_trace_selector",
                    )

                    trace = filtered_traces[selected_trace_index]

                    # ------------------------------------------------
                    # DECISION LINEAGE
                    # ------------------------------------------------

                    st.markdown("#### Decision lineage")

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.markdown("**SignalMerger**")

                        st.metric(
                            "Direction",
                            str(
                                trace.get(
                                    "signal_direction",
                                    "—",
                                )
                            ).upper(),
                        )

                        st.caption(
                            "Confidence: "
                            + fmt_num(
                                trace.get("signal_confidence"),
                                3,
                            )
                        )

                    with c2:
                        st.markdown("**RiskAgent**")

                        st.metric(
                            "Approved",
                            str(
                                trace.get(
                                    "risk_approved",
                                    "—",
                                )
                            ),
                        )

                        st.caption(
                            "Raw: "
                            + fmt_shares(trace.get("raw_shares"))
                            + " · Proposed: "
                            + fmt_shares(trace.get("proposed_shares"))
                        )

                    with c3:
                        st.markdown("**Coordinator**")

                        st.metric(
                            "Shares",
                            fmt_shares(trace.get("final_shares")),
                        )

                        st.caption(
                            "Approved: "
                            + str(
                                trace.get(
                                    "coordinator_approved",
                                    "—",
                                )
                            )
                        )

                    with c4:
                        st.markdown("**Execution**")

                        st.metric(
                            "Side",
                            str(
                                trace.get(
                                    "execution_side",
                                    "—",
                                )
                            ).upper(),
                        )

                        st.caption("Success: " + str(trace.get("execution_success")))

                    # ------------------------------------------------
                    # AGENT EVIDENCE
                    # ------------------------------------------------

                    st.markdown("#### Agent evidence")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown("**NewsAgent**")

                        news_rows = [
                            {
                                "Field": "Direction",
                                "Value": str(
                                    first_value(
                                        trace,
                                        "news_direction",
                                        default="—",
                                    )
                                ).upper(),
                            },
                            {
                                "Field": "Confidence",
                                "Value": fmt_num(
                                    first_value(
                                        trace,
                                        "news_confidence",
                                    ),
                                    3,
                                ),
                            },
                            {
                                "Field": "Availability",
                                "Value": first_value(
                                    trace,
                                    "news_availability",
                                    default="—",
                                ),
                            },
                            {
                                "Field": "Article count",
                                "Value": first_value(
                                    trace,
                                    "news_article_count",
                                    default="—",
                                ),
                            },
                            {
                                "Field": "Source",
                                "Value": first_value(
                                    trace,
                                    "news_source",
                                    default="—",
                                ),
                            },
                        ]

                        st.table(news_rows)

                    with c2:
                        st.markdown("**ChartAgent**")

                        chart_rows = [
                            {
                                "Field": "Direction",
                                "Value": str(
                                    first_value(
                                        trace,
                                        "chart_direction",
                                        default="—",
                                    )
                                ).upper(),
                            },
                            {
                                "Field": "Confidence",
                                "Value": fmt_num(
                                    first_value(
                                        trace,
                                        "chart_confidence",
                                    ),
                                    3,
                                ),
                            },
                        ]

                        st.table(chart_rows)

                    # ------------------------------------------------
                    # EXECUTION DETAILS
                    # ------------------------------------------------

                    st.markdown("#### Execution details")

                    execution_rows = [
                        {
                            "Field": "Attempted",
                            "Value": str(trace.get("execution_attempted")),
                        },
                        {
                            "Field": "Success",
                            "Value": str(trace.get("execution_success")),
                        },
                        {
                            "Field": "Side",
                            "Value": trace.get(
                                "execution_side",
                                "—",
                            ),
                        },
                        {
                            "Field": "Quantity",
                            "Value": fmt_shares(trace.get("execution_quantity")),
                        },
                        {
                            "Field": "Price",
                            "Value": fmt_money(trace.get("execution_price")),
                        },
                        {
                            "Field": "Order ID",
                            "Value": trace.get(
                                "order_id",
                                "—",
                            ),
                        },
                    ]

                    st.table(execution_rows)

                    # ------------------------------------------------
                    # ACCOUNTING
                    # ------------------------------------------------

                    st.markdown("#### Portfolio accounting")

                    accounting_rows = [
                        {
                            "Field": "Average cost",
                            "Value": fmt_money(trace.get("average_cost")),
                        },
                        {
                            "Field": "Realized P&L",
                            "Value": fmt_money(trace.get("realized_pnl")),
                        },
                        {
                            "Field": "Remaining shares",
                            "Value": fmt_shares(trace.get("remaining_shares")),
                        },
                        {
                            "Field": "Position closed",
                            "Value": str(trace.get("position_closed")),
                        },
                    ]

                    st.table(accounting_rows)

                    # ------------------------------------------------
                    # RISK CHECKS
                    # ------------------------------------------------

                    risk_checks = trace.get("risk_checks") or []

                    if risk_checks:
                        with st.expander(f"Risk checks ({len(risk_checks)})"):
                            for check in risk_checks:
                                st.caption("• " + str(check))

                    # ------------------------------------------------
                    # RAW TRACE
                    # ------------------------------------------------

                    with st.expander("Raw persisted TradeTrace"):
                        st.json(trace)

                    # ------------------------------------------------
                    # AUDIT INTERPRETATION
                    # ------------------------------------------------

                    st.markdown("#### Audit interpretation")

                    execution_attempted = bool_or_none(trace.get("execution_attempted"))

                    execution_success = bool_or_none(trace.get("execution_success"))

                    coordinator_approved = bool_or_none(
                        trace.get("coordinator_approved")
                    )

                    if execution_success is True:
                        audit_interpretation = (
                            "Execution succeeded and the "
                            "execution event is represented in "
                            "the normalized trade ledger."
                        )

                    elif execution_attempted is True:
                        audit_interpretation = (
                            "Execution was attempted but did "
                            "not produce a successful execution."
                        )

                    elif coordinator_approved is False:
                        audit_interpretation = (
                            "The coordinator blocked the decision "
                            "before execution. No broker execution "
                            "should be inferred from this trace."
                        )

                    else:
                        audit_interpretation = (
                            "The persisted trace records a decision "
                            "that did not result in a successful "
                            "execution."
                        )

                    st.info(audit_interpretation)

# ============================================================
# HISTORICAL LIFECYCLE DEMO
# ============================================================
# ============================================================
# HISTORICAL LIFECYCLE DEMO
# ============================================================

if "lifecycle_result" not in st.session_state:
    st.session_state.lifecycle_result = None

if "lifecycle_run_id" not in st.session_state:
    st.session_state.lifecycle_run_id = None

if "lifecycle_ticker" not in st.session_state:
    st.session_state.lifecycle_ticker = None

if "lifecycle_start" not in st.session_state:
    st.session_state.lifecycle_start = None

if "lifecycle_end" not in st.session_state:
    st.session_state.lifecycle_end = None


with st.expander(
    "📈 Historical Lifecycle Replay",
    expanded=False,
):
    st.markdown("### 📈 Historical Lifecycle Replay")

    st.caption(
        "Recruiter/demo proof of a stateful trading lifecycle. "
        "Historical sessions are replayed through the existing "
        "backtest runner and canonical run_tick() orchestration. "
        "The UI does not implement a second execution path."
    )

    display_html(
        """
        <div class="demo-banner">
            <div class="demo-banner-title">
                STATEFUL TRADING LIFECYCLE
            </div>

            <div class="demo-banner-text">
                Replay historical sessions and inspect the complete
                position lifecycle: BUY → OPEN → SELL → CLOSED,
                together with accounting state, realized P&L,
                and TradeTrace telemetry.
            </div>
        </div>
        """
    )

    lifecycle_c1, lifecycle_c2, lifecycle_c3 = st.columns(3)

    with lifecycle_c1:
        lifecycle_ticker = st.selectbox(
            "Lifecycle ticker",
            [
                "AAPL",
                "MSFT",
                "GOOGL",
                "JPM",
            ],
            key="lifecycle_ticker_selector",
        )

    with lifecycle_c2:
        lifecycle_start = st.date_input(
            "Start date",
            value=date(2024, 6, 3),
            key="lifecycle_start_date",
        )

    with lifecycle_c3:
        lifecycle_end = st.date_input(
            "End date",
            value=date(2024, 6, 10),
            key="lifecycle_end_date",
        )

    if lifecycle_start > lifecycle_end:
        st.error("Start date must be on or before end date.")

    else:
        st.caption(
            "Historical replay only exposes data available as of each "
            "simulated session. Future bars are not exposed to earlier "
            "sessions."
        )

        if st.button(
            "▶ Run Historical Lifecycle",
            width="stretch",
            type="primary",
            key="run_historical_lifecycle",
        ):
            try:
                with st.spinner(
                    f"Replaying {lifecycle_ticker} historical lifecycle..."
                ):
                    (
                        lifecycle_run_id,
                        lifecycle_result,
                    ) = run_historical_lifecycle_demo(
                        ticker=lifecycle_ticker,
                        start_date=lifecycle_start,
                        end_date=lifecycle_end,
                    )

                st.session_state.lifecycle_run_id = lifecycle_run_id
                st.session_state.lifecycle_result = lifecycle_result
                st.session_state.lifecycle_ticker = lifecycle_ticker
                st.session_state.lifecycle_start = lifecycle_start
                st.session_state.lifecycle_end = lifecycle_end

                st.success(f"Lifecycle replay completed: `{lifecycle_run_id}`")

            except Exception as exc:  # noqa: BLE001
                if LOGFIRE_OK:
                    logfire.exception(
                        "Historical lifecycle demo failed",
                        error_type=type(exc).__name__,
                    )

                st.error("Historical lifecycle replay failed.")

                with st.expander("Technical error"):
                    st.exception(exc)

    lifecycle_result = st.session_state.get("lifecycle_result")

    if lifecycle_result is not None:
        st.divider()

        tick_log = list(
            getattr(
                lifecycle_result,
                "tick_log",
                [],
            )
            or []
        )

        closed_trades = list(
            getattr(
                lifecycle_result,
                "closed_trades",
                [],
            )
            or []
        )

        trade_traces = list(
            getattr(
                lifecycle_result,
                "trade_traces",
                [],
            )
            or []
        )

        final_equity = getattr(
            lifecycle_result,
            "final_equity",
            None,
        )

        final_cash = getattr(
            lifecycle_result,
            "final_cash",
            None,
        )

        realized_pnl = getattr(
            lifecycle_result,
            "final_realized_pnl",
            None,
        )

        # --------------------------------------------------------
        # REPLAY IDENTITY
        # --------------------------------------------------------

        st.markdown("#### Replay identity")

        identity_rows = [
            {
                "Property": "Run ID",
                "Value": st.session_state.get("lifecycle_run_id") or "—",
            },
            {
                "Property": "Ticker",
                "Value": st.session_state.get("lifecycle_ticker") or "—",
            },
            {
                "Property": "Start date",
                "Value": str(st.session_state.get("lifecycle_start") or "—"),
            },
            {
                "Property": "End date",
                "Value": str(st.session_state.get("lifecycle_end") or "—"),
            },
            {
                "Property": "Replay boundary",
                "Value": "simulated_date",
            },
        ]

        st.dataframe(
            identity_rows,
            width="stretch",
            hide_index=True,
        )

        # --------------------------------------------------------
        # LIFECYCLE SUMMARY
        # --------------------------------------------------------

        st.markdown("#### Lifecycle summary")

        successful_executions = sum(
            1
            for entry in tick_log
            if isinstance(entry, dict) and entry.get("execution_success") is True
        )

        buy_events = sum(
            1
            for entry in tick_log
            if isinstance(entry, dict)
            and str(entry.get("execution_side") or "").lower() == "buy"
            and entry.get("execution_success") is True
        )

        sell_events = sum(
            1
            for entry in tick_log
            if isinstance(entry, dict)
            and str(entry.get("execution_side") or "").lower() == "sell"
            and entry.get("execution_success") is True
        )

        summary_c1, summary_c2, summary_c3, summary_c4, summary_c5 = st.columns(5)

        with summary_c1:
            st.metric(
                "Historical ticks",
                len(tick_log),
            )

        with summary_c2:
            st.metric(
                "Successful executions",
                successful_executions,
            )

        with summary_c3:
            st.metric(
                "BUY events",
                buy_events,
            )

        with summary_c4:
            st.metric(
                "SELL events",
                sell_events,
            )

        with summary_c5:
            st.metric(
                "Closed trades",
                len(closed_trades),
            )

        # --------------------------------------------------------
        # LIFECYCLE STATUS
        # --------------------------------------------------------

        has_buy = buy_events > 0
        has_sell = sell_events > 0
        has_closed_trade = len(closed_trades) > 0

        if has_buy and has_sell and has_closed_trade:
            st.success("Complete lifecycle observed: BUY → OPEN → SELL → CLOSED.")

        elif has_buy and not has_sell:
            st.warning(
                "A BUY execution was observed, but this replay window "
                "did not produce a completed SELL/CLOSED lifecycle."
            )

        elif has_sell and not has_buy:
            st.warning(
                "A SELL execution was observed, but no BUY/open position "
                "was observed within this replay window."
            )

        else:
            st.info(
                "This replay window did not produce a complete "
                "BUY → OPEN → SELL → CLOSED lifecycle."
            )

        # --------------------------------------------------------
        # POSITION LIFECYCLE
        # --------------------------------------------------------

        st.markdown("#### Position lifecycle")

        lifecycle_rows = []

        for entry in tick_log:
            if not isinstance(entry, dict):
                continue

            side = entry.get("execution_side")

            remaining_shares = safe_float(entry.get("remaining_shares"))

            position_state = entry.get("position_state")

            if not position_state:
                if entry.get("position_closed"):
                    position_state = "CLOSED"

                elif remaining_shares is not None and remaining_shares > 0:
                    position_state = "OPEN"

                else:
                    position_state = "NO POSITION"

            lifecycle_rows.append(
                {
                    "Date": entry.get(
                        "date",
                        "—",
                    ),
                    "Ticker": entry.get(
                        "ticker",
                        "—",
                    ),
                    "Action": (str(side).upper() if side else "HOLD"),
                    "Outcome": entry.get(
                        "outcome",
                        "—",
                    ),
                    "Price": (
                        fmt_money(entry.get("price"))
                        if entry.get("price") is not None
                        else "—"
                    ),
                    "Shares": fmt_shares(entry.get("shares")),
                    "Remaining": fmt_shares(entry.get("remaining_shares")),
                    "Position": position_state,
                    "Realized P&L": fmt_money(entry.get("realized_pnl")),
                }
            )

        if lifecycle_rows:
            st.dataframe(
                lifecycle_rows,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info("No lifecycle tick telemetry was produced.")

        # --------------------------------------------------------
        # CLOSED TRADE EVENTS
        # --------------------------------------------------------

        st.markdown("#### Closed trade events")

        if closed_trades:
            closed_rows = []

            for trade in closed_trades:
                if not isinstance(trade, dict):
                    continue

                closed_rows.append(
                    {
                        "Date": trade.get(
                            "date",
                            "—",
                        ),
                        "Ticker": trade.get(
                            "ticker",
                            "—",
                        ),
                        "Side": (
                            str(trade.get("side")).upper() if trade.get("side") else "—"
                        ),
                        "Shares": fmt_shares(trade.get("shares")),
                        "Exit price": fmt_money(trade.get("exit_price")),
                        "Average cost": fmt_money(trade.get("average_cost")),
                        "Realized P&L": fmt_money(trade.get("realized_pnl")),
                    }
                )

            if closed_rows:
                st.dataframe(
                    closed_rows,
                    width="stretch",
                    hide_index=True,
                )

        else:
            st.caption("No closed trade event was recorded in this replay.")

        # --------------------------------------------------------
        # ACCOUNTING PROOF
        # --------------------------------------------------------

        st.markdown("#### Accounting proof")

        accounting_rows = [
            {
                "Property": "Final cash",
                "Value": fmt_money(final_cash),
            },
            {
                "Property": "Final equity",
                "Value": fmt_money(final_equity),
            },
            {
                "Property": "Realized P&L",
                "Value": fmt_money(realized_pnl),
            },
            {
                "Property": "Closed trades",
                "Value": len(closed_trades),
            },
            {
                "Property": "TradeTrace events",
                "Value": len(trade_traces),
            },
            {
                "Property": "Historical boundary",
                "Value": "simulated_date",
            },
        ]

        st.dataframe(
            accounting_rows,
            width="stretch",
            hide_index=True,
        )

        # --------------------------------------------------------
        # RAW REPLAY RESULT
        # --------------------------------------------------------

        with st.expander("Raw lifecycle replay result"):
            st.json(
                {
                    "run_id": st.session_state.get("lifecycle_run_id"),
                    "tick_log": tick_log,
                    "closed_trades": closed_trades,
                    "trade_traces": trade_traces,
                    "final_equity": final_equity,
                    "final_cash": final_cash,
                    "final_realized_pnl": realized_pnl,
                }
            )

        st.caption(
            "This section is observational only. It surfaces the "
            "existing backtest result, TradeTrace telemetry, position "
            "state, and accounting outputs without implementing a "
            "second trading execution path."
        )

# ============================================================
# DATA & SYSTEM
# ============================================================

with tab_data:
    st.markdown("### Data quality")

    if backtest_runs:
        selected_run = st.selectbox(
            "Run",
            backtest_runs,
            key="data_run_selector",
        )

        run_data = load_run(selected_run)

        if run_data:
            quality = extract_news_quality(run_data)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "News coverage",
                fmt_pct(quality["coverage"]),
            )

            c2.metric(
                "Available",
                quality["available"],
            )

            c3.metric(
                "Unavailable",
                quality["unavailable"],
            )

            c4.metric(
                "Errors",
                quality["error"],
            )

            if quality["by_ticker"]:
                rows = [
                    {
                        "Ticker": ticker,
                        "Historical news": NEWS_LABELS.get(
                            status,
                            status,
                        ),
                    }
                    for ticker, status in sorted(quality["by_ticker"].items())
                ]

                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )

            if quality["unavailable"]:
                st.warning(
                    "Unavailable historical news means the "
                    "historical source/cache was not available. "
                    "It should not be interpreted as proof that "
                    "no news existed."
                )

    st.divider()

    st.markdown("### System components")

    components = [
        (
            "LangGraph",
            "Multi-agent orchestration",
        ),
        (
            "NewsAgent",
            "News-driven directional signal",
        ),
        (
            "ChartAgent",
            "Deterministic technical signal",
        ),
        (
            "SignalMerger",
            "Deterministic signal combination",
        ),
        (
            "RiskAgent",
            "ATR sizing + local portfolio constraints",
        ),
        (
            "PortfolioRiskCoordinator",
            "Cross-ticker book-level constraints",
        ),
        (
            "ExecutionAgent",
            "Broker abstraction + execution",
        ),
        (
            "PortfolioLedger",
            "Local portfolio source of truth",
        ),
        (
            "SimBroker",
            "Historical/paper execution",
        ),
        (
            "AlpacaBroker",
            "Live broker implementation",
        ),
        (
            "Historical Replay",
            "No-lookahead OHLCV/news replay",
        ),
        (
            "TradeTrace",
            "Deterministic decision lineage",
        ),
        (
            "SQLite Run Audit",
            "Persistent run + trade-trace observability",
        ),
    ]

    st.dataframe(
        [
            {
                "Component": name,
                "Responsibility": responsibility,
            }
            for name, responsibility in components
        ],
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.markdown("### Raw state")

    if raw_ledger:
        with st.expander("portfolio/ledger.json"):
            st.json(raw_ledger)

    if backtest_runs:
        selected_run = st.selectbox(
            "Raw result",
            backtest_runs,
            key="raw_run_selector",
        )

        raw_run = load_run(selected_run)

        if raw_run:
            with st.expander("Backtest JSON"):
                st.json(raw_run)

    if AUDIT_DB_PATH.exists():
        with st.expander("SQLite audit database"):
            st.write(f"Path: `{AUDIT_DB_PATH}`")

            counts = audit_table_counts(str(AUDIT_DB_PATH))

            if counts:
                st.dataframe(
                    [
                        {
                            "Table": table,
                            "Rows": count,
                        }
                        for table, count in counts.items()
                    ],
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.warning("No audit tables detected.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

display_html(
    """
    <div style="
        color:#626d7e;
        font-size:10px;
        padding:8px 0 20px;
    ">
        AI Multi-Agent Trading Firm · LangGraph orchestration ·
        deterministic risk gating · historical replay ·
        SimBroker / Alpaca · TradeTrace · SQLite persistent audit
    </div>
    """
)
