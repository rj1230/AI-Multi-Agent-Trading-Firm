"""
AI Multi-Agent Trading Firm · Command Console

Frontend for the multi-agent trading system.

Architecture:

    NewsAgent ─┐
               ▼
          SignalMerger
               │            (per ticker, concurrently)
    ChartAgent ┘
               │
               ▼
        RiskAgent (local)
               │
               ▼
    PortfolioRiskCoordinator      (book-level, all tickers this tick)
          /          \
   ExecutionAgent   HoldNode

Data sources:

    LIVE PORTFOLIO
        portfolio/ledger.json

    BACKTEST RUNS
        backtest/results/*.json
"""

from __future__ import annotations

from contextlib import nullcontext
import html
import json
import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st


# ============================================================
# ENVIRONMENT / PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LEDGER_PATH = Path(os.getenv("LEDGER_PATH", "portfolio/ledger.json"))
BACKTEST_RESULTS_DIR = Path(os.getenv("BACKTEST_RESULTS_DIR", "backtest/results"))

TRADING_MODE = os.getenv("TRADING_MODE", "backtest").lower()


# ============================================================
# LOGFIRE
# ============================================================

LOGFIRE_OK = False

try:
    import logfire

    logfire_token = os.getenv("LOGFIRE_TOKEN")

    if logfire_token:
        logfire.configure(token=logfire_token)
        LOGFIRE_OK = True

except Exception:
    pass


def _trace_context():
    if LOGFIRE_OK:
        return logfire.span("Trading console operation")
    return nullcontext()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Multi-Agent Trading Firm",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RENDERING HELPERS
# ============================================================


def display_html(markup: str):
    native = getattr(st, "html", None)

    if callable(native):
        native(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt_money(value, default="—") -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return default


def fmt_pct(value, default="—") -> str:
    if value is None:
        return default

    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return default


def fmt_num(value, decimals=3, default="—") -> str:
    if value is None:
        return default

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def fmt_shares(value, default="—") -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return default


# ============================================================
# NEWS AVAILABILITY HELPERS
# ============================================================


NEWS_STATUS_LABELS = {
    "available": "Available",
    "unavailable": "Unavailable",
    "error": "Error",
}


def normalize_news_status(value) -> str | None:
    """
    Normalize possible persisted availability representations.

    Supports:
        "available"
        "unavailable"
        "error"
        enum-like objects
        dictionaries containing {"availability": "..."}
    """
    if value is None:
        return None

    if isinstance(value, dict):
        value = value.get("availability")

    if hasattr(value, "value"):
        value = value.value

    value = str(value).strip().lower()

    if value in NEWS_STATUS_LABELS:
        return value

    return None


def extract_news_quality(run_data: dict) -> dict:
    """
    Read structured news availability information from a serialized
    backtest result.

    This function is intentionally defensive because the backend may
    evolve from a simple dictionary into a richer summary structure.

    Supported examples:

        "news_availability": {
            "AAPL": "available",
            "MSFT": "available",
            "GOOGL": "unavailable"
        }

    or:

        "news_availability": {
            "summary": {...},
            "by_ticker": {...}
        }

    Returns:

        {
            "by_ticker": {
                "AAPL": "available",
                ...
            },
            "available": int,
            "unavailable": int,
            "error": int,
            "total": int,
            "coverage": float | None
        }
    """

    raw = run_data.get("news_availability")

    if raw is None:
        return {
            "by_ticker": {},
            "available": 0,
            "unavailable": 0,
            "error": 0,
            "total": 0,
            "coverage": None,
        }

    by_ticker = {}

    # --------------------------------------------------------
    # Direct format:
    #
    # {
    #   "AAPL": "available",
    #   "GOOGL": "unavailable"
    # }
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Summary values
    # --------------------------------------------------------

    available = 0
    unavailable = 0
    error = 0
    total = 0
    coverage = None

    if by_ticker:
        available = sum(status == "available" for status in by_ticker.values())

        unavailable = sum(status == "unavailable" for status in by_ticker.values())

        error = sum(status == "error" for status in by_ticker.values())

        total = len(by_ticker)

    if isinstance(raw, dict):
        summary = raw.get("summary")

        if isinstance(summary, dict):
            available = int(summary.get("available", available) or 0)
            unavailable = int(summary.get("unavailable", unavailable) or 0)
            error = int(summary.get("error", error) or 0)
            total = int(summary.get("total", total) or 0)

            coverage = summary.get("coverage")

            if coverage is None:
                coverage = summary.get("coverage_pct")

        else:
            available = int(raw.get("available", available) or 0)
            unavailable = int(raw.get("unavailable", unavailable) or 0)
            error = int(raw.get("error", error) or 0)
            total = int(raw.get("total", total) or 0)

            coverage = raw.get("coverage")

            if coverage is None:
                coverage = raw.get("coverage_pct")

    if total <= 0 and by_ticker:
        total = len(by_ticker)

    if coverage is None and total > 0:
        coverage = available / total

    # Some serializers may persist coverage as 75 instead of 0.75.
    if coverage is not None:
        try:
            coverage = float(coverage)

            if coverage > 1:
                coverage /= 100.0

        except (TypeError, ValueError):
            coverage = None

    return {
        "by_ticker": by_ticker,
        "available": available,
        "unavailable": unavailable,
        "error": error,
        "total": total,
        "coverage": coverage,
    }


def render_news_status_badge(status: str | None) -> str:
    if status == "available":
        return '<span class="tf-news-badge available">● Available</span>'

    if status == "unavailable":
        return '<span class="tf-news-badge unavailable">● Unavailable</span>'

    if status == "error":
        return '<span class="tf-news-badge error">● Error</span>'

    return '<span class="tf-news-badge unknown">● Not reported</span>'


def render_news_quality_section(run_data: dict):
    """
    Render the historical-news quality section.

    Important:
    The UI never infers historical-news availability from the number
    of headlines. It only displays structured backend metadata.
    """

    st.markdown("#### News data quality")

    quality = extract_news_quality(run_data)

    by_ticker = quality["by_ticker"]
    available = quality["available"]
    unavailable = quality["unavailable"]
    error = quality["error"]
    total = quality["total"]
    coverage = quality["coverage"]

    if not by_ticker and total == 0:
        st.info(
            "This backtest result does not contain structured "
            "news-availability metadata yet. "
            "The UI will display it automatically once the backend "
            "persists `news_availability` in the serialized result."
        )
        return

    coverage_text = fmt_pct(coverage) if coverage is not None else "—"

    quality_kpis = [
        ("Coverage", coverage_text, ""),
        ("Available", str(available), "pos"),
        ("Unavailable", str(unavailable), "neg" if unavailable else ""),
        ("Errors", str(error), "neg" if error else ""),
        ("Tickers", str(total), ""),
    ]

    quality_html = "".join(
        f"""
        <div class="tf-kpi">
            <span class="label">{esc(label)}</span>
            <span class="value {cls}">{esc(value)}</span>
        </div>
        """
        for label, value, cls in quality_kpis
    )

    display_html(f'<div class="tf-kpi-row">{quality_html}</div>')

    if unavailable:
        st.warning(
            f"{unavailable} ticker(s) have unavailable historical "
            "news data for this backtest. Their NewsAgent signal "
            "should be treated as neutral rather than as evidence "
            "that no news existed."
        )

    if error:
        st.error(f"{error} ticker(s) reported a historical news-data source error.")

    if by_ticker:
        rows = []

        for ticker in sorted(by_ticker):
            status = by_ticker[ticker]

            rows.append(
                {
                    "Ticker": ticker,
                    "Historical news": NEWS_STATUS_LABELS.get(
                        status,
                        status.title(),
                    ),
                    "Status": status.upper(),
                }
            )

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
        )

    elif total:
        st.caption(
            "The backend reported aggregate news availability, "
            "but no per-ticker breakdown was persisted."
        )


# ============================================================
# PIPELINE STAGES
# ============================================================

PIPELINE_STAGES = [
    ("news", "NewsAgent", "LLM sentiment read"),
    ("chart", "ChartAgent", "Rule-based technicals"),
    ("merger", "SignalMerger", "Deterministic combine"),
    ("risk_local", "RiskAgent (local)", "Sizing + per-ticker gate"),
    (
        "coordinator",
        "PortfolioRiskCoordinator",
        "Book-level gate",
    ),
    ("execution", "Execution / Hold", "Broker fill or block"),
]


def visited_stages_for_outcome(outcome: str | None) -> set[str]:
    """Map a TickResult.outcome onto which pipeline stages were reached."""

    if outcome is None:
        return set()

    visited = {
        "news",
        "chart",
        "merger",
        "risk_local",
    }

    if outcome == "held_local_reject":
        return visited

    visited.add("coordinator")
    visited.add("execution")

    return visited


def render_pipeline_rail(
    outcome: str | None = None,
    active_ticker: str | None = None,
):
    visited = visited_stages_for_outcome(outcome)

    nodes = []

    for index, (key, title, subtitle) in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        active = key in visited
        active_class = " active" if active else ""

        current_subtitle = subtitle

        if key == "execution" and outcome == "executed":
            current_subtitle = "Filled via broker"

        elif key == "execution" and outcome == "held_book_reject":
            current_subtitle = "Blocked at book level"

        elif key == "risk_local" and outcome == "held_local_reject":
            current_subtitle = "Blocked locally"

        nodes.append(
            f"""
            <div class="tf-pipeline-node{active_class}">
                <div class="tf-node-index">{index}</div>
                <div class="tf-node-content">
                    <strong>{esc(title)}</strong>
                    <span>{esc(current_subtitle)}</span>
                </div>
            </div>
            """
        )

    header = ""

    if active_ticker:
        header = f'<div class="tf-pipeline-ticker">{esc(active_ticker)}</div>'

    return f"""
    {header}
    <div class="tf-pipeline">
        {"".join(nodes)}
    </div>
    """


# ============================================================
# DATA LOADING — LIVE LEDGER
# ============================================================


@st.cache_data(ttl=5, show_spinner=False)
def load_live_ledger(path_str: str, mtime: float):
    path = Path(path_str)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def normalize_ledger(raw: dict | None) -> dict:
    if not raw:
        return {
            "equity": None,
            "starting_equity": None,
            "cash": None,
            "positions": {},
            "raw": raw,
        }

    equity = raw.get("equity")

    starting_equity = raw.get("starting_equity") or raw.get("session_starting_equity")

    cash = raw.get("cash")
    positions = raw.get("positions") or {}

    if equity is None and cash is not None and isinstance(positions, dict):
        total_mv = 0.0
        any_mv = False

        for pos in positions.values():
            if isinstance(pos, dict) and "market_value" in pos:
                total_mv += float(pos.get("market_value") or 0.0)
                any_mv = True

        if any_mv:
            equity = float(cash) + total_mv

    return {
        "equity": equity,
        "starting_equity": starting_equity,
        "cash": cash,
        "positions": (positions if isinstance(positions, dict) else {}),
        "raw": raw,
    }


# ============================================================
# DATA LOADING — BACKTEST RESULTS
# ============================================================


@st.cache_data(ttl=5, show_spinner=False)
def list_backtest_runs(dir_str: str):
    directory = Path(dir_str)

    if not directory.exists():
        return []

    files = sorted(
        directory.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return [f.name for f in files]


@st.cache_data(ttl=5, show_spinner=False)
def load_backtest_run(dir_str: str, filename: str):
    path = Path(dir_str) / filename

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

if "last_backtest_run_id" not in st.session_state:
    st.session_state.last_backtest_run_id = None


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>

:root {
    --tf-bg: #0d0f14;
    --tf-panel: #14171f;
    --tf-panel-alt: #191d27;
    --tf-border: #282d39;
    --tf-border-soft: #20242e;

    --tf-text: #e8eaf0;
    --tf-muted: #8a90a3;
    --tf-muted-soft: #656c7e;

    --tf-accent: #5bc0be;
    --tf-accent-strong: #7ed9d7;
    --tf-accent-soft: rgba(91, 192, 190, .14);

    --tf-bull: #6ee7b7;
    --tf-bull-soft: rgba(110, 231, 183, .14);

    --tf-bear: #f0707a;
    --tf-bear-soft: rgba(240, 112, 122, .14);

    --tf-warning: #e0a95c;
    --tf-warning-soft: rgba(224, 169, 92, .14);
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: var(--tf-bg);
    color: var(--tf-text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

@keyframes tfFadeIn {
    from {
        opacity: 0;
        transform: translateY(4px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: var(--tf-panel);
    border-right: 1px solid var(--tf-border);
}

section[data-testid="stSidebar"] > div {
    padding-top: .6rem;
}

section[data-testid="stSidebar"] * {
    color: var(--tf-text);
}

.tf-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 17px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--tf-border);
}

.tf-brand strong {
    display: block;
    color: #ffffff;
    font-family: monospace;
    font-size: 15px;
    font-weight: 700;
}

.tf-brand small {
    display: block;
    margin-top: 3px;
    color: var(--tf-muted);
    font-size: 10.5px;
}

.tf-rail-label {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 18px 0 8px 2px;
    color: var(--tf-muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.tf-rail-label::before {
    content: "";
    width: 3px;
    height: 12px;
    border-radius: 3px;
    background: var(--tf-accent);
}

.tf-status-card {
    padding: 13px 14px;
    border: 1px solid var(--tf-border);
    border-radius: 10px;
    background: var(--tf-panel-alt);
}

.tf-status-card .kv {
    color: var(--tf-muted);
    font-family: monospace;
    font-size: 11px;
    line-height: 2;
}

.tf-status-card .kv b {
    color: var(--tf-text);
    font-weight: 500;
}

.tf-rail-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 22px;
    padding-top: 16px;
    border-top: 1px solid var(--tf-border);
}

.tf-rail-footer strong {
    display: block;
    color: var(--tf-text);
    font-size: 11.5px;
    font-weight: 600;
}

.tf-rail-footer small {
    display: block;
    margin-top: 3px;
    color: var(--tf-muted-soft);
    font-size: 10px;
}

.tf-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    flex: none;
    border-radius: 50%;
    background: var(--tf-muted-soft);
}

.tf-dot.success {
    background: var(--tf-bull);
    box-shadow: 0 0 12px var(--tf-bull-soft);
}

.tf-dot.warning {
    background: var(--tf-warning);
    box-shadow: 0 0 12px var(--tf-warning-soft);
}

.tf-dot.danger {
    background: var(--tf-bear);
    box-shadow: 0 0 12px var(--tf-bear-soft);
}

section[data-testid="stSidebar"] button {
    background: var(--tf-panel-alt) !important;
    border: 1px solid var(--tf-border) !important;
    color: var(--tf-text) !important;
    font-size: 12px !important;
}

section[data-testid="stSidebar"] button:hover {
    border-color: var(--tf-accent) !important;
}

/* TOPBAR */

.tf-topbar {
    padding-top: 4px;
}

.tf-kicker {
    margin-bottom: 7px;
    color: var(--tf-accent-strong);
    font-family: monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
}

.tf-topbar h1 {
    margin: 0 0 8px;
    color: var(--tf-text);
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -.02em;
}

.tf-topbar p {
    max-width: 800px;
    margin: 0;
    color: var(--tf-muted);
    font-size: 13px;
    line-height: 1.6;
}

.tf-engine-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-top: 13px;
    padding: 8px 13px;
    border: 1px solid var(--tf-border);
    border-radius: 999px;
    background: var(--tf-panel);
    color: var(--tf-muted);
    font-family: monospace;
    font-size: 11px;
}

.tf-signal-divider {
    margin: 15px 0 7px;
    color: var(--tf-border);
}

.tf-signal-divider svg {
    display: block;
    width: 100%;
    height: 18px;
}

/* PIPELINE */

.tf-pipeline-ticker {
    display: inline-block;
    margin: 4px 0 10px;
    padding: 4px 10px;
    border: 1px solid var(--tf-border);
    border-radius: 999px;
    background: var(--tf-panel-alt);
    color: var(--tf-accent-strong);
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
}

.tf-pipeline {
    display: flex;
    gap: 0;
    width: 100%;
    margin: 4px 0 22px;
}

.tf-pipeline-node {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 9px;
    flex: 1;
    min-width: 0;
    padding-right: 15px;
    opacity: .35;
}

.tf-pipeline-node.active {
    opacity: 1;
}

.tf-pipeline-node:not(:last-child)::after {
    content: "";
    position: absolute;
    top: 14px;
    left: calc(100% - 7px);
    width: calc(100% - 20px);
    height: 1px;
    background: var(--tf-border);
}

.tf-node-index {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    flex: none;
    border: 1px solid var(--tf-border);
    border-radius: 50%;
    background: var(--tf-panel);
    color: var(--tf-accent-strong);
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
}

.tf-pipeline-node.active .tf-node-index {
    border-color: var(--tf-accent);
    background: var(--tf-accent);
    color: #0a1414;
    box-shadow: 0 0 0 3px var(--tf-accent-soft);
}

.tf-node-content {
    min-width: 0;
}

.tf-node-content strong {
    display: block;
    color: var(--tf-text);
    font-size: 12px;
    font-weight: 700;
}

.tf-node-content span {
    display: block;
    margin-top: 3px;
    color: var(--tf-muted-soft);
    font-size: 10px;
    line-height: 1.35;
}

/* KPI */

.tf-kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 6px 0 20px;
}

.tf-kpi {
    display: inline-flex;
    flex-direction: column;
    gap: 3px;
    padding: 10px 14px;
    border: 1px solid var(--tf-border-soft);
    border-radius: 10px;
    background: var(--tf-panel-alt);
    min-width: 120px;
}

.tf-kpi .label {
    color: var(--tf-muted-soft);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.tf-kpi .value {
    color: var(--tf-text);
    font-family: monospace;
    font-size: 16px;
    font-weight: 700;
}

.tf-kpi .value.pos {
    color: var(--tf-bull);
}

.tf-kpi .value.neg {
    color: var(--tf-bear);
}

/* NEWS */

.tf-news-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 999px;
    font-family: monospace;
    font-size: 10px;
    font-weight: 700;
}

.tf-news-badge.available {
    color: var(--tf-bull);
    background: var(--tf-bull-soft);
    border: 1px solid var(--tf-bull-soft);
}

.tf-news-badge.unavailable {
    color: var(--tf-warning);
    background: var(--tf-warning-soft);
    border: 1px solid var(--tf-warning-soft);
}

.tf-news-badge.error {
    color: var(--tf-bear);
    background: var(--tf-bear-soft);
    border: 1px solid var(--tf-bear-soft);
}

.tf-news-badge.unknown {
    color: var(--tf-muted);
    background: var(--tf-panel);
    border: 1px solid var(--tf-border);
}

/* FEED */

.tf-tick-card {
    padding: 12px 14px;
    margin-bottom: 8px;
    border: 1px solid var(--tf-border-soft);
    border-left-width: 3px;
    border-radius: 4px 10px 10px 4px;
    background: var(--tf-panel-alt);
    animation: tfFadeIn .2s ease both;
}

.tf-tick-card.executed {
    border-left-color: var(--tf-bull);
}

.tf-tick-card.held_book_reject {
    border-left-color: var(--tf-warning);
}

.tf-tick-card.held_local_reject {
    border-left-color: var(--tf-bear);
}

.tf-tick-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 5px;
}

.tf-tick-head .left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.tf-tick-ticker {
    font-family: monospace;
    font-weight: 700;
    color: var(--tf-text);
    font-size: 13px;
}

.tf-tick-date {
    color: var(--tf-muted-soft);
    font-family: monospace;
    font-size: 10.5px;
}

.tf-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border: 1px solid var(--tf-border);
    border-radius: 999px;
    background: var(--tf-panel);
    color: var(--tf-muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .03em;
}

.tf-chip.executed {
    border-color: var(--tf-bull-soft);
    background: var(--tf-bull-soft);
    color: var(--tf-bull);
}

.tf-chip.held_book_reject {
    border-color: var(--tf-warning-soft);
    background: var(--tf-warning-soft);
    color: var(--tf-warning);
}

.tf-chip.held_local_reject {
    border-color: var(--tf-bear-soft);
    background: var(--tf-bear-soft);
    color: var(--tf-bear);
}

.tf-tick-notes {
    color: var(--tf-muted);
    font-size: 12px;
    line-height: 1.5;
}

.tf-tick-fill {
    margin-top: 6px;
    color: var(--tf-muted-soft);
    font-family: monospace;
    font-size: 10.5px;
}

/* POSITIONS */

.tf-position {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 13px;
    margin-bottom: 6px;
    border: 1px solid var(--tf-border-soft);
    border-radius: 8px;
    background: var(--tf-panel-alt);
}

.tf-position .ticker {
    font-family: monospace;
    font-weight: 700;
    color: var(--tf-text);
}

.tf-position .sector {
    color: var(--tf-muted-soft);
    font-size: 10.5px;
    margin-left: 8px;
}

.tf-position .value {
    font-family: monospace;
    color: var(--tf-accent-strong);
    font-weight: 600;
}

/* MOBILE */

@media (max-width: 900px) {
    .tf-topbar h1 {
        font-size: 23px;
    }

    .tf-pipeline {
        overflow-x: auto;
        padding-bottom: 8px;
    }

    .tf-pipeline-node {
        min-width: 150px;
    }
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SIDEBAR DATA
# ============================================================

ledger_exists = LEDGER_PATH.exists()

ledger_mtime = LEDGER_PATH.stat().st_mtime if ledger_exists else 0.0

raw_ledger = (
    load_live_ledger(
        str(LEDGER_PATH),
        ledger_mtime,
    )
    if ledger_exists
    else None
)

ledger = normalize_ledger(raw_ledger)

backtest_runs = list_backtest_runs(str(BACKTEST_RESULTS_DIR))

if ledger_exists and ledger["equity"] is not None:
    mode_dot = "success"
    mode_text = f"Ledger live ({TRADING_MODE})"

elif ledger_exists:
    mode_dot = "warning"
    mode_text = "Ledger found, shape uncertain"

else:
    mode_dot = "danger"
    mode_text = "No live ledger found"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    display_html(
        """
        <div class="tf-brand">
            <svg width="30" height="30"
                 viewBox="0 0 34 34" fill="none">
                <path d="M4 26 L12 16 L18 21 L30 8"
                      stroke="#5bc0be"
                      stroke-width="2.4"
                      fill="none"
                      stroke-linecap="round"
                      stroke-linejoin="round"/>
                <path d="M22 8 L30 8 L30 16"
                      stroke="#5bc0be"
                      stroke-width="2.4"
                      fill="none"
                      stroke-linecap="round"
                      stroke-linejoin="round"/>
            </svg>

            <div>
                <strong>Trading Firm</strong>
                <small>Multi-agent command console</small>
            </div>
        </div>

        <div class="tf-rail-label">Session</div>
        """
    )

    display_html(
        f"""
        <div class="tf-status-card">
            <div class="kv">
                <b>Session</b>&nbsp;
                {esc(st.session_state.session_id[:8])}<br/>

                <b>Started</b>&nbsp;
                {esc(st.session_state.session_started_at)}<br/>

                <b>Mode</b>&nbsp;
                {esc(TRADING_MODE.upper())}<br/>

                <b>Backtest runs</b>&nbsp;
                {len(backtest_runs)}<br/>

                <b>Ledger equity</b>&nbsp;
                {fmt_money(ledger["equity"])}
            </div>
        </div>
        """
    )

    display_html('<div class="tf-rail-label">Actions</div>')

    if st.button(
        "🔄 Refresh data",
        width="stretch",
    ):
        load_live_ledger.clear()
        list_backtest_runs.clear()
        load_backtest_run.clear()
        st.rerun()

    with st.expander("▶ Run new backtest"):
        bt_tickers = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL,MSFT,GOOGL",
        )

        bt_start = st.text_input(
            "Start date",
            value="2024-06-03",
        )

        bt_end = st.text_input(
            "End date",
            value="2024-06-10",
        )

        bt_run_id = st.text_input(
            "Run ID",
            value="",
        )

        bt_delay = st.number_input(
            "Tick delay (seconds)",
            min_value=0.0,
            value=1.5,
            step=0.5,
        )

        if st.button(
            "Run backtest",
            width="stretch",
        ):
            tickers_list = [
                t.strip().upper() for t in bt_tickers.split(",") if t.strip()
            ]

            run_id = bt_run_id.strip() or f"{bt_start}_{bt_end}"

            try:
                with st.spinner(f"Running backtest for {tickers_list}..."):
                    from backtest.runner import (
                        run_backtest,
                        _serialize_result,
                    )

                    result = run_backtest(
                        tickers=tickers_list,
                        start_date=bt_start,
                        end_date=bt_end,
                        tick_delay_seconds=bt_delay,
                    )

                    BACKTEST_RESULTS_DIR.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    out_path = BACKTEST_RESULTS_DIR / f"{run_id}.json"

                    out_path.write_text(
                        json.dumps(
                            _serialize_result(result),
                            indent=2,
                        )
                    )

                st.session_state.last_backtest_run_id = f"{run_id}.json"

                list_backtest_runs.clear()

                st.success(
                    f"Backtest complete: "
                    f"{len(result.tick_log)} tick(s), "
                    f"{len(result.trades)} executed trade(s)."
                )

                st.rerun()

            except Exception as exc:
                if LOGFIRE_OK:
                    logfire.exception(
                        "Backtest run failed",
                        error_type=type(exc).__name__,
                    )

                st.error(f"Backtest failed: {exc}")

    display_html(
        f"""
        <div class="tf-rail-footer">
            <span class="tf-dot {mode_dot}"></span>

            <div>
                <strong>{esc(mode_text)}</strong>
                <small>
                    LangGraph · Alpaca · SimBroker
                </small>
            </div>
        </div>
        """
    )


# ============================================================
# TOPBAR
# ============================================================

display_html(
    f"""
    <div class="tf-topbar">

        <div class="tf-kicker">
            Multi-agent risk-gated execution
        </div>

        <h1>
            📈 AI Multi-Agent Trading Firm
        </h1>

        <p>
            NewsAgent and ChartAgent independently read each ticker.
            SignalMerger combines them deterministically. A local
            RiskAgent gates position sizing, then the
            PortfolioRiskCoordinator checks every proposal against
            the whole book before ExecutionAgent places
            (or HoldNode blocks) the trade.
        </p>

        <div class="tf-engine-pill">
            <span class="tf-dot {mode_dot}"></span>
            {esc(mode_text)}
        </div>

    </div>

    <div class="tf-signal-divider">
        <svg viewBox="0 0 1200 20"
             preserveAspectRatio="none">

            <path
                d="M0,10 L360,10 L380,2 L400,18
                   L420,10 L1200,10"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
            />

        </svg>
    </div>
    """
)


# ============================================================
# TABS
# ============================================================

tab_overview, tab_feed, tab_report = st.tabs(
    [
        "Overview",
        "Reasoning Feed",
        "Backtest Report",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    if not ledger_exists:
        st.info(
            f"No live ledger found at `{LEDGER_PATH}`. "
            "Run the live/paper trading loop, or run a "
            "backtest and check the Backtest Report tab."
        )

    else:
        display_html(render_pipeline_rail())

        kpis = [
            (
                "Equity",
                fmt_money(ledger["equity"]),
                "",
            ),
            (
                "Starting equity",
                fmt_money(ledger["starting_equity"]),
                "",
            ),
            (
                "Cash",
                fmt_money(ledger["cash"]),
                "",
            ),
            (
                "Open positions",
                str(len(ledger["positions"])),
                "",
            ),
        ]

        kpi_html = "".join(
            f"""
            <div class="tf-kpi">
                <span class="label">
                    {esc(label)}
                </span>

                <span class="value {cls}">
                    {esc(value)}
                </span>
            </div>
            """
            for label, value, cls in kpis
        )

        display_html(f'<div class="tf-kpi-row">{kpi_html}</div>')

        st.markdown("#### Open positions")

        if not ledger["positions"]:
            st.caption("No open positions.")

        else:
            for ticker, pos in ledger["positions"].items():
                if not isinstance(pos, dict):
                    continue

                sector = pos.get("sector", "—")
                shares = fmt_shares(pos.get("shares"))
                market_value = fmt_money(pos.get("market_value"))

                display_html(
                    f"""
                    <div class="tf-position">

                        <div>
                            <span class="ticker">
                                {esc(ticker)}
                            </span>

                            <span class="sector">
                                {esc(sector)}
                            </span>
                        </div>

                        <div>
                            <span style="
                                color: var(--tf-muted-soft);
                                font-size: 11px;
                                margin-right: 10px;
                            ">
                                {esc(shares)} sh
                            </span>

                            <span class="value">
                                {esc(market_value)}
                            </span>
                        </div>

                    </div>
                    """
                )

        with st.expander("Raw ledger.json"):
            st.json(raw_ledger)


# ============================================================
# REASONING FEED
# ============================================================

with tab_feed:
    if not backtest_runs:
        st.info(
            f"No backtest runs found in "
            f"`{BACKTEST_RESULTS_DIR}`. "
            "Run one from the sidebar to populate "
            "the reasoning feed."
        )

    else:
        default_index = 0

        if st.session_state.last_backtest_run_id in backtest_runs:
            default_index = backtest_runs.index(st.session_state.last_backtest_run_id)

        selected_run = st.selectbox(
            "Backtest run",
            backtest_runs,
            index=default_index,
            key="feed_run_select",
        )

        run_data = load_backtest_run(
            str(BACKTEST_RESULTS_DIR),
            selected_run,
        )

        if not run_data:
            st.error("Could not read this run's JSON file.")

        else:
            # ------------------------------------------------
            # News quality summary
            # ------------------------------------------------

            news_quality = extract_news_quality(run_data)

            if news_quality["unavailable"]:
                st.warning(
                    f"Historical news unavailable for "
                    f"{news_quality['unavailable']} ticker(s) "
                    "in this run. The NewsAgent treats "
                    "unavailable historical news as neutral."
                )

            tick_log = run_data.get(
                "tick_log",
                [],
            )

            if not tick_log:
                st.caption("This run has no tick_log entries.")

            else:
                tickers_in_run = sorted({entry.get("ticker", "") for entry in tick_log})

                filter_ticker = st.selectbox(
                    "Filter by ticker",
                    ["All"] + tickers_in_run,
                    key="feed_ticker_filter",
                )

                filtered = [
                    e
                    for e in reversed(tick_log)
                    if (filter_ticker == "All" or e.get("ticker") == filter_ticker)
                ]

                if filtered:
                    display_html(
                        render_pipeline_rail(
                            filtered[0].get("outcome"),
                            filtered[0].get("ticker"),
                        )
                    )

                trades_by_key = {
                    (
                        t.get("date"),
                        t.get("ticker"),
                    ): t
                    for t in run_data.get(
                        "trades",
                        [],
                    )
                }

                st.markdown(f"#### {len(filtered)} tick event(s)")

                for entry in filtered:
                    outcome = entry.get(
                        "outcome",
                        "unknown",
                    )

                    ticker = entry.get(
                        "ticker",
                        "—",
                    )

                    date = entry.get(
                        "date",
                        "—",
                    )

                    notes = entry.get(
                        "notes",
                        "",
                    )

                    outcome_label = {
                        "executed": "Executed",
                        "held_local_reject": "Held (local)",
                        "held_book_reject": "Held (book)",
                    }.get(
                        outcome,
                        outcome,
                    )

                    fill_html = ""

                    trade = trades_by_key.get(
                        (
                            date,
                            ticker,
                        )
                    )

                    if trade:
                        fill_html = (
                            '<div class="tf-tick-fill">'
                            f"{fmt_shares(trade.get('shares'))}"
                            " sh @ "
                            f"{fmt_money(trade.get('price'))}"
                            "</div>"
                        )

                    display_html(
                        f"""
                        <div class="tf-tick-card {esc(outcome)}">

                            <div class="tf-tick-head">

                                <div class="left">

                                    <span class="tf-tick-ticker">
                                        {esc(ticker)}
                                    </span>

                                    <span class="tf-chip {esc(outcome)}">
                                        {esc(outcome_label)}
                                    </span>

                                </div>

                                <span class="tf-tick-date">
                                    {esc(date)}
                                </span>

                            </div>

                            <div class="tf-tick-notes">
                                {esc(notes)}
                            </div>

                            {fill_html}

                        </div>
                        """
                    )


# ============================================================
# BACKTEST REPORT
# ============================================================

with tab_report:
    if not backtest_runs:
        st.info(f"No backtest runs found in `{BACKTEST_RESULTS_DIR}`.")

    else:
        default_index = 0

        if st.session_state.last_backtest_run_id in backtest_runs:
            default_index = backtest_runs.index(st.session_state.last_backtest_run_id)

        selected_run = st.selectbox(
            "Backtest run",
            backtest_runs,
            index=default_index,
            key="report_run_select",
        )

        run_data = load_backtest_run(
            str(BACKTEST_RESULTS_DIR),
            selected_run,
        )

        if not run_data:
            st.error("Could not read this run's JSON file.")

        else:
            # =================================================
            # RUN HEADER
            # =================================================

            st.markdown(f"### Backtest: `{selected_run}`")

            # =================================================
            # HEADLINE METRICS
            # =================================================

            sharpe = run_data.get("sharpe_ratio")

            win_rate = run_data.get("win_rate")

            max_dd = run_data.get("max_drawdown")

            total_return = run_data.get("total_return")

            buy_hold_return = run_data.get("buy_hold_return")

            def _cls(value):

                try:
                    return "pos" if float(value) >= 0 else "neg"

                except (
                    TypeError,
                    ValueError,
                ):
                    return ""

            kpis = [
                (
                    "Sharpe ratio",
                    fmt_num(sharpe, 2),
                    "",
                ),
                (
                    "Win rate",
                    fmt_pct(win_rate),
                    "",
                ),
                (
                    "Max drawdown",
                    fmt_pct(max_dd),
                    "neg" if max_dd else "",
                ),
                (
                    "Total return",
                    fmt_pct(total_return),
                    _cls(total_return),
                ),
                (
                    "Buy & hold return",
                    fmt_pct(buy_hold_return),
                    _cls(buy_hold_return),
                ),
                (
                    "Trades",
                    str(
                        len(
                            run_data.get(
                                "trades",
                                [],
                            )
                        )
                    ),
                    "",
                ),
            ]

            kpi_html = "".join(
                f"""
                <div class="tf-kpi">

                    <span class="label">
                        {esc(label)}
                    </span>

                    <span class="value {cls}">
                        {esc(value)}
                    </span>

                </div>
                """
                for label, value, cls in kpis
            )

            display_html(f'<div class="tf-kpi-row">{kpi_html}</div>')

            # =================================================
            # WIN RATE EXPLANATION
            # =================================================

            if win_rate is None and run_data.get("trades"):
                st.info(
                    "Win rate is unavailable because the current "
                    "trade records represent entries rather than "
                    "completed entry→exit trades with realized P&L. "
                    "This is a metric-definition limitation, not "
                    "a 0% win rate."
                )

            # =================================================
            # NEWS DATA QUALITY
            # =================================================

            render_news_quality_section(run_data)

            # =================================================
            # EQUITY CURVE
            # =================================================

            equity_curve = run_data.get(
                "equity_curve",
                [],
            )

            if equity_curve:
                st.markdown("#### Equity curve")

                dates = [point[0] for point in equity_curve]

                values = [point[1] for point in equity_curve]

                st.line_chart(
                    {"equity": values},
                    x_label="tick",
                    y_label="equity ($)",
                )

                st.caption(f"{dates[0]} → {dates[-1]}")

            # =================================================
            # BENCHMARK
            # =================================================

            benchmark_curve = run_data.get(
                "buy_hold_curve",
                [],
            )

            if benchmark_curve:
                st.markdown("#### Strategy vs buy & hold")

                strategy_values = [point[1] for point in equity_curve]

                benchmark_values = [point[1] for point in benchmark_curve]

                max_len = min(
                    len(strategy_values),
                    len(benchmark_values),
                )

                if max_len:
                    st.line_chart(
                        {
                            "Strategy": strategy_values[:max_len],
                            "Buy & hold": benchmark_values[:max_len],
                        },
                        x_label="tick",
                        y_label="equity ($)",
                    )

                excess_return = run_data.get("excess_return_vs_buy_and_hold")

                st.caption(f"Excess return vs buy & hold: {fmt_pct(excess_return)}")

            # =================================================
            # EXECUTED TRADES
            # =================================================

            trades = run_data.get(
                "trades",
                [],
            )

            st.markdown("#### Executed trades")

            if not trades:
                st.caption("No trades executed in this run.")

            else:
                table_rows = [
                    {
                        "Date": t.get("date"),
                        "Ticker": t.get("ticker"),
                        "Shares": fmt_shares(t.get("shares")),
                        "Price": fmt_money(t.get("price")),
                        "Notes": t.get("notes"),
                    }
                    for t in trades
                ]

                st.dataframe(
                    table_rows,
                    width="stretch",
                    hide_index=True,
                )

            # =================================================
            # RAW JSON
            # =================================================

            with st.expander("Raw run JSON"):
                st.json(run_data)


# ============================================================
# FOOTER
# ============================================================

display_html(
    """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
        color: var(--tf-muted-soft);
        font-size: 10.5px;
    ">
        <span>AI Multi-Agent Trading Firm</span>
        <span>·</span>
        <span>LangGraph orchestration</span>
        <span>·</span>
        <span>Alpaca / SimBroker execution</span>
        <span>·</span>
        <span>Deterministic risk gating</span>
        <span>·</span>
        <span>Historical data integrity</span>
    </div>
    """
)
