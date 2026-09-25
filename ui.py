"""
AI Multi-Agent Trading Firm · Command Center

Purpose
-------
Interactive Streamlit console for understanding and monitoring the
multi-agent trading system.

The UI is observational. It does not change trading strategy logic.

Architecture

    NewsAgent ───────┐
                     │
    ChartAgent ──────┤
                     ▼
               SignalMerger
                     │
                     ▼
                 RiskAgent
                     │
                     ▼
          PortfolioRiskCoordinator
                /           \
               /             \
        APPROVED             REJECTED
           │                    │
           ▼                    ▼
    ExecutionAgent           HoldNode
           │
           ▼
       Broker / Fill

Persistence

    Backtest
        │
        ├── JSON result ───────► Backtest Lab
        │
        └── SQLite audit ─────► Run Audit
                                   │
                                   ├── runs
                                   └── trade_traces
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import nullcontext
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

except Exception:  # noqa: S110,BLE001
    pass


def trace_context():
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
    native = getattr(
        st,
        "html",
        None,
    )

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
    except (
        TypeError,
        ValueError,
    ):
        return default


def fmt_pct(
    value,
    default="—",
) -> str:
    if value is None:
        return default

    try:
        return f"{float(value):+.2%}"
    except (
        TypeError,
        ValueError,
    ):
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
    except (
        TypeError,
        ValueError,
    ):
        return default


def fmt_shares(
    value,
    default="—",
) -> str:
    try:
        return f"{float(value):.3f}"
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(value):
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def outcome_label(
    outcome: str | None,
) -> str:
    return {
        "executed": "EXECUTED",
        "held_local_reject": "LOCAL RISK BLOCK",
        "held_book_reject": "BOOK RISK BLOCK",
        "held_execution_reject": "BROKER EXECUTION FAILED",
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

/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

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

/* ---------------------------------------------------------
   ARCHITECTURE
--------------------------------------------------------- */

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

@media (max-width: 1100px) {
    .arch-grid {
        grid-template-columns: repeat(4, 1fr);
    }

    .arch-node::after {
        display: none;
    }
}

/* ---------------------------------------------------------
   KPI
--------------------------------------------------------- */

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin: 10px 0 20px;
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

@media (max-width: 1100px) {
    .kpi-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* ---------------------------------------------------------
   AGENT CARDS
--------------------------------------------------------- */

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

/* ---------------------------------------------------------
   DECISION CARD
--------------------------------------------------------- */

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

/* ---------------------------------------------------------
   PIPELINE
--------------------------------------------------------- */

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

@media (max-width: 900px) {
    .pipeline {
        overflow-x: auto;
    }

    .pipeline-step {
        min-width: 125px;
    }
}

/* ---------------------------------------------------------
   EXPLANATION
--------------------------------------------------------- */

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

/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #10141a;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] button {
    border-color: var(--border) !important;
}

/* ---------------------------------------------------------
   SECTION
--------------------------------------------------------- */

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

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
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

    except (
        json.JSONDecodeError,
        OSError,
    ):
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
    exists = LEDGER_PATH.exists()

    if not exists:
        return None

    mtime = LEDGER_PATH.stat().st_mtime

    return load_json_file(
        str(LEDGER_PATH),
        mtime,
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
# SQLITE AUDIT LOADERS
# ============================================================


def _sqlite_rows(
    query: str,
    params=(),
):
    """
    Execute a read-only SQLite query against the persistent audit DB.

    The UI treats SQLite as an observational data source.
    """
    if not AUDIT_DB_PATH.exists():
        return []

    connection = None

    try:
        connection = sqlite3.connect(AUDIT_DB_PATH)

        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            query,
            params,
        ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error:
        return []

    finally:
        if connection is not None:
            connection.close()


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def list_audit_runs(
    db_path_str: str,
):
    """
    Return persisted run records.

    The query intentionally selects the known run-audit fields instead
    of depending on RunAuditStore's Python API.
    """
    db_path = Path(db_path_str)

    if not db_path.exists():
        return []

    return _sqlite_rows(
        """
        SELECT
            run_id,
            status,
            started_at,
            finished_at,
            start_date,
            end_date,
            tickers,
            starting_equity,
            final_equity,
            realized_pnl,
            benchmark_return,
            trade_trace_count,
            metadata
        FROM runs
        ORDER BY
            COALESCE(
                finished_at,
                started_at
            ) DESC
        """
    )


@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def load_audit_run(
    db_path_str: str,
    run_id: str,
):
    """
    Load one persisted run and all of its trade traces.
    """
    db_path = Path(db_path_str)

    if not db_path.exists():
        return None

    runs = _sqlite_rows(
        """
        SELECT *
        FROM runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    )

    if not runs:
        return None

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


def parse_json_value(
    value,
    default=None,
):
    if value is None:
        return default

    if isinstance(
        value,
        (
            dict,
            list,
        ),
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        return value

    try:
        return json.loads(value)
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return default


def normalize_audit_trace(
    trace: dict,
) -> dict:
    """
    Normalize SQLite trade-trace columns into UI vocabulary.

    This keeps the dashboard tolerant of small storage naming differences
    while preserving the canonical trace semantics.
    """
    metadata = parse_json_value(
        trace.get("metadata"),
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    risk_checks = parse_json_value(
        trace.get("risk_checks"),
        [],
    )

    if not isinstance(
        risk_checks,
        list,
    ):
        risk_checks = []

    risk_notes = parse_json_value(
        trace.get("risk_notes"),
        [],
    )

    if not isinstance(
        risk_notes,
        list,
    ):
        risk_notes = []

    return {
        **trace,
        "metadata": metadata,
        "risk_checks": risk_checks,
        "risk_notes": risk_notes,
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
        "outcome": first_value(
            trace,
            "outcome",
            default="unknown",
        ),
        "signal_direction": first_value(
            trace,
            "signal_direction",
            default="neutral",
        ),
        "signal_confidence": first_value(
            trace,
            "signal_confidence",
            default=None,
        ),
        "merge_agreement": first_value(
            trace,
            "merge_agreement",
            default=None,
        ),
        "risk_decision": first_value(
            trace,
            "risk_decision",
            default=None,
        ),
        "proposed_shares": first_value(
            trace,
            "proposed_shares",
            default=None,
        ),
        "final_shares": first_value(
            trace,
            "final_shares",
            default=None,
        ),
        "execution_side": first_value(
            trace,
            "execution_side",
            default=None,
        ),
        "execution_attempted": first_value(
            trace,
            "execution_attempted",
            default=False,
        ),
        "execution_success": first_value(
            trace,
            "execution_success",
            default=None,
        ),
        "average_cost": first_value(
            trace,
            "average_cost",
            default=None,
        ),
        "realized_pnl": first_value(
            trace,
            "realized_pnl",
            default=None,
        ),
        "remaining_shares": first_value(
            trace,
            "remaining_shares",
            default=None,
        ),
        "position_closed": first_value(
            trace,
            "position_closed",
            default=None,
        ),
    }


def parse_run_metadata(
    run: dict,
):
    metadata = parse_json_value(
        run.get("metadata"),
        {},
    )

    if isinstance(
        metadata,
        dict,
    ):
        return metadata

    return {}


# ============================================================
# LEDGER NORMALIZATION
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

        for pos in positions.values():
            if isinstance(
                pos,
                dict,
            ):
                market_value += safe_float(pos.get("market_value")) or 0.0

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


def normalize_news_status(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        value = value.get("availability")

    if hasattr(
        value,
        "value",
    ):
        value = value.value

    value = str(value).lower().strip()

    return value if value in NEWS_LABELS else None


def extract_news_quality(
    run_data,
):
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

    if isinstance(
        raw,
        dict,
    ):
        candidate = raw.get("by_ticker")

        if candidate is None:
            candidate = raw.get("tickers")

        if isinstance(
            candidate,
            dict,
        ):
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

    result["available"] = sum(v == "available" for v in by_ticker.values())

    result["unavailable"] = sum(v == "unavailable" for v in by_ticker.values())

    result["error"] = sum(v == "error" for v in by_ticker.values())

    result["total"] = len(by_ticker)

    if isinstance(
        raw,
        dict,
    ):
        summary = raw.get("summary")

        if isinstance(
            summary,
            dict,
        ):
            source = summary
        else:
            source = raw

        for key in [
            "available",
            "unavailable",
            "error",
            "total",
        ]:
            if key in source:
                try:
                    result[key] = int(source[key] or 0)
                except (
                    TypeError,
                    ValueError,
                ):
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

            except (
                TypeError,
                ValueError,
            ):
                pass

    if result["coverage"] is None and result["total"] > 0:
        result["coverage"] = result["available"] / result["total"]

    return result


# ============================================================
# PIPELINE
# ============================================================

PIPELINE = [
    (
        "1",
        "NewsAgent",
        "Market/news",
    ),
    (
        "2",
        "ChartAgent",
        "Technicals",
    ),
    (
        "3",
        "SignalMerger",
        "Deterministic",
    ),
    (
        "4",
        "RiskAgent",
        "Position sizing",
    ),
    (
        "5",
        "Portfolio",
        "Book-level risk",
    ),
    (
        "6",
        "Execution",
        "Broker / Hold",
    ),
]


def render_architecture():
    nodes = []

    for (
        index,
        name,
        description,
    ) in PIPELINE:
        nodes.append(
            f"""
            <div class="arch-node">
                <div class="number">
                    {index}
                </div>
                <strong>
                    {esc(name)}
                </strong>
                <small>
                    {esc(description)}
                </small>
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

    for (
        index,
        name,
        description,
    ) in PIPELINE:
        nodes.append(
            f"""
            <div class="pipeline-step active">
                <div class="index">
                    {index}
                </div>
                <strong>
                    {esc(name)}
                </strong>
                <small>
                    {esc(description)}
                </small>
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
                <b>
                    {fmt_num(confidence, 3)}
                </b>
                <br/>
                {esc(detail)}
            </div>

        </div>
        """
    )


def render_decision(
    entry,
):
    outcome = entry.get("outcome")

    ticker = entry.get(
        "ticker",
        "—",
    )

    date = entry.get(
        "date",
        "—",
    )

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
                        {esc(date)}
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

        risk_html = f"""
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

        display_html(risk_html)

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

    notes = entry.get(
        "notes",
        "",
    )

    st.markdown("#### Why did the system make this decision?")

    display_html(
        f"""
        <div class="reason-box">
            <strong>Final outcome:</strong>
            {esc(outcome_label(outcome))}
            <br/><br/>
            {esc(notes)}
        </div>
        """
    )


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "session_started_at" not in st.session_state:
    st.session_state.session_started_at = time.strftime("%H:%M:%S")

if "selected_run" not in st.session_state:
    st.session_state.selected_run = None


# ============================================================
# SIDEBAR
# ============================================================

raw_ledger = load_ledger()

ledger = normalize_ledger(raw_ledger)

backtest_runs = list_backtest_runs(str(BACKTEST_RESULTS_DIR))

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
    )

    if st.button(
        "▶ Run backtest",
        width="stretch",
    ):
        tickers = [t.strip().upper() for t in bt_tickers.split(",") if t.strip()]

        run_id = bt_run_id.strip() or f"{bt_start}_{bt_end}"

        try:
            with st.spinner("Running multi-agent backtest..."):
                from backtest.runner import (
                    _serialize_result,
                    run_backtest,
                )

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

            st.success(f"Completed {len(result.tick_log)} ticks.")

            st.rerun()

        except Exception as exc:  # noqa: BLE001
            if LOGFIRE_OK:
                logfire.exception(
                    "Backtest failed",
                    error_type=type(exc).__name__,
                )

            st.error(f"Backtest failed: {exc}")

    st.divider()

    st.caption("LangGraph · Risk-gated execution · SimBroker / Alpaca")


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
            This console lets you trace how the trading system
            moves from market information to an actual broker
            decision. NewsAgent and ChartAgent generate independent
            evidence, SignalMerger combines it deterministically,
            RiskAgent performs position sizing and local risk checks,
            Portfolio Risk evaluates the whole book, and only then
            can ExecutionAgent submit a trade.
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
    tab_audit,
    tab_data,
) = st.tabs(
    [
        "🧠 Command Center",
        "🔎 Decision Explorer",
        "📊 Backtest Lab",
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
                for (
                    label,
                    value,
                    cls,
                ) in kpis
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

            for (
                ticker,
                position,
            ) in positions.items():
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
            "No live ledger found. Use the Backtest Lab to inspect historical runs."
        )


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
                st.markdown(f"### {selected_run}")

                tickers = sorted({x.get("ticker") for x in tick_log if x.get("ticker")})

                outcomes = sorted(
                    {x.get("outcome") for x in tick_log if x.get("outcome")}
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    ticker_filter = st.selectbox(
                        "Ticker",
                        ["All"] + tickers,
                    )

                with c2:
                    outcome_filter = st.selectbox(
                        "Outcome",
                        ["All"] + outcomes,
                    )

                with c3:
                    show_latest = st.number_input(
                        "Events to show",
                        min_value=1,
                        max_value=200,
                        value=20,
                    )

                filtered = [
                    x
                    for x in reversed(tick_log)
                    if (ticker_filter == "All" or x.get("ticker") == ticker_filter)
                    and (outcome_filter == "All" or x.get("outcome") == outcome_filter)
                ]

                st.caption(f"{len(filtered)} matching decision(s)")

                if filtered:
                    selected_index = st.selectbox(
                        "Inspect decision",
                        range(
                            min(
                                len(filtered),
                                int(show_latest),
                            )
                        ),
                        format_func=lambda i: (
                            f"{filtered[i].get('date', '—')} · "
                            f"{filtered[i].get('ticker', '—')} · "
                            f"{outcome_label(filtered[i].get('outcome'))}"
                        ),
                    )

                    render_decision(filtered[selected_index])


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
                    for (
                        label,
                        value,
                        cls,
                    ) in kpis
                )
                + "</div>"
            )

            st.markdown("#### Trading pipeline telemetry")

            telemetry = [
                (
                    "Ticks",
                    diagnostics.get(
                        "tick_count",
                        len(
                            run_data.get(
                                "tick_log",
                                [],
                            )
                        ),
                    ),
                ),
                (
                    "Risk approvals",
                    diagnostics.get(
                        "risk_approvals",
                        0,
                    ),
                ),
                (
                    "Local rejects",
                    diagnostics.get(
                        "local_rejections",
                        0,
                    ),
                ),
                (
                    "Book rejects",
                    diagnostics.get(
                        "book_rejections",
                        0,
                    ),
                ),
                (
                    "Execution attempts",
                    diagnostics.get(
                        "execution_attempts",
                        0,
                    ),
                ),
                (
                    "Execution failures",
                    diagnostics.get(
                        "execution_failures",
                        0,
                    ),
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
                    for (
                        label,
                        value,
                    ) in telemetry
                )
                + "</div>"
            )

            tick_log = run_data.get(
                "tick_log",
                [],
            )

            outcome_counts = {}

            for entry in tick_log:
                outcome = entry.get(
                    "outcome",
                    "unknown",
                )

                outcome_counts[outcome] = (
                    outcome_counts.get(
                        outcome,
                        0,
                    )
                    + 1
                )

            if outcome_counts:
                st.markdown("#### Decision outcomes")

                st.bar_chart(outcome_counts)

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
                rows = []

                for trade in trades:
                    rows.append(
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
                    )

                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.info("No executed trades.")


# ============================================================
# RUN AUDIT
# ============================================================

with tab_audit:
    st.markdown("### Persistent Run Audit")

    st.caption(
        "SQLite-backed audit history for reproducible trading "
        "decisions, execution outcomes, and portfolio accounting."
    )

    audit_runs = list_audit_runs(str(AUDIT_DB_PATH))

    if not audit_runs:
        st.info(
            "No persisted audit runs found. "
            "Run a backtest to create the first audit record."
        )

    else:
        run_ids = [str(run.get("run_id")) for run in audit_runs if run.get("run_id")]

        selected_result_stem = None

        if st.session_state.selected_run:
            selected_result_stem = Path(st.session_state.selected_run).stem

        if selected_result_stem in run_ids:
            default_index = run_ids.index(selected_result_stem)
        else:
            default_index = 0

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
            st.error("Unable to load the selected audit run.")

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

            trade_trace_count = run.get("trade_trace_count")

            if trade_trace_count is None:
                trade_trace_count = len(traces)

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
                    trade_trace_count,
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
                    for (
                        label,
                        value,
                        cls,
                    ) in summary
                )
                + "</div>"
            )

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

            st.dataframe(
                lifecycle_rows,
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------------
            # TRACE FILTERS
            # ------------------------------------------------

            st.markdown("#### Decision traces")

            if not traces:
                st.info("No trade traces were persisted for this run.")

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

                # ------------------------------------------------
                # TRACE TABLE
                # ------------------------------------------------

                rows = []

                for trace in filtered_traces:
                    rows.append(
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
                            "Risk": trace.get(
                                "risk_decision",
                                "—",
                            ),
                            "Shares": fmt_shares(trace.get("final_shares")),
                            "Execution": (trace.get("execution_side") or "—"),
                            "Outcome": outcome_label(trace.get("outcome")),
                        }
                    )

                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )

                # ------------------------------------------------
                # INDIVIDUAL TRACE
                # ------------------------------------------------

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

                    st.markdown("#### Decision lineage")

                    signal_direction = trace.get("signal_direction")

                    execution_side = trace.get("execution_side")

                    risk_decision = trace.get("risk_decision")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**SignalMerger**")

                        st.metric(
                            "Direction",
                            str(signal_direction or "—").upper(),
                        )

                        st.caption(
                            "Confidence: "
                            + fmt_num(
                                trace.get("signal_confidence"),
                                3,
                            )
                        )

                    with c2:
                        st.markdown("**Risk / Coordinator**")

                        st.metric(
                            "Risk",
                            str(risk_decision or "—").upper(),
                        )

                        st.caption(
                            "Proposed shares: "
                            + fmt_shares(trace.get("proposed_shares"))
                            + " · Final: "
                            + fmt_shares(trace.get("final_shares"))
                        )

                    with c3:
                        st.markdown("**Execution**")

                        st.metric(
                            "Side",
                            str(execution_side or "—").upper(),
                        )

                        st.caption("Success: " + str(trace.get("execution_success")))

                    # ------------------------------------------------
                    # NEWS / CHART
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

                        st.dataframe(
                            news_rows,
                            width="stretch",
                            hide_index=True,
                        )

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

                        st.dataframe(
                            chart_rows,
                            width="stretch",
                            hide_index=True,
                        )

                    # ------------------------------------------------
                    # ACCOUNTING
                    # ------------------------------------------------

                    st.markdown("#### Accounting")

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

                    st.dataframe(
                        accounting_rows,
                        width="stretch",
                        hide_index=True,
                    )

                    # ------------------------------------------------
                    # RISK CHECKS
                    # ------------------------------------------------

                    risk_checks = (
                        trace.get("risk_checks") or trace.get("risk_notes") or []
                    )

                    if risk_checks:
                        with st.expander(f"Risk checks ({len(risk_checks)})"):
                            for check in risk_checks:
                                st.caption("• " + str(check))

                    # ------------------------------------------------
                    # RAW TRACE
                    # ------------------------------------------------

                    with st.expander("Raw persisted trace"):
                        st.json(trace)


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
                    for (
                        ticker,
                        status,
                    ) in sorted(quality["by_ticker"].items())
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
            for (
                name,
                responsibility,
            ) in components
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

            audit_table_counts = {}

            for table in [
                "runs",
                "trade_traces",
                "trades",
                "agent_logs",
                "portfolio_snapshots",
            ]:
                rows = _sqlite_rows(f"SELECT COUNT(*) AS count FROM {table}")

                audit_table_counts[table] = rows[0]["count"] if rows else 0

            st.dataframe(
                [
                    {
                        "Table": table,
                        "Rows": count,
                    }
                    for (
                        table,
                        count,
                    ) in audit_table_counts.items()
                ],
                width="stretch",
                hide_index=True,
            )


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
        SimBroker / Alpaca · execution telemetry ·
        SQLite persistent audit
    </div>
    """
)
