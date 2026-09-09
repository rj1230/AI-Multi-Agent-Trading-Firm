"""
AI Multi-Agent Trading Firm — Dashboard
Phase 10: demo-ready UI over the live agent pipeline.

Run:
    streamlit run dashboard/app.py

Architecture (kept deliberately separated, like a real frontend):
    styles.py      -> design tokens + CSS injection
    components.py  -> reusable presentational functions (KPI cards, log rows...)
    data.py        -> all file I/O / data access, cached and swappable
    app.py         -> page layout + wiring only, no markup strings, no I/O
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dashboard.ui_components import (
    format_signed_currency,
    pnl_sentiment,
    render_agent_log_entry,
    render_header,
    render_kpi_row,
    render_section_header,
)
from dashboard.data import (
    has_real_data,
    load_agent_logs,
    load_backtest_results,
    load_positions,
)
from dashboard.styles import inject_css

st.set_page_config(
    page_title="AI Multi-Agent Trading Firm",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------------------------------------------------------------------------
# Sidebar — branding + navigation + mode indicator
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">📈 Trading Firm</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-tagline">Multi-agent · risk-gated · paper trading</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["Overview", "Live Agent Feed", "Positions & P&L", "Backtest Results"],
        label_visibility="collapsed",
    )

    st.divider()
    is_live = has_real_data()
    st.caption("DATA SOURCE")
    st.markdown("🟢 Live" if is_live else "🟡 Sample data")

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Agents: NewsAgent · ChartAgent · SignalMerger · "
        "RiskAgent · ExecutionAgent\n\nBroker: SimBroker / AlpacaBroker (paper)"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_header(
    title="AI Multi-Agent Trading Firm",
    subtitle="LangGraph agent pipeline · Alpaca paper trading · risk-gated execution",
    status_label="Live" if is_live else "Demo mode",
    status_ok=is_live,
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    positions_df = load_positions()
    equity_df, trades_df = load_backtest_results()
    logs_df = load_agent_logs(max_rows=5)

    total_unrealized = (
        positions_df["unrealized_pnl"].sum()
        if "unrealized_pnl" in positions_df
        else 0.0
    )
    start_eq = equity_df["equity"].iloc[0]
    end_eq = equity_df["equity"].iloc[-1]
    total_return_pct = (end_eq / start_eq - 1) * 100

    render_kpi_row(
        [
            {"label": "Open Positions", "value": str(len(positions_df))},
            {
                "label": "Unrealized P&L",
                "value": format_signed_currency(total_unrealized),
                "sentiment": pnl_sentiment(total_unrealized),
            },
            {
                "label": "Backtest Return",
                "value": f"{total_return_pct:+.1f}%",
                "sentiment": pnl_sentiment(total_return_pct),
            },
            {"label": "Equity (Backtest)", "value": f"${end_eq:,.0f}"},
        ]
    )

    st.write("")
    col_chart, col_feed = st.columns([2, 1])

    with col_chart:
        render_section_header("📊", "Backtest Equity Curve")
        st.line_chart(equity_df.set_index("date")["equity"], height=320)

    with col_feed:
        render_section_header("🧠", "Recent Agent Activity")
        for _, row in logs_df.iterrows():
            ts = row.get("timestamp")
            ts_str = (
                ts.strftime("%H:%M:%S")
                if ts is not None and not str(ts) == "NaT"
                else ""
            )
            render_agent_log_entry(
                ts_str,
                row.get("agent", ""),
                row.get("ticker", ""),
                row.get("message", ""),
            )

# ---------------------------------------------------------------------------
# Page: Live Agent Feed
# ---------------------------------------------------------------------------
elif page == "Live Agent Feed":
    render_section_header("🧠", "Live Agent Reasoning Feed")
    logs_df = load_agent_logs()

    agents_available = (
        sorted(logs_df["agent"].unique()) if "agent" in logs_df.columns else []
    )
    selected_agents = st.multiselect(
        "Filter by agent", options=agents_available, default=agents_available
    )
    filtered = (
        logs_df[logs_df["agent"].isin(selected_agents)] if selected_agents else logs_df
    )

    for _, row in filtered.iterrows():
        ts = row.get("timestamp")
        ts_str = (
            ts.strftime("%H:%M:%S") if ts is not None and not str(ts) == "NaT" else ""
        )
        render_agent_log_entry(
            ts_str, row.get("agent", ""), row.get("ticker", ""), row.get("message", "")
        )

    if filtered.empty:
        st.info("No log entries match the current filter.")

# ---------------------------------------------------------------------------
# Page: Positions & P&L
# ---------------------------------------------------------------------------
elif page == "Positions & P&L":
    render_section_header("💼", "Open Positions")
    positions_df = load_positions()

    if not positions_df.empty:
        total_unrealized = (
            positions_df["unrealized_pnl"].sum()
            if "unrealized_pnl" in positions_df
            else 0.0
        )
        render_kpi_row(
            [
                {"label": "Open Positions", "value": str(len(positions_df))},
                {
                    "label": "Unrealized P&L",
                    "value": format_signed_currency(total_unrealized),
                    "sentiment": pnl_sentiment(total_unrealized),
                },
                {
                    "label": "Sectors",
                    "value": str(positions_df["sector"].nunique())
                    if "sector" in positions_df.columns
                    else "—",
                },
            ]
        )
        st.write("")
        st.dataframe(positions_df, width="stretch", hide_index=True)
    else:
        st.info("No open positions.")

# ---------------------------------------------------------------------------
# Page: Backtest Results
# ---------------------------------------------------------------------------
elif page == "Backtest Results":
    render_section_header("📊", "Backtest Results")
    equity_df, trades_df = load_backtest_results()

    if not equity_df.empty:
        start_eq = equity_df["equity"].iloc[0]
        end_eq = equity_df["equity"].iloc[-1]
        total_return_pct = (end_eq / start_eq - 1) * 100

        render_kpi_row(
            [
                {"label": "Start Equity", "value": f"${start_eq:,.0f}"},
                {"label": "End Equity", "value": f"${end_eq:,.0f}"},
                {
                    "label": "Total Return",
                    "value": f"{total_return_pct:+.1f}%",
                    "sentiment": pnl_sentiment(total_return_pct),
                },
            ]
        )
        st.write("")
        st.line_chart(equity_df.set_index("date")["equity"], height=340)

    st.write("")
    render_section_header("📋", "Trade Log")
    st.dataframe(trades_df, width="stretch", hide_index=True)

st.divider()
st.caption(
    "Paper trading only — no real capital at risk. Built with LangGraph, Alpaca, and Streamlit."
)
