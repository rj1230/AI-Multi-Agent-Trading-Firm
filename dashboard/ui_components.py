"""
Reusable presentational components.

Each function renders one piece of UI via st.markdown(unsafe_allow_html=True).
Keeping these as small, named functions (rather than inlining HTML in app.py)
is what makes the page layout in app.py read like a page layout, not a wall
of markup.
"""

import html

import streamlit as st

from dashboard.styles import (
    AGENT_COLORS,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
)


def render_header(
    title: str, subtitle: str, status_label: str, status_ok: bool
) -> None:
    dot_color = COLOR_POSITIVE if status_ok else COLOR_WARNING
    st.markdown(
        f"""
        <div class="dash-header">
            <div>
                <h1>{html.escape(title)}</h1>
                <div class="subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="status-pill">
                <span class="status-dot" style="background:{dot_color};"></span>
                {html.escape(status_label)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">{icon} {html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str, value: str, delta: str | None = None, sentiment: str = "neutral"
) -> None:
    """sentiment: 'positive' | 'negative' | 'neutral' — controls delta color."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {sentiment}">{html.escape(delta)}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list[dict]) -> None:
    """cards: list of {label, value, delta?, sentiment?}"""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_kpi_card(
                card["label"],
                card["value"],
                card.get("delta"),
                card.get("sentiment", "neutral"),
            )


def render_agent_log_entry(
    time_str: str, agent: str, ticker: str, message: str
) -> None:
    color = AGENT_COLORS.get(agent, "#94A3B8")
    ticker_html = (
        f'<span class="log-ticker">· {html.escape(ticker)}</span>' if ticker else ""
    )
    st.markdown(
        f"""
        <div class="log-entry">
            <div class="log-time">{html.escape(time_str)}</div>
            <div>
                <span class="agent-badge" style="background:{color};">{html.escape(agent)}</span>
                {ticker_html}
                <div class="log-message">{html.escape(message)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pnl_sentiment(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def format_signed_currency(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.2f}"
