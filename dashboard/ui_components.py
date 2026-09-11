"""
dashboard/ui_components.py

Reusable, presentation-only components. Nothing in this file touches
disk or the network — every function takes plain data (from data.py)
and returns/renders markup. That split is what makes these testable
and reusable across Overview / Reasoning Feed / Backtest Report
without duplicating card markup three times.
"""

from __future__ import annotations

import streamlit as st

from dashboard.data import BacktestSummary, PortfolioSnapshot, Position, TickCard
from dashboard.styles import COLOR, SPACING, status_colors


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

def metric_row(snapshot: PortfolioSnapshot) -> None:
    """Top-row metric cards: equity, positions count, circuit-breaker status."""
    c1, c2, c3 = st.columns(3)

    pnl_color = COLOR["status_executed"] if snapshot.daily_pnl >= 0 else COLOR["status_error"]
    pnl_sign = "+" if snapshot.daily_pnl >= 0 else ""
    with c1:
        _metric_card(
            label="Current Equity",
            value=f"${snapshot.equity:,.2f}",
            sub=f"{pnl_sign}${snapshot.daily_pnl:,.2f} ({pnl_sign}{snapshot.daily_pnl_pct:.2%}) today",
            sub_color=pnl_color,
        )
    with c2:
        _metric_card(label="Open Positions", value=str(len(snapshot.positions)))
    with c3:
        halted = snapshot.circuit_breaker_halted
        badge_color = COLOR["status_error"] if halted else COLOR["status_executed"]
        badge_bg = COLOR["status_error_bg"] if halted else COLOR["status_executed_bg"]
        _metric_card(
            label="Circuit Breaker",
            value="HALTED" if halted else "OK",
            value_color=badge_color,
            value_bg=badge_bg,
        )


def _metric_card(label, value, sub=None, sub_color=None, value_color=None, value_bg=None) -> None:
    value_style = f"color:{value_color};" if value_color else f"color:{COLOR['text_primary']};"
    bg_style = f"background:{value_bg};display:inline-block;padding:2px 10px;border-radius:4px;" if value_bg else ""
    sub_html = (
        f"<div style='color:{sub_color or COLOR['text_secondary']};font-size:13px;margin-top:{SPACING['xs']};'>{sub}</div>"
        if sub
        else ""
    )
    st.markdown(
        f"""
        <div class="dash-card">
            <div style="color:{COLOR['text_secondary']};font-size:13px;">{label}</div>
            <div class="mono-fig" style="font-size:26px;font-weight:600;{value_style}{bg_style}margin-top:2px;">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def positions_table(positions: list[Position]) -> None:
    """Positions table with an inline per-ticker cap-usage bar — the
    single visual the design doc calls out as doing the most work to
    communicate 'risk-aware system'.
    """
    if not positions:
        st.markdown(
            f"<div style='color:{COLOR['text_muted']};padding:{SPACING['md']} 0;'>No open positions.</div>",
            unsafe_allow_html=True,
        )
        return

    header = st.columns([1.2, 1, 1.4, 1.4, 1, 2])
    for col, label in zip(header, ["Ticker", "Shares", "Sector", "Mkt Value", "% Port.", "Ticker cap used"]):
        col.markdown(f"<span style='color:{COLOR['text_secondary']};font-size:12px;'>{label}</span>", unsafe_allow_html=True)

    for p in sorted(positions, key=lambda x: x.market_value, reverse=True):
        row = st.columns([1.2, 1, 1.4, 1.4, 1, 2])
        row[0].markdown(f"**{p.ticker}**")
        row[1].markdown(f"<span class='mono-fig'>{p.shares:,.2f}</span>", unsafe_allow_html=True)
        row[2].markdown(p.sector)
        row[3].markdown(f"<span class='mono-fig'>${p.market_value:,.2f}</span>", unsafe_allow_html=True)
        row[4].markdown(f"<span class='mono-fig'>{p.pct_of_portfolio:.1%}</span>", unsafe_allow_html=True)
        row[5].markdown(_cap_bar_html(p.pct_of_ticker_cap), unsafe_allow_html=True)


def _cap_bar_html(fraction: float) -> str:
    fraction = max(0.0, min(fraction, 1.0))
    fill_color = COLOR["status_rejected"] if fraction > 0.85 else COLOR["accent"]
    return f"""
        <div style="background:{COLOR['surface_raised']};border-radius:3px;height:8px;width:100%;margin-top:6px;">
            <div style="background:{fill_color};width:{fraction * 100:.0f}%;height:8px;border-radius:3px;"></div>
        </div>
    """


def recent_activity_list(ticks: list[TickCard]) -> None:
    if not ticks:
        st.markdown(
            f"""<div style='color:{COLOR['text_muted']};padding:{SPACING['md']} 0;'>
            No live tick log yet — the live agent-log persistence hook hasn't been wired.
            See the Reasoning Feed tab for backtest-run history in the meantime.
            </div>""",
            unsafe_allow_html=True,
        )
        return
    for t in ticks:
        text_color, bg_color, glyph = status_colors(t.outcome)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:{SPACING['sm']};padding:{SPACING['sm']} 0;border-bottom:1px solid {COLOR['border']};">
                <span style="background:{bg_color};color:{text_color};border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600;min-width:78px;text-align:center;">
                    {glyph} {t.outcome.upper()}
                </span>
                <span style="font-weight:600;width:60px;">{t.ticker}</span>
                <span style="color:{COLOR['text_secondary']};font-size:13px;">{t.headline_reason}</span>
                <span style="margin-left:auto;color:{COLOR['text_muted']};font-size:12px;">{t.tick_date}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sector_exposure_bars(exposure: dict[str, float], cap: float = 0.40) -> None:
    """Grouped bar of sector exposure vs. the sector cap line — the
    visual counterpart to the PortfolioRiskCoordinator sector-cap test.
    """
    if not exposure:
        return
    for sector, pct in sorted(exposure.items(), key=lambda kv: kv[1], reverse=True):
        over = pct > cap
        fill_color = COLOR["status_rejected"] if over else COLOR["accent"]
        width = min(pct / (cap * 1.5), 1.0) * 100  # scale so the cap line sits at 2/3 width, headroom visible
        cap_pos = min(cap / (cap * 1.5), 1.0) * 100
        st.markdown(
            f"""
            <div style="margin-bottom:{SPACING['sm']};">
                <div style="display:flex;justify-content:space-between;font-size:13px;">
                    <span>{sector}</span>
                    <span class="mono-fig" style="color:{fill_color if over else COLOR['text_secondary']};">{pct:.1%} / {cap:.0%} cap</span>
                </div>
                <div style="position:relative;background:{COLOR['surface_raised']};border-radius:3px;height:10px;margin-top:4px;">
                    <div style="background:{fill_color};width:{width:.0f}%;height:10px;border-radius:3px;"></div>
                    <div style="position:absolute;left:{cap_pos:.0f}%;top:-2px;height:14px;width:2px;background:{COLOR['text_muted']};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Reasoning Feed tab
# ---------------------------------------------------------------------------

def reasoning_card(card: TickCard) -> None:
    text_color, bg_color, glyph = status_colors(card.outcome)
    with st.container():
        st.markdown(
            f"""
            <div class="dash-card" style="margin-bottom:{SPACING['md']};">
                <div style="display:flex;align-items:center;gap:{SPACING['sm']};border-bottom:1px solid {COLOR['border']};padding-bottom:{SPACING['sm']};margin-bottom:{SPACING['sm']};">
                    <span style="font-weight:600;font-size:15px;">{card.ticker}</span>
                    <span style="color:{COLOR['text_muted']};">·</span>
                    <span class="mono-fig" style="color:{COLOR['text_secondary']};font-size:13px;">{card.tick_date}</span>
                    <span style="margin-left:auto;background:{bg_color};color:{text_color};border-radius:4px;padding:2px 10px;font-size:12px;font-weight:600;">
                        {glyph} {card.outcome.upper()}
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        for step in card.steps:
            _reasoning_step_line(step)
        if not card.steps and card.headline_reason:
            st.markdown(
                f"<div style='color:{COLOR['text_secondary']};font-size:13px;'>{card.headline_reason}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _reasoning_step_line(step) -> None:
    if step.passed is None:
        line_color = COLOR["text_primary"]
        mark = ""
    elif step.passed:
        line_color = COLOR["status_executed"]
        mark = "✓ "
    else:
        # This is the single most important visual moment in the dashboard —
        # the failing rule must be unmistakably distinct from passing ones.
        line_color = COLOR["status_error"]
        mark = "✕ "
    st.markdown(
        f"""
        <div style="font-size:13px;padding:3px 0;color:{line_color};">
            {step.icon} <strong>{mark}{step.label}</strong>
            <span style="color:{COLOR['text_secondary']};"> — {step.detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def outcome_filter(ticks: list[TickCard], key: str = "reasoning_filter") -> list[TickCard]:
    """The highest-value filter per the design doc: 'show me only
    rejections' is the first thing a skeptical viewer clicks.
    """
    outcomes = sorted({t.outcome for t in ticks})
    selected = st.multiselect(
        "Filter by outcome",
        options=outcomes,
        default=outcomes,
        key=key,
        label_visibility="collapsed",
        placeholder="Filter by outcome…",
    )
    if not selected:
        return ticks
    return [t for t in ticks if t.outcome in selected]


# ---------------------------------------------------------------------------
# Backtest Report tab
# ---------------------------------------------------------------------------

def backtest_headline_metrics(summary: BacktestSummary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card(label="Sharpe Ratio", value=f"{summary.sharpe_ratio:.2f}")
    with c2:
        _metric_card(label="Win Rate", value=f"{summary.win_rate:.1%}")
    with c3:
        dd_color = COLOR["status_error"] if summary.max_drawdown < -0.15 else COLOR["text_primary"]
        _metric_card(label="Max Drawdown", value=f"{summary.max_drawdown:.1%}", value_color=dd_color)
    with c4:
        beat_market = summary.total_return > summary.buy_hold_return
        ret_color = COLOR["status_executed"] if beat_market else COLOR["status_rejected"]
        _metric_card(
            label="Total Return vs. Buy & Hold",
            value=f"{summary.total_return:.1%}",
            sub=f"Buy & hold: {summary.buy_hold_return:.1%}",
            value_color=ret_color,
        )


def limitations_caption(note: str) -> None:
    st.markdown(
        f"<div style='color:{COLOR['text_muted']};font-size:12px;padding-top:{SPACING['sm']};'>Note: {note}</div>",
        unsafe_allow_html=True,
    )
