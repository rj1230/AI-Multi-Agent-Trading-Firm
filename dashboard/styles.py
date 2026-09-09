"""
Design tokens + CSS injection for the trading dashboard.

Keeping this isolated from app.py means the visual system can be tuned in
one place without touching layout/data logic — the same separation you'd
want in any real frontend codebase.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLOR_BG = "#0E1117"
COLOR_SURFACE = "#161B22"
COLOR_SURFACE_ALT = "#1C2230"
COLOR_BORDER = "#2A2F3A"
COLOR_TEXT = "#E6E6E6"
COLOR_TEXT_MUTED = "#8A93A6"
COLOR_ACCENT = "#3B82F6"
COLOR_POSITIVE = "#22C55E"
COLOR_NEGATIVE = "#EF4444"
COLOR_WARNING = "#F59E0B"

AGENT_COLORS = {
    "NewsAgent": "#38BDF8",
    "ChartAgent": "#A78BFA",
    "SignalMerger": "#F472B6",
    "RiskAgent": "#F59E0B",
    "ExecutionAgent": "#22C55E",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, sans-serif;
            }}

            /* Tighten default Streamlit padding for a denser, app-like feel */
            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }}

            /* Header / branding bar */
            .dash-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding-bottom: 0.75rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid {COLOR_BORDER};
            }}
            .dash-header h1 {{
                font-size: 1.5rem;
                font-weight: 700;
                margin: 0;
                letter-spacing: -0.02em;
            }}
            .dash-header .subtitle {{
                color: {COLOR_TEXT_MUTED};
                font-size: 0.85rem;
                margin-top: 0.15rem;
            }}

            /* Status pill */
            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid {COLOR_BORDER};
                background: {COLOR_SURFACE};
            }}
            .status-dot {{
                width: 7px;
                height: 7px;
                border-radius: 50%;
                display: inline-block;
            }}

            /* KPI card */
            .kpi-card {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1rem 1.1rem;
                height: 100%;
            }}
            .kpi-label {{
                color: {COLOR_TEXT_MUTED};
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.35rem;
            }}
            .kpi-value {{
                font-size: 1.55rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: -0.01em;
            }}
            .kpi-delta {{
                font-size: 0.8rem;
                font-weight: 600;
                margin-top: 0.25rem;
            }}
            .kpi-delta.positive {{ color: {COLOR_POSITIVE}; }}
            .kpi-delta.negative {{ color: {COLOR_NEGATIVE}; }}
            .kpi-delta.neutral {{ color: {COLOR_TEXT_MUTED}; }}

            /* Agent log entries */
            .log-entry {{
                display: flex;
                gap: 0.75rem;
                padding: 0.65rem 0.8rem;
                border-radius: 8px;
                border: 1px solid {COLOR_BORDER};
                background: {COLOR_SURFACE};
                margin-bottom: 0.5rem;
            }}
            .log-time {{
                color: {COLOR_TEXT_MUTED};
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                white-space: nowrap;
                padding-top: 0.1rem;
            }}
            .agent-badge {{
                display: inline-block;
                padding: 0.12rem 0.5rem;
                border-radius: 5px;
                font-size: 0.72rem;
                font-weight: 700;
                color: #0E1117;
                white-space: nowrap;
            }}
            .log-message {{
                color: {COLOR_TEXT};
                font-size: 0.87rem;
                line-height: 1.4;
            }}
            .log-ticker {{
                color: {COLOR_TEXT_MUTED};
                font-weight: 600;
            }}

            /* Section header */
            .section-header {{
                font-size: 1rem;
                font-weight: 700;
                margin: 0.25rem 0 0.75rem 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}

            /* Sidebar branding */
            .sidebar-brand {{
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.1rem;
            }}
            .sidebar-tagline {{
                color: {COLOR_TEXT_MUTED};
                font-size: 0.75rem;
                margin-bottom: 1.25rem;
            }}

            /* Dataframe polish */
            [data-testid="stDataFrame"] {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                overflow: hidden;
            }}

            footer {{ visibility: hidden; }}
            #MainMenu {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
