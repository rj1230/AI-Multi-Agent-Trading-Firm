"""
dashboard/styles.py

Central design-token module. Every color, spacing, and font choice used
across the dashboard lives here — no component should hardcode a hex
value or px size. This is what keeps the three tabs visually consistent
as the app grows past Phase 10.

Palette rationale: this is a risk-gated trading system, not a generic
fintech SaaS. The palette leans toward a "control room" feel — dark
ledger background, monospace numerics for figures, and a strict
traffic-light semantic (green/gray/amber/red) that is NEVER used
decoratively. If a color appears, it means something specific.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

COLOR = {
    "bg": "#0F1115",  # near-black with a blue undertone
    "surface": "#161920",  # card background
    "surface_raised": "#1D2129",  # nested / hovered surface
    "border": "#2A2E38",
    "text_primary": "#E8E9ED",
    "text_secondary": "#8B90A0",
    "text_muted": "#5C6270",
    "accent": "#4C8DFF",  # interactive elements only (links, active tab) — never status
    # Semantic status colors — the single most important system in this app.
    # These are the ONLY colors allowed to convey pass/fail/neutral meaning.
    "status_executed": "#3DDC84",  # green — trade went through
    "status_executed_bg": "#173226",
    "status_held": "#8B90A0",  # gray — no opinion / neutral hold, NOT a failure state
    "status_held_bg": "#1B1E26",
    "status_rejected": "#E8A93D",  # amber — a real rule fired correctly, system protected itself
    "status_rejected_bg": "#332711",
    "status_error": "#F0555F",  # red — reserved for actual system errors, never for rejections
    "status_error_bg": "#341A1D",
}

FONT = {
    # A grotesk for UI chrome + labels, a monospace for anything numeric
    # (equity, prices, percentages) so figures read like ledger entries.
    "ui": "'IBM Plex Sans', -apple-system, sans-serif",
    "mono": "'IBM Plex Mono', 'SFMono-Regular', monospace",
}

SPACING = {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px"}

RADIUS = (
    "6px"  # small and deliberate — not zero-radius broadsheet, not pill-shaped SaaS
)

STATUS_MAP = {
    "executed": ("status_executed", "status_executed_bg", "✓"),
    "held": ("status_held", "status_held_bg", "–"),
    "rejected": ("status_rejected", "status_rejected_bg", "▲"),
    "error": ("status_error", "status_error_bg", "✕"),
}


def status_colors(outcome: str) -> tuple[str, str, str]:
    """Return (text_color, bg_color, glyph) for a tick outcome.

    Normalized case-insensitively. Unknown outcomes fall back to the
    neutral 'held' styling rather than red — an unrecognized outcome is
    a data issue, not necessarily a system error, and shouldn't read as
    a crash in the UI.
    """
    key = outcome.strip().lower()
    color_key, bg_key, glyph = STATUS_MAP.get(key, STATUS_MAP["held"])
    return COLOR[color_key], COLOR[bg_key], glyph


def inject_base_css() -> None:
    """Call once at the top of app.py, before rendering any tab content."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT["ui"]};
            color: {COLOR["text_primary"]};
        }}

        .stApp {{
            background-color: {COLOR["bg"]};
        }}

        /* Hide Streamlit's default chrome (hamburger menu, Deploy button,
           "Made with Streamlit" footer). This is a demo artifact for
           recruiters, not a deployed multi-user app — that bar reads as
           unfinished scaffolding, not a deliberate UI choice. */
        #MainMenu, header[data-testid="stHeader"], footer {{
            visibility: hidden;
            height: 0;
        }}

        /* With the default header hidden, block-container no longer needs
           to clear a fixed overlay — reclaim that space, but keep enough
           top padding that content doesn't sit flush against the browser
           chrome. */
        .block-container {{
            padding-top: {SPACING["xl"]};
            padding-bottom: {SPACING["xl"]};
            max-width: 1200px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: {SPACING["lg"]};
            border-bottom: 1px solid {COLOR["border"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            font-family: {FONT["ui"]};
            font-weight: 500;
            color: {COLOR["text_secondary"]};
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLOR["text_primary"]};
            border-bottom: 2px solid {COLOR["accent"]};
        }}

        .mono-fig {{
            font-family: {FONT["mono"]};
            font-variant-numeric: tabular-nums;
        }}

        .dash-card {{
            background-color: {COLOR["surface"]};
            border: 1px solid {COLOR["border"]};
            border-radius: {RADIUS};
            padding: {SPACING["md"]};
        }}

        hr {{
            border-color: {COLOR["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
