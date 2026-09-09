"""
Env-driven config: mode flag (live/backtest) and API keys.

Risk parameters live in config/risk_config.py, loaded from
mandates/*.yaml via load_risk_config() -- not here. This file only
holds env-derived values that aren't part of the risk mandate.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("TRADING_MODE", "backtest")  # "live" | "backtest"

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# LangSmith tracing (Phase 8). No code changes needed elsewhere -- LangGraph
# reads these three env vars directly and auto-instruments every node/edge
# in build_graph()/build_coordination_subgraph() once LANGCHAIN_TRACING_V2
# is set. Kept here (not just in .env) so `python -c "from config import settings"`
# fails loudly if a required var is missing, rather than tracing silently
# no-op'ing.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-multi-agent-trading-firm")

PAPER_STARTING_EQUITY = float(os.getenv("PAPER_STARTING_EQUITY", 100_000))
ATR_PERIOD = 14  # bars used for RiskAgent's ATR calculation (graph/nodes.py)
