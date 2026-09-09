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
LLM_API_KEY = os.getenv("LLM_API_KEY")

PAPER_STARTING_EQUITY = float(os.getenv("PAPER_STARTING_EQUITY", 100_000))
ATR_PERIOD = 14  # bars used for RiskAgent's ATR calculation (graph/nodes.py)
