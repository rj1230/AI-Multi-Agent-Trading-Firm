"""
Env-driven config: mode flag (live/backtest), API keys, and the tunable
risk parameters from the RiskAgent rule set.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("TRADING_MODE", "backtest")  # "live" | "backtest"

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Risk parameters (tunable)
RISK_PER_TRADE_PCT = 0.01
ATR_STOP_MULTIPLIER = 1.5
MAX_EXPOSURE_PER_TICKER_PCT = 0.20
MAX_EXPOSURE_PER_SECTOR_PCT = 0.40
MAX_CONCURRENT_POSITIONS = 3
DAILY_CIRCUIT_BREAKER_PCT = -0.03
CORRELATION_THRESHOLD = 0.7
