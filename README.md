# AI Multi-Agent Trading Firm

LangGraph-orchestrated system of four agents (News, Chart, Risk, Execution)
coordinating like a hedge fund team, with Alpaca paper trading, portfolio-
level risk management, backtesting, and a live Streamlit dashboard.

Status: Phase 1 skeleton — structure and stubs only, no logic implemented yet.

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

## Verify
```
python -c "import langgraph, yfinance, alpaca"
```

See the full architecture + 10-phase build plan for what comes next.

![tests](https://github.com/rj1230/AI-Multi-Agent-Trading-Firm/actions/workflows/tests.yml/badge.svg)

