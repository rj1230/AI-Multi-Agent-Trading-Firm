# 📈 AI Multi-Agent Trading Firm

**Multi-agent trading research platform built with LangGraph, Groq, deterministic risk controls, and a broker-agnostic portfolio ledger — for auditable, reproducible AI-assisted trading research.**

> **Paper trading and historical backtesting only — no real capital at risk.**

AI Multi-Agent Trading Firm separates *judgment* from *correctness*. LLM agents interpret financial news and technical signals; deterministic software handles signal merging, risk authorization, position sizing, execution, portfolio accounting, backtesting, and evaluation. The result is a trading pipeline that stays testable, traceable, and reproducible — not an LLM with its hands on the order button.

> **Use AI where judgment is useful; use deterministic software where correctness, risk, and accounting matter.**

---

## ✨ Key Features

- **LangGraph multi-agent pipeline** — News and Chart agents run in parallel per ticker → Signal Merger → Risk Agent, with multi-ticker fan-out, not a single-shot LLM call
- **Deterministic Signal Merger** — resolves agreement and conflict between News and Chart signals with no LLM in the loop
- **Risk authorization ≠ signal generation** — ATR-based sizing, per-ticker and per-sector caps, correlation checks, and a daily circuit breaker; a bullish signal never automatically becomes an order
- **Portfolio Risk Coordinator** — book-level arbitration across all tickers in a tick; highest-conviction proposals win the shared exposure budget
- **Broker / ledger separation** — `SimBroker` (or Alpaca paper) executes; `PortfolioLedger` accounts (weighted average cost, partial sells, realized P&L) and updates only after a confirmed fill
- **Historical backtesting engine** — replays real cached sessions with simulated dates, historical data cutoffs, a fresh isolated portfolio per run, and a buy-and-hold benchmark
- **News provenance** — missing news is explicit data (`AVAILABLE` / `UNAVAILABLE` / `ERROR` / `AVAILABLE_WITH_ZERO_ELIGIBLE_ARTICLES`), never silently treated as neutral
- **Full observability** — LangSmith traces across every node plus tick-level telemetry (signal, risk decision, execution, fills, news provenance) saved as structured JSON
- **Streamlit dashboard** — run backtests and inspect pipeline stages, portfolio state, and results
- **Correctness-first test suite** — 168 automated tests with GitHub Actions CI

---

## 🏗️ Architecture

```
                 Market Data (OHLCV + News · live or historical replay)
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
             ┌─────────────┐                     ┌─────────────┐
             │ News Agent  │                     │ Chart Agent │
             │ (LLM)       │                     │ (technical) │
             └──────┬──────┘                     └──────┬──────┘
                    └─────────────────┬─────────────────┘
                                      ▼
                             ┌─────────────────┐
                             │  Signal Merger  │  deterministic
                             └────────┬────────┘
                                      ▼
                             ┌─────────────────┐
                             │   Risk Agent    │  ATR sizing · caps · checks
                             └────────┬────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Portfolio Risk Coordinator│  book-level, all tickers
                        └─────────────┬─────────────┘
                              ┌───────┴────────┐
                              ▼                ▼
                        ┌──────────┐    ┌───────────────┐
                        │ HoldNode │    │ Execution     │
                        │          │    │ Agent         │
                        └──────────┘    └───────┬───────┘
                                                ▼
                                   SimBroker / Alpaca (paper)
                                                ▼
                                        PortfolioLedger
                                                ▼
                                Backtest Engine → Performance Metrics
```

**Flow:** Market data enters per-ticker subgraphs that run concurrently → News Agent interprets headlines while Chart Agent reads technical signals → Signal Merger deterministically combines them → Risk Agent authorizes or rejects with sizing based on ATR → Portfolio Risk Coordinator arbitrates across all tickers in the tick → approved trades go to the Execution Agent and broker, rejected ones to HoldNode → confirmed fills are recorded in the PortfolioLedger → the backtest engine turns the session history into an equity curve and metrics. Every node emits traces and telemetry.

---

## 🧩 Tech Stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM inference | Groq (gpt-oss models) |
| Output validation | Pydantic (retry-then-fallback) + NeMo Guardrails |
| Market data | Alpaca (with yfinance fallback) |
| News data | NewsAPI |
| Broker | SimBroker (backtest) / Alpaca paper trading |
| Portfolio accounting | JSON-backed `PortfolioLedger` (weighted average cost) |
| Observability | LangSmith tracing + tick-level telemetry |
| Evaluation | Sharpe, max drawdown, win rate, profit factor, buy-and-hold benchmark |
| Demo UI | Streamlit |
| Testing / CI | pytest, ruff, GitHub Actions |
| Runtime | Python 3.13, uv |

---

## 📂 Project Structure

```
AI-Multi-Agent-Trading-Firm/
├── agents/            # News / Chart / Risk / Execution agents, signal merger, guardrails, retry helper
├── graph/             # LangGraph state, nodes, and graph builder
├── orchestrator/      # tick_runner.py — multi-ticker fan-out + coordinator + execution
├── portfolio/         # PortfolioLedger, risk coordinator, correlation, circuit breaker
├── broker/            # Broker protocol, SimBroker, AlpacaBroker
├── data_sources/      # Live + historical replay sources, shared Pydantic schemas
├── data_cache/        # Cached historical OHLCV + news (git-ignored)
├── backtest/          # runner.py, metrics.py, results/
├── mandates/          # default.yaml — risk mandate
├── config/            # risk_config.py, sectors.py, settings.py, guardrails/
├── dashboard/         # Streamlit UI components
├── harness/
├── scheduler/
├── storage/
├── telemetry/
├── tools/
├── scripts/
├── tests/             # ledger, backtest runner, tick runner, guardrails, nodes, ...
├── ui.py              # Streamlit entrypoint
├── run_one_tick.py    # Run a single live tick
├── smoke_test.py      # API connectivity check
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure .env
GROQ_API_KEY=...
NEWSAPI_KEY=...
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
TRADING_MODE=backtest          # or: live (paper)
LANGCHAIN_TRACING_V2=true      # optional — LangSmith tracing
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=...

# 3. Run a historical backtest
python -m backtest.runner --tickers AAPL,MSFT,GOOGL,JPM --start 2026-08-25 --end 2026-09-20 --run-id sep2026

# 4. Launch the dashboard
streamlit run ui.py

# 5. Run the test suite
pytest tests/ -v
```

Backtests replay cached sessions, so populate the historical OHLCV/news cache first (see `seed_news_cache.py`). Results are written to `backtest/results/<run-id>.json`.

---

## 🛡️ Guardrails & Risk Controls

- **LLM output gate** — agent outputs are validated against strict Pydantic schemas; on failure the agent retries, then falls back to a neutral signal instead of passing bad data downstream
- **Deterministic risk authorization** — the default mandate enforces 1% equity risk per trade with a 1.5× ATR stop, 20% max per ticker, 40% max per sector, 3 concurrent positions max, a −3% daily circuit breaker, and correlation checks that reject or downsize new positions above 0.7
- **Book-level arbitration** — when the shared sector/exposure budget runs out, the lowest-conviction proposals lose their slot
- **Fail-safe accounting** — rejected or failed orders never mutate the ledger; order idempotency prevents duplicate fills
- **Lookahead protection** — backtest mode refuses to run without a simulated date, and every data fetch is cut off at that date

---

## 📊 Backtesting & Evaluation

Each backtest runs the same multi-agent pipeline against a **fresh portfolio state** through `SimBroker` and never touches the live/paper ledger. Results include:

- Equity curve and equal-weight buy-and-hold benchmark
- Execution events and closed round-trip trades
- Tick-level telemetry: signal direction/confidence/agreement, ATR, entry price, proposed size, risk decision and notes, execution outcome, fills
- News availability, article count, source, and timestamp provenance
- Orchestration diagnostics: risk approvals, local/book rejections, execution attempts/successes/failures
- Metrics: Sharpe ratio, max drawdown, total return, excess return vs benchmark, realized P&L, profit factor, win rate

`win_rate = None` means there are no scored closed trades — not a 0% win rate.

**Weighted average cost example**

```
10 shares @ $100 + 10 shares @ $120  →  average cost = $110
Sell 5 @ $140  →  realized P&L = 5 × ($140 − $110) = $150
```

Partial sells preserve the remaining position's average cost, and mark-to-market updates market value without overwriting acquisition cost.

**Example output**

```
============================================================
BACKTEST COMPLETE
============================================================
Sessions      : 5
Executed      : 3
Final equity  : $100,463.28
Benchmark     : 5 points
News tickers  : 4
Risk approvals: 7
Exec attempts : 3
Exec successes: 3
Exec failures : 0
============================================================
```

Short backtests are for validating execution correctness, state isolation, accounting, and orchestration — not for judging strategy performance.
