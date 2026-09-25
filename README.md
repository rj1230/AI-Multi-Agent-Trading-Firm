# AI Multi-Agent Trading Firm

## to run this project :- python -m backtest.runner --tickers AAPL,MSFT,GOOGL,JPM --start 2026-08-25 --end 2026-09-20 --run-id sep2026

> **Paper trading and historical backtesting only — no real capital at risk.**

A production-oriented **multi-agent trading research platform** that combines LLM-based market interpretation with deterministic trading infrastructure.

The system is designed around a simple principle:

> **Use AI where judgment is useful; use deterministic software where correctness, risk, and accounting matter.**

LLM-powered agents interpret market information such as **financial news and technical/chart signals**. Deterministic components handle **signal merging, risk validation, position sizing, portfolio accounting, order execution, backtesting, and performance evaluation**.

This separation makes the system easier to test, audit, reproduce, and evolve than an architecture where an LLM directly controls the entire trading loop.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │     Market Data      │
                         │   OHLCV + News Data  │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │     News Agent    │               │    Chart Agent    │
        │                   │               │                   │
        │ LLM-assisted      │               │ Technical/chart   │
        │ headline analysis │               │ interpretation    │
        └─────────┬─────────┘               └─────────┬─────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   Signal Merger      │
                         │                      │
                         │ Deterministic        │
                         │ agreement / conflict │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Risk Agent       │
                         │                      │
                         │ ATR / sizing /       │
                         │ exposure / checks    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                  ┌──────────────┐      ┌──────────────┐
                  │   HoldNode   │      │ Execution    │
                  │              │      │    Agent     │
                  │ Deterministic│      │              │
                  │ rejection    │      │ Broker API   │
                  └──────────────┘      └──────┬───────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │    SimBroker    │
                                      │                 │
                                      │ Order execution │
                                      │ + idempotency   │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ PortfolioLedger │
                                      │                 │
                                      │ Cash            │
                                      │ Positions       │
                                      │ Average cost    │
                                      │ Market value    │
                                      │ Realized P&L    │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Backtest Engine │
                                      │                 │
                                      │ Historical      │
                                      │ sessions        │
                                      │ isolated state  │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Performance     │
                                      │ Evaluation      │
                                      │                 │
                                      │ Sharpe          │
                                      │ Drawdown        │
                                      │ Returns         │
                                      │ Win rate        │
                                      │ Benchmark       │
                                      └─────────────────┘
```

---

## Core Design Philosophy

### 1. LLMs do not control the entire trading system

LLMs are used for tasks where interpretation is valuable:

* Financial-news interpretation
* Market sentiment/context extraction
* Chart-pattern interpretation
* Natural-language reasoning

Critical financial decisions are handled by deterministic code.

```text
LLM reasoning
     ↓
Structured signal
     ↓
Deterministic validation
     ↓
Risk controls
     ↓
Broker execution
     ↓
Ledger accounting
```

This reduces the surface area where nondeterministic model behavior can directly affect execution.

---

## 2. Deterministic Risk Controls

The system separates **signal generation** from **risk authorization**.

The Risk Agent is responsible for checks such as:

* ATR-based risk calculations
* Entry price validation
* Position sizing
* Available capital
* Portfolio constraints
* Risk configuration
* Trade approval/rejection

A bullish model signal does not automatically result in an order.

```text
Signal
  │
  ▼
Risk validation
  │
  ├── rejected ──► Hold
  │
  └── approved ─► Execution
```

---

## 3. Broker and Portfolio Accounting Are Separated

The system intentionally separates:

### Broker

Responsible for execution.

`SimBroker` provides:

* Simulated order execution
* Fill prices
* Order IDs
* Order idempotency
* Cash/execution constraints

### PortfolioLedger

Responsible for accounting.

The ledger is the local source of truth for:

* Cash
* Positions
* Shares
* Average acquisition cost
* Market value
* Realized P&L
* Session state

A successful broker fill is recorded in the ledger only after execution succeeds.

```text
Order
  ↓
Broker
  ↓
Confirmed Fill
  ↓
PortfolioLedger
```

This prevents rejected orders from accidentally changing portfolio state.

---

# Historical Backtesting

The project includes a historical multi-agent backtesting engine designed to replay actual market sessions.

The backtester:

* Advances through historical OHLCV sessions
* Uses simulated dates
* Applies historical data cutoffs
* Runs the same multi-agent decision pipeline
* Uses a fresh portfolio state for every run
* Uses `SimBroker` instead of real execution
* Records detailed tick-level telemetry
* Tracks news availability and provenance
* Generates an equity curve
* Generates a buy-and-hold benchmark
* Produces structured JSON results

A backtest does **not** reuse the live/paper portfolio state.

```text
Historical Session
       │
       ▼
Multi-Agent Pipeline
       │
       ▼
SimBroker
       │
       ▼
Fresh PortfolioLedger
       │
       ▼
Equity Curve
       │
       ▼
Performance Metrics
```

---

## Backtest Result Telemetry

Each backtest records more than simply whether a trade occurred.

The system captures information such as:

* Date
* Ticker
* Signal direction
* Signal confidence
* Signal agreement
* ATR
* Entry price
* Proposed position size
* Risk decision
* Risk notes
* Execution attempt
* Execution success
* Execution side
* Fill information
* News availability
* News article count
* News source
* News timestamp/provenance

This allows the system to be evaluated as an **orchestration pipeline**, not merely as a final P&L number.

---

# Portfolio Accounting

The portfolio ledger maintains explicit cost-basis accounting.

### Weighted average cost

Multiple purchases are handled using weighted average acquisition cost.

Example:

```text
10 shares @ $100
10 shares @ $120

Average cost = $110
```

Selling 5 shares at $140 produces:

```text
Realized P&L
= 5 × ($140 - $110)
= $150
```

Partial sells preserve the remaining position's average cost.

Mark-to-market operations update **market value** without overwriting historical acquisition cost.

This distinction is important for reliable realized/unrealized P&L calculations.

---

# Performance Evaluation

The backtesting layer currently evaluates the strategy using metrics including:

* Sharpe ratio
* Maximum drawdown
* Total return
* Buy-and-hold return
* Excess return versus benchmark
* Number of executions
* Closed trades
* Win rate

The architecture is also designed to support richer trade-level accounting such as:

* Realized P&L
* Winning trades
* Losing trades
* Profit factor
* Trade-level attribution

Metrics intentionally distinguish between:

```text
No closed trades
```

and:

```text
Closed trades with 0% win rate
```

For example, `win_rate = None` means there are currently no scored closed trades rather than incorrectly implying zero profitable trades.

---

# News Provenance

News availability is treated as explicit data rather than silently assuming that missing news means neutral news.

The system distinguishes states such as:

```text
AVAILABLE
UNAVAILABLE
ERROR
AVAILABLE_WITH_ZERO_ELIGIBLE_ARTICLES
```

Backtest results preserve news metadata so researchers can determine whether a trading decision was made with:

* Available news
* Missing historical news
* Empty eligible article sets
* Data-source errors

This is particularly important for avoiding accidental look-ahead or hidden-data assumptions during historical evaluation.

---

# Safety and Execution Boundaries

The project currently operates in **paper/simulated trading mode**.

There is no requirement for real capital to validate the architecture.

The system separates:

```text
Research
   ↓
Historical Backtesting
   ↓
Paper Execution
   ↓
Evaluation
```

from any future real-money execution environment.

The broker abstraction allows execution infrastructure to remain separate from strategy logic.

---

# Testing

The project has a comprehensive automated test suite covering the core trading infrastructure.

Current baseline:

```text
156 tests passed
```

The tests cover areas including:

* Portfolio ledger behavior
* Fresh backtest isolation
* Position accounting
* Weighted average cost
* Partial sells
* Realized P&L
* Ledger persistence
* Broker behavior
* Backtest execution
* Tick processing
* Risk logic
* Guardrails
* Trading-state behavior
* Agent orchestration

The project follows a correctness-first workflow:

```text
Implement
   ↓
Write / update tests
   ↓
Run focused tests
   ↓
Run integration tests
   ↓
Run full suite
   ↓
Only then expand architecture
```

---

# Project Structure

```text
AI-Multi-Agent-Trading-Firm/
│
├── agents/
│   ├── news_agent.py
│   ├── chart_agent.py
│   ├── risk_agent.py
│   └── ...
│
├── broker/
│   ├── protocol.py
│   ├── sim_broker.py
│   └── alpaca_broker.py
│
├── portfolio/
│   ├── ledger.py
│   ├── state.py
│   └── correlation.py
│
├── graph/
│   ├── nodes.py
│   ├── state.py
│   └── ...
│
├── backtest/
│   ├── runner.py
│   ├── metrics.py
│   ├── results/
│   └── ...
│
├── config/
│   ├── risk_config.py
│   ├── sectors.py
│   └── settings.py
│
├── tests/
│   ├── test_ledger.py
│   ├── test_backtest_runner.py
│   ├── test_tick_runner.py
│   ├── test_guardrails.py
│   └── ...
│
└── README.md
```

---

# Example Backtest

Example command:

```powershell
python -m backtest.runner `
  --start 2024-06-03 `
  --end 2024-06-10 `
  --tickers AAPL,MSFT,GOOGL,JPM
```

The run produces structured results under:

```text
backtest/results/
```

including:

* Equity curve
* Benchmark curve
* Trade records
* Tick-level telemetry
* Diagnostics
* News availability
* Performance metrics

---

# Example Backtest Output

A short historical run can produce output similar to:

```text
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

Short backtests should not be interpreted as statistically meaningful strategy validation. They are primarily useful for validating execution correctness, state isolation, accounting, and orchestration behavior.

---

# Engineering Principles

The project intentionally follows several engineering principles.

### Separation of concerns

```text
Agents       → interpretation
Merger       → deterministic signal combination
Risk         → authorization
Broker       → execution
Ledger       → accounting
Backtester   → historical replay
Metrics      → evaluation
```

### Reproducibility

Historical runs use:

* Explicit dates
* Historical data
* Simulated time
* Fresh portfolio state
* Deterministic infrastructure where possible

### Observability

The system records:

* Agent decisions
* Risk decisions
* Execution outcomes
* News provenance
* Portfolio state
* Backtest diagnostics

### Testability

Infrastructure components are independently testable rather than tightly coupled to LLM calls or broker APIs.

### Fail-safe behavior

A failed or rejected execution should not mutate portfolio accounting.

---

# Current Development Status

| Component                      | Status                |
| ------------------------------ | --------------------- |
| Multi-agent architecture       | ✅                     |
| News Agent                     | ✅                     |
| Chart/Technical Agent          | ✅                     |
| Signal merger                  | ✅                     |
| Risk controls                  | ✅                     |
| Simulated broker               | ✅                     |
| Order idempotency              | ✅                     |
| Portfolio ledger               | ✅                     |
| Weighted average cost          | ✅                     |
| Partial-sell accounting        | ✅                     |
| Realized P&L accounting        | ✅                     |
| Fresh backtest isolation       | ✅                     |
| Historical backtesting         | ✅                     |
| News provenance                | ✅                     |
| Backtest diagnostics           | ✅                     |
| Performance metrics            | 🟢 Active development |
| Closed-trade attribution       | 🟡 Next               |
| Profit factor                  | 🟡 Next               |
| Advanced strategy optimization | ⏳ Future              |
| Production UI/observability    | ⏳ Future              |

---

# Roadmap

### Phase 1 — Trading Infrastructure

* [x] Multi-agent architecture
* [x] Deterministic signal merging
* [x] Risk controls
* [x] Broker abstraction
* [x] Simulated execution
* [x] Portfolio ledger
* [x] Backtest isolation

### Phase 2 — Accounting & Evaluation

* [x] Average-cost accounting
* [x] Partial-sell handling
* [x] Realized P&L persistence
* [x] Equity curves
* [x] Benchmark comparison
* [x] Drawdown analysis
* [ ] Closed-trade attribution
* [ ] Profit factor
* [ ] Trade-level P&L attribution

### Phase 3 — Research Platform

* [ ] Walk-forward evaluation
* [ ] Multiple market regimes
* [ ] Transaction-cost modeling
* [ ] Slippage modeling
* [ ] Robustness testing
* [ ] Parameter sensitivity analysis
* [ ] Strategy comparison framework

### Phase 4 — Observability

* [ ] Agent traces
* [ ] Decision-level observability
* [ ] Backtest dashboards
* [ ] Strategy diagnostics
* [ ] Experiment tracking

### Phase 5 — Advanced Agentic Research

* [ ] Strategy research agents
* [ ] Automated experiment generation
* [ ] Evaluation agents
* [ ] Research memory
* [ ] Agent-to-agent research workflows

---

# Important Disclaimer

This project is intended for **research, software engineering, and educational purposes**.

It currently supports **paper/simulated trading and historical backtesting only**.

Historical backtest performance does not guarantee future results. Backtests can be affected by data quality, missing information, survivorship bias, look-ahead bias, transaction costs, slippage, market-regime changes, and model behavior.

**No real capital is required or assumed by this project.**

---

## Author

Built as an engineering-focused exploration of:

* Multi-agent systems
* LLM-assisted financial reasoning
* LangGraph orchestration
* Deterministic risk systems
* Algorithmic trading infrastructure
* Portfolio accounting
* Historical backtesting
* AI system evaluation
* Production-oriented ML/AI engineering
