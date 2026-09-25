-- Persistence schema for trades, agent activity, traces, and backtest runs.

PRAGMA foreign_keys = ON;

-- Trades executed in either live or backtest mode.
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    qty REAL NOT NULL CHECK (qty > 0),
    price REAL CHECK (price IS NULL OR price >= 0),
    order_id TEXT,
    status TEXT,
    mode TEXT NOT NULL CHECK (mode IN ('live', 'backtest')),
    created_at TEXT NOT NULL
);

-- Every graph-node execution for the dashboard's agent reasoning feed.
CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    ticker TEXT,
    input_summary TEXT,
    output_summary TEXT,
    latency_ms REAL CHECK (
        latency_ms IS NULL
        OR latency_ms >= 0
    ),
    token_usage INTEGER CHECK (
        token_usage IS NULL
        OR token_usage >= 0
    ),
    created_at TEXT NOT NULL
);

-- Full node-level traces keyed by trace_id, enabling replayability.
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    input_state TEXT,
    output_state TEXT,
    created_at TEXT NOT NULL
);

-- Portfolio-level snapshots.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equity REAL NOT NULL,
    total_exposure_pct REAL CHECK (
        total_exposure_pct IS NULL
        OR total_exposure_pct >= 0
    ),
    sector_concentration TEXT,
    circuit_breaker_tripped INTEGER NOT NULL DEFAULT 0 CHECK (
        circuit_breaker_tripped IN (0, 1)
    ),
    created_at TEXT NOT NULL
);

-- One row per backtest or live experiment run.
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'pending',
            'running',
            'completed',
            'failed',
            'cancelled'
        )
    ),
    start_date TEXT,
    end_date TEXT,
    tickers_json TEXT NOT NULL,
    starting_equity REAL CHECK (
        starting_equity IS NULL
        OR starting_equity >= 0
    ),
    final_equity REAL CHECK (
        final_equity IS NULL
        OR final_equity >= 0
    ),
    realized_pnl REAL,
    benchmark_return REAL,
    trade_trace_count INTEGER NOT NULL DEFAULT 0 CHECK (
        trade_trace_count >= 0
    ),
    metadata_json TEXT
);

-- One complete deterministic trade trace per ticker/tick.
CREATE TABLE IF NOT EXISTS trade_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trace_key TEXT NOT NULL UNIQUE,
    tick_id TEXT,
    ticker TEXT NOT NULL,
    simulated_date TEXT,
    outcome TEXT,

    -- Signal direction is distinct from execution side.
    -- Signal direction represents market interpretation, not an order action.
    signal_direction TEXT CHECK (
        signal_direction IS NULL
        OR signal_direction IN (
            'bullish',
            'bearish',
            'neutral'
        )
    ),
    signal_confidence REAL CHECK (
        signal_confidence IS NULL
        OR (
            signal_confidence >= 0
            AND signal_confidence <= 1
        )
    ),

    -- Risk-stage decision and sizing.
    risk_approved INTEGER CHECK (
        risk_approved IS NULL
        OR risk_approved IN (0, 1)
    ),
    risk_raw_shares REAL CHECK (
        risk_raw_shares IS NULL
        OR risk_raw_shares >= 0
    ),
    risk_proposed_shares REAL CHECK (
        risk_proposed_shares IS NULL
        OR risk_proposed_shares >= 0
    ),
    risk_final_shares REAL CHECK (
        risk_final_shares IS NULL
        OR risk_final_shares >= 0
    ),

    -- Portfolio coordinator decision.
    coordinator_approved INTEGER CHECK (
        coordinator_approved IS NULL
        OR coordinator_approved IN (0, 1)
    ),
    coordinator_shares REAL CHECK (
        coordinator_shares IS NULL
        OR coordinator_shares >= 0
    ),

    -- Broker execution outcome.
    execution_attempted INTEGER CHECK (
        execution_attempted IS NULL
        OR execution_attempted IN (0, 1)
    ),
    execution_success INTEGER CHECK (
        execution_success IS NULL
        OR execution_success IN (0, 1)
    ),
    execution_side TEXT CHECK (
        execution_side IS NULL
        OR execution_side IN ('BUY', 'SELL')
    ),
    execution_quantity REAL CHECK (
        execution_quantity IS NULL
        OR execution_quantity >= 0
    ),
    execution_price REAL CHECK (
        execution_price IS NULL
        OR execution_price >= 0
    ),
    execution_order_id TEXT,

    -- Portfolio accounting result.
    realized_pnl REAL,
    remaining_shares REAL CHECK (
        remaining_shares IS NULL
        OR remaining_shares >= 0
    ),
    position_closed INTEGER CHECK (
        position_closed IS NULL
        OR position_closed IN (0, 1)
    ),

    -- Canonical serialized TradeTrace for replay and audit fidelity.
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

-- Query-performance indexes.
CREATE INDEX IF NOT EXISTS idx_trades_trace_id
    ON trades (trace_id);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_created_at
    ON trades (ticker, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_logs_trace_id
    ON agent_logs (trace_id);

CREATE INDEX IF NOT EXISTS idx_agent_logs_created_at
    ON agent_logs (created_at);

CREATE INDEX IF NOT EXISTS idx_traces_trace_id
    ON traces (trace_id);

CREATE INDEX IF NOT EXISTS idx_trade_traces_run_id
    ON trade_traces (run_id);

CREATE INDEX IF NOT EXISTS idx_trade_traces_ticker
    ON trade_traces (ticker);

CREATE INDEX IF NOT EXISTS idx_trade_traces_simulated_date
    ON trade_traces (simulated_date);

CREATE INDEX IF NOT EXISTS idx_trade_traces_outcome
    ON trade_traces (outcome);

CREATE INDEX IF NOT EXISTS idx_trade_traces_run_ticker_date
    ON trade_traces (run_id, ticker, simulated_date);