-- Trades executed (live or backtest)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL,
    order_id TEXT,
    status TEXT,
    mode TEXT NOT NULL,           -- 'live' | 'backtest'
    created_at TEXT NOT NULL
);

-- Every graph node execution, for the dashboard's agent reasoning feed
CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    ticker TEXT,
    input_summary TEXT,
    output_summary TEXT,
    latency_ms REAL,
    token_usage INTEGER,
    created_at TEXT NOT NULL
);

-- Full node-level traces, keyed by trace_id, for replayability
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    input_state TEXT,             -- JSON blob
    output_state TEXT,            -- JSON blob
    created_at TEXT NOT NULL
);

-- Portfolio-level snapshots (equity, exposure, correlation summary)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equity REAL NOT NULL,
    total_exposure_pct REAL,
    sector_concentration TEXT,    -- JSON blob
    circuit_breaker_tripped BOOLEAN,
    created_at TEXT NOT NULL
);
