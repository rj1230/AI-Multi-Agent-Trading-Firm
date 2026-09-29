"""
Persistent run-level audit storage.

RunAuditStore persists backtest/live run metadata and deterministic
TradeTrace records into SQLite.

This module is observational only. It must never participate in trading
decisions, risk sizing, execution, or portfolio accounting.

Execution lifecycle semantics:

    requested quantity != filled quantity

A normalized trade is created only when:

    filled_quantity > 0

This allows full fills and partial fills to be represented without
over-accounting requested-but-unfilled quantity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from storage.db import get_connection, init_db, insert_trade
from telemetry.trade_trace import TradeTrace

_EXECUTION_STATUSES = {
    "filled",
    "pending",
    "partially_filled",
    "rejected",
    "canceled",
    "expired",
}


class RunAuditStore:
    """SQLite persistence layer for trading runs and TradeTrace records."""

    def __init__(self, db_path: str = "storage/trading_firm.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as conn:
            init_db(conn)
            _ensure_execution_lifecycle_columns(conn)

    # ---------- Public API ----------

    def start_run(
        self,
        run_id: str,
        *,
        started_at: datetime | str,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        tickers: Iterable[str] = (),
        starting_equity: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new audit run."""
        if not run_id.strip():
            raise ValueError("run_id must not be empty")

        tickers_json = json.dumps(list(tickers), separators=(",", ":"))

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    started_at,
                    status,
                    start_date,
                    end_date,
                    tickers_json,
                    starting_equity,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _isoformat(started_at),
                    "running",
                    _isoformat(start_date),
                    _isoformat(end_date),
                    tickers_json,
                    starting_equity,
                    _json_dumps(metadata or {}),
                ),
            )

    def record_trade_trace(self, trace: TradeTrace) -> None:
        """
        Persist one complete TradeTrace, upserting by trace_key.

        Normalized trade persistence is fill-aware:

            filled_quantity > 0
                -> normalized trade exists

            filled_quantity == 0
                -> no normalized trade

        For legacy traces created before filled_quantity existed,
        successful executions fall back to their requested quantity.
        """
        if not isinstance(trace, TradeTrace):
            raise TypeError("trace must be a TradeTrace instance")

        payload = trace.to_dict()

        execution = payload.get("execution") or {}
        risk = payload.get("risk") or {}
        signal = payload.get("signal") or {}
        coordinator = payload.get("coordinator") or {}
        accounting = payload.get("accounting") or {}

        metadata = payload.get("metadata") or {}
        tick_id = metadata.get("tick_id")
        outcome = metadata.get("outcome")

        execution_status = _normalize_execution_status(
            execution.get("status"),
            success=execution.get("success"),
            filled_quantity=execution.get("filled_quantity"),
        )

        requested_quantity = _as_non_negative_float(execution.get("quantity"))

        filled_quantity = _resolve_filled_quantity(
            execution=execution,
            execution_status=execution_status,
        )

        trace_key = _trace_key(
            run_id=trace.run_id,
            ticker=trace.ticker,
            simulated_date=trace.simulated_date,
            tick_id=tick_id,
        )

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_traces (
                    run_id,
                    trace_key,
                    tick_id,
                    ticker,
                    simulated_date,
                    outcome,
                    signal_direction,
                    signal_confidence,
                    risk_approved,
                    risk_raw_shares,
                    risk_proposed_shares,
                    risk_final_shares,
                    coordinator_approved,
                    coordinator_shares,
                    execution_attempted,
                    execution_success,
                    execution_status,
                    execution_side,
                    execution_quantity,
                    execution_filled_quantity,
                    execution_price,
                    execution_order_id,
                    realized_pnl,
                    remaining_shares,
                    position_closed,
                    trace_json,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                ON CONFLICT(trace_key) DO UPDATE SET
                    tick_id = excluded.tick_id,
                    ticker = excluded.ticker,
                    simulated_date = excluded.simulated_date,
                    outcome = excluded.outcome,
                    signal_direction = excluded.signal_direction,
                    signal_confidence = excluded.signal_confidence,
                    risk_approved = excluded.risk_approved,
                    risk_raw_shares = excluded.risk_raw_shares,
                    risk_proposed_shares = excluded.risk_proposed_shares,
                    risk_final_shares = excluded.risk_final_shares,
                    coordinator_approved = excluded.coordinator_approved,
                    coordinator_shares = excluded.coordinator_shares,
                    execution_attempted = excluded.execution_attempted,
                    execution_success = excluded.execution_success,
                    execution_status = excluded.execution_status,
                    execution_side = excluded.execution_side,
                    execution_quantity = excluded.execution_quantity,
                    execution_filled_quantity = excluded.execution_filled_quantity,
                    execution_price = excluded.execution_price,
                    execution_order_id = excluded.execution_order_id,
                    realized_pnl = excluded.realized_pnl,
                    remaining_shares = excluded.remaining_shares,
                    position_closed = excluded.position_closed,
                    trace_json = excluded.trace_json,
                    created_at = excluded.created_at
                """,
                (
                    trace.run_id,
                    trace_key,
                    tick_id,
                    trace.ticker,
                    _isoformat(trace.simulated_date),
                    outcome,
                    _normalize_signal_direction(signal.get("direction")),
                    signal.get("confidence"),
                    _bool_to_int(risk.get("approved")),
                    risk.get("raw_shares"),
                    risk.get("proposed_shares"),
                    risk.get("final_shares"),
                    _bool_to_int(coordinator.get("approved")),
                    coordinator.get("shares"),
                    _bool_to_int(execution.get("attempted")),
                    _bool_to_int(execution.get("success")),
                    execution_status,
                    _normalize_execution_side(execution.get("side")),
                    requested_quantity,
                    filled_quantity,
                    execution.get("price"),
                    execution.get("order_id"),
                    accounting.get("realized_pnl"),
                    accounting.get("remaining_shares"),
                    _bool_to_int(accounting.get("position_closed")),
                    _json_dumps(payload),
                    _isoformat(trace.timestamp),
                ),
            )

            self._sync_normalized_trade(
                conn=conn,
                trace=trace,
                trace_key=trace_key,
                execution=execution,
                execution_status=execution_status,
                filled_quantity=filled_quantity,
            )

            conn.execute(
                """
                UPDATE runs
                SET trade_trace_count = (
                    SELECT COUNT(*)
                    FROM trade_traces
                    WHERE trade_traces.run_id = runs.run_id
                )
                WHERE run_id = ?
                """,
                (trace.run_id,),
            )

    def _sync_normalized_trade(
        self,
        *,
        conn: Any,
        trace: TradeTrace,
        trace_key: str,
        execution: dict[str, Any],
        execution_status: str | None,
        filled_quantity: float,
    ) -> None:
        """
        Synchronize the normalized trade ledger with actual filled quantity.

        The normalized ledger represents actual fills, not requested orders.
        Therefore zero-fill lifecycle states must never produce a trade row.
        """
        existing_trade = conn.execute(
            """
            SELECT id
            FROM trades
            WHERE trace_id = ?
            LIMIT 1
            """,
            (trace_key,),
        ).fetchone()

        side = _normalize_execution_side(execution.get("side"))
        price = execution.get("price")
        order_id = execution.get("order_id")

        if filled_quantity <= 0:
            if existing_trade is not None:
                conn.execute(
                    """
                    DELETE FROM trades
                    WHERE trace_id = ?
                    """,
                    (trace_key,),
                )
            return

        trade_status = (
            "filled" if execution_status == "filled" else execution_status or "filled"
        )

        trade_payload = {
            "trace_id": trace_key,
            "ticker": trace.ticker,
            "side": side,
            "qty": filled_quantity,
            "price": price,
            "order_id": order_id,
            "status": trade_status,
            "mode": "backtest",
            "created_at": _isoformat(trace.timestamp),
        }

        if existing_trade is None:
            insert_trade(conn, trade_payload)
            return

        conn.execute(
            """
            UPDATE trades
            SET
                ticker = ?,
                side = ?,
                qty = ?,
                price = ?,
                order_id = ?,
                status = ?,
                mode = ?,
                created_at = ?
            WHERE trace_id = ?
            """,
            (
                trace.ticker,
                side,
                filled_quantity,
                price,
                order_id,
                trade_status,
                "backtest",
                _isoformat(trace.timestamp),
                trace_key,
            ),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: datetime | str,
        status: str = "completed",
        final_equity: float | None = None,
        realized_pnl: float | None = None,
        benchmark_return: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a run complete and persist its final summary."""
        valid_statuses = {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        if status not in valid_statuses:
            raise ValueError(
                f"Unsupported run status: {status!r}. "
                f"Expected one of: {sorted(valid_statuses)}"
            )

        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE runs
                SET
                    finished_at = ?,
                    status = ?,
                    final_equity = ?,
                    realized_pnl = ?,
                    benchmark_return = ?,
                    metadata_json = COALESCE(?, metadata_json),
                    trade_trace_count = (
                        SELECT COUNT(*)
                        FROM trade_traces
                        WHERE trade_traces.run_id = runs.run_id
                    )
                WHERE run_id = ?
                """,
                (
                    _isoformat(finished_at),
                    status,
                    final_equity,
                    realized_pnl,
                    benchmark_return,
                    _json_dumps(metadata) if metadata is not None else None,
                    run_id,
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run summary."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["tickers"] = json.loads(result.pop("tickers_json"))
        result["metadata"] = json.loads(result.pop("metadata_json") or "{}")

        return result

    def list_runs(self) -> list[dict[str, Any]]:
        """Return run summaries ordered newest first."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC",
            ).fetchall()

        return [
            {
                **{
                    key: value
                    for key, value in dict(row).items()
                    if key not in {"tickers_json", "metadata_json"}
                },
                "tickers": json.loads(row["tickers_json"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def get_trade_traces(self, run_id: str) -> list[dict[str, Any]]:
        """Return complete serialized TradeTrace records for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT trace_json
                FROM trade_traces
                WHERE run_id = ?
                ORDER BY simulated_date, ticker, id
                """,
                (run_id,),
            ).fetchall()

        return [json.loads(row["trace_json"]) for row in rows]

    def get_run_audit(self, run_id: str) -> dict[str, Any]:
        """
        Reconstruct and validate the complete persisted audit lifecycle
        for one run.

        This method is observational only. It never modifies persistence
        state or participates in trading decisions.

        Reconstruction path:

            runs
                ↓
            trade_traces
                ↓
            trades

        Normalized trades represent actual filled quantity only.
        """
        with self._connection() as conn:
            run_row = conn.execute(
                """
                SELECT *
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if run_row is None:
                raise KeyError(f"Unknown run_id: {run_id}")

            trace_rows = conn.execute(
                """
                SELECT *
                FROM trade_traces
                WHERE run_id = ?
                ORDER BY simulated_date, ticker, id
                """,
                (run_id,),
            ).fetchall()

            trade_rows = conn.execute(
                """
                SELECT
                    t.*
                FROM trades AS t
                INNER JOIN trade_traces AS tt
                    ON tt.trace_key = t.trace_id
                WHERE tt.run_id = ?
                ORDER BY t.created_at, t.id
                """,
                (run_id,),
            ).fetchall()

        run = dict(run_row)
        run["tickers"] = json.loads(run.pop("tickers_json"))
        run["metadata"] = json.loads(run.pop("metadata_json") or "{}")

        trades = [dict(row) for row in trade_rows]

        trades_by_trace: dict[str, list[dict[str, Any]]] = {}

        for trade in trades:
            trace_id = trade["trace_id"]
            trades_by_trace.setdefault(trace_id, []).append(trade)

        traces: list[dict[str, Any]] = []

        successful_execution_count = 0
        filled_execution_count = 0
        partial_fill_count = 0
        zero_fill_count = 0
        rejected_or_unexecuted_count = 0

        invalid_trade_link_count = 0
        execution_trade_mismatch_count = 0
        invalid_fill_quantity_count = 0
        execution_status_mismatch_count = 0

        for row in trace_rows:
            trace_key = row["trace_key"]
            trace_payload = json.loads(row["trace_json"])
            execution = trace_payload.get("execution") or {}

            normalized_trades = trades_by_trace.get(trace_key, [])

            execution_success = row["execution_success"] == 1

            execution_status = _normalize_execution_status(
                row["execution_status"],
                success=execution_success,
                filled_quantity=row["execution_filled_quantity"],
            )

            requested_quantity = _as_non_negative_float(row["execution_quantity"])

            filled_quantity = _resolve_persisted_filled_quantity(
                row=row,
                execution=execution,
                execution_status=execution_status,
            )

            if execution_success:
                successful_execution_count += 1

            if filled_quantity < 0:
                invalid_fill_quantity_count += 1

            if (
                requested_quantity is not None
                and filled_quantity > requested_quantity + 1e-9
            ):
                invalid_fill_quantity_count += 1

            if filled_quantity > 0:
                filled_execution_count += 1

                if execution_status == "partially_filled":
                    partial_fill_count += 1

                if len(normalized_trades) != 1:
                    invalid_trade_link_count += 1
            else:
                zero_fill_count += 1
                rejected_or_unexecuted_count += 1

                if normalized_trades:
                    invalid_trade_link_count += len(normalized_trades)

            trace_execution_status = _normalize_execution_status(
                execution.get("status"),
                success=execution.get("success"),
                filled_quantity=execution.get("filled_quantity"),
            )

            if (
                execution_status is not None
                and trace_execution_status is not None
                and execution_status != trace_execution_status
            ):
                execution_status_mismatch_count += 1

            if filled_quantity > 0 and len(normalized_trades) == 1:
                trade = normalized_trades[0]

                expected_side = _normalize_execution_side(execution.get("side"))

                expected_price = execution.get("price")
                expected_order_id = execution.get("order_id")

                fields_match = (
                    trade["ticker"] == trace_payload.get("ticker")
                    and trade["side"] == expected_side
                    and _numbers_equal(
                        trade["qty"],
                        filled_quantity,
                    )
                    and _numbers_equal(
                        trade["price"],
                        expected_price,
                    )
                    and trade["order_id"] == expected_order_id
                )

                if not fields_match:
                    execution_trade_mismatch_count += 1

            traces.append(
                {
                    "trace_key": trace_key,
                    "tick_id": row["tick_id"],
                    "ticker": row["ticker"],
                    "simulated_date": row["simulated_date"],
                    "outcome": row["outcome"],
                    "execution_success": execution_success,
                    "execution_status": execution_status,
                    "execution_quantity": requested_quantity,
                    "execution_filled_quantity": filled_quantity,
                    "trace": trace_payload,
                    "normalized_trades": normalized_trades,
                }
            )

        persisted_trace_count = len(trace_rows)
        normalized_trade_count = len(trades)

        run_trace_count_matches = int(run["trade_trace_count"]) == persisted_trace_count

        filled_execution_count_matches_trade_count = (
            filled_execution_count == normalized_trade_count
        )

        zero_fill_traces_have_no_trade = all(
            not item["normalized_trades"]
            for item in traces
            if item["execution_filled_quantity"] <= 0
        )

        filled_traces_have_exactly_one_trade = all(
            len(item["normalized_trades"]) == 1
            for item in traces
            if item["execution_filled_quantity"] > 0
        )

        normalized_trade_qty_matches_filled_quantity = (
            execution_trade_mismatch_count == 0
        )

        execution_fields_match_trade = execution_trade_mismatch_count == 0
        execution_status_matches_trace = execution_status_mismatch_count == 0
        valid_fill_quantities = invalid_fill_quantity_count == 0
        no_invalid_trade_links = invalid_trade_link_count == 0

        # Backward-compatible invariant name retained for existing consumers.
        #
        # It remains true for the historical full-fill model. For partial
        # fills, the new fill-aware invariant above is authoritative.
        successful_execution_count_matches_trade_count = (
            successful_execution_count == normalized_trade_count
        )

        successful_traces_have_exactly_one_trade = all(
            len(item["normalized_trades"]) == 1
            for item in traces
            if item["execution_success"]
        )

        rejected_traces_have_no_trade = all(
            not item["normalized_trades"]
            for item in traces
            if not item["execution_success"] and item["execution_filled_quantity"] <= 0
        )

        valid = all(
            (
                run_trace_count_matches,
                filled_execution_count_matches_trade_count,
                zero_fill_traces_have_no_trade,
                filled_traces_have_exactly_one_trade,
                normalized_trade_qty_matches_filled_quantity,
                execution_fields_match_trade,
                execution_status_matches_trace,
                valid_fill_quantities,
                no_invalid_trade_links,
            )
        )

        return {
            "run": run,
            "summary": {
                "trace_count": persisted_trace_count,
                "successful_executions": successful_execution_count,
                "filled_executions": filled_execution_count,
                "partial_fills": partial_fill_count,
                "zero_fill_executions": zero_fill_count,
                "normalized_trades": normalized_trade_count,
                "rejected_or_unexecuted": rejected_or_unexecuted_count,
            },
            "traces": traces,
            "trades": trades,
            "invariants": {
                "run_trace_count_matches": run_trace_count_matches,
                "filled_execution_count_matches_trade_count": (
                    filled_execution_count_matches_trade_count
                ),
                "zero_fill_traces_have_no_trade": (zero_fill_traces_have_no_trade),
                "filled_traces_have_exactly_one_trade": (
                    filled_traces_have_exactly_one_trade
                ),
                "normalized_trade_qty_matches_filled_quantity": (
                    normalized_trade_qty_matches_filled_quantity
                ),
                "execution_fields_match_trade": execution_fields_match_trade,
                "execution_status_matches_trace": execution_status_matches_trace,
                "valid_fill_quantities": valid_fill_quantities,
                "no_invalid_trade_links": no_invalid_trade_links,
                # Backward-compatible invariants.
                "successful_execution_count_matches_trade_count": (
                    successful_execution_count_matches_trade_count
                ),
                "rejected_traces_have_no_trade": (rejected_traces_have_no_trade),
                "successful_traces_have_exactly_one_trade": (
                    successful_traces_have_exactly_one_trade
                ),
            },
            "diagnostics": {
                "invalid_trade_link_count": invalid_trade_link_count,
                "execution_trade_mismatch_count": (execution_trade_mismatch_count),
                "invalid_fill_quantity_count": invalid_fill_quantity_count,
                "execution_status_mismatch_count": (execution_status_mismatch_count),
            },
            "valid": valid,
        }

    def delete_run(self, run_id: str) -> None:
        """Delete one run and its persisted trade traces."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM runs WHERE run_id = ?",
                (run_id,),
            )

            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run_id: {run_id}")

    # ---------- Internal helpers ----------

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Yield a SQLite connection with transactional semantics."""
        conn = get_connection(self.db_path)

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------- Module-level helpers ----------


def _ensure_execution_lifecycle_columns(conn: Any) -> None:
    """
    Add lifecycle columns to existing SQLite databases.

    schema.sql will eventually contain these columns for fresh databases,
    but existing databases require an explicit migration.
    """
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(trade_traces)").fetchall()
    }

    if "execution_status" not in columns:
        conn.execute(
            """
            ALTER TABLE trade_traces
            ADD COLUMN execution_status TEXT
            """
        )

    if "execution_filled_quantity" not in columns:
        conn.execute(
            """
            ALTER TABLE trade_traces
            ADD COLUMN execution_filled_quantity REAL
            """
        )


def _resolve_filled_quantity(
    *,
    execution: dict[str, Any],
    execution_status: str | None,
) -> float:
    """
    Resolve actual filled quantity from a TradeTrace payload.

    New traces use execution.filled_quantity.

    Legacy traces did not have this field. For a historical successful
    execution, requested quantity is treated as filled quantity so existing
    audit records remain reconstructable.
    """
    explicit = execution.get("filled_quantity")

    if explicit is not None:
        return max(float(explicit), 0.0)

    if execution_status == "filled" or execution.get("success") is True:
        requested = execution.get("quantity")
        if requested is not None:
            return max(float(requested), 0.0)

    return 0.0


def _resolve_persisted_filled_quantity(
    *,
    row: Any,
    execution: dict[str, Any],
    execution_status: str | None,
) -> float:
    """Resolve filled quantity from persisted columns with legacy fallback."""
    persisted = row["execution_filled_quantity"]

    if persisted is not None:
        return max(float(persisted), 0.0)

    return _resolve_filled_quantity(
        execution=execution,
        execution_status=execution_status,
    )


def _normalize_execution_status(
    value: Any,
    *,
    success: bool | None = None,
    filled_quantity: float | None = None,
) -> str | None:
    """
    Normalize broker execution lifecycle status.

    Legacy traces without a status are inferred from their existing
    success/fill fields.
    """
    if value is not None:
        if hasattr(value, "value"):
            value = value.value

        status = str(value).strip().lower()

        if status.startswith("orderstatus."):
            status = status.rsplit(".", maxsplit=1)[-1]

        if status in _EXECUTION_STATUSES:
            return status

        raise ValueError(
            "Unsupported execution status. "
            f"Expected one of {sorted(_EXECUTION_STATUSES)}; "
            f"received {value!r}."
        )

    if filled_quantity is not None and float(filled_quantity) > 0:
        if success is True:
            return "filled"

        return "partially_filled"

    if success is True:
        return "filled"

    return None


def _as_non_negative_float(value: Any) -> float | None:
    """Convert a numeric value to a non-negative float."""
    if value is None:
        return None

    numeric = float(value)

    if numeric < 0:
        raise ValueError(f"Expected a non-negative quantity, received {value!r}")

    return numeric


def _numbers_equal(
    left: float | None,
    right: float | None,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Compare numeric persistence fields without brittle float equality."""
    if left is None or right is None:
        return left is right

    return abs(float(left) - float(right)) <= tolerance


def _isoformat(value: datetime | str | None) -> str | None:
    """Normalize a datetime or string to ISO-8601 text."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def _bool_to_int(value: bool | None) -> int | None:
    """Convert an optional boolean to a SQLite-compatible 0/1 integer."""
    if value is None:
        return None

    return int(value)


def _json_dumps(obj: Any) -> str:
    """Serialize an object as compact JSON."""
    return json.dumps(
        obj,
        separators=(",", ":"),
        default=str,
    )


def _normalize_signal_direction(value: Any) -> str | None:
    """
    Normalize a signal direction to the values allowed by trade_traces.

    Database values:
    - bullish
    - bearish
    - neutral
    """
    if value is None:
        return None

    if hasattr(value, "value"):
        value = value.value

    direction = str(value).strip().lower()

    if direction.startswith("signaldirection."):
        direction = direction.rsplit(".", maxsplit=1)[-1]

    aliases = {
        "buy": "bullish",
        "long": "bullish",
        "bull": "bullish",
        "bullish": "bullish",
        "sell": "bearish",
        "short": "bearish",
        "bear": "bearish",
        "bearish": "bearish",
        "hold": "neutral",
        "flat": "neutral",
        "none": "neutral",
        "neutral": "neutral",
    }

    normalized = aliases.get(direction)

    if normalized is None:
        raise ValueError(
            "Unsupported signal direction. "
            f"Expected bullish, bearish, or neutral; received {value!r}."
        )

    return normalized


def _normalize_execution_side(value: Any) -> str | None:
    """
    Normalize an execution side to the values allowed by trade_traces.

    Database values:
    - BUY
    - SELL
    """
    if value is None:
        return None

    if hasattr(value, "value"):
        value = value.value

    side = str(value).strip().upper()

    if side.startswith("ORDERSIDE."):
        side = side.rsplit(".", maxsplit=1)[-1]

    aliases = {
        "BUY": "BUY",
        "LONG": "BUY",
        "SELL": "SELL",
        "SHORT": "SELL",
    }

    normalized = aliases.get(side)

    if normalized is None:
        raise ValueError(
            f"Unsupported execution side. Expected BUY or SELL; received {value!r}."
        )

    return normalized


def _trace_key(
    *,
    run_id: str,
    ticker: str,
    simulated_date: datetime | None,
    tick_id: Any,
) -> str:
    """
    Build a stable idempotency key for one ticker/session trace.

    A TradeTrace must be unique for its run, ticker, and historical
    simulated session. tick_id is retained as additional context.
    """
    session_identity = (
        _isoformat(simulated_date) if simulated_date is not None else "unscheduled"
    )

    tick_identity = str(tick_id) if tick_id is not None else "no-tick-id"

    return f"{run_id}:{ticker}:{session_identity}:{tick_identity}"
