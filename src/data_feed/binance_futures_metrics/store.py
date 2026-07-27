#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SQLite persistence for normalized Binance futures metrics."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    METRIC_COLUMNS,
    BinanceMetricsCoverage,
    BinanceMetricsDayResult,
    parse_archive_day,
)


class BinanceMetricsStore:
    CACHE_SCHEMA_VERSION = 1

    def __init__(self, symbol: str, db_path: Path) -> None:
        self.symbol = symbol
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def day_is_complete(self, day: date) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, rows FROM futures_metrics_coverage WHERE symbol = ? AND day_utc = ?",
                (self.symbol, day.isoformat()),
            ).fetchone()
        return bool(row and str(row[0]) in {"complete", "partial"} and int(row[1] or 0) > 0)

    def save_day(self, result: BinanceMetricsDayResult) -> None:
        if result.frame is None or result.frame.empty:
            raise ValueError("cannot save empty metrics day")
        frame = result.frame.copy()
        db_columns = [
            "symbol",
            "timestamp",
            "source_timestamp_utc",
            "period",
            *METRIC_COLUMNS,
            "source_day_utc",
            "source",
        ]
        values = [self._clean_row(db_columns, row) for row in frame[db_columns].itertuples(index=False, name=None)]
        placeholders = ",".join("?" for _ in db_columns)
        updates = ",".join(
            f"{column}=excluded.{column}" for column in db_columns if column not in {"symbol", "timestamp"}
        )
        sql = (
            f"INSERT INTO futures_metrics_5m ({','.join(db_columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol,timestamp) DO UPDATE SET {updates}"
        )
        first_ts = self._format_timestamp(frame["timestamp"].min())
        last_ts = self._format_timestamp(frame["timestamp"].max())

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM futures_metrics_5m WHERE symbol = ? AND source_day_utc = ?",
                (self.symbol, result.day_utc.isoformat()),
            )
            conn.executemany(sql, values)
            self._upsert_coverage(conn, result, first_ts=first_ts, last_ts=last_ts)

    def save_coverage_result(self, result: BinanceMetricsDayResult) -> None:
        with self._connect() as conn:
            self._upsert_coverage(conn, result, first_ts=None, last_ts=None)

    def load_timestamp_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT * FROM futures_metrics_5m
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                """,
                conn,
                params=(self.symbol, self._format_timestamp(start), self._format_timestamp(end)),
                parse_dates=["timestamp", "source_timestamp_utc"],
            )
        if frame.empty:
            return pd.DataFrame()
        return frame.drop(columns=["symbol"], errors="ignore").sort_values("timestamp").reset_index(drop=True)

    def load_archive_days(self, start_date: Any, end_date: Any) -> pd.DataFrame:
        start_day = parse_archive_day(start_date).isoformat()
        end_day = parse_archive_day(end_date).isoformat()
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT * FROM futures_metrics_5m
                WHERE symbol = ? AND source_day_utc >= ? AND source_day_utc <= ?
                ORDER BY timestamp
                """,
                conn,
                params=(self.symbol, start_day, end_day),
                parse_dates=["timestamp", "source_timestamp_utc"],
            )
        if frame.empty:
            return pd.DataFrame()
        return frame.drop(columns=["symbol"], errors="ignore")

    def coverage(self) -> BinanceMetricsCoverage:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM futures_metrics_5m WHERE symbol = ?",
                (self.symbol,),
            ).fetchone()
            status_rows = conn.execute(
                "SELECT status, COUNT(*) FROM futures_metrics_coverage WHERE symbol = ? GROUP BY status",
                (self.symbol,),
            ).fetchall()
        counts = {str(status): int(count) for status, count in status_rows}
        return BinanceMetricsCoverage(
            symbol=self.symbol,
            rows=int(row[0] or 0),
            start=pd.Timestamp(row[1]) if row[1] else None,
            end=pd.Timestamp(row[2]) if row[2] else None,
            complete_days=counts.get("complete", 0),
            partial_days=counts.get("partial", 0),
            missing_days=counts.get("missing", 0),
            error_days=counts.get("error", 0),
        )

    def coverage_by_day(self, start_date: Any | None = None, end_date: Any | None = None) -> pd.DataFrame:
        clauses = ["symbol = ?"]
        params: list[Any] = [self.symbol]
        if start_date is not None:
            clauses.append("day_utc >= ?")
            params.append(parse_archive_day(start_date).isoformat())
        if end_date is not None:
            clauses.append("day_utc <= ?")
            params.append(parse_archive_day(end_date).isoformat())
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM futures_metrics_coverage WHERE {' AND '.join(clauses)} ORDER BY day_utc",
                conn,
                params=tuple(params),
                parse_dates=["first_timestamp", "last_timestamp", "updated_at"],
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS futures_metrics_5m (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source_timestamp_utc TEXT NOT NULL,
                    period TEXT NOT NULL,
                    sum_open_interest REAL,
                    sum_open_interest_value REAL,
                    count_toptrader_long_short_ratio REAL,
                    sum_toptrader_long_short_ratio REAL,
                    count_long_short_ratio REAL,
                    sum_taker_long_short_vol_ratio REAL,
                    source_day_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY(symbol, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_binance_metrics_symbol_ts
                    ON futures_metrics_5m(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_binance_metrics_symbol_day
                    ON futures_metrics_5m(symbol, source_day_utc);
                CREATE TABLE IF NOT EXISTS futures_metrics_coverage (
                    symbol TEXT NOT NULL,
                    day_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows INTEGER NOT NULL DEFAULT 0,
                    first_timestamp TEXT,
                    last_timestamp TEXT,
                    source_url TEXT,
                    checksum_sha256 TEXT,
                    checksum_verified INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(symbol, day_utc)
                );
                CREATE INDEX IF NOT EXISTS idx_binance_metrics_coverage_status
                    ON futures_metrics_coverage(symbol, status, day_utc);
                """
            )

    def _upsert_coverage(
        self,
        conn: sqlite3.Connection,
        result: BinanceMetricsDayResult,
        *,
        first_ts: str | None,
        last_ts: str | None,
    ) -> None:
        updated_at = pd.Timestamp.now("UTC").tz_convert(None).strftime("%Y-%m-%d %H:%M:%S.%f")
        conn.execute(
            """
            INSERT INTO futures_metrics_coverage (
                symbol, day_utc, status, rows, first_timestamp, last_timestamp,
                source_url, checksum_sha256, checksum_verified, error,
                updated_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, day_utc) DO UPDATE SET
                status=excluded.status,
                rows=excluded.rows,
                first_timestamp=excluded.first_timestamp,
                last_timestamp=excluded.last_timestamp,
                source_url=excluded.source_url,
                checksum_sha256=excluded.checksum_sha256,
                checksum_verified=excluded.checksum_verified,
                error=excluded.error,
                updated_at=excluded.updated_at,
                schema_version=excluded.schema_version
            """,
            (
                self.symbol,
                result.day_utc.isoformat(),
                result.status,
                int(result.rows),
                first_ts,
                last_ts,
                result.source_url,
                result.checksum_sha256,
                int(result.checksum_verified),
                result.error,
                updated_at,
                self.CACHE_SCHEMA_VERSION,
            ),
        )

    @classmethod
    def _clean_row(cls, columns: list[str], row: tuple[Any, ...]) -> tuple[Any, ...]:
        clean: list[Any] = []
        for column, value in zip(columns, row):
            if column in {"timestamp", "source_timestamp_utc"}:
                clean.append(cls._format_timestamp(value))
            elif pd.isna(value):
                clean.append(None)
            elif hasattr(value, "item"):
                clean.append(value.item())
            else:
                clean.append(value)
        return tuple(clean)

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")
