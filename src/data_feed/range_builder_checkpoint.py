#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistent checkpoints for path-dependent OKX range-bar builders."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

from src.data_feed.okx_range_bar_loader import RangeBarBuilder, range_code


class RangeBuilderCheckpointStore:
    """Store exact end-of-UTC-day builder states in a small SQLite database."""

    TABLE_NAME = "range_builder_checkpoints"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def cache_key(
        *,
        symbol: str,
        range_pct: float,
        price_step: float | None,
        contract_value: float,
        large_trade_notional_threshold: float,
    ) -> str:
        step = "none" if price_step is None else format(float(price_step), ".12g")
        return (
            f"{symbol}|{range_code(range_pct)}|step={step}"
            f"|cv={format(float(contract_value), '.12g')}"
            f"|large={format(float(large_trade_notional_threshold), '.12g')}"
        )

    def save(
        self,
        *,
        cache_key: str,
        utc_day: date,
        symbol: str,
        range_pct: float,
        price_step: float | None,
        builder: RangeBarBuilder,
    ) -> None:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        state_json = json.dumps(builder.export_state(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME}
                    (cache_key, utc_day, symbol, range_pct, price_step, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key, utc_day) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    cache_key,
                    utc_day.isoformat(),
                    symbol,
                    float(range_pct),
                    None if price_step is None else float(price_step),
                    state_json,
                    now,
                    now,
                ),
            )
            conn.commit()

    def load(self, *, cache_key: str, utc_day: date) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT state_json FROM {self.TABLE_NAME} WHERE cache_key = ? AND utc_day = ?",
                (cache_key, utc_day.isoformat()),
            ).fetchone()
        return None if row is None else json.loads(str(row[0]))

    def latest_common_day(self, cache_keys: Iterable[str], *, before_day: date) -> date | None:
        keys = list(dict.fromkeys(cache_keys))
        if not keys:
            return None
        placeholders = ",".join("?" for _ in keys)
        params = [*keys, before_day.isoformat(), len(keys)]
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT utc_day
                FROM {self.TABLE_NAME}
                WHERE cache_key IN ({placeholders}) AND utc_day <= ?
                GROUP BY utc_day
                HAVING COUNT(DISTINCT cache_key) = ?
                ORDER BY utc_day DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return None if row is None else date.fromisoformat(str(row[0]))

    def delete_from(self, cache_keys: Iterable[str], *, utc_day: date) -> None:
        keys = list(dict.fromkeys(cache_keys))
        if not keys:
            return
        placeholders = ",".join("?" for _ in keys)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self.TABLE_NAME} WHERE cache_key IN ({placeholders}) AND utc_day >= ?",
                [*keys, utc_day.isoformat()],
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    cache_key TEXT NOT NULL,
                    utc_day TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    range_pct REAL NOT NULL,
                    price_step REAL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (cache_key, utc_day)
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_day ON {self.TABLE_NAME}(utc_day)"
            )
            conn.commit()
