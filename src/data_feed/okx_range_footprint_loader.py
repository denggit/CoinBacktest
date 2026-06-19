#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build and load OKX Range Bar Footprint data with SQLite cache.

The footprint table stores price-bucket order-flow statistics inside each range
bar.  It shares the same range-bar construction logic as
``okx_range_bar_loader.py`` so ``bar_id`` is stable across the range-bar and
footprint databases when built over the same raw trades in the same order.

Default ETH table names:
    ETH_USDT_SWAP_range_footprint_r0020_step1
    ETH_USDT_SWAP_range_footprint_r0015_step1
    ETH_USDT_SWAP_range_footprint_r0025_step1
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

try:
    from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader
    from src.data_feed.okx_range_bar_loader import (
        BAR_ID_MULT,
        DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
        DEFAULT_RANGE_PCTS,
        RangeBarBuilder,
        date_range,
        iter_trade_csv_chunks,
        normalize_trade_chunk_fast,
        parse_date,
        parse_timezone_offset_hours,
        range_code,
        safe_symbol,
        timestamp_to_db_text,
    )
except ImportError:  # pragma: no cover
    from okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader
    from okx_range_bar_loader import (
        BAR_ID_MULT,
        DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
        DEFAULT_RANGE_PCTS,
        RangeBarBuilder,
        date_range,
        iter_trade_csv_chunks,
        normalize_trade_chunk_fast,
        parse_date,
        parse_timezone_offset_hours,
        range_code,
        safe_symbol,
        timestamp_to_db_text,
    )

try:
    from src.utils.log import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def price_step_code(price_step: float) -> str:
    step = float(price_step)
    if step.is_integer():
        return f"step{int(step)}"
    text = (f"{step:.8f}").rstrip("0").rstrip(".").replace(".", "_")
    return f"step{text}"


def table_name_for_range_footprint(symbol: str, range_pct: float, price_step: float) -> str:
    return f"{safe_symbol(symbol)}_range_footprint_{range_code(range_pct)}_{price_step_code(price_step)}"


class OKXRangeFootprintLoader:
    """Load/build price-bucket footprints inside OKX range bars."""

    BASE_COLUMNS = [
        "bar_id",
        "start_ts",
        "end_ts",
        "price_bucket",
        "volume",
        "notional",
        "trades_count",
        "buy_volume",
        "sell_volume",
        "buy_notional",
        "sell_notional",
        "buy_trades_count",
        "sell_trades_count",
        "delta_volume",
        "delta_notional",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
        "large_trades_count",
        "max_trade_notional",
    ]

    NUMERIC_COLUMNS = [c for c in BASE_COLUMNS if c not in {"bar_id", "start_ts", "end_ts"}]

    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        range_pct: float = 0.0020,
        price_step: float = 1.0,
        data_dir: str | os.PathLike[str] | None = None,
        db_name: str = "okx_range_footprints.db",
        trades_url_template: str = DEFAULT_OKX_TRADES_URL_TEMPLATE,
        contract_value: float | None = None,
        large_trade_notional_threshold: float = DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
        align_with_okx_loader_timezone: bool = True,
    ):
        self.symbol = symbol
        self.range_pct = float(range_pct)
        self.price_step = float(price_step)
        if self.price_step <= 0:
            raise ValueError("price_step must be > 0")
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_name
        self.trades_url_template = trades_url_template
        self.contract_value = self._infer_contract_value(symbol) if contract_value is None else float(contract_value)
        self.large_trade_notional_threshold = float(large_trade_notional_threshold)
        self.align_with_okx_loader_timezone = bool(align_with_okx_loader_timezone)
        self.timezone_offset_hours = parse_timezone_offset_hours(TIMEZONE) if self.align_with_okx_loader_timezone else 0
        self.tick_loader = OKXTickLoader(symbol=self.symbol, data_dir=self.data_dir, trades_url_template=self.trades_url_template)
        self.table_name = table_name_for_range_footprint(self.symbol, self.range_pct, self.price_step)
        self.coverage_table_name = "range_footprint_coverage"
        self._init_db()

    def fetch_data_by_date_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        if end_ts < start_ts:
            raise ValueError(f"end_date must be >= start_date, got {start_ts} -> {end_ts}")
        self.ensure_cached_range(start_ts, end_ts, chunksize=chunksize, force_rebuild=force_rebuild)
        return self.load_local_data(start_date=start_ts, end_date=end_ts)

    def fetch_historical_data(
        self,
        limit: int = 500_000,
        *,
        end_date: str | datetime | pd.Timestamp | None = None,
        lookback_days: int = 30,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        end_ts = self._normalize_query_timestamp(end_date or pd.Timestamp.now(), is_end=True)
        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
        df = self.fetch_data_by_date_range(start_ts, end_ts, chunksize=chunksize, force_rebuild=force_rebuild)
        return df.tail(limit)

    def ensure_cached_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> None:
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        days = list(self._required_utc_days_for_local_range(start_ts, end_ts))
        missing = [d for d in days if force_rebuild or not self._has_coverage(d)]
        if missing:
            self.build_range_footprint_to_db(days, chunksize=chunksize, force_rebuild=force_rebuild)

    def ensure_cached_days(self, utc_days: Iterable[date], *, chunksize: int = 300_000, force_rebuild: bool = False) -> None:
        days = [parse_date(d) for d in utc_days]
        missing = [d for d in days if force_rebuild or not self._has_coverage(d)]
        if missing:
            self.build_range_footprint_to_db(days, chunksize=chunksize, force_rebuild=force_rebuild)

    def build_range_footprint_to_db(self, utc_days: Sequence[date], *, chunksize: int = 300_000, force_rebuild: bool = False) -> dict[str, int]:
        days = [parse_date(d) for d in utc_days]
        if not days:
            return {"footprints_written": 0, "chunks_read": 0, "days": 0}
        if force_rebuild:
            self._delete_cached_days(days)

        builder = RangeBarBuilder(
            range_pct=self.range_pct,
            contract_value=self.contract_value,
            large_trade_notional_threshold=self.large_trade_notional_threshold,
            price_step=self.price_step,
        )
        footprints_written = 0
        bars_closed = 0
        chunks_read = 0
        for day in days:
            if not force_rebuild and self._has_coverage(day):
                logger.info(
                    "[RANGE-FOOTPRINT-SKIP] symbol=%s range=%s step=%s utc_day=%s",
                    self.symbol,
                    range_code(self.range_pct),
                    self.price_step,
                    day,
                )
                continue
            raw_file = self._ensure_raw_trade_file(day)
            logger.info(
                "[RANGE-FOOTPRINT-DAY-START] symbol=%s range=%s step=%s utc_day=%s raw=%s",
                self.symbol,
                range_code(self.range_pct),
                self.price_step,
                day,
                raw_file,
            )
            day_rows = 0
            day_bars = 0
            for raw in iter_trade_csv_chunks(raw_file, chunksize=chunksize):
                chunks_read += 1
                chunk = normalize_trade_chunk_fast(raw)
                bars, footprints = builder.process_chunk(chunk)
                if footprints:
                    df = self._footprints_to_frame(footprints)
                    self._upsert_footprints(df)
                    footprints_written += len(df)
                    day_rows += len(df)
                    day_bars += len(bars)
            bars_closed += day_bars
            self._mark_coverage(day, rows=day_rows, bars=day_bars)
            logger.info(
                "[RANGE-FOOTPRINT-DAY-DONE] symbol=%s range=%s step=%s utc_day=%s bars=%s rows=%s",
                self.symbol,
                range_code(self.range_pct),
                self.price_step,
                day,
                day_bars,
                day_rows,
            )
        return {"footprints_written": footprints_written, "bars_closed": bars_closed, "chunks_read": chunks_read, "days": len(days)}

    def load_local_data(self, start_date: Any | None = None, end_date: Any | None = None, bar_ids: Sequence[int] | None = None) -> pd.DataFrame:
        where: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            where.append("end_ts >= ?")
            params.append(timestamp_to_db_text(start_date))
        if end_date is not None:
            where.append("start_ts <= ?")
            params.append(timestamp_to_db_text(end_date))
        if bar_ids:
            placeholders = ",".join(["?"] * len(bar_ids))
            where.append(f"bar_id IN ({placeholders})")
            params.extend([int(x) for x in bar_ids])
        sql = f"SELECT {', '.join(self.BASE_COLUMNS)} FROM {self.table_name}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY bar_id, price_bucket"
        with self._get_db_connection() as conn:
            try:
                df = pd.read_sql_query(sql, conn, params=params, parse_dates=["start_ts", "end_ts"])
            except Exception as exc:
                logger.warning("读取 range footprint DB 失败 table=%s error=%s", self.table_name, exc)
                return pd.DataFrame(columns=self.BASE_COLUMNS)
        return self._finalize_return_df(df)

    def delete_cache(self) -> None:
        with self._get_db_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
            conn.execute(f"DELETE FROM {self.coverage_table_name} WHERE cache_key = ?", (self._cache_key(),))
            conn.commit()

    def _ensure_raw_trade_file(self, day: date) -> Path:
        raw_file = self.tick_loader.find_local_trade_file(day, template=self.trades_url_template)
        if raw_file is not None and raw_file.exists() and raw_file.stat().st_size > 0:
            return raw_file
        return self.tick_loader.download_official_trade_file(day, self.trades_url_template)

    def _footprints_to_frame(self, footprints: list[dict[str, Any]]) -> pd.DataFrame:
        if not footprints:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        df = pd.DataFrame(footprints)
        if self.timezone_offset_hours:
            df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True) + pd.Timedelta(hours=self.timezone_offset_hours)
            df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True) + pd.Timedelta(hours=self.timezone_offset_hours)
        else:
            df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
            df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
        df["start_ts"] = df["start_ts"].dt.tz_localize(None)
        df["end_ts"] = df["end_ts"].dt.tz_localize(None)
        return self._finalize_return_df(df)

    def _finalize_return_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        out = df.copy()
        for col in ["start_ts", "end_ts"]:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
        for col in self.NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out = out.dropna(subset=["start_ts", "end_ts"]).sort_values(["bar_id", "price_bucket"])
        return out[self.BASE_COLUMNS]

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")  # about 256 MiB page cache when possible
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA wal_autocheckpoint=10000")
        return conn

    def _init_db(self) -> None:
        numeric_cols_sql = ",\n                    ".join(f"{col} REAL" for col in self.NUMERIC_COLUMNS)
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    bar_id INTEGER NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    {numeric_cols_sql},
                    PRIMARY KEY (bar_id, price_bucket)
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_end_ts ON {self.table_name}(end_ts)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_bar_id ON {self.table_name}(bar_id)")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.coverage_table_name} (
                    cache_key TEXT NOT NULL,
                    utc_day TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    range_pct REAL NOT NULL,
                    price_step REAL NOT NULL,
                    table_name TEXT NOT NULL,
                    rows INTEGER NOT NULL DEFAULT 0,
                    bars INTEGER NOT NULL DEFAULT 0,
                    params_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (cache_key, utc_day)
                )
                """
            )
            conn.commit()

    def _upsert_footprints(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        clean = df.reset_index(drop=True).copy()
        clean["start_ts"] = pd.to_datetime(clean["start_ts"]).map(timestamp_to_db_text)
        clean["end_ts"] = pd.to_datetime(clean["end_ts"]).map(timestamp_to_db_text)
        clean["bar_id"] = pd.to_numeric(clean["bar_id"], errors="coerce").astype("int64")
        for col in self.NUMERIC_COLUMNS:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)
        db_cols = self.BASE_COLUMNS
        placeholders = ",".join(["?"] * len(db_cols))
        update_cols = [c for c in db_cols if c not in {"bar_id", "price_bucket"}]
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in update_cols])
        sql = f"""
            INSERT INTO {self.table_name} ({', '.join(db_cols)})
            VALUES ({placeholders})
            ON CONFLICT(bar_id, price_bucket) DO UPDATE SET {update_sql}
        """
        with self._get_db_connection() as conn:
            conn.executemany(sql, clean[db_cols].itertuples(index=False, name=None))
            conn.commit()

    def _delete_cached_days(self, days: Sequence[date]) -> None:
        with self._get_db_connection() as conn:
            for day in days:
                prefix = int(day.strftime("%Y%m%d")) * BAR_ID_MULT
                conn.execute(f"DELETE FROM {self.table_name} WHERE bar_id >= ? AND bar_id < ?", (prefix, prefix + BAR_ID_MULT))
                conn.execute(f"DELETE FROM {self.coverage_table_name} WHERE cache_key = ? AND utc_day = ?", (self._cache_key(), day.isoformat()))
            conn.commit()

    def _has_coverage(self, utc_day: date) -> bool:
        with self._get_db_connection() as conn:
            row = conn.execute(
                f"SELECT rows FROM {self.coverage_table_name} WHERE cache_key = ? AND utc_day = ?",
                (self._cache_key(), utc_day.isoformat()),
            ).fetchone()
        return row is not None

    def _mark_coverage(self, utc_day: date, *, rows: int, bars: int) -> None:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        params = {
            "symbol": self.symbol,
            "range_pct": self.range_pct,
            "price_step": self.price_step,
            "contract_value": self.contract_value,
            "large_trade_notional_threshold": self.large_trade_notional_threshold,
            "align_with_okx_loader_timezone": self.align_with_okx_loader_timezone,
            "timezone": TIMEZONE,
        }
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.coverage_table_name}
                    (cache_key, utc_day, symbol, range_pct, price_step, table_name, rows, bars, params_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key, utc_day) DO UPDATE SET
                    rows=excluded.rows,
                    bars=excluded.bars,
                    params_json=excluded.params_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self._cache_key(),
                    utc_day.isoformat(),
                    self.symbol,
                    self.range_pct,
                    self.price_step,
                    self.table_name,
                    int(rows),
                    int(bars),
                    json.dumps(params, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()

    def _cache_key(self) -> str:
        return f"{self.symbol}|{range_code(self.range_pct)}|{self.price_step}|cv={self.contract_value}|large={self.large_trade_notional_threshold}|tz={self.timezone_offset_hours}"

    def _required_utc_days_for_local_range(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        if self.timezone_offset_hours:
            raw_start = start_ts - pd.Timedelta(hours=self.timezone_offset_hours)
            raw_end = end_ts - pd.Timedelta(hours=self.timezone_offset_hours)
        else:
            raw_start = start_ts
            raw_end = end_ts
        yield from date_range(raw_start.date(), raw_end.date())

    def _normalize_query_timestamp(self, value: Any, *, is_end: bool) -> pd.Timestamp:
        if isinstance(value, str) and len(value) == 10:
            suffix = " 23:59:59.999" if is_end else " 00:00:00.000"
            return pd.Timestamp(value + suffix)
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts

    def _infer_contract_value(self, symbol: str) -> float:
        s = str(symbol).upper()
        if s.endswith("-SWAP") and s.startswith("ETH-"):
            return 0.1
        return 1.0
