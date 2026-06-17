#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX tick/trades data loader for backtests.

Usage in backtest:

    loader = OKXTickLoader(
        symbol="ETH-USDT-SWAP",
        trades_url_template="https://www.okx.com/cdn/okex/traderecords/trades/daily/{yyyymmdd}/{symbol}-trades-{date}.zip",
    )

    for day, chunk in loader.iter_trades("2026-06-01", "2026-06-10", chunksize=100_000):
        backtest_on_tick_chunk(chunk)
        del chunk

Design:
- Local first: read local daily sqlite db if complete.
- Lazy fetch: if a completed day is missing, download OKX official trade file,
  stream it into local sqlite, then yield it.
- Memory safe: never loads more than one day; read/write uses chunks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

UTC = timezone.utc


class OKXTickLoader:
    def __init__(self, symbol: str = "ETH-USDT-SWAP", data_dir: str | os.PathLike[str] | None = None, trades_url_template: str = ""):
        self.symbol = symbol
        self.trades_url_template = trades_url_template
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.raw_dir = self.data_dir / "okx" / "raw" / "trades" / self._safe(symbol)
        self.db_root = self.data_dir / "okx" / "ticks" / self._safe(symbol)

    def iter_trades(
        self,
        start_date: str | date,
        end_date: str | date,
        *,
        chunksize: int = 100_000,
        trades_url_template: str | None = None,
    ) -> Iterator[tuple[date, pd.DataFrame]]:
        """Yield tick data by day/chunk; auto-fetch missing completed days."""

        template = trades_url_template if trades_url_template is not None else self.trades_url_template
        for day in self._date_range(start_date, end_date):
            if not self.has_complete_day(day):
                if not self._is_completed_day(day):
                    logger.info("skip caching incomplete/current day: %s", day)
                    continue
                if not template:
                    raise RuntimeError(
                        "local tick cache missing and trades_url_template is empty. "
                        "Set OKX official trade file URL template before running tick backtest."
                    )
                raw_file = self.download_official_trade_file(day, template)
                self.save_official_trade_file_to_db(day, raw_file, chunksize=chunksize)

            for chunk in self.read_day(day, chunksize=chunksize):
                yield day, chunk

    def has_complete_day(self, day: str | date) -> bool:
        db = self.db_path(day)
        if not db.exists() or db.stat().st_size <= 0:
            return False
        try:
            with sqlite3.connect(db) as conn:
                row = conn.execute("SELECT value FROM meta WHERE key='complete'").fetchone()
                return bool(row and json.loads(row[0]) is True)
        except Exception:
            return False

    def read_day(self, day: str | date, *, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
        db = self.db_path(day)
        if not db.exists():
            return
        conn = sqlite3.connect(db)
        try:
            query = "SELECT ts_ms, timestamp, symbol, trade_id, price, size, side, raw_json FROM ticks ORDER BY ts_ms ASC"
            for chunk in pd.read_sql_query(query, conn, chunksize=chunksize, parse_dates=["timestamp"]):
                yield chunk
        finally:
            conn.close()

    def download_official_trade_file(self, day: str | date, url_template: str, *, timeout: int = 60, retries: int = 3) -> Path:
        d = self._parse_date(day)
        url = url_template.format(date=d.isoformat(), yyyymmdd=d.strftime("%Y%m%d"), symbol=self.symbol, kind="trades")
        filename = Path(urlparse(url).path).name or f"{self.symbol}-trades-{d.isoformat()}.zip"
        out = self.raw_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            return out

        tmp = out.with_suffix(out.suffix + ".part")
        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "CoinBacktest/okx-tick-loader"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
                    while True:
                        block = resp.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                tmp.replace(out)
                logger.info("downloaded OKX tick file: %s", out)
                return out
            except Exception as exc:
                last_error = repr(exc)
                logger.warning("download retry %s/%s url=%s error=%s", attempt, retries, url, last_error)
                time.sleep(min(2 ** attempt, 30))
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download OKX tick file: {last_error}")

    def save_official_trade_file_to_db(self, day: str | date, path: str | Path, *, chunksize: int = 100_000) -> Path:
        d = self._parse_date(day)
        if not self._is_completed_day(d):
            raise RuntimeError(f"refuse to save incomplete/current tick day: {d}")

        db = self.db_path(d)
        db.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with self._connect(db) as conn:
            self._init_db(conn)
            conn.execute("DELETE FROM ticks")
            for chunk in self._iter_trade_file(path, chunksize=chunksize):
                if chunk.empty:
                    continue
                chunk.to_sql("ticks", conn, if_exists="append", index=False, chunksize=chunksize, method="multi")
                total += len(chunk)
                del chunk
            self._set_meta(conn, complete=total > 0, rows=total, symbol=self.symbol, date=d.isoformat(), source=str(path), sha256=self._sha256(path))
        logger.info("saved tick db: %s rows=%s", db, total)
        return db

    def db_path(self, day: str | date) -> Path:
        d = self._parse_date(day)
        return self.db_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.db"

    def _iter_trade_file(self, path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
        p = Path(path)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    with zf.open(name) as f:
                        for raw in pd.read_csv(f, chunksize=chunksize):
                            yield self._normalize_trades(raw)
        else:
            for raw in pd.read_csv(p, chunksize=chunksize):
                yield self._normalize_trades(raw)

    def _normalize_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rename = {}
        for col in out.columns:
            low = str(col).strip().lower()
            if low in {"ts", "timestamp", "time"}:
                rename[col] = "ts_ms"
            elif low in {"tradeid", "trade_id", "id"}:
                rename[col] = "trade_id"
            elif low in {"px", "price"}:
                rename[col] = "price"
            elif low in {"sz", "size", "qty", "amount"}:
                rename[col] = "size"
            elif low == "side":
                rename[col] = "side"
        out = out.rename(columns=rename)
        if "ts_ms" not in out.columns:
            raise ValueError(f"trade file has no timestamp column: {list(df.columns)}")

        numeric_ts = pd.to_numeric(out["ts_ms"], errors="coerce")
        if numeric_ts.notna().all():
            if numeric_ts.max() < 10_000_000_000:
                numeric_ts = numeric_ts * 1000
            out["ts_ms"] = numeric_ts.astype("int64")
            out["timestamp"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True).astype(str)
        else:
            ts = pd.to_datetime(out["ts_ms"], utc=True, errors="coerce")
            out["timestamp"] = ts.astype(str)
            out["ts_ms"] = (ts.astype("int64") // 1_000_000).astype("int64")

        for col in ["trade_id", "price", "size", "side"]:
            if col not in out.columns:
                out[col] = None
        out["symbol"] = self.symbol
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out["size"] = pd.to_numeric(out["size"], errors="coerce")
        out["side"] = out["side"].astype(str).str.lower()

        base = {"ts_ms", "timestamp", "symbol", "trade_id", "price", "size", "side"}
        extra_cols = [c for c in out.columns if c not in base]
        if extra_cols:
            out["raw_json"] = out[extra_cols].apply(lambda row: json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=str), axis=1)
        else:
            out["raw_json"] = None
        out = out[["ts_ms", "timestamp", "symbol", "trade_id", "price", "size", "side", "raw_json"]]
        out = out.dropna(subset=["ts_ms", "timestamp"]).drop_duplicates(subset=["ts_ms", "trade_id", "price", "size", "side"], keep="last")
        return out.sort_values("ts_ms").reset_index(drop=True)

    @contextmanager
    def _connect(self, db: Path) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-65536")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                ts_ms INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_id TEXT,
                price REAL,
                size REAL,
                side TEXT,
                raw_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts_ms ON ticks(ts_ms)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_key ON ticks(ts_ms, trade_id, price, size, side)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _set_meta(self, conn: sqlite3.Connection, **kwargs) -> None:
        for key, value in kwargs.items():
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False, default=str)))

    def _date_range(self, start: str | date, end: str | date) -> Iterator[date]:
        cur = self._parse_date(start)
        last = self._parse_date(end)
        if last < cur:
            raise ValueError("end_date must be >= start_date")
        while cur <= last:
            yield cur
            cur += timedelta(days=1)

    def _parse_date(self, value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    def _is_completed_day(self, day: str | date) -> bool:
        d = self._parse_date(day)
        end = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(days=1)
        return datetime.now(UTC) >= end

    def _safe(self, value: str) -> str:
        return str(value).replace("/", "_").replace("\\", "_").strip() or "default"

    def _sha256(self, path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
