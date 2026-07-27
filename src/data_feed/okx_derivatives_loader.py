#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local-first OKX public derivatives data loader for liquidation-map research.

The loader owns all external data interaction for the estimated liquidation
heatmap.  Research, plugins and models consume normalized local frames and do
not call OKX endpoints directly.

Data sources (public, no API key):
- contract-level open-interest history (with aggregate OI kept only as a diagnostic fallback)
- funding-rate history
- mark-price history candles
- locally captured/imported liquidation events (no public historical REST backfill)

The public history depth differs by endpoint.  The downloader therefore stores
whatever OKX returns, resumes idempotently, and never fabricates missing
history.  CSV import is provided for externally archived data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
import requests

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


OKX_BASE_URL = "https://www.okx.com"

# OKX documents that contract OI history returns only the latest 1,440
# entries for the selected period.  The loader therefore auto-selects the
# finest period whose rolling history window can still reach the requested
# start time.
OI_HISTORY_MAX_ENTRIES = 1_440
OI_PERIODS: tuple[str, ...] = ("5m", "15m", "30m", "1H", "2H", "4H", "1D")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("-", "_").lower()


def _timezone_offset() -> pd.Timedelta:
    value = str(TIMEZONE).strip()
    if value.startswith("+"):
        return pd.Timedelta(hours=float(value[1:] or 0))
    if value.startswith("-"):
        return -pd.Timedelta(hours=float(value[1:] or 0))
    return pd.Timedelta(0)


def _ms_to_local_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(int(float(value)), unit="ms", utc=True).tz_convert(None)
    return ts + _timezone_offset()


def _timeframe_delta(value: str) -> pd.Timedelta:
    text = str(value).strip()
    if text.endswith("m") and text[:-1].isdigit():
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("H") and text[:-1].isdigit():
        return pd.Timedelta(hours=int(text[:-1]))
    if text.endswith("D") and text[:-1].isdigit():
        return pd.Timedelta(days=int(text[:-1]))
    if text.endswith("s") and text[:-1].isdigit():
        return pd.Timedelta(seconds=int(text[:-1]))
    raise ValueError(f"unsupported timeframe: {value}")


def _local_naive_to_ms(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    utc_naive = ts - _timezone_offset()
    return int(utc_naive.timestamp() * 1000)


@dataclass(frozen=True)
class DerivativesCoverage:
    dataset: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None


class OKXAPIError(RuntimeError):
    """Structured OKX public API error with the original response context."""

    def __init__(
        self,
        *,
        endpoint: str,
        params: Mapping[str, Any],
        status_code: int | None,
        code: str,
        message: str,
    ) -> None:
        self.endpoint = endpoint
        self.params = dict(params)
        self.status_code = status_code
        self.code = str(code or "")
        self.message = str(message or "unknown error")
        status_text = f"HTTP {status_code}" if status_code is not None else "HTTP error"
        super().__init__(
            f"OKX {status_text} code={self.code or 'unknown'} msg={self.message} "
            f"endpoint={endpoint} params={self.params}"
        )


class OKXDerivativesLoader:
    """Download and load public derivatives inputs through one SQLite cache."""

    # Contract-level OI history is the correct source for a symbol-specific
    # liquidation model.  The currency aggregate endpoint is intentionally not
    # used for arbitrary historical backfills: several OKX deployments reject
    # begin/end ranges on that endpoint with code 50030.
    OPEN_INTEREST_HISTORY_ENDPOINT = "/api/v5/rubik/stat/contracts/open-interest-history"
    OPEN_INTEREST_VOLUME_ENDPOINT = "/api/v5/rubik/stat/contracts/open-interest-volume"
    OPEN_INTEREST_ENDPOINT = OPEN_INTEREST_HISTORY_ENDPOINT
    FUNDING_ENDPOINT = "/api/v5/public/funding-rate-history"
    MARK_PRICE_ENDPOINT = "/api/v5/market/history-mark-price-candles"
    LIQUIDATION_ENDPOINT = "/api/v5/public/liquidation-orders"

    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        *,
        data_dir: str | Path | None = None,
        session: requests.Session | None = None,
        base_url: str = OKX_BASE_URL,
        timeout: int = 20,
    ) -> None:
        self.symbol = symbol
        self.currency = symbol.split("-")[0]
        self.instrument_family = "-".join(symbol.split("-")[:2])
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "okx_derivatives.db"
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public local reads
    # ------------------------------------------------------------------
    def load_open_interest(self, start: Any, end: Any, *, period: str | None = None) -> pd.DataFrame:
        frame = self._load_table(
            "open_interest",
            start,
            end,
            extra_where="period = ?" if period else "",
            extra_params=(period,) if period else (),
        )
        if frame.empty or period or "period" not in frame.columns:
            return frame
        return self._select_best_oi_period(frame, start=start, end=end)

    def load_funding_rates(self, start: Any, end: Any) -> pd.DataFrame:
        return self._load_table("funding_rate", start, end)

    def load_mark_prices(self, start: Any, end: Any, *, timeframe: str = "1m") -> pd.DataFrame:
        return self._load_table("mark_price", start, end, extra_where="timeframe = ?", extra_params=(timeframe,))

    def load_liquidations(self, start: Any, end: Any) -> pd.DataFrame:
        return self._load_table("liquidation", start, end)

    def coverage(self) -> list[DerivativesCoverage]:
        out: list[DerivativesCoverage] = []
        with self._connect() as conn:
            for table in ("open_interest", "funding_rate", "mark_price", "liquidation"):
                row = conn.execute(
                    f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table} WHERE symbol = ?",
                    (self.symbol,),
                ).fetchone()
                rows = int(row[0] or 0)
                out.append(
                    DerivativesCoverage(
                        dataset=table,
                        rows=rows,
                        start=pd.Timestamp(row[1]) if row[1] else None,
                        end=pd.Timestamp(row[2]) if row[2] else None,
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Public remote fetches
    # ------------------------------------------------------------------
    def fetch_open_interest_history(
        self,
        start: Any,
        end: Any,
        *,
        period: str = "5m",
        page_limit: int = 100,
        sleep_seconds: float = 0.30,
        progress: Callable[[int], None] | None = None,
        auto_select_period: bool = True,
        reference_time: Any | None = None,
    ) -> pd.DataFrame:
        """Fetch symbol-specific OKX contract open-interest history.

        OKX exposes only the latest 1,440 entries for each period.  A 5-minute
        request therefore reaches roughly five days back, while 1-hour data
        reaches roughly sixty days.  When ``auto_select_period`` is enabled,
        the loader transparently upgrades to the finest coarser period that can
        still cover the requested start timestamp.

        Pagination uses only ``end`` because the endpoint defines it as
        "records earlier than ts".  The requested lower bound is enforced
        locally; this avoids ambiguous begin/end interactions while preserving
        strict date filtering.
        """
        requested_period = str(period)
        if requested_period not in OI_PERIODS:
            raise ValueError(f"unsupported OI period: {requested_period}")

        begin_ms = _local_naive_to_ms(start)
        end_ms = _local_naive_to_ms(end)
        if end_ms < begin_ms:
            raise ValueError("end must be greater than or equal to start")

        effective_period = requested_period
        if auto_select_period:
            effective_period = self._choose_oi_period(
                requested_period,
                start=start,
                reference_time=reference_time,
            )
        period_changed = effective_period != requested_period
        if period_changed:
            logger.warning(
                "OKX contract OI keeps only the latest %s entries; "
                "requested period %s cannot reach %s, using %s instead.",
                OI_HISTORY_MAX_ENTRIES,
                requested_period,
                pd.Timestamp(start),
                effective_period,
            )

        cursor_end = end_ms + 1  # ``end`` is exclusive; include a bar exactly at end_ms.
        limit = str(min(100, max(1, int(page_limit))))
        records: list[dict[str, Any]] = []
        seen: set[int] = set()

        while cursor_end > begin_ms:
            params = {
                "instId": self.symbol,
                "period": effective_period,
                "end": str(cursor_end),
                "limit": limit,
            }
            payload = self._get_json(self.OPEN_INTEREST_HISTORY_ENDPOINT, params)
            rows = payload.get("data") or []
            if not rows:
                break

            parsed_rows: list[tuple[int, float, float, float]] = []
            for raw in rows:
                parsed = self._parse_contract_open_interest_row(raw)
                if parsed is not None:
                    parsed_rows.append(parsed)
            if not parsed_rows:
                break

            oldest = min(item[0] for item in parsed_rows)
            newest = max(item[0] for item in parsed_rows)
            for ts_ms, oi, oi_ccy, oi_usd in parsed_rows:
                if ts_ms in seen or ts_ms < begin_ms or ts_ms > end_ms:
                    continue
                seen.add(ts_ms)
                records.append(
                    {
                        "timestamp": _ms_to_local_naive(ts_ms),
                        "oi_contracts": oi,
                        "oi_ccy": oi_ccy,
                        "oi_usd": oi_usd,
                        "period": effective_period,
                    }
                )

            if progress:
                progress(len(records))

            # Once the page has crossed the requested lower bound, no older
            # page can add an in-range record.
            if oldest <= begin_ms or newest < begin_ms:
                break
            if oldest >= cursor_end:
                break
            cursor_end = oldest
            time.sleep(max(0.0, sleep_seconds))

        frame = _frame(records)
        frame.attrs["oi_source"] = "okx_contract_history"
        frame.attrs["requested_period"] = requested_period
        frame.attrs["effective_period"] = effective_period
        frame.attrs["auto_period_changed"] = period_changed
        if frame.empty:
            coverage = _timeframe_delta(effective_period) * OI_HISTORY_MAX_ENTRIES
            frame.attrs["availability_note"] = (
                f"OKX contract OI returned no rows. The endpoint exposes only its latest "
                f"{OI_HISTORY_MAX_ENTRIES:,} {effective_period} entries "
                f"(about {coverage}); requested range may be outside that rolling window."
            )
            return frame

        frame.attrs["availability_note"] = (
            f"OKX contract-level OI for {self.symbol}; requested_period={requested_period}, "
            f"effective_period={effective_period}. oi_usd is used directly."
        )
        self._save_open_interest(frame)
        return frame

    @staticmethod
    def _choose_oi_period(
        requested_period: str,
        *,
        start: Any,
        reference_time: Any | None = None,
    ) -> str:
        """Choose the finest supported period whose latest 1,440 bars reach start."""
        requested_period = str(requested_period)
        if requested_period not in OI_PERIODS:
            raise ValueError(f"unsupported OI period: {requested_period}")

        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        if reference_time is None:
            now_utc = pd.Timestamp.now(tz="UTC").tz_convert(None)
            reference_ts = now_utc + _timezone_offset()
        else:
            reference_ts = pd.Timestamp(reference_time)
            if reference_ts.tzinfo is not None:
                reference_ts = reference_ts.tz_convert(None)

        age = max(pd.Timedelta(0), reference_ts - start_ts)
        requested_index = OI_PERIODS.index(requested_period)
        for candidate in OI_PERIODS[requested_index:]:
            # Keep a two-bar safety margin because the newest bar may still be
            # forming and endpoint boundaries are exclusive.
            usable_entries = OI_HISTORY_MAX_ENTRIES - 2
            if _timeframe_delta(candidate) * usable_entries >= age:
                return candidate
        return OI_PERIODS[-1]

    @staticmethod
    def _select_best_oi_period(frame: pd.DataFrame, *, start: Any, end: Any) -> pd.DataFrame:
        """Return one coherent OI resolution when several periods overlap."""
        if frame.empty or "period" not in frame.columns:
            return frame
        target_start = pd.Timestamp(start)
        target_end = pd.Timestamp(end)
        candidates: list[tuple[int, float, float, str, pd.DataFrame]] = []
        for period, group in frame.groupby("period", sort=False):
            period_text = str(period)
            if period_text not in OI_PERIODS or group.empty:
                continue
            group = group.sort_index()
            delta = _timeframe_delta(period_text)
            covers_start = group.index.min() <= target_start + delta
            covers_end = group.index.max() >= target_end - delta
            full_cover = int(covers_start and covers_end)
            covered_seconds = max(0.0, (min(group.index.max(), target_end) - max(group.index.min(), target_start)).total_seconds())
            resolution_seconds = delta.total_seconds()
            candidates.append((full_cover, covered_seconds, -resolution_seconds, period_text, group))
        if not candidates:
            return frame.sort_index()
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected = candidates[0][4].copy()
        selected.attrs["selected_period"] = candidates[0][3]
        return selected.sort_index()

    @staticmethod
    def _parse_contract_open_interest_row(raw: Any) -> tuple[int, float, float, float] | None:
        """Normalize documented mapping rows and defensive array variants."""
        if isinstance(raw, Mapping):
            ts_value = raw.get("ts", raw.get("timestamp", 0))
            if not ts_value:
                return None
            return (
                int(float(ts_value)),
                _to_float(raw.get("oi", raw.get("openInterest"))),
                _to_float(raw.get("oiCcy")),
                _to_float(raw.get("oiUsd")),
            )

        values = list(raw)
        if len(values) < 2:
            return None
        # Defensive compatibility: [ts, oi, oiCcy, oiUsd].  Missing optional
        # fields remain NaN and are never invented.
        return (
            int(float(values[0])),
            _to_float(values[1] if len(values) > 1 else None),
            _to_float(values[2] if len(values) > 2 else None),
            _to_float(values[3] if len(values) > 3 else None),
        )

    def fetch_funding_rate_history(
        self,
        start: Any,
        end: Any,
        *,
        page_limit: int = 100,
        sleep_seconds: float = 0.12,
        progress: Callable[[int], None] | None = None,
    ) -> pd.DataFrame:
        begin_ms = _local_naive_to_ms(start)
        end_ms = _local_naive_to_ms(end)
        cursor_after = end_ms
        records: list[dict[str, Any]] = []
        seen: set[int] = set()
        while cursor_after >= begin_ms:
            payload = self._get_json(
                self.FUNDING_ENDPOINT,
                {
                    "instId": self.symbol,
                    "after": str(cursor_after),
                    "limit": str(min(100, max(1, int(page_limit)))),
                },
            )
            rows = payload.get("data") or []
            if not rows:
                break
            oldest = cursor_after
            added = 0
            for raw in rows:
                ts_ms = int(float(raw.get("fundingTime", raw.get("ts", 0))))
                oldest = min(oldest, ts_ms)
                if ts_ms in seen or ts_ms < begin_ms or ts_ms > end_ms:
                    continue
                seen.add(ts_ms)
                records.append(
                    {
                        "timestamp": _ms_to_local_naive(ts_ms),
                        "funding_rate": _to_float(raw.get("fundingRate")),
                        "realized_rate": _to_float(raw.get("realizedRate")),
                        "method": str(raw.get("method", "")),
                    }
                )
                added += 1
            if progress:
                progress(len(records))
            if oldest >= cursor_after or added == 0:
                break
            cursor_after = oldest - 1
            time.sleep(max(0.0, sleep_seconds))
        frame = _frame(records)
        self._save_funding(frame)
        return frame

    def fetch_mark_price_history(
        self,
        start: Any,
        end: Any,
        *,
        timeframe: str = "1m",
        page_limit: int = 100,
        sleep_seconds: float = 0.12,
        progress: Callable[[int], None] | None = None,
    ) -> pd.DataFrame:
        begin_ms = _local_naive_to_ms(start)
        end_ms = _local_naive_to_ms(end)
        cursor_after = end_ms
        records: list[dict[str, Any]] = []
        seen: set[int] = set()
        while cursor_after >= begin_ms:
            payload = self._get_json(
                self.MARK_PRICE_ENDPOINT,
                {
                    "instId": self.symbol,
                    "bar": timeframe,
                    "after": str(cursor_after),
                    "limit": str(min(100, max(1, int(page_limit)))),
                },
            )
            rows = payload.get("data") or []
            if not rows:
                break
            oldest = cursor_after
            added = 0
            for raw in rows:
                values = list(raw)
                if len(values) < 5:
                    continue
                ts_ms = int(float(values[0]))
                oldest = min(oldest, ts_ms)
                if ts_ms in seen or ts_ms < begin_ms or ts_ms > end_ms:
                    continue
                seen.add(ts_ms)
                records.append(
                    {
                        "timestamp": _ms_to_local_naive(ts_ms) + _timeframe_delta(timeframe),
                        "open": _to_float(values[1]),
                        "high": _to_float(values[2]),
                        "low": _to_float(values[3]),
                        "close": _to_float(values[4]),
                        "confirm": int(float(values[5])) if len(values) > 5 and str(values[5]) else 1,
                        "timeframe": timeframe,
                    }
                )
                added += 1
            if progress:
                progress(len(records))
            if oldest >= cursor_after or added == 0:
                break
            cursor_after = oldest - 1
            time.sleep(max(0.0, sleep_seconds))
        frame = _frame(records)
        self._save_mark(frame, timeframe=timeframe)
        return frame

    def fetch_liquidation_orders(
        self,
        start: Any,
        end: Any,
        *,
        page_limit: int = 100,
        sleep_seconds: float = 0.12,
        progress: Callable[[int], None] | None = None,
    ) -> pd.DataFrame:
        """Return locally captured/imported liquidation events.

        OKX removed the historical liquidation-orders REST endpoint from its
        public API documentation.  Historical backfill is therefore not
        attempted: public liquidation events must be captured prospectively
        from the liquidation-orders WebSocket channel or imported from an
        external archive.  The unused pagination arguments remain for backward
        compatibility with V1 callers.
        """
        del page_limit, sleep_seconds
        frame = self.load_liquidations(start, end)
        frame.attrs["remote_history_available"] = False
        frame.attrs["availability_note"] = (
            "OKX does not provide public historical liquidation REST backfill; "
            "only locally captured or imported events are available."
        )
        if progress:
            progress(len(frame))
        logger.warning(frame.attrs["availability_note"] )
        return frame

    def fetch_all(self, start: Any, end: Any, *, oi_period: str = "5m", mark_timeframe: str = "1m") -> dict[str, pd.DataFrame]:
        return {
            "open_interest": self.fetch_open_interest_history(start, end, period=oi_period),
            "funding_rate": self.fetch_funding_rate_history(start, end),
            "mark_price": self.fetch_mark_price_history(start, end, timeframe=mark_timeframe),
            "liquidation": self.fetch_liquidation_orders(start, end),
        }

    # ------------------------------------------------------------------
    # CSV import for externally archived history
    # ------------------------------------------------------------------
    def import_csv(self, dataset: str, path: str | Path, *, timeframe: str = "1m") -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "timestamp" not in frame.columns:
            raise ValueError("CSV must contain timestamp")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp").sort_index()
        if dataset == "open_interest":
            self._save_open_interest(frame)
        elif dataset == "funding_rate":
            self._save_funding(frame)
        elif dataset == "mark_price":
            self._save_mark(frame, timeframe=timeframe)
        elif dataset == "liquidation":
            self._save_liquidations(frame)
        else:
            raise ValueError("dataset must be open_interest/funding_rate/mark_price/liquidation")
        return frame

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS open_interest (
                    symbol TEXT NOT NULL, timestamp TEXT NOT NULL, period TEXT NOT NULL,
                    oi_contracts REAL, oi_ccy REAL, oi_usd REAL,
                    PRIMARY KEY(symbol, timestamp, period)
                );
                CREATE INDEX IF NOT EXISTS idx_oi_symbol_ts ON open_interest(symbol, timestamp);
                CREATE TABLE IF NOT EXISTS funding_rate (
                    symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
                    funding_rate REAL, realized_rate REAL, method TEXT,
                    PRIMARY KEY(symbol, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts ON funding_rate(symbol, timestamp);
                CREATE TABLE IF NOT EXISTS mark_price (
                    symbol TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, confirm INTEGER,
                    PRIMARY KEY(symbol, timeframe, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_mark_symbol_ts ON mark_price(symbol, timeframe, timestamp);
                CREATE TABLE IF NOT EXISTS liquidation (
                    symbol TEXT NOT NULL, timestamp TEXT NOT NULL, side TEXT NOT NULL,
                    position_side TEXT, price REAL, size REAL, notional REAL, bankruptcy_loss REAL,
                    PRIMARY KEY(symbol, timestamp, side, price, size)
                );
                CREATE INDEX IF NOT EXISTS idx_liq_symbol_ts ON liquidation(symbol, timestamp);
                """
            )

    def _load_table(
        self,
        table: str,
        start: Any,
        end: Any,
        *,
        extra_where: str = "",
        extra_params: tuple[Any, ...] = (),
    ) -> pd.DataFrame:
        where = "symbol = ? AND timestamp >= ? AND timestamp <= ?"
        params: tuple[Any, ...] = (
            self.symbol,
            pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S.%f"),
            pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S.%f"),
        )
        if extra_where:
            where += " AND " + extra_where
            params += tuple(extra_params)
        with self._connect() as conn:
            frame = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE {where} ORDER BY timestamp",
                conn,
                params=params,
                parse_dates=["timestamp"],
            )
        if frame.empty:
            return pd.DataFrame()
        frame = frame.drop(columns=["symbol"], errors="ignore").set_index("timestamp")
        return frame.sort_index()

    def _save_open_interest(self, frame: pd.DataFrame) -> None:
        self._upsert(
            "open_interest",
            frame,
            columns=("period", "oi_contracts", "oi_ccy", "oi_usd"),
            key_columns=("symbol", "timestamp", "period"),
            defaults={"period": "5m"},
        )

    def _save_funding(self, frame: pd.DataFrame) -> None:
        self._upsert(
            "funding_rate",
            frame,
            columns=("funding_rate", "realized_rate", "method"),
            key_columns=("symbol", "timestamp"),
            defaults={"method": ""},
        )

    def _save_mark(self, frame: pd.DataFrame, *, timeframe: str) -> None:
        self._upsert(
            "mark_price",
            frame,
            columns=("timeframe", "open", "high", "low", "close", "confirm"),
            key_columns=("symbol", "timeframe", "timestamp"),
            defaults={"timeframe": timeframe, "confirm": 1},
        )

    def _save_liquidations(self, frame: pd.DataFrame) -> None:
        self._upsert(
            "liquidation",
            frame,
            columns=("side", "position_side", "price", "size", "notional", "bankruptcy_loss"),
            key_columns=("symbol", "timestamp", "side", "price", "size"),
            defaults={"side": "unknown", "position_side": ""},
        )

    def _upsert(
        self,
        table: str,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...],
        key_columns: tuple[str, ...],
        defaults: Mapping[str, Any],
    ) -> None:
        if frame is None or frame.empty:
            return
        work = frame.copy()
        if "timestamp" in work.columns:
            work["timestamp"] = pd.to_datetime(work["timestamp"])
        else:
            work.index = pd.to_datetime(work.index)
            work.index.name = "timestamp"
            work = work.reset_index()
        work["symbol"] = self.symbol
        for col in columns:
            if col not in work.columns:
                work[col] = defaults.get(col)
        db_columns = list(dict.fromkeys([*key_columns, *columns]))
        values = []
        for row in work[db_columns].itertuples(index=False, name=None):
            clean = []
            for col, value in zip(db_columns, row):
                if col == "timestamp":
                    clean.append(pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f"))
                elif pd.isna(value):
                    clean.append(None)
                elif hasattr(value, "item"):
                    clean.append(value.item())
                else:
                    clean.append(value)
            values.append(tuple(clean))
        placeholders = ",".join("?" for _ in db_columns)
        updates = [col for col in db_columns if col not in key_columns]
        update_sql = ",".join(f"{col}=excluded.{col}" for col in updates)
        sql = (
            f"INSERT INTO {table} ({','.join(db_columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT({','.join(key_columns)}) DO UPDATE SET {update_sql}"
        )
        with self._connect() as conn:
            conn.executemany(sql, values)

    def _get_json(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        max_rate_limit_retries: int = 5,
    ) -> dict[str, Any]:
        request_params = dict(params)
        retries = max(0, int(max_rate_limit_retries))

        for attempt in range(retries + 1):
            response = self.session.get(self.base_url + endpoint, params=request_params, timeout=self.timeout)
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                payload = {}

            status_code = getattr(response, "status_code", None)
            code = str(payload.get("code", "")) if isinstance(payload, Mapping) else ""
            message = str(payload.get("msg", "")) if isinstance(payload, Mapping) else ""
            is_rate_limited = (
                (status_code is not None and int(status_code) == 429)
                or code == "50011"
            )
            if is_rate_limited and attempt < retries:
                headers = getattr(response, "headers", {}) or {}
                retry_after_raw = headers.get("Retry-After") if isinstance(headers, Mapping) else None
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw is not None else 0.0
                except (TypeError, ValueError):
                    retry_after = 0.0
                delay = max(retry_after, min(12.0, 1.0 * (2**attempt)))
                logger.warning(
                    "OKX rate limit %s on %s; retrying in %.1fs (%d/%d)",
                    code or status_code,
                    endpoint,
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
                continue

            if status_code is not None and int(status_code) >= 400:
                raise OKXAPIError(
                    endpoint=endpoint,
                    params=request_params,
                    status_code=int(status_code),
                    code=code,
                    message=message or getattr(response, "text", "") or "HTTP request failed",
                )
            # Keep compatibility with lightweight test doubles that only implement
            # raise_for_status(), while still preserving OKX's JSON error details.
            if status_code is None:
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise OKXAPIError(
                        endpoint=endpoint,
                        params=request_params,
                        status_code=None,
                        code=code,
                        message=message or str(exc),
                    ) from exc
            if code and code != "0":
                raise OKXAPIError(
                    endpoint=endpoint,
                    params=request_params,
                    status_code=int(status_code) if status_code is not None else None,
                    code=code,
                    message=message or "OKX returned a non-zero code",
                )
            return dict(payload)

        raise AssertionError("unreachable")


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
