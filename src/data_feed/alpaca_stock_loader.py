#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Historical US-stock minute bars from Alpaca Market Data v2.

This is a reusable data-feed adapter.  Research code must not call Alpaca HTTP
endpoints directly.  The loader uses the official historical stock-bars API,
paginates on ``next_page_token``, and persists normalized OHLCV locally.

Timestamp contract
------------------
The public methods return a timezone-aware UTC ``DatetimeIndex`` named
``timestamp_utc``.  Keeping source timestamps in UTC avoids ambiguous DST/local
conversions; research callers can convert explicitly to America/New_York.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class AlpacaDataError(RuntimeError):
    pass


class AlpacaStockLoader:
    BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
    MAX_LIMIT = 10_000

    def __init__(
        self,
        symbol: str = "SOXL",
        timeframe: str = "1Min",
        *,
        feed: str = "sip",
        adjustment: str = "raw",
        data_dir: str | os.PathLike[str] = "data",
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        session: requests.Session | Any | None = None,
    ) -> None:
        if timeframe not in {"1Min", "2Min", "5Min", "15Min"}:
            raise ValueError("supported timeframe must be one of 1Min,2Min,5Min,15Min")
        if feed not in {"sip", "iex", "boats"}:
            raise ValueError("feed must be sip, iex, or boats")
        self.symbol = str(symbol).upper().strip()
        self.timeframe = timeframe
        self.feed = feed
        self.adjustment = adjustment

        # CoinBacktest's existing config/env_loader.py reads the repository
        # root .env into a dict; it intentionally does not export values into
        # os.environ.  Respect explicit args first, then process environment
        # variables, then fall back to the project's .env loader.
        project_env = self._load_project_env()
        self.api_key_id = (
            api_key_id
            or os.getenv("APCA_API_KEY_ID")
            or os.getenv("ALPACA_API_KEY_ID")
            or project_env.get("APCA_API_KEY_ID")
            or project_env.get("ALPACA_API_KEY_ID")
            or project_env.get("ALPACA_API_KEY")
        )
        self.api_secret_key = (
            api_secret_key
            or os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_API_SECRET_KEY")
            or project_env.get("APCA_API_SECRET_KEY")
            or project_env.get("ALPACA_API_SECRET_KEY")
            or project_env.get("ALPACA_SECRET_KEY")
        )
        self.session = session or requests.Session()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "alpaca_stock_history.db"
        safe = "_".join([self.symbol, self.timeframe, self.feed, self.adjustment]).replace("-", "_").replace(",", "_")
        self.table_name = f"ALPACA_{safe}"


    @staticmethod
    def _load_project_env() -> dict[str, str]:
        """Read CoinBacktest's repository-root .env without mutating it."""
        try:
            from config.env_loader import load_env_config

            config = load_env_config()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

    def _headers(self) -> dict[str, str]:
        if not self.api_key_id or not self.api_secret_key:
            raise AlpacaDataError(
                "Alpaca API credentials missing. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                "(or ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)."
            )
        return {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
        }

    @staticmethod
    def _as_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(ts)
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return value

    @staticmethod
    def _normalize_bars(payload: list[dict[str, Any]]) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
        frame = pd.DataFrame(payload).rename(
            columns={"t": "timestamp_utc", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"}
        )
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
            if col not in frame.columns:
                frame[col] = pd.NA
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame[["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]
        frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close"]).drop_duplicates("timestamp_utc", keep="last")
        return frame.sort_values("timestamp_utc").set_index("timestamp_utc")

    def fetch_remote(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        max_retries: int = 5,
        request_pause_seconds: float = 0.0,
    ) -> pd.DataFrame:
        start_utc = self._as_utc(start)
        end_utc = self._as_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")

        params: dict[str, Any] = {
            "symbols": self.symbol,
            "timeframe": self.timeframe,
            "start": start_utc.isoformat().replace("+00:00", "Z"),
            "end": end_utc.isoformat().replace("+00:00", "Z"),
            "limit": self.MAX_LIMIT,
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
        }
        page_token: str | None = None
        parts: list[pd.DataFrame] = []

        while True:
            if page_token:
                params["page_token"] = page_token
            else:
                params.pop("page_token", None)

            last_error: Exception | None = None
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.session.get(self.BASE_URL, params=params, headers=self._headers(), timeout=30)
                    if response.status_code == 429:
                        retry_after = float(response.headers.get("Retry-After", "1") or 1)
                        time.sleep(max(retry_after, 1.0))
                        continue
                    response.raise_for_status()
                    break
                except Exception as exc:  # pragma: no cover - exact requests exception varies
                    last_error = exc
                    if attempt + 1 < max_retries:
                        time.sleep(min(2 ** attempt, 8))
            else:
                raise AlpacaDataError(f"Alpaca historical bars request failed: {last_error}")

            data = response.json()
            bars_by_symbol = data.get("bars", {}) or {}
            raw = bars_by_symbol.get(self.symbol, []) if isinstance(bars_by_symbol, dict) else []
            part = self._normalize_bars(raw)
            if not part.empty:
                parts.append(part)
            page_token = data.get("next_page_token")
            if not page_token:
                break
            if request_pause_seconds > 0:
                time.sleep(float(request_pause_seconds))

        if not parts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
        out = pd.concat(parts).sort_index()
        return out.loc[~out.index.duplicated(keep="last")]

    def load_local_data(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,)
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            frame = pd.read_sql_query(f'SELECT * FROM "{self.table_name}" ORDER BY timestamp_utc', conn)
        if frame.empty:
            return frame
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return frame.set_index("timestamp_utc").sort_index()

    def load_local_data_by_date_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Load only a UTC slice from the local cache.

        This keeps overlap audits fast when the cache contains many years of
        1-minute bars. Research callers still access SQLite only through the
        data-feed adapter.
        """
        start_utc = self._as_utc(start)
        end_utc = self._as_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")
        if not self.db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (self.table_name,),
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            # The first range query builds a persistent timestamp index. This
            # is especially useful after multi-year prebuilds (1M+ rows).
            index_name = f"idx_{self.table_name}_timestamp_utc"
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self.table_name}" (timestamp_utc)'
            )
            frame = pd.read_sql_query(
                f'SELECT * FROM "{self.table_name}" WHERE timestamp_utc >= ? AND timestamp_utc <= ? ORDER BY timestamp_utc',
                conn,
                params=(str(start_utc), str(end_utc)),
            )
        if frame.empty:
            return frame
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return frame.set_index("timestamp_utc").sort_index()

    def save_local_data(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        work = frame.copy()
        idx = pd.DatetimeIndex(work.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        work.index = idx
        work.index.name = "timestamp_utc"
        existing = self.load_local_data()
        if not existing.empty:
            work = pd.concat([existing, work]).sort_index()
            work = work.loc[~work.index.duplicated(keep="last")]
        payload = work.reset_index()
        payload["timestamp_utc"] = pd.to_datetime(payload["timestamp_utc"], utc=True).astype(str)
        with sqlite3.connect(self.db_path) as conn:
            payload.to_sql(self.table_name, conn, if_exists="replace", index=False)

    def fetch_data_by_date_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        local_only: bool = False,
    ) -> pd.DataFrame:
        start_utc = self._as_utc(start)
        end_utc = self._as_utc(end)
        local_slice = self.load_local_data_by_date_range(start_utc, end_utc)
        if local_only:
            return local_slice

        # Fetch the requested range.  Alpaca pagination is efficient and the
        # merge is idempotent, so this remains simple and resume-safe.
        remote = self.fetch_remote(start_utc, end_utc)
        if not remote.empty:
            self.save_local_data(remote)
        merged = pd.concat([local_slice, remote]) if not local_slice.empty or not remote.empty else pd.DataFrame()
        if merged.empty:
            return merged
        merged = merged.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
        return merged.loc[(merged.index >= start_utc) & (merged.index <= end_utc)].copy()
