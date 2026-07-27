#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate local OKX official trades ZIP files into cached OHLCV/order-flow bars.

This module is the bridge between raw OKX trades files and strategy-friendly bar
features:

- Cache-first by default: strategy-facing reads come from ``data/okx_trade_bars.db``
  when the requested UTC days are already covered.
- Missing-day build: only genuinely missing UTC coverage triggers raw trades
  download/aggregation before the DB read.
- Schema/feature migrations never force a rebuild by themselves; new columns are
  added with defaults and can be backfilled explicitly with force rebuild.
- Memory safe: aggregation is done one UTC day at a time.
- Interface style: close to ``OKXDataLoader.fetch_data_by_date_range``.

Timestamp policy:
The project's current ``OKXDataLoader`` stores timestamps as timezone-shifted
naive datetimes according to ``config.loader.TIMEZONE``. This loader follows the
same default behavior so trade-derived 1m/5m bars can align with normal K-line
bars. Set ``align_with_okx_loader_timezone=False`` if you want pure UTC-naive
bar timestamps instead.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover - project fallback
    TIMEZONE = "+8"

try:
    from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader
except ImportError:  # pragma: no cover - direct file execution fallback
    from okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader

try:
    from src.utils.log import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - standalone fallback
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


Timeframe = str
CVDMode = str


class OKXTradeBarLoader:
    """Load trade-aggregated OHLCV/order-flow bars with SQLite cache.

    Parameters
    ----------
    symbol:
        OKX instrument id, e.g. ``ETH-USDT-SWAP``.
    timeframe:
        Target bar timeframe. Supports second/minute/hour/day bars such as
        ``1s``, ``5s``, ``30s``, ``1m``, ``5m``, ``1H`` and ``1D``.
    data_dir:
        Project data directory. Defaults to ``<project_root>/data``.
    db_name:
        SQLite DB filename under ``data_dir``.
    trades_url_template:
        Official OKX trades ZIP template. Leave default unless OKX changes its
        CDN path.
    contract_value:
        Multiplier used only for quote notional columns. For OKX
        ``ETH-USDT-SWAP`` the historical trade ``size`` is contract count, and
        the contract face value is normally 0.1 ETH. If your raw data already
        uses base-asset size, set this to 1.0.
    large_trade_notional_threshold:
        A trade is treated as "large" when ``price * size * contract_value`` is
        greater than or equal to this threshold.
    align_with_okx_loader_timezone:
        If True, store/return timestamps shifted by ``config.loader.TIMEZONE``
        as naive datetimes, matching ``OKXDataLoader``. If False, use UTC-naive.
    """

    CACHE_SCHEMA_VERSION = 2

    # Common presets kept for help text/backward compatibility. The loader is
    # intentionally not limited to these values; any positive integer
    # second/minute/hour/day timeframe such as 1s, 5s, 10s, 30s, 1m, 2H is valid.
    FREQ_MAP: dict[str, str] = {
        "1s": "1s",
        "5s": "5s",
        "10s": "10s",
        "15s": "15s",
        "30s": "30s",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1H": "1h",
        "4H": "4h",
        "1D": "1D",
    }

    BASE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades_count",
        "buy_volume",
        "sell_volume",
        "notional",
        "buy_notional",
        "sell_notional",
        "buy_trades_count",
        "sell_trades_count",
        "delta_volume",
        "delta_notional",
        "cvd_volume",
        "cvd_notional",
        "taker_buy_ratio",
        "avg_trade_size",
        "vwap",
        "large_buy_notional",
        "large_sell_notional",
        "large_buy_trades_count",
        "large_sell_trades_count",
        "large_delta_notional",
        "large_trades_count",
        "max_trade_notional",
        "max_trade_size",
    ]

    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        timeframe: Timeframe = "1m",
        data_dir: str | os.PathLike[str] | None = None,
        db_name: str = "okx_trade_bars.db",
        trades_url_template: str = DEFAULT_OKX_TRADES_URL_TEMPLATE,
        contract_value: float | None = None,
        large_trade_notional_threshold: float = 100_000.0,
        align_with_okx_loader_timezone: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = self._normalize_timeframe(timeframe)
        self.freq = self._freq_from_timeframe(self.timeframe)

        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_name

        self.trades_url_template = trades_url_template
        self.contract_value = self._infer_contract_value(symbol) if contract_value is None else float(contract_value)
        self.large_trade_notional_threshold = float(large_trade_notional_threshold)
        self.align_with_okx_loader_timezone = bool(align_with_okx_loader_timezone)
        self.timezone_offset_hours = self._parse_timezone_offset_hours(TIMEZONE) if self.align_with_okx_loader_timezone else 0

        self.tick_loader = OKXTickLoader(
            symbol=self.symbol,
            data_dir=self.data_dir,
            trades_url_template=self.trades_url_template,
        )

        self.table_name = self._build_table_name()
        self.coverage_table_name = "trade_bar_coverage"
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_data_by_date_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
        cvd_mode: CVDMode = "range",
        build_missing: bool = True,
    ) -> pd.DataFrame:
        """Return trade-aggregated bars for ``[start_date, end_date]``.

        This is the main strategy-facing interface. It is cache-first by
        default: existing covered UTC days are read from SQLite, while genuinely
        missing UTC days are downloaded/aggregated before the read. Schema
        upgrades, added feature columns, or older coverage metadata do not force
        rebuilds by themselves. Pass ``build_missing=False`` for a pure DB read,
        or ``force_rebuild=True`` for explicit backfill/rebuild.

        ``cvd_mode``:
        - ``range``: CVD starts from zero at ``start_date``. Best for isolated
          backtests and factor windows.
        - ``db``: CVD is calculated from the earliest cached bar in the DB up to
          ``end_date``, then sliced to the requested range.
        """
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        if end_ts < start_ts:
            raise ValueError(f"end_date must be >= start_date, got: {start_ts} -> {end_ts}")
        if cvd_mode not in {"range", "db"}:
            raise ValueError("cvd_mode must be 'range' or 'db'")

        if build_missing or force_rebuild:
            required_utc_days = list(self._required_utc_days_for_local_range(start_ts, end_ts))
            self.ensure_cached_days(required_utc_days, chunksize=chunksize, force_rebuild=force_rebuild)

        if cvd_mode == "db":
            df = self.load_local_data(end_date=end_ts)
            if not df.empty:
                df = self._recompute_cvd(df)
                df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        else:
            df = self.load_local_data(start_date=start_ts, end_date=end_ts)
            if not df.empty:
                df = self._recompute_cvd(df)

        return self._finalize_return_df(df)

    def fetch_historical_data(
        self,
        limit: int = 50_000,
        *,
        end_date: str | datetime | pd.Timestamp | None = None,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
        cvd_mode: CVDMode = "range",
        build_missing: bool = True,
    ) -> pd.DataFrame:
        """Return the latest ``limit`` cached bars ending at ``end_date``.

        By default this is cache-first and will build only missing UTC coverage.
        Pass ``build_missing=False`` for a pure DB read, or ``force_rebuild=True``
        for explicit backfill/rebuild.
        """
        if limit <= 0:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        end_ts = self._normalize_query_timestamp(end_date or pd.Timestamp.now(), is_end=True)
        seconds = self._get_seconds(self.timeframe)
        # Add a small buffer so missing/no-trade intervals do not under-fill the result.
        start_ts = end_ts - pd.Timedelta(seconds=seconds * int(limit * 1.15 + 10))
        df = self.fetch_data_by_date_range(
            start_ts,
            end_ts,
            chunksize=chunksize,
            force_rebuild=force_rebuild,
            cvd_mode=cvd_mode,
            build_missing=build_missing,
        )
        return df.tail(limit)

    def ensure_cached_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> None:
        """Ensure DB has bars for the raw UTC days needed by a query range."""
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        self.ensure_cached_days(
            self._required_utc_days_for_local_range(start_ts, end_ts),
            chunksize=chunksize,
            force_rebuild=force_rebuild,
        )

    def ensure_cached_days(
        self,
        utc_days: Iterator[date] | list[date],
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> None:
        """Aggregate missing UTC raw-trade days into SQLite."""
        for day in utc_days:
            d = self._parse_date(day)
            if not force_rebuild and self._has_coverage(d):
                continue
            logger.info("聚合 OKX trades -> %s %s day=%s", self.symbol, self.timeframe, d)
            bars = self.aggregate_day(d, chunksize=chunksize)
            if not bars.empty:
                self._upsert_bars(bars)
            self._mark_coverage(d, rows=len(bars))

    def aggregate_day(self, day: str | date, *, chunksize: int = 300_000) -> pd.DataFrame:
        """Aggregate one raw UTC trade day into OHLCV/order-flow bars.

        This function reads one official OKX trades ZIP in streaming chunks and
        keeps only partial bars in memory. It intentionally does not call
        ``OKXTickLoader.iter_trades`` for parsing because that normalizer keeps
        extra raw JSON columns that are useful for tick replay but unnecessarily
        slow for bar aggregation. ``OKXTickLoader`` is still reused for
        local-file discovery and missing-file download.
        """
        d = self._parse_date(day)
        raw_file = self._ensure_raw_trade_file(d)
        partials: list[pd.DataFrame] = []
        for raw in self._iter_trade_csv_chunks(raw_file, chunksize=chunksize):
            chunk = self._normalize_trade_chunk_fast(raw)
            if chunk.empty:
                continue
            partial = self._aggregate_trade_chunk_partial(chunk)
            if not partial.empty:
                partials.append(partial)

        if not partials:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        return self._combine_partial_bars(partials)

    def _ensure_raw_trade_file(self, day: date) -> Path:
        raw_file = self.tick_loader.find_local_trade_file(day, template=self.trades_url_template)
        if raw_file is not None and raw_file.exists() and raw_file.stat().st_size > 0:
            return raw_file
        return self.tick_loader.download_official_trade_file(day, self.trades_url_template)

    def _iter_trade_csv_chunks(self, path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
        wanted = {
            "ts", "timestamp", "time", "datetime", "created_time", "createdtime",
            "create_time", "created_at", "createdat", "px", "price", "sz", "size",
            "qty", "amount", "side",
        }

        def use_column(name: str) -> bool:
            return str(name).strip().lower() in wanted

        p = Path(path)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                members = [name for name in zf.namelist() if not name.endswith("/")]
                if not members:
                    raise RuntimeError(f"empty OKX trade ZIP: {p}")
                for name in members:
                    with zf.open(name) as f:
                        yield from pd.read_csv(f, chunksize=chunksize, usecols=use_column)
            return
        yield from pd.read_csv(p, chunksize=chunksize, usecols=use_column)

    def _normalize_trade_chunk_fast(self, raw: pd.DataFrame) -> pd.DataFrame:
        rename: dict[str, str] = {}
        for col in raw.columns:
            low = str(col).strip().lower()
            if low in {"ts", "timestamp", "time", "datetime", "created_time", "createdtime", "create_time", "created_at", "createdat"}:
                rename[col] = "ts_ms"
            elif low in {"px", "price"}:
                rename[col] = "price"
            elif low in {"sz", "size", "qty", "amount"}:
                rename[col] = "size"
            elif low == "side":
                rename[col] = "side"
        df = raw.rename(columns=rename)
        required = {"ts_ms", "price", "size"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"trade file missing columns {sorted(missing)}; columns={list(raw.columns)}")

        ts_num = pd.to_numeric(df["ts_ms"], errors="coerce")
        price = pd.to_numeric(df["price"], errors="coerce")
        size = pd.to_numeric(df["size"], errors="coerce")
        ok = ts_num.notna() & price.notna() & size.notna()
        if not ok.any():
            return pd.DataFrame(columns=["timestamp", "price", "size", "side"])

        ts_num = ts_num.loc[ok]
        # Seconds timestamps are uncommon for OKX historical files but supported.
        if ts_num.max() < 10_000_000_000:
            ts_num = ts_num * 1000

        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_num.astype("int64"), unit="ms", utc=True),
                "price": price.loc[ok].astype(float),
                "size": size.loc[ok].astype(float),
                "side": df.get("side", "").loc[ok].astype(str).str.lower() if "side" in df.columns else "",
            }
        )
        return out.sort_values("timestamp").reset_index(drop=True)

    def _aggregate_trade_chunk_partial(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        df = trades.copy()
        is_buy = df["side"].eq("buy")
        is_sell = df["side"].eq("sell")
        df["notional"] = df["price"] * df["size"] * self.contract_value
        df["price_size"] = df["price"] * df["size"]
        df["buy_volume"] = df["size"].where(is_buy, 0.0)
        df["sell_volume"] = df["size"].where(is_sell, 0.0)
        df["buy_notional"] = df["notional"].where(is_buy, 0.0)
        df["sell_notional"] = df["notional"].where(is_sell, 0.0)
        df["buy_trades_count"] = is_buy.astype("int64")
        df["sell_trades_count"] = is_sell.astype("int64")
        is_large = df["notional"].abs() >= self.large_trade_notional_threshold
        df["large_buy_notional"] = df["notional"].where(is_large & is_buy, 0.0)
        df["large_sell_notional"] = df["notional"].where(is_large & is_sell, 0.0)
        df["large_trades_count"] = is_large.astype("int64")
        df["large_buy_trades_count"] = (is_large & is_buy).astype("int64")
        df["large_sell_trades_count"] = (is_large & is_sell).astype("int64")

        df = df.sort_values("timestamp").set_index("timestamp")
        grouped = df.resample(self.freq, label="left", closed="left")
        partial = pd.DataFrame(
            {
                "open": grouped["price"].first(),
                "high": grouped["price"].max(),
                "low": grouped["price"].min(),
                "close": grouped["price"].last(),
                "volume": grouped["size"].sum(),
                "trades_count": grouped["price"].count(),
                "buy_volume": grouped["buy_volume"].sum(),
                "sell_volume": grouped["sell_volume"].sum(),
                "notional": grouped["notional"].sum(),
                "buy_notional": grouped["buy_notional"].sum(),
                "sell_notional": grouped["sell_notional"].sum(),
                "buy_trades_count": grouped["buy_trades_count"].sum(),
                "sell_trades_count": grouped["sell_trades_count"].sum(),
                "price_size_sum": grouped["price_size"].sum(),
                "large_buy_notional": grouped["large_buy_notional"].sum(),
                "large_sell_notional": grouped["large_sell_notional"].sum(),
                "large_buy_trades_count": grouped["large_buy_trades_count"].sum(),
                "large_sell_trades_count": grouped["large_sell_trades_count"].sum(),
                "large_trades_count": grouped["large_trades_count"].sum(),
                "max_trade_notional": grouped["notional"].max(),
                "max_trade_size": grouped["size"].max(),
            }
        )
        return partial.dropna(subset=["open", "high", "low", "close"])

    def _combine_partial_bars(self, partials: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(partials).sort_index(kind="stable")
        grouped = combined.groupby(level=0, sort=True)
        bars = pd.DataFrame(
            {
                "open": grouped["open"].first(),
                "high": grouped["high"].max(),
                "low": grouped["low"].min(),
                "close": grouped["close"].last(),
                "volume": grouped["volume"].sum(),
                "trades_count": grouped["trades_count"].sum(),
                "buy_volume": grouped["buy_volume"].sum(),
                "sell_volume": grouped["sell_volume"].sum(),
                "notional": grouped["notional"].sum(),
                "buy_notional": grouped["buy_notional"].sum(),
                "sell_notional": grouped["sell_notional"].sum(),
                "buy_trades_count": grouped["buy_trades_count"].sum(),
                "sell_trades_count": grouped["sell_trades_count"].sum(),
                "price_size_sum": grouped["price_size_sum"].sum(),
                "large_buy_notional": grouped["large_buy_notional"].sum(),
                "large_sell_notional": grouped["large_sell_notional"].sum(),
                "large_buy_trades_count": grouped["large_buy_trades_count"].sum(),
                "large_sell_trades_count": grouped["large_sell_trades_count"].sum(),
                "large_trades_count": grouped["large_trades_count"].sum(),
                "max_trade_notional": grouped["max_trade_notional"].max(),
                "max_trade_size": grouped["max_trade_size"].max(),
            }
        )
        if bars.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        bars["delta_volume"] = bars["buy_volume"] - bars["sell_volume"]
        bars["delta_notional"] = bars["buy_notional"] - bars["sell_notional"]
        bars["cvd_volume"] = bars["delta_volume"].cumsum()
        bars["cvd_notional"] = bars["delta_notional"].cumsum()
        bars["taker_buy_ratio"] = self._safe_divide(bars["buy_volume"], bars["volume"])
        bars["avg_trade_size"] = self._safe_divide(bars["volume"], bars["trades_count"])
        bars["vwap"] = self._safe_divide(bars["price_size_sum"], bars["volume"])
        bars["large_delta_notional"] = bars["large_buy_notional"] - bars["large_sell_notional"]
        bars = bars.drop(columns=["price_size_sum"])

        if self.timezone_offset_hours:
            bars.index = bars.index + pd.Timedelta(hours=self.timezone_offset_hours)
        if getattr(bars.index, "tz", None) is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "timestamp"
        return self._finalize_return_df(bars)

    def aggregate_trades_df(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a normalized trades DataFrame into bars.

        Expected normalized columns are compatible with ``OKXTickLoader``:
        ``timestamp``, ``price``, ``size``, ``side``.
        """
        if trades.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        df = trades.copy()
        if "timestamp" not in df.columns:
            raise ValueError(f"trades data has no timestamp column: {list(df.columns)}")
        if "price" not in df.columns or "size" not in df.columns:
            raise ValueError(f"trades data must contain price and size columns: {list(df.columns)}")

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        ts = ts.loc[df.index]
        df["timestamp"] = ts
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df = df.dropna(subset=["timestamp", "price", "size"])
        if df.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        df["side"] = df.get("side", "").astype(str).str.lower()
        # Official OKX files may use BUY/SELL. After lower-casing, buy means
        # taker buy and sell means taker sell.
        is_buy = df["side"].eq("buy")
        is_sell = df["side"].eq("sell")

        df["notional"] = df["price"] * df["size"] * self.contract_value
        df["price_size"] = df["price"] * df["size"]
        df["buy_volume"] = df["size"].where(is_buy, 0.0)
        df["sell_volume"] = df["size"].where(is_sell, 0.0)
        df["buy_notional"] = df["notional"].where(is_buy, 0.0)
        df["sell_notional"] = df["notional"].where(is_sell, 0.0)
        df["buy_trades_count"] = is_buy.astype("int64")
        df["sell_trades_count"] = is_sell.astype("int64")
        is_large = df["notional"].abs() >= self.large_trade_notional_threshold
        df["large_buy_notional"] = df["notional"].where(is_large & is_buy, 0.0)
        df["large_sell_notional"] = df["notional"].where(is_large & is_sell, 0.0)
        df["large_trades_count"] = is_large.astype("int64")
        df["large_buy_trades_count"] = (is_large & is_buy).astype("int64")
        df["large_sell_trades_count"] = (is_large & is_sell).astype("int64")

        df = df.sort_values("timestamp").set_index("timestamp")
        grouped = df.resample(self.freq, label="left", closed="left")

        bars = pd.DataFrame(
            {
                "open": grouped["price"].first(),
                "high": grouped["price"].max(),
                "low": grouped["price"].min(),
                "close": grouped["price"].last(),
                "volume": grouped["size"].sum(),
                "trades_count": grouped["price"].count(),
                "buy_volume": grouped["buy_volume"].sum(),
                "sell_volume": grouped["sell_volume"].sum(),
                "notional": grouped["notional"].sum(),
                "buy_notional": grouped["buy_notional"].sum(),
                "sell_notional": grouped["sell_notional"].sum(),
                "buy_trades_count": grouped["buy_trades_count"].sum(),
                "sell_trades_count": grouped["sell_trades_count"].sum(),
                "price_size_sum": grouped["price_size"].sum(),
                "large_buy_notional": grouped["large_buy_notional"].sum(),
                "large_sell_notional": grouped["large_sell_notional"].sum(),
                "large_buy_trades_count": grouped["large_buy_trades_count"].sum(),
                "large_sell_trades_count": grouped["large_sell_trades_count"].sum(),
                "large_trades_count": grouped["large_trades_count"].sum(),
                "max_trade_notional": grouped["notional"].max(),
                "max_trade_size": grouped["size"].max(),
            }
        )

        bars = bars.dropna(subset=["open", "high", "low", "close"])
        if bars.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        bars["delta_volume"] = bars["buy_volume"] - bars["sell_volume"]
        bars["delta_notional"] = bars["buy_notional"] - bars["sell_notional"]
        bars["cvd_volume"] = bars["delta_volume"].cumsum()
        bars["cvd_notional"] = bars["delta_notional"].cumsum()
        bars["taker_buy_ratio"] = self._safe_divide(bars["buy_volume"], bars["volume"])
        bars["avg_trade_size"] = self._safe_divide(bars["volume"], bars["trades_count"])
        bars["vwap"] = self._safe_divide(bars["price_size_sum"], bars["volume"])
        bars["large_delta_notional"] = bars["large_buy_notional"] - bars["large_sell_notional"]
        bars = bars.drop(columns=["price_size_sum"])

        # Match OKXDataLoader timestamp style: UTC -> configured local offset -> naive.
        if self.timezone_offset_hours:
            bars.index = bars.index + pd.Timedelta(hours=self.timezone_offset_hours)
        if getattr(bars.index, "tz", None) is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "timestamp"

        bars = bars[self.BASE_COLUMNS]
        return self._finalize_return_df(bars)

    def load_local_data(
        self,
        start_date: str | datetime | pd.Timestamp | None = None,
        end_date: str | datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load cached bars from SQLite. Returns timestamp-indexed DataFrame."""
        self._init_db()
        where: list[str] = []
        params: list[str] = []
        if start_date is not None:
            start_ts = self._normalize_query_timestamp(start_date, is_end=False)
            where.append("timestamp >= ?")
            params.append(self._timestamp_to_db_text(start_ts))
        if end_date is not None:
            end_ts = self._normalize_query_timestamp(end_date, is_end=True)
            where.append("timestamp <= ?")
            params.append(self._timestamp_to_db_text(end_ts))

        sql = f"SELECT * FROM {self.table_name}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp ASC"

        with self._get_db_connection() as conn:
            try:
                df = pd.read_sql_query(sql, conn, params=params, parse_dates=["timestamp"])
            except Exception as exc:
                logger.warning("读取 trades 聚合 DB 失败 table=%s error=%s", self.table_name, exc)
                return pd.DataFrame(columns=self.BASE_COLUMNS)

        if df.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        df = df.set_index("timestamp")
        df.index.name = "timestamp"
        return self._finalize_return_df(df)

    def save_local_data(self, df: pd.DataFrame) -> None:
        """Replace the whole SQLite table with ``df``.

        Usually you should not need this. Normal cache writes use upsert by day.
        """
        self._init_db()
        if df.empty:
            return
        clean = self._prepare_bars_for_db(df)
        with self._get_db_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
            self._insert_prepared_bars(conn, clean)
            conn.commit()
        logger.info("保存 trades 聚合 bars 到 DB: table=%s rows=%s", self.table_name, len(clean))

    def delete_cache(self) -> None:
        """Delete this loader's cached bars and coverage records."""
        with self._get_db_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
            keys = tuple(self._compatible_cache_keys())
            placeholders = ",".join(["?"] * len(keys))
            conn.execute(f"DELETE FROM {self.coverage_table_name} WHERE cache_key IN ({placeholders})", keys)
            conn.commit()

    # ------------------------------------------------------------------
    # DB internals
    # ------------------------------------------------------------------
    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")  # about 256 MiB page cache when possible
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA wal_autocheckpoint=10000")
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    def _init_db(self) -> None:
        cols_sql = ",\n                ".join(f"{col} REAL" for col in self.BASE_COLUMNS)
        # Count columns should be integer-like, but REAL keeps schema simple and
        # robust with pandas/numpy dtypes. They are cast back on return.
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    timestamp TEXT PRIMARY KEY,
                    {cols_sql}
                )
                """
            )
            self._ensure_bar_table_columns(conn)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.coverage_table_name} (
                    cache_key TEXT NOT NULL,
                    utc_day TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    rows INTEGER NOT NULL DEFAULT 0,
                    params_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (cache_key, utc_day)
                )
                """
            )
            conn.commit()


    def _ensure_bar_table_columns(self, conn: sqlite3.Connection) -> None:
        """Migrate existing trade-bar tables when new feature columns are added."""
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()}
        for col in self.BASE_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {col} REAL NOT NULL DEFAULT 0")
                existing.add(col)

    def _upsert_bars(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        clean = self._prepare_bars_for_db(df)
        with self._get_db_connection() as conn:
            self._insert_prepared_bars(conn, clean)
            conn.commit()
        logger.debug("写入 trades 聚合 bars: table=%s rows=%s", self.table_name, len(clean))

    def _insert_prepared_bars(self, conn: sqlite3.Connection, clean: pd.DataFrame) -> None:
        db_cols = ["timestamp", *self.BASE_COLUMNS]
        placeholders = ",".join(["?"] * len(db_cols))
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in self.BASE_COLUMNS])
        sql = f"""
            INSERT INTO {self.table_name} ({', '.join(db_cols)})
            VALUES ({placeholders})
            ON CONFLICT(timestamp) DO UPDATE SET {update_sql}
        """
        conn.executemany(sql, clean[db_cols].itertuples(index=False, name=None))

    def _prepare_bars_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = self._finalize_return_df(df).copy()
        if clean.empty:
            return pd.DataFrame(columns=["timestamp", *self.BASE_COLUMNS])
        clean = clean.reset_index()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"]).map(self._timestamp_to_db_text)
        for col in self.BASE_COLUMNS:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)
        return clean[["timestamp", *self.BASE_COLUMNS]]

    def _has_coverage(self, utc_day: date) -> bool:
        """Return True when the UTC raw-trade day is already cached.

        Coverage is intentionally based on the physical cached day, not on the
        feature/schema version. Adding columns such as max_trade_notional should
        not make normal reads rebuild every old day. If the day is not covered,
        normal cache-first reads still build/download it via ensure_cached_days.
        """
        d = self._parse_date(utc_day)
        keys = tuple(self._compatible_cache_keys())
        placeholders = ",".join(["?"] * len(keys))
        with self._get_db_connection() as conn:
            row = conn.execute(
                f"""
                SELECT rows
                FROM {self.coverage_table_name}
                WHERE utc_day = ? AND cache_key IN ({placeholders})
                LIMIT 1
                """,
                (d.isoformat(), *keys),
            ).fetchone()
        return row is not None

    def _mark_coverage(self, utc_day: date, *, rows: int) -> None:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        params = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "contract_value": self.contract_value,
            "large_trade_notional_threshold": self.large_trade_notional_threshold,
            "align_with_okx_loader_timezone": self.align_with_okx_loader_timezone,
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
            "timezone": TIMEZONE,
        }
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.coverage_table_name}
                    (cache_key, utc_day, symbol, timeframe, table_name, rows, params_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key, utc_day) DO UPDATE SET
                    rows=excluded.rows,
                    params_json=excluded.params_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self._cache_key(),
                    utc_day.isoformat(),
                    self.symbol,
                    self.timeframe,
                    self.table_name,
                    int(rows),
                    json.dumps(params, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Timestamp/range helpers
    # ------------------------------------------------------------------
    def _required_utc_days_for_local_range(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Iterator[date]:
        """Map query timestamps to the raw OKX UTC daily files needed."""
        if self.timezone_offset_hours:
            raw_start = start_ts - pd.Timedelta(hours=self.timezone_offset_hours)
            raw_end = end_ts - pd.Timedelta(hours=self.timezone_offset_hours)
        else:
            raw_start = start_ts
            raw_end = end_ts
        cur = raw_start.date()
        last = raw_end.date()
        while cur <= last:
            yield cur
            cur += timedelta(days=1)

    def _normalize_query_timestamp(self, value: str | datetime | pd.Timestamp, *, is_end: bool) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            if self.align_with_okx_loader_timezone:
                target_tz = timezone(timedelta(hours=self.timezone_offset_hours))
                ts = ts.tz_convert(target_tz).tz_localize(None)
            else:
                ts = ts.tz_convert("UTC").tz_localize(None)
        # Date-only strings should mean whole day range, not exactly midnight at end.
        if is_end and isinstance(value, str) and len(value.strip()) == 10:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return ts

    def _timestamp_to_db_text(self, ts: pd.Timestamp | datetime | str) -> str:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert(None)
        return t.strftime("%Y-%m-%d %H:%M:%S")

    def _parse_date(self, value: str | date | datetime | pd.Timestamp) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    def _parse_timezone_offset_hours(self, tz_text: str) -> int:
        text = str(tz_text).strip()
        if text.upper() in {"UTC", "Z", "+0", "-0", "0"}:
            return 0
        try:
            if text.startswith("+"):
                return int(text[1:].split(":")[0])
            if text.startswith("-"):
                return -int(text[1:].split(":")[0])
            return int(text)
        except Exception:
            logger.warning("无法解析 TIMEZONE=%r，trades 聚合 timestamp 不做偏移", tz_text)
            return 0

    def _normalize_timeframe(self, timeframe: str) -> str:
        text = str(timeframe).strip()
        if not text:
            raise ValueError("timeframe cannot be empty")

        unit = text[-1]
        amount_text = text[:-1]
        if not amount_text.isdigit():
            raise ValueError(
                f"invalid timeframe: {timeframe!r}. Use positive integer + unit, e.g. 1s, 5s, 30s, 1m, 5m, 1H, 1D."
            )
        amount = int(amount_text)
        if amount <= 0:
            raise ValueError(f"timeframe must be positive, got: {timeframe!r}")

        unit_low = unit.lower()
        if unit_low == "s":
            return f"{amount}s"
        if unit_low == "m":
            return f"{amount}m"
        if unit_low == "h":
            return f"{amount}H"
        if unit_low == "d":
            return f"{amount}D"
        raise ValueError(
            f"unsupported timeframe unit: {timeframe!r}. Supported units: s(seconds), m(minutes), H(hours), D(days)."
        )

    def _freq_from_timeframe(self, timeframe: str) -> str:
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        if unit == "s":
            return f"{amount}s"
        if unit == "m":
            return f"{amount}min"
        if unit == "H":
            return f"{amount}h"
        if unit == "D":
            return f"{amount}D"
        raise ValueError(f"unsupported normalized timeframe: {timeframe!r}")

    def _get_seconds(self, timeframe: str) -> int:
        normalized = self._normalize_timeframe(timeframe)
        amount = int(normalized[:-1])
        unit = normalized[-1]
        multipliers = {"s": 1, "m": 60, "H": 3600, "D": 86400}
        return amount * multipliers[unit]

    # ------------------------------------------------------------------
    # Formatting/math helpers
    # ------------------------------------------------------------------
    def _finalize_return_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        out = df.copy()
        if "timestamp" in out.columns:
            out = out.set_index("timestamp")
        out.index = pd.to_datetime(out.index)
        out.index.name = "timestamp"
        out = out.sort_index()

        for col in self.BASE_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        count_cols = [
            "trades_count",
            "buy_trades_count",
            "sell_trades_count",
            "large_buy_trades_count",
            "large_sell_trades_count",
            "large_trades_count",
        ]
        for col in count_cols:
            out[col] = out[col].round().astype("int64")

        return out[self.BASE_COLUMNS]

    def _recompute_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._finalize_return_df(df)
        if out.empty:
            return out
        out["cvd_volume"] = out["delta_volume"].cumsum()
        out["cvd_notional"] = out["delta_notional"].cumsum()
        return out

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        result = numerator.astype(float) / denominator.astype(float).replace(0, pd.NA)
        return result.fillna(0.0)

    def _build_table_name(self) -> str:
        contract_tag = self._number_tag(self.contract_value)
        large_tag = self._number_tag(self.large_trade_notional_threshold)
        tz_tag = f"tz{self.timezone_offset_hours:+d}" if self.align_with_okx_loader_timezone else "utc"
        return self._safe_sql_identifier(f"{self.symbol}_trade_bars_{self.timeframe}_cv{contract_tag}_lg{large_tag}_{tz_tag}")

    def _cache_key(self) -> str:
        """Stable coverage key for this physical cache table.

        Do not include CACHE_SCHEMA_VERSION here. Schema/features can evolve via
        ALTER TABLE without invalidating historical day coverage. Use
        force_rebuild=True when a real feature backfill is needed.
        """
        return self._cache_key_for_schema(None)

    def _compatible_cache_keys(self) -> list[str]:
        keys = [self._cache_key()]
        # Backward compatibility for previous coverage keys that included the
        # schema version. This prevents a normal read from rebuilding old days
        # just because the feature schema changed.
        for schema_version in range(int(self.CACHE_SCHEMA_VERSION), 0, -1):
            legacy = self._cache_key_for_schema(schema_version)
            if legacy not in keys:
                keys.append(legacy)
        return keys

    def _cache_key_for_schema(self, schema_version: int | None) -> str:
        parts = [
            self.symbol,
            self.timeframe,
            f"contract_value={self.contract_value:g}",
            f"large={self.large_trade_notional_threshold:g}",
        ]
        if schema_version is not None:
            parts.append(f"schema={schema_version}")
        parts.append(f"tz_offset={self.timezone_offset_hours if self.align_with_okx_loader_timezone else 0}")
        return "|".join(parts)

    def _safe_sql_identifier(self, value: str) -> str:
        text = str(value).replace("-", "_").replace("/", "_").replace("\\", "_").replace(".", "p").replace("+", "plus")
        text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)
        if not text or text[0].isdigit():
            text = "t_" + text
        return text

    def _number_tag(self, value: float) -> str:
        text = f"{float(value):g}"
        return text.replace(".", "p").replace("-", "m")

    def _infer_contract_value(self, symbol: str) -> float:
        # OKX ETH-USDT-SWAP face value is normally 0.1 ETH per contract. The raw
        # historical trades file stores size as contract count, so using 0.1 makes
        # buy_notional/sell_notional close to quote-currency turnover.
        if symbol.upper() == "ETH-USDT-SWAP":
            return 0.1
        return 1.0


if __name__ == "__main__":
    # Small manual smoke test:
    #   python src/data_feed/okx_trade_bar_loader.py
    loader = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="1m")
    sample = loader.fetch_data_by_date_range("2022-01-01", "2022-01-01", chunksize=300_000)
    print(sample.head())
    print(sample.tail())
