#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stream compact event-local trade bars from OKX official raw trade archives.

This loader exists for research that needs second-level data only around sparse
historical events.  Building a complete multi-year 1s SQLite table is often
unnecessarily large.  Instead, this module reads each required UTC raw-trade day
once, extracts only requested local-time windows, aggregates those slices, and
yields one bounded daily batch.

Causality and timestamp semantics
---------------------------------
- Input ``start_time``/``end_time`` values follow the CoinBacktest project
  timezone convention (UTC+8 naive by default).
- Output ``timestamp`` is the left edge of the bar.
- Output ``available_time`` is ``timestamp + timeframe``.  Research code must
  use ``available_time`` when generating a signal.
- No future fill is performed here.  Missing/no-trade seconds remain absent and
  can be regularized by the caller using only prior prices.

Memory/speed boundaries
-----------------------
- Raw files are streamed in chunks.
- Each required UTC day is read at most once.
- Only slices overlapping event windows are retained in memory.
- Windows crossing a UTC-day boundary are reported as unsupported by this
  daily streaming path; callers should use sufficiently short windows or
  exclude those rare events explicitly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator, Mapping

import numpy as np
import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader

try:
    from src.utils.log import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


BAR_COLUMNS: tuple[str, ...] = (
    "window_id",
    "timestamp",
    "available_time",
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
    "taker_buy_ratio",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_buy_trades_count",
    "large_sell_trades_count",
    "large_trades_count",
    "max_trade_notional",
    "max_trade_size",
    "vwap",
)


@dataclass(frozen=True)
class DailyEventWindowBatch:
    """One UTC day's compact event-window bars and coverage audit."""

    utc_day: date
    bars: pd.DataFrame
    coverage: pd.DataFrame


class OKXEventTradeWindowLoader:
    """Extract sparse event-local second/minute trade bars from raw OKX ZIPs."""

    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        *,
        data_dir: str | os.PathLike[str] | None = None,
        trades_url_template: str = DEFAULT_OKX_TRADES_URL_TEMPLATE,
        contract_value: float | None = None,
        large_trade_notional_threshold: float = 100_000.0,
        align_with_okx_loader_timezone: bool = True,
    ) -> None:
        self.symbol = str(symbol)
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.trades_url_template = str(trades_url_template)
        self.contract_value = self._infer_contract_value(symbol) if contract_value is None else float(contract_value)
        self.large_trade_notional_threshold = float(large_trade_notional_threshold)
        self.align_with_okx_loader_timezone = bool(align_with_okx_loader_timezone)
        self.timezone_offset_hours = self._parse_timezone_offset_hours(TIMEZONE) if self.align_with_okx_loader_timezone else 0
        self.tick_loader = OKXTickLoader(
            symbol=self.symbol,
            data_dir=self.data_dir,
            trades_url_template=self.trades_url_template,
        )

    def iter_daily_window_bars(
        self,
        windows: pd.DataFrame,
        *,
        timeframe: str = "1s",
        chunksize: int = 300_000,
        allow_download_missing: bool = False,
        progress_callback: Callable[[int, int, date], None] | None = None,
    ) -> Iterator[DailyEventWindowBatch]:
        """Yield compact bars for requested event windows, one raw UTC day at a time.

        Required input columns are ``window_id``, ``start_time`` and ``end_time``.
        Time intervals are left-closed/right-open. Duplicate ``window_id`` values
        are rejected so downstream feature tables remain one-to-one.
        """

        prepared, pre_audit = self.prepare_windows(windows)
        if prepared.empty:
            if not pre_audit.empty:
                yield DailyEventWindowBatch(
                    utc_day=date(1970, 1, 1),
                    bars=pd.DataFrame(columns=BAR_COLUMNS),
                    coverage=pre_audit,
                )
            return

        freq, delta = self._parse_timeframe(timeframe)
        groups = list(prepared.groupby("utc_day", sort=True))
        total = len(groups)
        if not pre_audit.empty:
            # Emit unsupported/cross-day rows once, without creating a fake
            # research value.  The sentinel day is explicit in the audit.
            yield DailyEventWindowBatch(
                utc_day=date(1970, 1, 1),
                bars=pd.DataFrame(columns=BAR_COLUMNS),
                coverage=pre_audit,
            )

        for done, (utc_day, day_windows) in enumerate(groups, start=1):
            batch = self._load_one_day(
                utc_day=utc_day,
                day_windows=day_windows,
                freq=freq,
                timeframe_delta=delta,
                chunksize=int(chunksize),
                allow_download_missing=bool(allow_download_missing),
            )
            yield batch
            if progress_callback is not None:
                progress_callback(done, total, utc_day)

    def prepare_windows(self, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"window_id", "start_time", "end_time"}
        missing = sorted(required - set(windows.columns))
        if missing:
            raise ValueError(f"event windows missing columns: {missing}")
        frame = windows.loc[:, ["window_id", "start_time", "end_time"]].copy()
        frame["window_id"] = frame["window_id"].astype(str)
        if frame["window_id"].duplicated().any():
            dup = frame.loc[frame["window_id"].duplicated(), "window_id"].head(5).tolist()
            raise ValueError(f"window_id must be unique; duplicates={dup}")
        frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
        frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")

        bad_time = frame["start_time"].isna() | frame["end_time"].isna() | (frame["end_time"] <= frame["start_time"])
        bad_rows = frame.loc[bad_time].copy()
        bad_rows["status"] = "invalid_window"
        frame = frame.loc[~bad_time].copy()

        offset = pd.Timedelta(hours=self.timezone_offset_hours)
        frame["start_utc"] = frame["start_time"] - offset
        frame["end_utc"] = frame["end_time"] - offset
        start_day = frame["start_utc"].dt.date
        # Right-open interval: subtract one microsecond before choosing the final day.
        end_day = (frame["end_utc"] - pd.Timedelta(microseconds=1)).dt.date
        cross = start_day != end_day
        cross_rows = frame.loc[cross, ["window_id", "start_time", "end_time"]].copy()
        cross_rows["status"] = "cross_utc_day_skipped"
        frame = frame.loc[~cross].copy()
        frame["utc_day"] = frame["start_utc"].dt.date
        frame["start_ms"] = (frame["start_utc"].astype("int64") // 1_000_000).astype("int64")
        frame["end_ms"] = (frame["end_utc"].astype("int64") // 1_000_000).astype("int64")

        audit_parts = []
        for part in (bad_rows, cross_rows):
            if not part.empty:
                audit_parts.append(part.loc[:, ["window_id", "start_time", "end_time", "status"]])
        audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame(
            columns=["window_id", "start_time", "end_time", "status"]
        )
        return frame.sort_values(["utc_day", "start_ms", "window_id"], kind="mergesort"), audit

    def _load_one_day(
        self,
        *,
        utc_day: date,
        day_windows: pd.DataFrame,
        freq: str,
        timeframe_delta: pd.Timedelta,
        chunksize: int,
        allow_download_missing: bool,
    ) -> DailyEventWindowBatch:
        raw_path = self.tick_loader.find_local_trade_file(utc_day, template=self.trades_url_template)
        if raw_path is None and allow_download_missing:
            raw_path = self.tick_loader.download_official_trade_file(utc_day, self.trades_url_template)
        if raw_path is None:
            coverage = day_windows.loc[:, ["window_id", "start_time", "end_time"]].copy()
            coverage["status"] = "missing_raw_day"
            coverage["raw_utc_day"] = str(utc_day)
            coverage["raw_trade_rows"] = 0
            coverage["bar_rows"] = 0
            return DailyEventWindowBatch(utc_day, pd.DataFrame(columns=BAR_COLUMNS), coverage)

        ids = day_windows["window_id"].astype(str).to_numpy()
        starts = day_windows["start_ms"].to_numpy(dtype=np.int64)
        ends = day_windows["end_ms"].to_numpy(dtype=np.int64)
        slices: dict[str, list[pd.DataFrame]] = {window_id: [] for window_id in ids}
        raw_counts = np.zeros(len(ids), dtype=np.int64)

        for chunk in self.tick_loader.read_zip(raw_path, chunksize=max(10_000, chunksize), minimal=True):
            if chunk.empty:
                continue
            ts = chunk["ts_ms"].to_numpy(dtype=np.int64, copy=False)
            for i, window_id in enumerate(ids):
                left = int(np.searchsorted(ts, starts[i], side="left"))
                right = int(np.searchsorted(ts, ends[i], side="left"))
                if right <= left:
                    continue
                part = chunk.iloc[left:right].copy()
                raw_counts[i] += len(part)
                slices[window_id].append(part)

        bar_parts: list[pd.DataFrame] = []
        coverage_rows: list[dict[str, object]] = []
        meta = day_windows.set_index("window_id", drop=False)
        for i, window_id in enumerate(ids):
            parts = slices[window_id]
            if parts:
                trades = pd.concat(parts, ignore_index=True)
                bars = self._aggregate_window(window_id, trades, freq=freq, timeframe_delta=timeframe_delta)
                if not bars.empty:
                    bar_parts.append(bars)
                status = "complete" if raw_counts[i] > 0 else "no_trades_in_window"
                bar_rows = int(len(bars))
            else:
                status = "no_trades_in_window"
                bar_rows = 0
            row = meta.loc[window_id]
            coverage_rows.append(
                {
                    "window_id": window_id,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "status": status,
                    "raw_utc_day": str(utc_day),
                    "raw_trade_rows": int(raw_counts[i]),
                    "bar_rows": bar_rows,
                }
            )

        bars_out = pd.concat(bar_parts, ignore_index=True) if bar_parts else pd.DataFrame(columns=BAR_COLUMNS)
        coverage = pd.DataFrame(coverage_rows)
        return DailyEventWindowBatch(utc_day=utc_day, bars=bars_out, coverage=coverage)

    def _aggregate_window(
        self,
        window_id: str,
        trades: pd.DataFrame,
        *,
        freq: str,
        timeframe_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = trades.copy()
        frame["timestamp"] = pd.to_datetime(frame["ts_ms"], unit="ms", utc=True)
        if self.timezone_offset_hours:
            frame["timestamp"] = frame["timestamp"] + pd.Timedelta(hours=self.timezone_offset_hours)
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "price", "size"]).sort_values("timestamp", kind="mergesort")
        if frame.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        side = frame["side"].astype(str).str.lower()
        buy = side.eq("buy")
        sell = side.eq("sell")
        frame["notional"] = frame["price"] * frame["size"] * self.contract_value
        frame["price_size"] = frame["price"] * frame["size"]
        frame["buy_volume"] = frame["size"].where(buy, 0.0)
        frame["sell_volume"] = frame["size"].where(sell, 0.0)
        frame["buy_notional"] = frame["notional"].where(buy, 0.0)
        frame["sell_notional"] = frame["notional"].where(sell, 0.0)
        frame["buy_trades_count"] = buy.astype("int64")
        frame["sell_trades_count"] = sell.astype("int64")
        large = frame["notional"].abs() >= self.large_trade_notional_threshold
        frame["large_buy_notional"] = frame["notional"].where(large & buy, 0.0)
        frame["large_sell_notional"] = frame["notional"].where(large & sell, 0.0)
        frame["large_buy_trades_count"] = (large & buy).astype("int64")
        frame["large_sell_trades_count"] = (large & sell).astype("int64")
        frame["large_trades_count"] = large.astype("int64")
        frame = frame.set_index("timestamp")
        grouped = frame.resample(freq, label="left", closed="left")
        out = pd.DataFrame(
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
        ).dropna(subset=["open", "high", "low", "close"])
        if out.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        out["delta_volume"] = out["buy_volume"] - out["sell_volume"]
        out["delta_notional"] = out["buy_notional"] - out["sell_notional"]
        out["taker_buy_ratio"] = self._safe_div(out["buy_volume"], out["volume"])
        out["large_delta_notional"] = out["large_buy_notional"] - out["large_sell_notional"]
        out["vwap"] = self._safe_div(out["price_size_sum"], out["volume"])
        out = out.drop(columns=["price_size_sum"])
        out.insert(0, "timestamp", out.index)
        out.insert(0, "window_id", str(window_id))
        out.insert(2, "available_time", out["timestamp"] + timeframe_delta)
        for name in BAR_COLUMNS:
            if name not in out.columns:
                out[name] = 0.0
        return out.loc[:, BAR_COLUMNS].reset_index(drop=True)

    @staticmethod
    def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
        return pd.to_numeric(numerator, errors="coerce") / den

    @staticmethod
    def _parse_timeframe(value: str) -> tuple[str, pd.Timedelta]:
        text = str(value).strip()
        if len(text) < 2 or not text[:-1].isdigit():
            raise ValueError(f"invalid timeframe: {value!r}")
        amount = int(text[:-1])
        unit = text[-1].lower()
        if amount <= 0 or unit not in {"s", "m"}:
            raise ValueError("event-window timeframe supports positive seconds or minutes, e.g. 1s, 5s, 1m")
        if unit == "s":
            return f"{amount}s", pd.Timedelta(seconds=amount)
        return f"{amount}min", pd.Timedelta(minutes=amount)

    @staticmethod
    def _parse_timezone_offset_hours(value: str) -> int:
        text = str(value).strip().upper().replace("UTC", "")
        if not text:
            return 0
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"unsupported fixed timezone offset: {value!r}") from exc

    @staticmethod
    def _infer_contract_value(symbol: str) -> float:
        return 0.1 if str(symbol).upper() == "ETH-USDT-SWAP" else 1.0


__all__ = ["BAR_COLUMNS", "DailyEventWindowBatch", "OKXEventTradeWindowLoader"]
