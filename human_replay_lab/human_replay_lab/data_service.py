from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader

SUPPORTED_TIMEFRAMES = ("1m", "2m", "5m", "15m", "30m", "1H", "4H", "1D")
TIMEFRAME_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "1D": pd.Timedelta(days=1),
}
RESAMPLE_RULE = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}

NEW_YORK_TZ = "America/New_York"
BEIJING_TZ = "Asia/Shanghai"
REPLAY_START_TIME = time(7, 30)
REPLAY_END_TIME = time(16, 0)
MARKET_OPEN_TIME = time(9, 30)
DATA_SOURCE = "Alpaca SIP / split"
DEFAULT_SYMBOL = "SOXL"
DEFAULT_PREFETCH_DAYS = 50


def _ny_wall_ts(value: Any) -> pd.Timestamp:
    """Timezone-naive timestamp interpreted as America/New_York wall time."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NEW_YORK_TZ).tz_localize(None)
    return ts


def _ny_to_utc(value: Any) -> pd.Timestamp:
    wall = _ny_wall_ts(value)
    return wall.tz_localize(NEW_YORK_TZ).tz_convert("UTC")


def _date_only(value: Any) -> pd.Timestamp:
    return _ny_wall_ts(value).normalize()


@dataclass(frozen=True)
class CandleWindow:
    symbol: str
    timeframe: str
    cursor_time: str
    bars: list[dict[str, Any]]
    source: str


@dataclass
class _DayCache:
    day_key: str
    start: pd.Timestamp
    end: pd.Timestamp
    raw_1m: pd.DataFrame
    frames: dict[str, pd.DataFrame]


class ReplayDataService:
    """Local-only SOXL replay source backed by ``AlpacaStockLoader``.

    The day is prefetched once into memory (including later bars that remain
    inaccessible behind a strict available-time gate). Normal playback then
    slices cached frames instead of re-querying SQLite and re-resampling four
    charts every minute.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.loader = AlpacaStockLoader(
            symbol=DEFAULT_SYMBOL,
            timeframe="1Min",
            feed="sip",
            adjustment="split",
            data_dir=data_dir,
        )
        self.db_path = self.loader.db_path
        self._trading_day_cache: dict[str, bool] = {}
        self._day_caches: dict[str, _DayCache] = {}

    @staticmethod
    def _require_symbol(symbol: str) -> str:
        normalized = str(symbol or DEFAULT_SYMBOL).upper().strip()
        if normalized != DEFAULT_SYMBOL:
            raise ValueError(f"SOXL replay mode only supports {DEFAULT_SYMBOL}; got {normalized}")
        return normalized

    @staticmethod
    def _episode_start_for_date(day: Any) -> pd.Timestamp:
        date = _date_only(day)
        return date + pd.Timedelta(hours=7, minutes=30)

    @staticmethod
    def _episode_end_for_date(day: Any) -> pd.Timestamp:
        date = _date_only(day)
        return date + pd.Timedelta(hours=16)

    def _load_1m(self, start_ny: Any, end_ny: Any) -> pd.DataFrame:
        start = _ny_wall_ts(start_ny)
        end = _ny_wall_ts(end_ny)
        if end < start:
            return pd.DataFrame()
        frame = self.loader.load_local_data_by_date_range(_ny_to_utc(start), _ny_to_utc(end))
        if frame.empty:
            return frame
        out = frame.copy()
        idx = pd.DatetimeIndex(out.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        out.index = idx.tz_convert(NEW_YORK_TZ).tz_localize(None)
        out.index.name = "timestamp_et"
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
        out = out[cols].copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in out.columns:
                out[col] = 0.0 if col == "volume" else pd.NA
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.sort_index()

    def is_trading_day(self, symbol: str, day: Any) -> bool:
        self._require_symbol(symbol)
        date = _date_only(day)
        key = date.strftime("%Y-%m-%d")
        if key in self._trading_day_cache:
            return self._trading_day_cache[key]
        if date.dayofweek >= 5:
            self._trading_day_cache[key] = False
            return False
        # Holidays / missing data are excluded too. We only need one bar in the
        # US extended-hours-to-close window to establish that local data exists.
        frame = self._load_1m(date + pd.Timedelta(hours=4), date + pd.Timedelta(hours=16))
        result = not frame.empty
        self._trading_day_cache[key] = result
        return result

    def random_cursor(self, symbol: str, start: str = "2023-01-01", end: str = "2026-06-30") -> pd.Timestamp:
        self._require_symbol(symbol)
        lo, hi = _date_only(start), _date_only(end)
        if hi < lo:
            raise ValueError("random end date must be >= start date")
        weekdays = list(pd.date_range(lo, hi, freq="B"))
        random.shuffle(weekdays)
        for day in weekdays:
            if self.is_trading_day(symbol, day):
                return self._episode_start_for_date(day)
        raise ValueError("no local SOXL Alpaca trading day found in requested range")

    def cursor_for_date(self, symbol: str, day: Any) -> pd.Timestamp:
        self._require_symbol(symbol)
        date = _date_only(day)
        if date.dayofweek >= 5:
            raise ValueError(f"weekend is disabled for SOXL replay: {date:%Y-%m-%d}")
        if not self.is_trading_day(symbol, date):
            raise ValueError(f"no local SOXL Alpaca data for trading day: {date:%Y-%m-%d}")
        return self._episode_start_for_date(date)

    def validate_cursor(self, symbol: str, cursor: str | pd.Timestamp) -> pd.Timestamp:
        self._require_symbol(symbol)
        ts = _ny_wall_ts(cursor)
        if ts.dayofweek >= 5:
            raise ValueError(f"weekend replay is disabled: {ts:%Y-%m-%d}")
        if not self.is_trading_day(symbol, ts):
            raise ValueError(f"no local SOXL Alpaca data for trading day: {ts:%Y-%m-%d}")
        start, end = self._episode_start_for_date(ts), self._episode_end_for_date(ts)
        if ts < start or ts > end:
            raise ValueError(f"SOXL replay cursor must stay within 07:30-16:00 ET: {ts}")
        return ts

    @staticmethod
    def _resample(raw_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe == "1m":
            return raw_1m
        numeric = raw_1m[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        out = numeric.resample(
            RESAMPLE_RULE[timeframe], label="left", closed="left", origin="start_day"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        return out.dropna(subset=["open", "high", "low", "close"])

    @staticmethod
    def _lookback_for(timeframes: list[str], limit: int) -> pd.Timedelta:
        limit = max(30, min(int(limit), 1500))
        max_delta = max((TIMEFRAME_DELTA[tf] for tf in timeframes), default=pd.Timedelta(minutes=30))
        # Stocks have overnight/weekend gaps. Multiplying by 3 keeps enough
        # calendar history for the visible bar budget without reading years for
        # the normal 30m/15m/2m/1m workspace.
        calculated = max_delta * (limit * 3 + 12)
        return max(pd.Timedelta(days=DEFAULT_PREFETCH_DAYS), calculated)

    def prepare_episode(self, symbol: str, cursor: str | pd.Timestamp, timeframes: list[str] | None = None, limit: int = 700) -> None:
        cursor_ts = self.validate_cursor(symbol, cursor)
        tfs = [tf for tf in (timeframes or ["30m", "15m", "2m", "1m"]) if tf in SUPPORTED_TIMEFRAMES]
        day_key = cursor_ts.strftime("%Y-%m-%d")
        required_start = cursor_ts.normalize() - self._lookback_for(tfs, limit)
        required_end = self._episode_end_for_date(cursor_ts) + pd.Timedelta(minutes=1)
        cache = self._day_caches.get(day_key)
        if cache is None or cache.start > required_start or cache.end < required_end:
            raw = self._load_1m(required_start, required_end)
            cache = _DayCache(day_key, required_start, required_end, raw, {"1m": raw})
            self._day_caches[day_key] = cache
            # Replay is single-user/local. Keep memory bounded to the latest 3 days.
            while len(self._day_caches) > 3:
                oldest = next(iter(self._day_caches))
                if oldest == day_key:
                    break
                self._day_caches.pop(oldest, None)
        for tf in tfs:
            if tf not in cache.frames:
                cache.frames[tf] = self._resample(cache.raw_1m, tf)

    def _frame(self, symbol: str, timeframe: str, cursor: pd.Timestamp, limit: int) -> tuple[pd.DataFrame, str]:
        self.prepare_episode(symbol, cursor, [timeframe], limit)
        cache = self._day_caches[cursor.strftime("%Y-%m-%d")]
        frame = cache.frames[timeframe]
        source = "alpaca_sip_split_1m_cached" if timeframe == "1m" else "cached_resample_from_alpaca_sip_split_1m"
        return frame, source

    @staticmethod
    def _rows_to_bars(frame: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
        delta = TIMEFRAME_DELTA[timeframe]
        bars: list[dict[str, Any]] = []
        for ts, row in frame.iterrows():
            bar_start = _ny_wall_ts(ts)
            bars.append({
                "time": bar_start.strftime("%Y-%m-%d %H:%M:%S"),
                "available_time": (bar_start + delta).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            })
        return bars

    def candles(self, symbol: str, timeframe: str, cursor: str | pd.Timestamp, limit: int = 320) -> CandleWindow:
        self._require_symbol(symbol)
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        limit = max(30, min(int(limit), 1500))
        cursor_ts = self.validate_cursor(symbol, cursor)
        frame, source = self._frame(symbol, timeframe, cursor_ts, limit)
        delta = TIMEFRAME_DELTA[timeframe]
        visible = frame[(frame.index + delta) <= cursor_ts].tail(limit)
        return CandleWindow(symbol, timeframe, cursor_ts.strftime("%Y-%m-%d %H:%M:%S"), self._rows_to_bars(visible, timeframe), source)

    def incremental_bars(
        self,
        symbol: str,
        timeframes: list[str],
        old_cursor: str | pd.Timestamp,
        new_cursor: str | pd.Timestamp,
    ) -> dict[str, list[dict[str, Any]]]:
        """Bars that became visible in ``(old_cursor, new_cursor]`` only."""
        old_ts, new_ts = _ny_wall_ts(old_cursor), self.validate_cursor(symbol, new_cursor)
        cleaned = [tf for tf in dict.fromkeys(timeframes) if tf in SUPPORTED_TIMEFRAMES]
        self.prepare_episode(symbol, new_ts, cleaned, 700)
        cache = self._day_caches[new_ts.strftime("%Y-%m-%d")]
        updates: dict[str, list[dict[str, Any]]] = {}
        for tf in cleaned:
            delta = TIMEFRAME_DELTA[tf]
            frame = cache.frames[tf]
            available = frame.index + delta
            rows = frame[(available > old_ts) & (available <= new_ts)]
            updates[tf] = self._rows_to_bars(rows, tf)
        return updates

    def execution_open(self, symbol: str, cursor: str | pd.Timestamp) -> float:
        cursor_ts = self.validate_cursor(symbol, cursor)
        self.prepare_episode(symbol, cursor_ts, ["1m"], 700)
        raw = self._day_caches[cursor_ts.strftime("%Y-%m-%d")].raw_1m
        exact = raw.loc[raw.index == cursor_ts]
        if exact.empty:
            raise ValueError(f"no SOXL 1m execution bar at {cursor_ts} ET")
        return float(exact.iloc[0]["open"])

    def can_step_to(self, symbol: str, cursor: str | pd.Timestamp) -> bool:
        try:
            ts = _ny_wall_ts(cursor)
            self._require_symbol(symbol)
            if ts.dayofweek >= 5 or not self.is_trading_day(symbol, ts):
                return False
            return self._episode_start_for_date(ts) <= ts <= self._episode_end_for_date(ts)
        except Exception:
            return False

    def clock_info(self, cursor: str | pd.Timestamp) -> dict[str, str]:
        wall = _ny_wall_ts(cursor)
        aware_ny = wall.tz_localize(ZoneInfo(NEW_YORK_TZ))
        beijing = aware_ny.tz_convert(ZoneInfo(BEIJING_TZ))
        local_time = wall.time()
        phase = "PREMARKET" if local_time < MARKET_OPEN_TIME else ("RTH" if local_time < REPLAY_END_TIME else "CLOSE")
        return {
            "timezone": NEW_YORK_TZ,
            "new_york": aware_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "beijing": beijing.strftime("%Y-%m-%d %H:%M:%S CST"),
            "market_phase": phase,
            "market_open_et": "09:30",
            "episode_start_et": "07:30",
            "episode_end_et": "16:00",
            "weekdays_only": "true",
            "source": DATA_SOURCE,
        }
