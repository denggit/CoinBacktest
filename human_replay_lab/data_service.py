from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE

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
DATA_SOURCE = "OKX local 1m"
CHART_CONTEXT_MODE = "all_available_okx_bars"
DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
DEFAULT_PREFETCH_DAYS = 50
CONTINUOUS_FORWARD_PREFETCH_DAYS = 7
CONTINUOUS_24X7_SYMBOLS = frozenset({"ETH-USDT-SWAP", "XAU-USDT-SWAP"})


def _data_source(symbol: str) -> str:
    return f"OKX {str(symbol).upper().strip()} · local 1m"


def _source_offset_hours(text: str) -> int:
    value = str(text or "+8").strip().upper().replace("UTC", "")
    if value.startswith("+") or value.startswith("-"):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return int(value)
    except ValueError:
        return 8


def _source_wall_to_new_york(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    source_tz = timezone(timedelta(hours=_source_offset_hours(OKX_LOADER_TIMEZONE)))
    aware = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=source_tz))
    return aware.tz_convert(ZoneInfo(NEW_YORK_TZ)).tz_localize(None)


def _ny_wall_to_source_naive(value: Any) -> pd.Timestamp:
    wall = _ny_wall_ts(value)
    aware_ny = pd.Timestamp(wall.to_pydatetime().replace(tzinfo=ZoneInfo(NEW_YORK_TZ)))
    source_tz = timezone(timedelta(hours=_source_offset_hours(OKX_LOADER_TIMEZONE)))
    return aware_ny.tz_convert(source_tz).tz_localize(None)


def _ny_wall_ts(value: Any) -> pd.Timestamp:
    """Timezone-naive timestamp interpreted as America/New_York wall time."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NEW_YORK_TZ).tz_localize(None)
    return ts


def _date_only(value: Any) -> pd.Timestamp:
    return _ny_wall_ts(value).normalize()


def _beijing_wall_to_new_york(value: Any) -> pd.Timestamp:
    """Convert a timezone-naive Beijing wall timestamp to internal New York wall time."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(BEIJING_TZ).tz_localize(None)
    aware_bjt = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=ZoneInfo(BEIJING_TZ)))
    return aware_bjt.tz_convert(ZoneInfo(NEW_YORK_TZ)).tz_localize(None)


def _ny_wall_to_beijing(value: Any) -> pd.Timestamp:
    """Convert an internal timezone-naive New York wall timestamp to Beijing time.

    Replay state stays in New York wall time because the current blind-replay
    session profile is anchored there. This helper is presentation-only: it never changes causal ordering,
    Episode boundaries, fills, or persisted event timestamps.
    """
    wall = _ny_wall_ts(value)
    aware_ny = pd.Timestamp(wall.to_pydatetime().replace(tzinfo=ZoneInfo(NEW_YORK_TZ)))
    return aware_ny.tz_convert(ZoneInfo(BEIJING_TZ))


def _beijing_text(value: Any, *, with_zone: bool = False) -> str:
    ts = _ny_wall_to_beijing(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S CST" if with_zone else "%Y-%m-%d %H:%M:%S")


def _source_naive_to_new_york(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert OKX loader's configured source-wall timestamps to NY wall time.

    ``OKXDataLoader`` stores timestamps as timezone-naive values after applying
    the project ``TIMEZONE`` offset (normally UTC+8). Replay semantics are always
    New York wall time, so conversion happens once when a local replay window
    is loaded.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        source_tz = timezone(timedelta(hours=_source_offset_hours(OKX_LOADER_TIMEZONE)))
        idx = idx.tz_localize(source_tz)
    out.index = idx.tz_convert(NEW_YORK_TZ).tz_localize(None)
    out.index.name = "timestamp_et"
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[cols].copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.sort_index()


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
    """Local-only multi-symbol OKX replay source.

    Symbols are discovered from existing ``*_1m`` tables in
    ``data/crypto_history.db``.  Replay never downloads data.  Episode chart
    windows are range-loaded once and cached, which keeps large 24/7 histories
    from being read in full on every server start or replay step.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        probe = OKXDataLoader(symbol=DEFAULT_SYMBOL, timeframe="1m", db_dir=str(self.data_dir))
        self.db_path = probe.db_path
        self._loaders: dict[str, OKXDataLoader] = {}
        self._coverage_cache: dict[str, dict[str, Any]] = {}
        self._symbols_cache: list[str] | None = None
        self._day_caches: dict[tuple[str, str], _DayCache] = {}

    def available_symbols(self) -> list[str]:
        if self._symbols_cache is not None:
            return list(self._symbols_cache)
        symbols = OKXDataLoader.list_local_symbols(str(self.data_dir), timeframe="1m")
        preferred = [DEFAULT_SYMBOL, "ETH-USDT-SWAP", "XAU-USDT-SWAP"]
        ordered: list[str] = []
        for symbol in preferred + symbols:
            normalized = str(symbol).upper().strip()
            if normalized in symbols and normalized not in ordered:
                ordered.append(normalized)
        self._symbols_cache = ordered
        return list(ordered)

    def _require_symbol(self, symbol: str) -> str:
        normalized = str(symbol or DEFAULT_SYMBOL).upper().strip()
        symbols = self.available_symbols()
        if normalized not in symbols:
            available = ", ".join(symbols) if symbols else "(none)"
            raise ValueError(
                f"no local OKX 1m table for {normalized}; available symbols: {available}. "
                "Replay Lab is local-only and will not download missing history."
            )
        return normalized

    @staticmethod
    def is_24x7_symbol(symbol: str) -> bool:
        return str(symbol or "").upper().strip() in CONTINUOUS_24X7_SYMBOLS

    def session_profile(self, symbol: str) -> str:
        return "crypto_24x7_until_bracket_exit" if self.is_24x7_symbol(symbol) else "weekday_0730_1600_et"

    def auto_close_on_bracket_exit(self, symbol: str) -> bool:
        return self.is_24x7_symbol(symbol)

    def _loader(self, symbol: str) -> OKXDataLoader:
        normalized = self._require_symbol(symbol)
        loader = self._loaders.get(normalized)
        if loader is None:
            loader = OKXDataLoader(symbol=normalized, timeframe="1m", db_dir=str(self.data_dir))
            self._loaders[normalized] = loader
        return loader

    @staticmethod
    def _episode_start_for_date(day: Any) -> pd.Timestamp:
        date = _date_only(day)
        return date + pd.Timedelta(hours=7, minutes=30)

    @staticmethod
    def _episode_end_for_date(day: Any) -> pd.Timestamp:
        date = _date_only(day)
        return date + pd.Timedelta(hours=16)

    def _coverage_raw(self, symbol: str) -> dict[str, Any]:
        normalized = self._require_symbol(symbol)
        cached = self._coverage_cache.get(normalized)
        if cached is None:
            cached = self._loader(normalized).get_local_data_coverage()
            self._coverage_cache[normalized] = cached
        return cached

    def coverage(self, symbol: str | None = None) -> dict[str, Any]:
        normalized = self._require_symbol(symbol or DEFAULT_SYMBOL)
        raw = self._coverage_raw(normalized)
        if not raw.get("rows") or not raw.get("start") or not raw.get("end"):
            raise RuntimeError(f"No local OKX {normalized} 1m data found in {self.db_path}")
        start_et = _source_wall_to_new_york(raw["start"])
        end_et = _source_wall_to_new_york(raw["end"])
        if self.is_24x7_symbol(normalized):
            first_day = _ny_wall_to_beijing(start_et).tz_localize(None).normalize()
            last_day = _ny_wall_to_beijing(end_et).tz_localize(None).normalize()
        else:
            first_day = start_et.normalize()
            last_day = end_et.normalize()
            while first_day.dayofweek >= 5:
                first_day += pd.Timedelta(days=1)
            while last_day.dayofweek >= 5:
                last_day -= pd.Timedelta(days=1)
        return {
            "symbol": normalized,
            "source": _data_source(normalized),
            "rows_1m": int(raw["rows"]),
            "available_start_et": start_et.strftime("%Y-%m-%d %H:%M:%S"),
            "available_end_et": end_et.strftime("%Y-%m-%d %H:%M:%S"),
            "available_start_bjt": _beijing_text(start_et),
            "available_end_bjt": _beijing_text(end_et),
            "first_episode_date": first_day.strftime("%Y-%m-%d"),
            "last_episode_date": last_day.strftime("%Y-%m-%d"),
            "episode_weekdays": False if self.is_24x7_symbol(normalized) else True,
            "session_profile": self.session_profile(normalized),
            "chart_context_mode": CHART_CONTEXT_MODE,
            "chart_context_filters_session": False,
            "weekday_off_hours_rows_1m": None,
            "weekend_rows_1m": None,
        }

    def _load_1m(self, symbol: str, start_ny: Any, end_ny: Any) -> pd.DataFrame:
        normalized = self._require_symbol(symbol)
        start = _ny_wall_ts(start_ny)
        end = _ny_wall_ts(end_ny)
        if end < start:
            return pd.DataFrame()
        source_start = _ny_wall_to_source_naive(start)
        source_end = _ny_wall_to_source_naive(end)
        raw = self._loader(normalized).load_local_data_range(source_start, source_end)
        return _source_naive_to_new_york(raw) if not raw.empty else raw

    def _available_day_keys_in_range(self, symbol: str, start: Any, end: Any) -> set[str]:
        normalized = self._require_symbol(symbol)
        lo = _date_only(start)
        hi = _date_only(end)
        if hi < lo:
            return set()
        frame = self._load_1m(normalized, lo, hi + pd.Timedelta(days=1))
        if frame.empty:
            return set()
        idx = pd.DatetimeIndex(frame.index)
        minute = idx.hour * 60 + idx.minute
        mask = (idx.dayofweek < 5) & (minute >= 7 * 60 + 30) & (minute <= 16 * 60)
        return set(idx[mask].strftime("%Y-%m-%d"))

    def is_trading_day(self, symbol: str, day: Any) -> bool:
        normalized = self._require_symbol(symbol)
        date = _date_only(day)
        if date.dayofweek >= 5:
            return False
        start = self._episode_start_for_date(date)
        end = self._episode_end_for_date(date)
        frame = self._load_1m(normalized, start, end)
        return not frame.empty

    def random_cursor(self, symbol: str, start: str = "2026-05-20", end: str = "2026-08-15") -> pd.Timestamp:
        normalized = self._require_symbol(symbol)
        if self.is_24x7_symbol(normalized):
            # UI date bounds are Beijing calendar dates. Pick a blind 30-minute
            # decision point anywhere in the requested 24/7 interval.
            start_bjt = pd.Timestamp(start).normalize()
            end_bjt = pd.Timestamp(end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            lo = _beijing_wall_to_new_york(start_bjt)
            hi = _beijing_wall_to_new_york(end_bjt)
            coverage = self.coverage(normalized)
            cov_lo = _ny_wall_ts(coverage["available_start_et"])
            cov_hi = _ny_wall_ts(coverage["available_end_et"])
            lo = max(lo, cov_lo)
            hi = min(hi, cov_hi)
            lo = lo.ceil("30min")
            hi = hi.floor("30min")
            if hi < lo:
                raise ValueError("requested 24/7 random range has no local OKX data")
            slots = int((hi - lo) / pd.Timedelta(minutes=30)) + 1
            for _ in range(min(128, max(16, slots))):
                candidate = lo + pd.Timedelta(minutes=30 * random.randrange(slots))
                if not self._load_1m(normalized, candidate, candidate + pd.Timedelta(minutes=1)).empty:
                    return candidate
            # Dense 24/7 data normally succeeds above. Fallback to one bounded
            # range scan only when the local database has gaps.
            frame = self._load_1m(normalized, lo, hi + pd.Timedelta(minutes=1))
            if frame.empty:
                raise ValueError("no local 24/7 1m bars in requested random range")
            candidates = pd.DatetimeIndex(frame.index).floor("30min").unique()
            return _ny_wall_ts(random.choice(list(candidates)))

        lo, hi = _date_only(start), _date_only(end)
        if hi < lo:
            raise ValueError("random end date must be >= start date")
        candidates = sorted(self._available_day_keys_in_range(normalized, lo, hi))
        if not candidates:
            coverage = self.coverage(normalized)
            raise ValueError(
                f"no local OKX {normalized} weekday data found in requested range; "
                f"available episode dates={coverage.get('first_episode_date')} -> {coverage.get('last_episode_date')}"
            )
        return self._episode_start_for_date(pd.Timestamp(random.choice(candidates)))

    def _first_available_24x7_at_or_after(self, symbol: str, cursor: pd.Timestamp) -> pd.Timestamp:
        """Find the first local 1m bar at/after ``cursor`` without loading full history.

        Sequential replay uses this only to advance the blind decision cursor across
        genuine local-data gaps. Price/OHLC values from the searched future bars are
        never returned to the caller; only the timestamp of the first available row is
        used as the next Episode start.
        """
        normalized = self._require_symbol(symbol)
        coverage = self.coverage(normalized)
        end = _ny_wall_ts(coverage["available_end_et"])
        probe = max(_ny_wall_ts(cursor).floor("min"), _ny_wall_ts(coverage["available_start_et"]))
        if probe > end:
            raise ValueError(f"no later local OKX {normalized} 1m data for sequential replay")
        chunk = pd.Timedelta(days=7)
        while probe <= end:
            window_end = min(end + pd.Timedelta(minutes=1), probe + chunk)
            frame = self._load_1m(normalized, probe, window_end)
            if not frame.empty:
                idx = pd.DatetimeIndex(frame.index)
                idx = idx[idx >= probe]
                if len(idx):
                    return _ny_wall_ts(idx[0])
            probe = window_end
        raise ValueError(f"no later local OKX {normalized} 1m data for sequential replay")

    def sequential_start_cursor(self, symbol: str, value: Any) -> pd.Timestamp:
        """Resolve the first Episode cursor for chronological/sequential replay.

        * ETH/XAU: the UI supplies Beijing wall time; start at the first available
          local 1m bar at or after that moment.
        * SOXL/session symbols: the UI supplies a calendar date; start at the first
          locally available weekday Episode (07:30 ET) on or after that date.
        """
        normalized = self._require_symbol(symbol)
        if self.is_24x7_symbol(normalized):
            requested = _beijing_wall_to_new_york(pd.Timestamp(value)).floor("min")
            return self._first_available_24x7_at_or_after(normalized, requested)

        day = _date_only(value)
        coverage = self.coverage(normalized)
        last = pd.Timestamp(coverage["last_episode_date"]).normalize()
        while day <= last:
            if day.dayofweek < 5 and self.is_trading_day(normalized, day):
                return self._episode_start_for_date(day)
            day += pd.Timedelta(days=1)
        raise ValueError(f"no local OKX {normalized} replay day at/after {value}")

    def next_sequential_cursor(self, symbol: str, previous_cursor: Any) -> pd.Timestamp:
        """Return the next chronological Episode start after a closed Episode.

        For 24/7 symbols this is the first available 1m bar strictly after the prior
        cursor. For SOXL/session symbols the next Episode begins on the next locally
        available weekday at 07:30 ET so the pre-open setup workflow is preserved.
        """
        normalized = self._require_symbol(symbol)
        previous = _ny_wall_ts(previous_cursor)
        if self.is_24x7_symbol(normalized):
            return self._first_available_24x7_at_or_after(normalized, previous + pd.Timedelta(minutes=1))

        day = previous.normalize() + pd.Timedelta(days=1)
        coverage = self.coverage(normalized)
        last = pd.Timestamp(coverage["last_episode_date"]).normalize()
        while day <= last:
            if day.dayofweek < 5 and self.is_trading_day(normalized, day):
                return self._episode_start_for_date(day)
            day += pd.Timedelta(days=1)
        raise ValueError(f"no later local OKX {normalized} replay day after {previous:%Y-%m-%d}")

    def cursor_for_date(self, symbol: str, day: Any) -> pd.Timestamp:
        normalized = self._require_symbol(symbol)
        if self.is_24x7_symbol(normalized):
            # Date-only callers are kept backward compatible by selecting the
            # first locally available 24/7 bar inside that Beijing calendar day.
            day_bjt = pd.Timestamp(day).normalize()
            start_ny = _beijing_wall_to_new_york(day_bjt)
            end_ny = _beijing_wall_to_new_york(day_bjt + pd.Timedelta(days=1))
            frame = self._load_1m(normalized, start_ny, end_ny)
            if frame.empty:
                raise ValueError(f"no local OKX {normalized} data for Beijing date: {day_bjt:%Y-%m-%d}")
            return _ny_wall_ts(frame.index[0])
        date = _date_only(day)
        if date.dayofweek >= 5:
            raise ValueError(f"weekend is disabled for replay: {date:%Y-%m-%d}")
        if not self.is_trading_day(normalized, date):
            coverage = self.coverage(normalized)
            raise ValueError(
                f"no local OKX {normalized} data for weekday: {date:%Y-%m-%d}; "
                f"available={coverage.get('first_episode_date')} -> {coverage.get('last_episode_date')}"
            )
        return self._episode_start_for_date(date)

    def cursor_for_start(self, symbol: str, value: Any) -> pd.Timestamp:
        normalized = self._require_symbol(symbol)
        if not self.is_24x7_symbol(normalized):
            return self.cursor_for_date(normalized, value)
        # 24/7 specific start is entered/displayed in Beijing time.
        cursor = _beijing_wall_to_new_york(pd.Timestamp(value)).floor("min")
        cursor = self.validate_cursor(normalized, cursor)
        if self._load_1m(normalized, cursor, cursor + pd.Timedelta(minutes=1)).empty:
            raise ValueError(f"no local OKX {normalized} 1m bar at requested Beijing start time")
        return cursor

    def validate_cursor(self, symbol: str, cursor: str | pd.Timestamp) -> pd.Timestamp:
        normalized = self._require_symbol(symbol)
        ts = _ny_wall_ts(cursor)
        if self.is_24x7_symbol(normalized):
            coverage = self.coverage(normalized)
            lo = _ny_wall_ts(coverage["available_start_et"])
            hi = _ny_wall_ts(coverage["available_end_et"]) + pd.Timedelta(minutes=1)
            if ts < lo or ts > hi:
                raise ValueError(f"24/7 replay cursor outside local OKX coverage for {normalized}: {ts}")
            return ts
        if ts.dayofweek >= 5:
            raise ValueError(f"weekend replay is disabled: {ts:%Y-%m-%d}")
        start, end = self._episode_start_for_date(ts), self._episode_end_for_date(ts)
        if ts < start or ts > end:
            raise ValueError(f"replay cursor must stay within 07:30-16:00 ET: {ts}")
        cache = self._day_caches.get((normalized, ts.strftime("%Y-%m-%d")))
        if cache is None and not self.is_trading_day(normalized, ts):
            raise ValueError(f"no local OKX {normalized} data for weekday: {ts:%Y-%m-%d}")
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
        calculated = max_delta * (limit + 24)
        return max(pd.Timedelta(days=DEFAULT_PREFETCH_DAYS), calculated)

    def prepare_episode(
        self,
        symbol: str,
        cursor: str | pd.Timestamp,
        timeframes: list[str] | None = None,
        limit: int = 700,
    ) -> None:
        cursor_ts = self.validate_cursor(symbol, cursor)
        tfs = [tf for tf in (timeframes or ["30m", "15m", "2m", "1m"]) if tf in SUPPORTED_TIMEFRAMES]
        day_key = cursor_ts.strftime("%Y-%m-%d")
        required_start = cursor_ts.normalize() - self._lookback_for(tfs, limit)
        if self.is_24x7_symbol(symbol):
            coverage_end = _ny_wall_ts(self.coverage(symbol)["available_end_et"]) + pd.Timedelta(minutes=1)
            required_end = min(coverage_end, cursor_ts + pd.Timedelta(days=CONTINUOUS_FORWARD_PREFETCH_DAYS))
        else:
            required_end = self._episode_end_for_date(cursor_ts) + pd.Timedelta(minutes=1)
        cache_key = (self._require_symbol(symbol), day_key)
        cache = self._day_caches.get(cache_key)
        if cache is None or cache.start > required_start or cache.end < required_end:
            # IMPORTANT: chart context is intentionally NOT filtered by weekday,
            # premarket, RTH, or the 07:30-16:00 decision window. Episode
            # eligibility and stepping are restricted separately; every local OKX
            # bar in the lookback remains visible, including overnight/off-hours
            # and weekend bars when the source table contains them.
            raw = self._load_1m(symbol, required_start, required_end)
            cache = _DayCache(day_key, required_start, required_end, raw, {"1m": raw})
            self._day_caches[cache_key] = cache
            # Local/single-user tool: retain only a few replay days worth of resampled frames.
            while len(self._day_caches) > 3:
                oldest = next(iter(self._day_caches))
                if oldest == cache_key:
                    break
                self._day_caches.pop(oldest, None)
        for tf in tfs:
            if tf not in cache.frames:
                cache.frames[tf] = self._resample(cache.raw_1m, tf)

    def _frame(self, symbol: str, timeframe: str, cursor: pd.Timestamp, limit: int) -> tuple[pd.DataFrame, str]:
        self.prepare_episode(symbol, cursor, [timeframe], limit)
        cache = self._day_caches[(self._require_symbol(symbol), cursor.strftime("%Y-%m-%d"))]
        frame = cache.frames[timeframe]
        source = "okx_local_1m_memory_cache" if timeframe == "1m" else "cached_resample_from_okx_local_1m"
        return frame, source

    @staticmethod
    def _rows_to_bars(frame: pd.DataFrame, timeframe: str) -> list[dict[str, Any]]:
        delta = TIMEFRAME_DELTA[timeframe]
        bars: list[dict[str, Any]] = []
        for ts, row in frame.iterrows():
            bar_start = _ny_wall_ts(ts)
            bars.append({
                "time": bar_start.strftime("%Y-%m-%d %H:%M:%S"),
                "time_bjt": _beijing_text(bar_start),
                "available_time": (bar_start + delta).strftime("%Y-%m-%d %H:%M:%S"),
                "available_time_bjt": _beijing_text(bar_start + delta),
                "observed_through": (bar_start + delta).strftime("%Y-%m-%d %H:%M:%S"),
                "observed_through_bjt": _beijing_text(bar_start + delta),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
                "is_closed": True,
                "is_partial": False,
            })
        return bars

    @staticmethod
    def _bucket_start(cursor: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        if timeframe == "1D":
            return cursor.normalize()
        return cursor.floor(RESAMPLE_RULE[timeframe])

    def _partial_bar(
        self,
        raw_1m: pd.DataFrame,
        timeframe: str,
        cursor: pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Build the currently-forming HTF candle from causally closed 1m children.

        The cache may contain the entire replay day, but this method never reads a
        child bar whose own close/available time is after ``cursor``. This gives the
        trader the live-looking 15m/30m candle without leaking its future high, low
        or close.
        """
        if timeframe == "1m":
            return None
        delta = TIMEFRAME_DELTA[timeframe]
        start = self._bucket_start(cursor, timeframe)
        end = start + delta
        child_available = raw_1m.index + TIMEFRAME_DELTA["1m"]
        children = raw_1m[
            (raw_1m.index >= start)
            & (raw_1m.index < end)
            & (child_available <= cursor)
        ]
        # At an exact replay cursor the current 1m OPEN is already observable,
        # while that 1m bar's high/low/close/volume are not. Using only this one
        # field makes the HTF candle appear immediately at a new bucket boundary
        # and mirrors a live chart without leaking the rest of the current minute.
        live_open_row = raw_1m[(raw_1m.index == cursor) & (raw_1m.index >= start) & (raw_1m.index < end)]
        live_open = None if live_open_row.empty else float(live_open_row.iloc[0]["open"])
        if children.empty and live_open is None:
            return None

        if children.empty:
            open_ = high = low = close = float(live_open)
            volume = 0.0
        else:
            first = children.iloc[0]
            last = children.iloc[-1]
            open_ = float(first["open"])
            high = float(children["high"].max())
            low = float(children["low"].min())
            close = float(last["close"])
            volume = float(children["volume"].sum())
            if live_open is not None:
                high = max(high, live_open)
                low = min(low, live_open)
                close = live_open

        expected_seen = max(0, min(int(delta / TIMEFRAME_DELTA["1m"]), int((cursor - start) / TIMEFRAME_DELTA["1m"])))
        return {
            "time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "time_bjt": _beijing_text(start),
            "available_time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "available_time_bjt": _beijing_text(end),
            "observed_through": cursor.strftime("%Y-%m-%d %H:%M:%S"),
            "observed_through_bjt": _beijing_text(cursor),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "is_closed": False,
            "is_partial": True,
            "child_bars": int(len(children)),
            "expected_child_bars_seen": int(expected_seen),
            "includes_live_open": live_open is not None,
        }

    def candles(self, symbol: str, timeframe: str, cursor: str | pd.Timestamp, limit: int = 320) -> CandleWindow:
        self._require_symbol(symbol)
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        limit = max(30, min(int(limit), 1500))
        cursor_ts = self.validate_cursor(symbol, cursor)
        frame, source = self._frame(symbol, timeframe, cursor_ts, limit)
        delta = TIMEFRAME_DELTA[timeframe]
        closed = frame[(frame.index + delta) <= cursor_ts]
        bars = self._rows_to_bars(closed.tail(limit), timeframe)
        if timeframe != "1m":
            cache = self._day_caches[(self._require_symbol(symbol), cursor_ts.strftime("%Y-%m-%d"))]
            partial = self._partial_bar(cache.raw_1m, timeframe, cursor_ts)
            if partial is not None:
                bars = [bar for bar in bars if bar["time"] != partial["time"]]
                bars.append(partial)
                bars = bars[-limit:]
        return CandleWindow(
            symbol,
            timeframe,
            cursor_ts.strftime("%Y-%m-%d %H:%M:%S"),
            bars,
            source,
        )

    def incremental_bars(
        self,
        symbol: str,
        timeframes: list[str],
        old_cursor: str | pd.Timestamp,
        new_cursor: str | pd.Timestamp,
    ) -> dict[str, list[dict[str, Any]]]:
        """Small chart upserts for ``(old_cursor, new_cursor]``.

        Fully closed bars that become available are emitted once. For timeframes
        above 1m, the currently-forming candle is also emitted as an upsert keyed
        by its start time. The browser replaces that same candle on every replay
        minute until it transitions to ``is_closed=True``.
        """
        old_ts, new_ts = _ny_wall_ts(old_cursor), self.validate_cursor(symbol, new_cursor)
        cleaned = [tf for tf in dict.fromkeys(timeframes) if tf in SUPPORTED_TIMEFRAMES]
        self.prepare_episode(symbol, new_ts, cleaned, 700)
        cache = self._day_caches[(self._require_symbol(symbol), new_ts.strftime("%Y-%m-%d"))]
        updates: dict[str, list[dict[str, Any]]] = {}
        for tf in cleaned:
            delta = TIMEFRAME_DELTA[tf]
            frame = cache.frames[tf]
            available = frame.index + delta
            rows = frame[(available > old_ts) & (available <= new_ts)]
            bars = self._rows_to_bars(rows, tf)
            if tf != "1m":
                partial = self._partial_bar(cache.raw_1m, tf, new_ts)
                if partial is not None:
                    bars = [bar for bar in bars if bar["time"] != partial["time"]]
                    bars.append(partial)
            updates[tf] = bars
        return updates

    def execution_open(self, symbol: str, cursor: str | pd.Timestamp) -> float:
        cursor_ts = self.validate_cursor(symbol, cursor)
        self.prepare_episode(symbol, cursor_ts, ["1m"], 700)
        raw = self._day_caches[(self._require_symbol(symbol), cursor_ts.strftime("%Y-%m-%d"))].raw_1m
        exact = raw.loc[raw.index == cursor_ts]
        if exact.empty:
            raise ValueError(f"no OKX {self._require_symbol(symbol)} 1m execution bar at {cursor_ts} ET")
        return float(exact.iloc[0]["open"])

    def limit_order_fill(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        start_cursor: str | pd.Timestamp,
        end_cursor: str | pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Return the earliest causal 1m fill in [start_cursor, end_cursor).

        The replay only calls this after advancing to ``end_cursor``. Therefore
        every inspected 1m bar is already closed/available to the user. For a
        resting buy limit we fill at the limit when ``low <= limit``; if a bar
        opens below the limit, the simulated fill receives the better open. A
        sell limit is symmetric.
        """
        self._require_symbol(symbol)
        side = str(side).upper().strip()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("limit order side must be LONG or SHORT")
        price = float(limit_price)
        if not price > 0:
            raise ValueError("limit_price must be > 0")
        start_ts = _ny_wall_ts(start_cursor)
        end_ts = self.validate_cursor(symbol, end_cursor)
        if end_ts <= start_ts:
            return None
        self.prepare_episode(symbol, end_ts, ["1m"], 700)
        raw = self._day_caches[(self._require_symbol(symbol), end_ts.strftime("%Y-%m-%d"))].raw_1m
        rows = raw[(raw.index >= start_ts) & (raw.index < end_ts)]
        for ts, row in rows.iterrows():
            open_ = float(row["open"]); high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            if side == "LONG":
                if open_ <= price:
                    fill_price, reason = open_, "gap_or_open_better_than_limit"
                elif low <= price:
                    fill_price, reason = price, "intrabar_touch_limit"
                else:
                    continue
            else:
                if open_ >= price:
                    fill_price, reason = open_, "gap_or_open_better_than_limit"
                elif high >= price:
                    fill_price, reason = price, "intrabar_touch_limit"
                else:
                    continue
            return {
                "fill_price": float(fill_price),
                "trigger_bar_time": _ny_wall_ts(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "fill_reason": reason,
                "trigger_bar": {"open": open_, "high": high, "low": low, "close": close},
            }
        return None

    def closed_1m_frame(
        self,
        symbol: str,
        start_cursor: str | pd.Timestamp,
        end_cursor: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return a cache-backed OHLC frame for closed 1m bars in [start, end).

        This is the fast-path primitive used by chunked replay.  The returned
        frame may be inspected internally to locate the *earliest* order/fill or
        bracket event, but callers must stop the visible replay cursor at that
        event before returning chart data.  In other words, vectorized scanning
        changes runtime cost, not information available to the trader.
        """
        self._require_symbol(symbol)
        start_ts = _ny_wall_ts(start_cursor)
        end_ts = self.validate_cursor(symbol, end_cursor)
        if end_ts <= start_ts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.prepare_episode(symbol, end_ts, ["1m"], 700)
        raw = self._day_caches[(self._require_symbol(symbol), end_ts.strftime("%Y-%m-%d"))].raw_1m
        return raw[(raw.index >= start_ts) & (raw.index < end_ts)]

    def fast_forward_target(
        self,
        symbol: str,
        cursor: str | pd.Timestamp,
        minutes: int,
    ) -> pd.Timestamp:
        """Clamp a requested replay jump to the symbol's causal session/data boundary."""
        normalized = self._require_symbol(symbol)
        start = self.validate_cursor(normalized, cursor)
        requested = start + pd.Timedelta(minutes=max(0, int(minutes)))
        if self.is_24x7_symbol(normalized):
            coverage = self.coverage(normalized)
            end = _ny_wall_ts(coverage["available_end_et"]) + pd.Timedelta(minutes=1)
        else:
            end = self._episode_end_for_date(start)
        return min(requested, end)

    def closed_1m_bars(
        self,
        symbol: str,
        start_cursor: str | pd.Timestamp,
        end_cursor: str | pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """Return causally closed 1m bars in ``[start_cursor, end_cursor)``.

        ``end_cursor`` is the replay decision cursor, so a row whose start time
        is strictly before it is already closed and safe for order/exit replay.
        This method is intentionally small and cache-backed; it never performs a
        fresh full-table scan during playback.
        """
        self._require_symbol(symbol)
        start_ts = _ny_wall_ts(start_cursor)
        end_ts = self.validate_cursor(symbol, end_cursor)
        if end_ts <= start_ts:
            return []
        self.prepare_episode(symbol, end_ts, ["1m"], 700)
        raw = self._day_caches[(self._require_symbol(symbol), end_ts.strftime("%Y-%m-%d"))].raw_1m
        rows = raw[(raw.index >= start_ts) & (raw.index < end_ts)]
        out: list[dict[str, Any]] = []
        for ts, row in rows.iterrows():
            wall = _ny_wall_ts(ts)
            out.append({
                "time": wall.strftime("%Y-%m-%d %H:%M:%S"),
                "time_bjt": _beijing_text(wall),
                "available_time": (wall + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })
        return out

    def can_step_to(self, symbol: str, cursor: str | pd.Timestamp) -> bool:
        try:
            ts = _ny_wall_ts(cursor)
            normalized = self._require_symbol(symbol)
            if self.is_24x7_symbol(normalized):
                coverage = self.coverage(normalized)
                return _ny_wall_ts(coverage["available_start_et"]) <= ts <= (_ny_wall_ts(coverage["available_end_et"]) + pd.Timedelta(minutes=1))
            if ts.dayofweek >= 5 or not self.is_trading_day(normalized, ts):
                return False
            return self._episode_start_for_date(ts) <= ts <= self._episode_end_for_date(ts)
        except Exception:
            return False

    @staticmethod
    def beijing_display(value: Any, *, with_zone: bool = False) -> str:
        return _beijing_text(value, with_zone=with_zone)

    def clock_info(self, cursor: str | pd.Timestamp, symbol: str | None = None) -> dict[str, str]:
        wall = _ny_wall_ts(cursor)
        aware_ny = wall.tz_localize(ZoneInfo(NEW_YORK_TZ))
        beijing = aware_ny.tz_convert(ZoneInfo(BEIJING_TZ))
        normalized = str(symbol or DEFAULT_SYMBOL).upper().strip()
        if self.is_24x7_symbol(normalized):
            return {
                "timezone": BEIJING_TZ,
                "display_timezone": BEIJING_TZ,
                "internal_session_timezone": NEW_YORK_TZ,
                "beijing": beijing.strftime("%Y-%m-%d %H:%M:%S CST"),
                "beijing_plain": beijing.strftime("%Y-%m-%d %H:%M:%S"),
                "new_york": aware_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "market_phase": "24/7",
                "market_open_bjt": "-",
                "episode_start_bjt": _beijing_text(wall)[11:16],
                "episode_end_bjt": "TP/SL",
                "market_open_et": "-",
                "episode_start_et": wall.strftime("%H:%M"),
                "episode_end_et": "TP/SL",
                "weekdays_only": "false",
                "session_profile": "crypto_24x7_until_bracket_exit",
                "chart_context": "all available OKX bars; 24/7",
                "source": _data_source(normalized),
            }
        local_time = wall.time()
        phase = "PREMARKET" if local_time < MARKET_OPEN_TIME else ("RTH" if local_time < REPLAY_END_TIME else "CLOSE")
        day = wall.normalize()
        start_bjt = _ny_wall_to_beijing(day + pd.Timedelta(hours=7, minutes=30))
        open_bjt = _ny_wall_to_beijing(day + pd.Timedelta(hours=9, minutes=30))
        end_bjt = _ny_wall_to_beijing(day + pd.Timedelta(hours=16))
        return {
            "timezone": BEIJING_TZ,
            "display_timezone": BEIJING_TZ,
            "internal_session_timezone": NEW_YORK_TZ,
            "beijing": beijing.strftime("%Y-%m-%d %H:%M:%S CST"),
            "beijing_plain": beijing.strftime("%Y-%m-%d %H:%M:%S"),
            "new_york": aware_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "market_phase": phase,
            "market_open_bjt": open_bjt.strftime("%H:%M"),
            "episode_start_bjt": start_bjt.strftime("%H:%M"),
            "episode_end_bjt": end_bjt.strftime("%H:%M"),
            "market_open_et": "09:30",
            "episode_start_et": "07:30",
            "episode_end_et": "16:00",
            "weekdays_only": "true",
            "session_profile": "weekday_0730_1600_et",
            "chart_context": "all available OKX bars; no session/weekend filter",
            "source": _data_source(normalized),
        }
