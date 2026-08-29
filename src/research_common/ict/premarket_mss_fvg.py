#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal primitives for SOXL premarket sweep -> MSS -> FVG research.

The module is deliberately research-only and has no exchange/network access.
All market data must be supplied by the entrypoint after loading through
``src.data_feed``.

Timing model
------------
* Source 1m timestamps are bar-start timestamps.
* Every 1m observation becomes usable at ``bar_start + 1 minute``.
* Aggregated 2m/5m/15m bars are left-labelled and become usable only at
  ``bar_start + timeframe``.
* Premarket liquidity is frozen at 08:30 America/New_York.
* A short-term pivot is usable only after its right-side confirmation bar has
  closed.
* MSS/FVG signals are generated from a completed execution-timeframe bar.
* The limit order becomes active no earlier than the next 1m bar start, which
  equals the aggregate bar's available time when delay=0.
* 1m OHLC cannot reveal intrabar path. Ambiguous stop/target cases are resolved
  pessimistically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, time as dtime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    DateOffset,
    Easter,
    FR,
    Holiday,
    MO,
    TH,
    nearest_workday,
)

NY_TZ = "America/New_York"
EPS = 1e-12

PREMARKET_START = dtime(4, 0)
PREMARKET_END = dtime(8, 30)
TRADE_START = dtime(8, 30)
TRADE_END = dtime(16, 30)


class USEquityHolidayCalendar(AbstractHolidayCalendar):
    """NYSE-style full-day holidays needed by this research.

    This intentionally models only full-day closures. Early-close sessions are
    still kept because the requested strategy window explicitly runs to 16:30;
    those sessions are surfaced in data-quality/session outputs for review.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=DateOffset(weekday=MO(3)),
            start_date="1998-01-01",
        ),
        Holiday("Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday("Good Friday", month=1, day=1, offset=[Easter(), DateOffset(days=-2)]),
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@dataclass(frozen=True)
class ResearchConfig:
    execution_timeframes: tuple[int, ...] = (1, 2, 5)
    displacement_body_mult: float = 1.50
    displacement_body_window: int = 20
    displacement_min_periods: int = 10
    displacement_close_location: float = 0.75
    mss_pivot_left: int = 1
    mss_pivot_right: int = 1
    premarket_pivot_left: int = 2
    premarket_pivot_right: int = 2
    required_day_coverage: float = 0.995
    round_trip_cost: float = 0.0011
    risk_fraction: float = 0.01
    max_notional_multiple: float = 2.0


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    cost_multiple: float = 1.0
    order_delay_minutes: int = 0


BASE_SCENARIO = ReplayScenario("base", 1.0, 0)


REQUIRED_OHLC = ("open", "high", "low", "close")


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= EPS:
        return np.nan
    return float(num / den)


def _profit_factor(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    wins = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= EPS:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def _max_consecutive_losses(values: Iterable[float]) -> int:
    best = current = 0
    for value in values:
        if np.isfinite(value) and float(value) < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _top_winner_share(values: Iterable[float], count: int = 5) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)[::-1]
    total = float(x.sum())
    return float(x[:count].sum() / total) if total > EPS else np.nan


def ensure_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("1m bars must use a DatetimeIndex")
    missing = [c for c in REQUIRED_OHLC if c not in frame.columns]
    if missing:
        raise KeyError(f"1m bars missing OHLC columns: {missing}")
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.loc[~out.index.duplicated(keep="last")]
    for col in REQUIRED_OHLC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out


def source_naive_to_new_york(
    bars: pd.DataFrame,
    *,
    source_offset_hours: int = 8,
) -> pd.DataFrame:
    """Convert project-local naive bar-start timestamps to New York time."""

    out = ensure_ohlc(bars)
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        fixed = timezone(timedelta(hours=int(source_offset_hours)))
        idx = idx.tz_localize(fixed)
    idx = idx.tz_convert(NY_TZ)
    out.index = idx
    out.index.name = "bar_start_ny"
    return out


def ny_date_bounds_to_source_naive(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    source_offset_hours: int = 8,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map inclusive New York calendar dates to the loader's naive timezone."""

    start_ny = pd.Timestamp(start_date).normalize().tz_localize(NY_TZ)
    end_ny = (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
    fixed = timezone(timedelta(hours=int(source_offset_hours)))
    start_source = start_ny.tz_convert(fixed).tz_localize(None)
    end_source_exclusive = end_ny.tz_convert(fixed).tz_localize(None)
    # Loader date ranges are inclusive, so subtract one minute.
    return start_source, end_source_exclusive - pd.Timedelta(minutes=1)


def us_equity_holidays(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> set[date]:
    cal = USEquityHolidayCalendar()
    values = cal.holidays(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date))
    return {pd.Timestamp(x).date() for x in values}


def eligible_ny_dates(
    bars_ny: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    exclude_equity_holidays: bool = True,
) -> list[date]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be >= start_date")
    holidays = us_equity_holidays(start_date, end_date) if exclude_equity_holidays else set()
    # Generate the full requested calendar rather than deriving dates from the
    # data. Otherwise a completely missing weekday would silently disappear
    # from the coverage audit instead of failing it with zero coverage.
    calendar_days = [pd.Timestamp(x).date() for x in pd.date_range(start, end, freq="D")]
    return [d for d in calendar_days if d.weekday() < 5 and d not in holidays]


def _time_mask(index: pd.DatetimeIndex, start: dtime, end: dtime) -> np.ndarray:
    mins = index.hour * 60 + index.minute
    lo = start.hour * 60 + start.minute
    hi = end.hour * 60 + end.minute
    return (mins >= lo) & (mins < hi)


def slice_ny_day(bars_ny: pd.DataFrame, day: date, start: dtime, end: dtime) -> pd.DataFrame:
    """Fast half-open New York day slice ``[start, end)``.

    Long-history ICT studies call this helper thousands of times.  The old
    implementation rebuilt a full-length boolean date mask on every call,
    turning a multi-year study into repeated O(days * rows) scans.  For a
    monotonic DatetimeIndex we use binary-search boundaries instead; the
    returned rows and causal semantics are identical.
    """
    if bars_ny.empty:
        return bars_ny.copy()
    idx = pd.DatetimeIndex(bars_ny.index)
    if idx.tz is None:
        raise ValueError("slice_ny_day expects a timezone-aware New York index")
    anchor = pd.Timestamp(day).tz_localize(idx.tz)
    start_ts = anchor + pd.Timedelta(hours=start.hour, minutes=start.minute)
    end_ts = anchor + pd.Timedelta(hours=end.hour, minutes=end.minute)
    if idx.is_monotonic_increasing:
        left = int(idx.searchsorted(start_ts, side="left"))
        right = int(idx.searchsorted(end_ts, side="left"))
        return bars_ny.iloc[left:right].copy()
    day_mask = np.asarray([ts.date() == day for ts in idx], dtype=bool)
    time_mask = _time_mask(idx, start, end)
    return bars_ny.loc[day_mask & time_mask].copy()


def session_coverage_row(bars_ny: pd.DataFrame, day: date) -> dict[str, object]:
    pre = slice_ny_day(bars_ny, day, PREMARKET_START, PREMARKET_END)
    trade = slice_ny_day(bars_ny, day, TRADE_START, TRADE_END)
    expected_pre = 270
    expected_trade = 480
    return {
        "ny_date": str(day),
        "weekday": day.strftime("%A"),
        "premarket_rows": int(len(pre)),
        "premarket_expected_rows": expected_pre,
        "premarket_coverage": float(len(pre) / expected_pre),
        "trade_rows": int(len(trade)),
        "trade_expected_rows": expected_trade,
        "trade_coverage": float(len(trade) / expected_trade),
        "full_window_rows": int(len(pre) + len(trade)),
        "full_window_expected_rows": expected_pre + expected_trade,
        "full_window_coverage": float((len(pre) + len(trade)) / (expected_pre + expected_trade)),
        "first_bar": pre.index.min() if len(pre) else pd.NaT,
        "last_bar": trade.index.max() if len(trade) else pd.NaT,
    }


def build_data_quality_table(
    bars_ny: pd.DataFrame,
    days: Sequence[date],
    *,
    required_coverage: float,
) -> pd.DataFrame:
    rows = []
    for day in days:
        row = session_coverage_row(bars_ny, day)
        row["coverage_pass"] = bool(
            float(row["premarket_coverage"]) >= required_coverage
            and float(row["trade_coverage"]) >= required_coverage
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_closed_bars(one_minute: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Aggregate 1m bars into left-labelled closed bars without leakage."""

    tf = int(timeframe_minutes)
    if tf <= 0:
        raise ValueError("timeframe_minutes must be positive")
    frame = ensure_ohlc(one_minute)
    if frame.index.tz is None:
        raise ValueError("aggregate_closed_bars expects timezone-aware index")

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        agg["volume"] = "sum"
    work = frame.copy()
    work["_bar_count"] = 1
    agg["_bar_count"] = "sum"
    out = work.resample(f"{tf}min", origin="start_day", label="left", closed="left").agg(agg)
    out = out.loc[out["_bar_count"] == tf].copy()
    out.rename(columns={"_bar_count": "bar_count"}, inplace=True)
    out["available_time"] = out.index + pd.Timedelta(minutes=tf)
    out["timeframe_minutes"] = tf
    return out


def confirmed_pivots(
    frame: pd.DataFrame,
    *,
    left: int,
    right: int,
) -> pd.DataFrame:
    """Return causally confirmed swing highs/lows from completed bars."""

    if left < 1 or right < 1:
        raise ValueError("pivot left/right must be >= 1")
    if frame.empty:
        return pd.DataFrame()
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    idx = pd.DatetimeIndex(frame.index)
    available = pd.to_datetime(frame["available_time"])
    rows: list[dict[str, object]] = []
    for pos in range(left, len(frame) - right):
        h = highs[pos]
        l = lows[pos]
        left_h = highs[pos - left : pos]
        right_h = highs[pos + 1 : pos + right + 1]
        left_l = lows[pos - left : pos]
        right_l = lows[pos + 1 : pos + right + 1]
        if np.isfinite(h) and np.isfinite(left_h).all() and np.isfinite(right_h).all():
            neighbor_high = float(max(np.max(left_h), np.max(right_h)))
            if h > neighbor_high:
                rows.append(
                    {
                        "pivot_side": "high",
                        "pivot_time": idx[pos],
                        "pivot_price": float(h),
                        "pivot_pos": pos,
                        "confirmation_available_time": pd.Timestamp(available.iloc[pos + right]),
                        "local_prominence_abs": float(h - neighbor_high),
                    }
                )
        if np.isfinite(l) and np.isfinite(left_l).all() and np.isfinite(right_l).all():
            neighbor_low = float(min(np.min(left_l), np.min(right_l)))
            if l < neighbor_low:
                rows.append(
                    {
                        "pivot_side": "low",
                        "pivot_time": idx[pos],
                        "pivot_price": float(l),
                        "pivot_pos": pos,
                        "confirmation_available_time": pd.Timestamp(available.iloc[pos + right]),
                        "local_prominence_abs": float(neighbor_low - l),
                    }
                )
    return pd.DataFrame(rows)


def _day_anchor(day: date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=NY_TZ)


def build_premarket_liquidity_levels(
    bars_ny: pd.DataFrame,
    day: date,
    *,
    pivot_left: int = 2,
    pivot_right: int = 2,
) -> pd.DataFrame:
    """Freeze premarket extremes plus the strongest confirmed 15m internal swings."""

    pm = slice_ny_day(bars_ny, day, PREMARKET_START, PREMARKET_END)
    if pm.empty:
        return pd.DataFrame()
    pm15 = aggregate_closed_bars(pm, 15)
    if pm15.empty:
        return pd.DataFrame()
    available_at = _day_anchor(day, 8, 30)
    pm_range = float(pm["high"].max() - pm["low"].min())

    high_time = pd.Timestamp(pm["high"].idxmax())
    low_time = pd.Timestamp(pm["low"].idxmin())
    rows: list[dict[str, object]] = [
        {
            "ny_date": str(day),
            "liquidity_side": "high",
            "level_type": "premarket_extreme",
            "level_price": float(pm["high"].max()),
            "source_bar_time": high_time,
            "level_available_time": available_at,
            "prominence_abs": np.nan,
            "prominence_frac_of_premarket_range": np.nan,
        },
        {
            "ny_date": str(day),
            "liquidity_side": "low",
            "level_type": "premarket_extreme",
            "level_price": float(pm["low"].min()),
            "source_bar_time": low_time,
            "level_available_time": available_at,
            "prominence_abs": np.nan,
            "prominence_frac_of_premarket_range": np.nan,
        },
    ]

    pivots = confirmed_pivots(pm15, left=pivot_left, right=pivot_right)
    if not pivots.empty:
        pivots = pivots.loc[pd.to_datetime(pivots["confirmation_available_time"]) <= available_at].copy()
        for side in ("high", "low"):
            candidates = pivots.loc[pivots["pivot_side"] == side].copy()
            if candidates.empty:
                continue
            candidates["prominence_frac"] = candidates["local_prominence_abs"] / pm_range if pm_range > EPS else np.nan
            candidates = candidates.sort_values(
                ["local_prominence_abs", "pivot_time"], ascending=[False, True], kind="mergesort"
            )
            best = candidates.iloc[0]
            # If the strongest internal swing is literally the same bar as the
            # absolute extreme, the extreme already represents it; avoid a
            # duplicate level without imposing an arbitrary price-distance filter.
            extreme_time = high_time if side == "high" else low_time
            if pd.Timestamp(best["pivot_time"]) == extreme_time.floor("15min"):
                continue
            rows.append(
                {
                    "ny_date": str(day),
                    "liquidity_side": side,
                    "level_type": "major_15m_swing",
                    "level_price": float(best["pivot_price"]),
                    "source_bar_time": pd.Timestamp(best["pivot_time"]),
                    "level_available_time": pd.Timestamp(best["confirmation_available_time"]),
                    "prominence_abs": float(best["local_prominence_abs"]),
                    "prominence_frac_of_premarket_range": float(best["prominence_frac"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["premarket_high"] = float(pm["high"].max())
    out["premarket_low"] = float(pm["low"].min())
    out["premarket_range"] = pm_range
    out["premarket_range_pct"] = _safe_ratio(pm_range, float(pm["close"].iloc[-1]))
    out["premarket_close"] = float(pm["close"].iloc[-1])
    out["premarket_15m_bars"] = int(len(pm15))
    return out.sort_values(["liquidity_side", "level_type", "level_price"]).reset_index(drop=True)


def build_all_premarket_levels(
    bars_ny: pd.DataFrame,
    days: Sequence[date],
    *,
    pivot_left: int,
    pivot_right: int,
) -> pd.DataFrame:
    parts = [
        build_premarket_liquidity_levels(
            bars_ny,
            day,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
        )
        for day in days
    ]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_sweep_events(
    bars_ny: pd.DataFrame,
    levels: pd.DataFrame,
) -> pd.DataFrame:
    """Detect the first 1m sweep of each frozen premarket liquidity level."""

    if levels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day_text, day_levels in levels.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        trade = slice_ny_day(bars_ny, day, TRADE_START, TRADE_END)
        if trade.empty:
            continue
        highs = pd.to_numeric(trade["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(trade["low"], errors="coerce").to_numpy(float)
        for level in day_levels.to_dict("records"):
            side = str(level["liquidity_side"])
            price = float(level["level_price"])
            mask = highs > price if side == "high" else lows < price
            hit_positions = np.flatnonzero(mask)
            if hit_positions.size == 0:
                continue
            pos = int(hit_positions[0])
            bar_start = pd.Timestamp(trade.index[pos])
            sweep_time = bar_start + pd.Timedelta(minutes=1)
            extreme = float(highs[pos] if side == "high" else lows[pos])
            rows.append(
                {
                    **level,
                    "event_id": f"{day_text}|{side}|{level['level_type']}|{price:.8f}",
                    "trade_side": "SHORT" if side == "high" else "LONG",
                    "sweep_bar_start": bar_start,
                    "sweep_time": sweep_time,
                    "sweep_price_extreme_initial": extreme,
                    "sweep_distance_pct": abs(extreme / price - 1.0) if price > EPS else np.nan,
                    "sweep_minute_of_session": int((bar_start.hour * 60 + bar_start.minute) - (8 * 60 + 30)),
                    "sweep_hour_ny": int(bar_start.hour),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["sweep_time", "liquidity_side", "level_price"]).reset_index(drop=True)


def _add_displacement_features(
    frame: pd.DataFrame,
    *,
    body_window: int,
    min_periods: int,
) -> pd.DataFrame:
    out = frame.copy()
    body = (pd.to_numeric(out["close"], errors="coerce") - pd.to_numeric(out["open"], errors="coerce")).abs()
    rng = pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")
    out["body_abs"] = body
    out["range_abs"] = rng
    # Shift before rolling: the current displacement bar cannot influence its
    # own baseline threshold.
    out["prior_body_median"] = body.shift(1).rolling(body_window, min_periods=min_periods).median()
    out["body_vs_prior_median"] = body / out["prior_body_median"].replace(0.0, np.nan)
    out["close_location"] = (pd.to_numeric(out["close"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")) / rng.replace(0.0, np.nan)
    return out


def _latest_pivot_before(
    pivots: pd.DataFrame,
    *,
    side: str,
    available_by: pd.Timestamp,
) -> Mapping[str, object] | None:
    if pivots.empty:
        return None
    p = pivots.loc[
        (pivots["pivot_side"] == side)
        & (pd.to_datetime(pivots["confirmation_available_time"]) <= available_by)
    ]
    if p.empty:
        return None
    return p.sort_values(["pivot_time", "confirmation_available_time"], kind="mergesort").iloc[-1].to_dict()


def _one_minute_path_between(
    day_1m: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(day_1m.index)
    # Each 1m bar is known at start+1m. Include bars whose completed data are
    # available by end_time and that started at/after start_time-1m.
    return day_1m.loc[(idx + pd.Timedelta(minutes=1) >= start_time) & (idx + pd.Timedelta(minutes=1) <= end_time)]


def build_signal_attempts_for_timeframe(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    timeframe_minutes: int,
    displacement_body_mult: float,
    body_window: int,
    min_periods: int,
    close_location_threshold: float,
    pivot_left: int,
    pivot_right: int,
) -> pd.DataFrame:
    """Build earliest strict MSS+displacement-FVG signal for each sweep event.

    Strict interpretation: the third FVG candle is also the completed bar that
    closes through the frozen short-term MSS pivot. This intentionally avoids a
    loose hindsight definition of a multi-bar displacement leg.
    """

    if sweeps.empty:
        return pd.DataFrame()
    tf = int(timeframe_minutes)
    rows: list[dict[str, object]] = []

    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, PREMARKET_START, TRADE_END)
        if day_1m.empty:
            continue
        exec_frame = aggregate_closed_bars(day_1m, tf)
        exec_frame = _add_displacement_features(exec_frame, body_window=body_window, min_periods=min_periods)
        pivots = confirmed_pivots(exec_frame, left=pivot_left, right=pivot_right)
        if exec_frame.empty or pivots.empty:
            continue

        idx = pd.DatetimeIndex(exec_frame.index)
        available = pd.to_datetime(exec_frame["available_time"])
        highs = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
        closes = pd.to_numeric(exec_frame["close"], errors="coerce").to_numpy(float)
        opens = pd.to_numeric(exec_frame["open"], errors="coerce").to_numpy(float)
        body_ratio = pd.to_numeric(exec_frame["body_vs_prior_median"], errors="coerce").to_numpy(float)
        close_loc = pd.to_numeric(exec_frame["close_location"], errors="coerce").to_numpy(float)

        for sweep in day_sweeps.to_dict("records"):
            sweep_time = pd.Timestamp(sweep["sweep_time"])
            trade_side = str(sweep["trade_side"])
            is_long = trade_side == "LONG"
            ref_side = "high" if is_long else "low"
            reference = _latest_pivot_before(pivots, side=ref_side, available_by=sweep_time)
            if reference is None:
                continue
            ref_price = float(reference["pivot_price"])
            ref_available = pd.Timestamp(reference["confirmation_available_time"])
            target = float(sweep["premarket_high"] if is_long else sweep["premarket_low"])

            # Candidate aggregate bars must have closed after the sweep became
            # known. We start at i>=2 because a 3-candle FVG is required.
            start_positions = np.flatnonzero((available > sweep_time).to_numpy(dtype=bool))
            if start_positions.size == 0:
                continue
            first_pos = max(2, int(start_positions[0]))
            chosen: dict[str, object] | None = None
            for pos in range(first_pos, len(exec_frame)):
                signal_time = pd.Timestamp(available.iloc[pos])
                if signal_time.time() > TRADE_END or signal_time > _day_anchor(day, 16, 30):
                    break
                # All three FVG candles must start after the sweep was known;
                # this makes the reversal sequence strictly ordered.
                if pd.Timestamp(idx[pos - 2]) < sweep_time:
                    continue
                if not np.isfinite(body_ratio[pos]) or body_ratio[pos] < float(displacement_body_mult):
                    continue
                if is_long:
                    mss_break = closes[pos] > ref_price
                    fvg = lows[pos] > highs[pos - 2]
                    close_quality = close_loc[pos] >= float(close_location_threshold)
                    entry_limit = lows[pos]
                    fvg_far_edge = highs[pos - 2]
                else:
                    mss_break = closes[pos] < ref_price
                    fvg = highs[pos] < lows[pos - 2]
                    close_quality = close_loc[pos] <= 1.0 - float(close_location_threshold)
                    entry_limit = highs[pos]
                    fvg_far_edge = lows[pos - 2]
                if not (mss_break and fvg and close_quality):
                    continue

                path_to_signal = _one_minute_path_between(
                    day_1m,
                    start_time=sweep_time,
                    end_time=signal_time,
                )
                if path_to_signal.empty:
                    continue
                stop_extreme = float(path_to_signal["low"].min() if is_long else path_to_signal["high"].max())
                target_already_touched = bool(
                    (path_to_signal["high"] >= target).any() if is_long else (path_to_signal["low"] <= target).any()
                )
                risk_abs = (entry_limit - stop_extreme) if is_long else (stop_extreme - entry_limit)
                reward_abs = (target - entry_limit) if is_long else (entry_limit - target)
                if not np.isfinite(risk_abs) or risk_abs <= EPS or not np.isfinite(reward_abs) or reward_abs <= EPS:
                    continue
                if target_already_touched:
                    continue

                chosen = {
                    **sweep,
                    "execution_tf": f"{tf}m",
                    "execution_tf_minutes": tf,
                    "displacement_body_mult": float(displacement_body_mult),
                    "mss_reference_side": ref_side,
                    "mss_reference_time": pd.Timestamp(reference["pivot_time"]),
                    "mss_reference_price": ref_price,
                    "mss_reference_available_time": ref_available,
                    "signal_bar_start": pd.Timestamp(idx[pos]),
                    "signal_time": signal_time,
                    "signal_open": float(opens[pos]),
                    "signal_high": float(highs[pos]),
                    "signal_low": float(lows[pos]),
                    "signal_close": float(closes[pos]),
                    "signal_body_vs_prior_median": float(body_ratio[pos]),
                    "signal_close_location": float(close_loc[pos]),
                    "fvg_near_edge_entry": float(entry_limit),
                    "fvg_far_edge": float(fvg_far_edge),
                    "fvg_size_abs": abs(float(entry_limit - fvg_far_edge)),
                    "fvg_size_pct": abs(float(entry_limit / fvg_far_edge - 1.0)) if abs(fvg_far_edge) > EPS else np.nan,
                    "stop_price": stop_extreme,
                    "target_price": target,
                    "risk_abs": float(risk_abs),
                    "risk_pct": float(risk_abs / entry_limit),
                    "planned_reward_abs": float(reward_abs),
                    "planned_rr": float(reward_abs / risk_abs),
                    "sweep_to_signal_minutes": float((signal_time - sweep_time).total_seconds() / 60.0),
                    "target_already_touched_before_signal": False,
                    "strict_break_bar_fvg": True,
                }
                break
            if chosen is not None:
                rows.append(chosen)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["attempt_id"] = (
        out["event_id"].astype(str)
        + "|tf="
        + out["execution_tf"].astype(str)
        + "|disp="
        + out["displacement_body_mult"].map(lambda x: f"{float(x):.2f}")
    )
    return out.sort_values(["signal_time", "attempt_id"], kind="mergesort").reset_index(drop=True)


def build_signal_attempts(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    *,
    config: ResearchConfig,
    displacement_body_multipliers: Sequence[float] | None = None,
) -> pd.DataFrame:
    multipliers = tuple(displacement_body_multipliers or (config.displacement_body_mult,))
    parts: list[pd.DataFrame] = []
    for mult in multipliers:
        for tf in config.execution_timeframes:
            part = build_signal_attempts_for_timeframe(
                bars_ny,
                sweeps,
                timeframe_minutes=tf,
                displacement_body_mult=float(mult),
                body_window=config.displacement_body_window,
                min_periods=config.displacement_min_periods,
                close_location_threshold=config.displacement_close_location,
                pivot_left=config.mss_pivot_left,
                pivot_right=config.mss_pivot_right,
            )
            if not part.empty:
                parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _pending_and_trade_path(
    day_1m: pd.DataFrame,
    attempt: Mapping[str, object],
    scenario: ReplayScenario,
    *,
    round_trip_cost: float,
    risk_fraction: float,
    max_notional_multiple: float,
) -> dict[str, object]:
    is_long = str(attempt["trade_side"]) == "LONG"
    signal_time = pd.Timestamp(attempt["signal_time"])
    order_active_time = signal_time + pd.Timedelta(minutes=int(scenario.order_delay_minutes))
    session_end = _day_anchor(pd.Timestamp(attempt["ny_date"]).date(), 16, 30)
    limit_price = float(attempt["fvg_near_edge_entry"])
    stop = float(attempt["stop_price"])
    target = float(attempt["target_price"])
    risk_abs = (limit_price - stop) if is_long else (stop - limit_price)
    risk_pct = _safe_ratio(risk_abs, limit_price)

    result: dict[str, object] = {
        **attempt,
        "scenario": scenario.name,
        "cost_multiple": float(scenario.cost_multiple),
        "order_delay_minutes": int(scenario.order_delay_minutes),
        "order_active_time": order_active_time,
        "order_status": "pending",
        "cancel_reason": "",
        "fill_time": pd.NaT,
        "entry_price": np.nan,
        "exit_time": pd.NaT,
        "exit_price": np.nan,
        "exit_reason": "",
        "gross_return": np.nan,
        "net_return": np.nan,
        "gross_r": np.nan,
        "net_r": np.nan,
        "mfe_r": np.nan,
        "mae_r": np.nan,
        "bars_held_1m": 0,
        "same_bar_entry_stop_ambiguous": False,
        "same_bar_entry_target_ambiguous": False,
        "same_bar_stop_target_ambiguous": False,
        "lifecycle_end_time": session_end,
        "filled": False,
    }
    if not np.isfinite(risk_abs) or risk_abs <= EPS or not np.isfinite(risk_pct) or risk_pct <= 0:
        result.update(order_status="invalid", cancel_reason="invalid_risk", lifecycle_end_time=order_active_time)
        return result
    if order_active_time >= session_end:
        result.update(order_status="cancelled", cancel_reason="order_activated_at_or_after_session_end", lifecycle_end_time=session_end)
        return result

    idx = pd.DatetimeIndex(day_1m.index)
    path = day_1m.loc[(idx >= order_active_time) & (idx < session_end)]
    if path.empty:
        result.update(order_status="cancelled", cancel_reason="no_1m_path_after_order", lifecycle_end_time=session_end)
        return result

    filled = False
    entry_bar_pos = -1
    exit_bar_pos = -1
    entry_time = pd.NaT
    exit_time = pd.NaT
    exit_price = np.nan
    exit_reason = ""
    mfe_r = 0.0
    mae_r = 0.0

    opens = pd.to_numeric(path["open"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(path["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(path["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(path["close"], errors="coerce").to_numpy(float)
    pidx = pd.DatetimeIndex(path.index)

    for pos in range(len(path)):
        bar_start = pd.Timestamp(pidx[pos])
        high = highs[pos]
        low = lows[pos]
        if not filled:
            entry_touch = low <= limit_price if is_long else high >= limit_price
            stop_touch = low <= stop if is_long else high >= stop
            target_touch = high >= target if is_long else low <= target

            if target_touch and entry_touch:
                # User explicitly wants the pending order cancelled if price has
                # already reached the opposite side. With 1m OHLC the sequence is
                # unknown, so cancel rather than assume a profitable fill first.
                result.update(
                    order_status="cancelled",
                    cancel_reason="target_and_entry_same_bar_ambiguous_cancel",
                    same_bar_entry_target_ambiguous=True,
                    lifecycle_end_time=bar_start + pd.Timedelta(minutes=1),
                )
                return result
            if target_touch:
                result.update(
                    order_status="cancelled",
                    cancel_reason="opposite_premarket_extreme_reached_before_fill",
                    lifecycle_end_time=bar_start + pd.Timedelta(minutes=1),
                )
                return result
            if stop_touch and not entry_touch:
                result.update(
                    order_status="cancelled",
                    cancel_reason="sweep_extreme_invalidated_before_fill",
                    lifecycle_end_time=bar_start + pd.Timedelta(minutes=1),
                )
                return result
            if not entry_touch:
                continue

            filled = True
            entry_bar_pos = pos
            entry_time = bar_start
            result["filled"] = True
            result["order_status"] = "filled"
            result["fill_time"] = entry_time
            result["entry_price"] = limit_price

            if stop_touch:
                # Entry/stop order within one minute has unknowable path. Assume
                # the adverse sequence: entry first, then stop.
                result["same_bar_entry_stop_ambiguous"] = True
                exit_bar_pos = pos
                exit_time = bar_start + pd.Timedelta(minutes=1)
                exit_price = stop
                exit_reason = "entry_then_stop_same_bar_conservative"
                mae_r = -1.0
                mfe_r = 0.0
                break
            # Do not credit same-bar favourable excursion because it may have
            # occurred before the limit order actually filled.
            continue

        stop_touch = low <= stop if is_long else high >= stop
        target_touch = high >= target if is_long else low <= target
        if stop_touch and target_touch:
            result["same_bar_stop_target_ambiguous"] = True
            exit_bar_pos = pos
            exit_time = bar_start + pd.Timedelta(minutes=1)
            exit_price = stop
            exit_reason = "stop_first_same_bar_both_conservative"
            mae_r = min(mae_r, -1.0)
            break
        if stop_touch:
            exit_bar_pos = pos
            exit_time = bar_start + pd.Timedelta(minutes=1)
            exit_price = stop
            exit_reason = "structural_sweep_extreme_stop"
            mae_r = min(mae_r, -1.0)
            break
        if target_touch:
            exit_bar_pos = pos
            exit_time = bar_start + pd.Timedelta(minutes=1)
            exit_price = target
            exit_reason = "opposite_premarket_extreme_target"
            mfe_r = max(mfe_r, float((target - limit_price) / risk_abs if is_long else (limit_price - target) / risk_abs))
            break

        favourable = (high - limit_price) / risk_abs if is_long else (limit_price - low) / risk_abs
        adverse = (low - limit_price) / risk_abs if is_long else (limit_price - high) / risk_abs
        mfe_r = max(mfe_r, float(favourable))
        mae_r = min(mae_r, float(adverse))

    if not filled:
        result.update(
            order_status="cancelled",
            cancel_reason="session_end_unfilled",
            lifecycle_end_time=session_end,
        )
        return result

    if not exit_reason:
        # The requested strategy trades only inside 08:30-16:30. Exit at the
        # last completed 1m close, which is the 16:29 bar close / 16:30 boundary.
        last_pos = len(path) - 1
        exit_bar_pos = last_pos
        exit_time = session_end
        exit_price = float(closes[last_pos])
        exit_reason = "session_1630_close"
        favourable = (highs[last_pos] - limit_price) / risk_abs if is_long else (limit_price - lows[last_pos]) / risk_abs
        adverse = (lows[last_pos] - limit_price) / risk_abs if is_long else (limit_price - highs[last_pos]) / risk_abs
        mfe_r = max(mfe_r, float(favourable))
        mae_r = min(mae_r, float(adverse))

    gross = (exit_price / limit_price - 1.0) * (1.0 if is_long else -1.0)
    cost = float(round_trip_cost) * float(scenario.cost_multiple)
    net = gross - cost
    gross_r = gross / risk_pct
    net_r = net / risk_pct
    notional_multiple = min(float(max_notional_multiple), float(risk_fraction) / risk_pct)
    account_return = net * notional_multiple

    result.update(
        exit_time=exit_time,
        exit_price=float(exit_price),
        exit_reason=exit_reason,
        gross_return=float(gross),
        net_return=float(net),
        gross_r=float(gross_r),
        net_r=float(net_r),
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        bars_held_1m=int(max(1, exit_bar_pos - entry_bar_pos + 1)),
        lifecycle_end_time=exit_time,
        notional_multiple=float(notional_multiple),
        account_return=float(account_return),
        round_trip_cost=float(cost),
    )
    return result


def replay_attempts(
    bars_ny: pd.DataFrame,
    attempts: pd.DataFrame,
    *,
    scenario: ReplayScenario,
    round_trip_cost: float,
    risk_fraction: float,
    max_notional_multiple: float,
) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    day_cache: dict[str, pd.DataFrame] = {}
    for attempt in attempts.to_dict("records"):
        day_text = str(attempt["ny_date"])
        if day_text not in day_cache:
            day_cache[day_text] = slice_ny_day(
                bars_ny,
                pd.Timestamp(day_text).date(),
                TRADE_START,
                TRADE_END,
            )
        rows.append(
            _pending_and_trade_path(
                day_cache[day_text],
                attempt,
                scenario,
                round_trip_cost=round_trip_cost,
                risk_fraction=risk_fraction,
                max_notional_multiple=max_notional_multiple,
            )
        )
    return pd.DataFrame(rows)


def filter_liquidity_mode(attempts: pd.DataFrame, mode: str) -> pd.DataFrame:
    if attempts.empty:
        return attempts.copy()
    if mode == "extremes_only":
        return attempts.loc[attempts["level_type"] == "premarket_extreme"].copy()
    if mode == "extremes_plus_major_swing":
        return attempts.loc[attempts["level_type"].isin(["premarket_extreme", "major_15m_swing"])].copy()
    raise ValueError(f"unknown liquidity mode: {mode}")


def enforce_single_lifecycle(replays: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only one pending order/position lifecycle at a time per variant."""

    if replays.empty:
        return replays.copy(), 0
    work = replays.sort_values(["signal_time", "planned_rr", "attempt_id"], ascending=[True, False, True], kind="mergesort")
    kept: list[int] = []
    next_free = pd.Timestamp("1900-01-01", tz=NY_TZ)
    skipped = 0
    for idx, row in work.iterrows():
        signal_time = pd.Timestamp(row["signal_time"])
        if signal_time < next_free:
            skipped += 1
            continue
        kept.append(idx)
        lifecycle_end = pd.Timestamp(row["lifecycle_end_time"])
        next_free = lifecycle_end
    return work.loc[kept].reset_index(drop=True), skipped


def compound_account(
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.sort_values("fill_time", kind="mergesort").copy()
    capital = float(initial_capital)
    rows: list[dict[str, object]] = []
    peak = capital
    for row in work.to_dict("records"):
        account_return = float(row.get("account_return", np.nan))
        if not np.isfinite(account_return):
            continue
        before = capital
        pnl = before * account_return
        capital = before + pnl
        peak = max(peak, capital)
        drawdown = capital / peak - 1.0 if peak > 0 else np.nan
        fee_dollars = before * float(row.get("notional_multiple", 0.0)) * float(row.get("round_trip_cost", 0.0))
        rows.append(
            {
                **row,
                "capital_before": before,
                "pnl": pnl,
                "fee": fee_dollars,
                "capital": capital,
                "drawdown": drawdown,
            }
        )
    return pd.DataFrame(rows)


def summarize_variant(
    lifecycle: pd.DataFrame,
    *,
    skipped_overlap: int = 0,
    initial_capital: float = 10_000.0,
) -> dict[str, object]:
    if lifecycle.empty:
        return {
            "attempts": 0,
            "lifecycle_kept": 0,
            "skipped_overlap": int(skipped_overlap),
            "filled_trades": 0,
        }
    filled = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    cancellations = lifecycle.loc[~lifecycle["filled"].fillna(False).astype(bool)].copy()
    summary: dict[str, object] = {
        "attempts": int(len(lifecycle) + skipped_overlap),
        "lifecycle_kept": int(len(lifecycle)),
        "skipped_overlap": int(skipped_overlap),
        "filled_trades": int(len(filled)),
        "fill_rate": float(len(filled) / len(lifecycle)) if len(lifecycle) else np.nan,
        "cancelled_orders": int(len(cancellations)),
        "target_before_fill_cancel_rate": float(
            cancellations["cancel_reason"].astype(str).str.contains("target|opposite_premarket", regex=True).mean()
        ) if len(cancellations) else 0.0,
    }
    if filled.empty:
        return summary
    x = pd.to_numeric(filled["net_return"], errors="coerce").dropna()
    nr = pd.to_numeric(filled["net_r"], errors="coerce").dropna()
    account = compound_account(filled, initial_capital=initial_capital)
    wins = x[x > 0]
    losses = x[x < 0]
    summary.update(
        {
            "mean_net_return": float(x.mean()) if len(x) else np.nan,
            "median_net_return": float(x.median()) if len(x) else np.nan,
            "mean_net_r": float(nr.mean()) if len(nr) else np.nan,
            "median_net_r": float(nr.median()) if len(nr) else np.nan,
            "win_rate": float((x > 0).mean()) if len(x) else np.nan,
            "profit_factor": _profit_factor(x),
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
            "target_hit_rate": float((filled["exit_reason"] == "opposite_premarket_extreme_target").mean()),
            "stop_hit_rate": float(filled["exit_reason"].astype(str).str.contains("stop").mean()),
            "session_close_rate": float((filled["exit_reason"] == "session_1630_close").mean()),
            "same_bar_ambiguity_rate": float(
                filled[[
                    "same_bar_entry_stop_ambiguous",
                    "same_bar_entry_target_ambiguous",
                    "same_bar_stop_target_ambiguous",
                ]].fillna(False).any(axis=1).mean()
            ),
            "median_planned_rr": float(pd.to_numeric(filled["planned_rr"], errors="coerce").median()),
            "median_sweep_to_signal_minutes": float(pd.to_numeric(filled["sweep_to_signal_minutes"], errors="coerce").median()),
            "median_bars_held_1m": float(pd.to_numeric(filled["bars_held_1m"], errors="coerce").median()),
            "median_mae_r": float(pd.to_numeric(filled["mae_r"], errors="coerce").median()),
            "median_mfe_r": float(pd.to_numeric(filled["mfe_r"], errors="coerce").median()),
            "max_consecutive_losses": _max_consecutive_losses(x),
            "top5_winner_share": _top_winner_share(x, 5),
            "initial_capital": float(initial_capital),
            "final_capital": float(account["capital"].iloc[-1]) if not account.empty else float(initial_capital),
            "account_total_return": float(account["capital"].iloc[-1] / initial_capital - 1.0) if not account.empty else 0.0,
            "account_max_drawdown": float(pd.to_numeric(account["drawdown"], errors="coerce").min()) if not account.empty else 0.0,
        }
    )
    return summary


def build_causal_audit(attempts: pd.DataFrame, lifecycle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if attempts.empty:
        return pd.DataFrame([{"check": "attempts_non_empty", "passed": False, "violations": 0, "detail": "no signal attempts"}])

    def add(check: str, mask: pd.Series, detail: str) -> None:
        bad = int((~mask.fillna(False)).sum())
        rows.append({"check": check, "passed": bad == 0, "violations": bad, "detail": detail})

    signal = pd.to_datetime(attempts["signal_time"])
    sweep = pd.to_datetime(attempts["sweep_time"])
    level_avail = pd.to_datetime(attempts["level_available_time"])
    ref_avail = pd.to_datetime(attempts["mss_reference_available_time"])
    signal_start = pd.to_datetime(attempts["signal_bar_start"])
    tf_delta = pd.to_timedelta(pd.to_numeric(attempts["execution_tf_minutes"]), unit="m")

    add("premarket_level_available_before_sweep", level_avail <= sweep, "08:30-frozen liquidity must be known before the sweep event")
    add("mss_reference_available_before_sweep", ref_avail <= sweep, "short-term pivot is frozen using only causally confirmed structure")
    add("signal_after_sweep", signal > sweep, "MSS/FVG signal must occur strictly after sweep is known")
    add("signal_uses_closed_execution_bar", signal == signal_start + tf_delta, "execution bar is usable only at bar_start + timeframe")
    add("positive_risk", pd.to_numeric(attempts["risk_abs"], errors="coerce") > 0, "stop must be beyond the FVG entry")
    add("positive_reward", pd.to_numeric(attempts["planned_reward_abs"], errors="coerce") > 0, "opposite premarket target must be beyond entry")

    if not lifecycle.empty:
        order_active = pd.to_datetime(lifecycle["order_active_time"])
        signal_life = pd.to_datetime(lifecycle["signal_time"])
        add("order_not_active_before_signal", order_active >= signal_life, "limit order cannot exist before completed MSS/FVG signal")
        filled = lifecycle["filled"].fillna(False).astype(bool)
        if filled.any():
            fill_time = pd.to_datetime(lifecycle.loc[filled, "fill_time"])
            active = pd.to_datetime(lifecycle.loc[filled, "order_active_time"])
            add("fill_not_before_order_active", fill_time >= active, "1m fill search starts only after order activation")
    return pd.DataFrame(rows)


def summarize_by_group(trades: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if trades.empty or group_col not in trades.columns:
        return pd.DataFrame()
    filled = trades.loc[trades["filled"].fillna(False).astype(bool)].copy()
    if filled.empty:
        return pd.DataFrame()
    rows = []
    for key, group in filled.groupby(group_col, dropna=False, sort=True):
        x = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        rows.append(
            {
                group_col: key,
                "trades": int(len(group)),
                "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                "mean_net_return": float(x.mean()) if len(x) else np.nan,
                "median_net_return": float(x.median()) if len(x) else np.nan,
                "profit_factor": _profit_factor(x),
                "target_hit_rate": float((group["exit_reason"] == "opposite_premarket_extreme_target").mean()),
                "stop_hit_rate": float(group["exit_reason"].astype(str).str.contains("stop").mean()),
                "median_planned_rr": float(pd.to_numeric(group["planned_rr"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def add_analysis_dimensions(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return lifecycle.copy()
    out = lifecycle.copy()
    sweep_time = pd.to_datetime(out["sweep_time"])
    out["weekday"] = sweep_time.dt.day_name()
    out["month"] = sweep_time.dt.strftime("%Y-%m")
    mins = sweep_time.dt.hour * 60 + sweep_time.dt.minute
    out["sweep_time_bucket"] = pd.cut(
        mins,
        bins=[8 * 60 + 29, 9 * 60 + 30, 11 * 60 + 30, 13 * 60 + 30, 15 * 60, 16 * 60 + 31],
        labels=["08:30-09:29", "09:30-11:29", "11:30-13:29", "13:30-14:59", "15:00-16:30"],
        include_lowest=True,
        right=False,
    ).astype(str)
    return out


def report_trade_history(account: pd.DataFrame) -> list[dict[str, object]]:
    """Convert account rows to ``src.utils.report.print_full_report`` schema."""

    if account.empty:
        return []
    out: list[dict[str, object]] = []
    for row in account.to_dict("records"):
        out.append(
            {
                "type": str(row["trade_side"]),
                "entry_time": pd.Timestamp(row["fill_time"]).tz_localize(None),
                "exit_time": pd.Timestamp(row["exit_time"]).tz_localize(None),
                "entry": float(row["entry_price"]),
                "exit": float(row["exit_price"]),
                "pnl": float(row["pnl"]),
                "fee": float(row["fee"]),
                "capital": float(row["capital"]),
                "mfe_r": float(row.get("mfe_r", np.nan)),
                "mae_r": float(row.get("mae_r", np.nan)),
                "exit_reason": str(row.get("exit_reason", "")),
                "attempt_id": str(row.get("attempt_id", "")),
            }
        )
    return out


def make_synthetic_ict_day(day: str = "2026-06-02") -> pd.DataFrame:
    """Deterministic synthetic day containing one bullish sweep/MSS/FVG path."""

    start = pd.Timestamp(f"{day} 04:00", tz=NY_TZ)
    idx = pd.date_range(start, periods=750, freq="1min")
    price = np.full(len(idx), 100.0, dtype=float)
    # Premarket: 100-110 range with enough oscillation for pivots.
    for i in range(270):
        price[i] = 105.0 + 4.0 * math.sin(i / 22.0) + 0.003 * i
    # Session starts near 104. Create a W-like low sweep around 09:00.
    price[270:] = 104.0
    session = np.arange(len(idx) - 270)
    price[270:] += 0.25 * np.sin(session / 13.0)
    # Build a short-term high, then sweep below PM low, then displacement.
    anchors = {
        286: 103.0,
        289: 104.2,
        292: 102.8,
        295: 99.2,
        296: 98.8,
        297: 99.4,
        298: 100.1,
        299: 101.0,
        300: 102.2,
        301: 103.6,
        302: 105.0,
        303: 106.2,
        304: 105.5,
        305: 104.8,
        306: 104.2,
        307: 103.8,
    }
    points = sorted(anchors)
    for a, b in zip(points[:-1], points[1:]):
        price[a : b + 1] = np.linspace(anchors[a], anchors[b], b - a + 1)
    price[points[-1] :] = 104.5 + 0.15 * np.sin(np.arange(len(price) - points[-1]) / 9.0)

    open_ = np.r_[price[0], price[:-1]]
    close = price.copy()
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    # Freeze a short-term high before the sweep so MSS has a causally
    # confirmed reference by 08:55. Index 285 is 08:45 NY.
    high[285] = max(high[285], 105.10)
    # Force a clean sweep extreme and bullish 1m FVG around the MSS break.
    low[296] = min(low[296], 98.65)
    # Three-candle bullish FVG: low[303] > high[301].
    high[301] = min(high[301], 103.7)
    low[303] = max(low[303], 104.4)
    high[303] = max(high[303], 106.4)
    close[303] = max(close[303], 106.1)
    open_[303] = min(open_[303], 104.5)
    # After the FVG retrace fills the limit order, touch the frozen opposite
    # premarket extreme so the lifecycle exercises a target exit.
    high[310] = max(high[310], 110.0)
    volume = np.full(len(idx), 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
