#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vectorized causal feature construction for the clean-sheet RL dataset.

These functions deliberately produce observable state descriptors rather than
hand-coded trade signals.  Every row is a feature of the current or historical
market path; no future outcome enters the state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


_EPS = 1e-12


def _detach_ambiguous_named_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose index names cannot collide with column labels.

    Some public ``src.data_feed`` loaders intentionally return an observable
    timestamp both as the index and as a retained column (for example
    ``OKXRangeBarLoader`` uses ``set_index("end_ts", drop=False)``). Pandas
    treats operations such as ``sort_values(["end_ts", ...])`` as ambiguous
    when the same name is both an index level and a column label.  The event
    feature builders operate on explicit timestamp columns before constructing
    their own availability index, so retaining the incoming named index adds no
    information.  Drop only the *index labels* here; never drop the timestamp
    column itself.
    """

    out = frame.copy()
    index_names = {name for name in out.index.names if name is not None}
    if index_names.intersection(out.columns):
        out = out.reset_index(drop=True)
    return out


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = _detach_ambiguous_named_index(frame)
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0.0, np.nan)


def _prepare_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"bar frame missing columns: {missing}")
    out = _numeric(frame, required)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    valid = (
        (out[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out["low"] <= out[["open", "close"]].min(axis=1))
    )
    return out.loc[valid]



def resample_ohlcv_from_1m_bars(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    daily_offset: pd.Timedelta | str = "8h",
) -> pd.DataFrame:
    """Causally aggregate left-labeled 1m OHLCV bars into fixed OHLCV bars.

    The returned index is the *bar start*.  Availability is still enforced later
    by :func:`align_left_labeled_bars`, so a 5m bucket starting at 15:10 is not
    visible until 15:15.  ``daily_offset`` matches the project convention where
    OKX UTC daily candles are stored as naive local timestamps (UTC+8 => 08:00).

    R00 uses the locally prebuilt official 1m K-line cache as the single K-line
    base and derives every higher timeframe from it.  This avoids independent
    HTF cache freshness drift while keeping K-lines distinct from tick-derived
    trade bars.  The helper never rebuilds or mutates ``src.data_feed`` data.
    """

    bars = _prepare_bar_frame(frame)
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    mapping = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h", "1D": "1D"}
    if timeframe not in mapping:
        raise ValueError(f"unsupported fixed timeframe for 1m OHLCV resample: {timeframe}")

    rule = mapping[timeframe]
    daily_shift = pd.Timedelta(daily_offset) if timeframe == "1D" else pd.Timedelta(0)

    # Pandas 3.x ignores ``offset`` for calendar-like daily frequencies and emits
    # a RuntimeWarning.  Shift-resample-unshift is explicit and cross-version:
    # a local +08:00 trading day becomes a midnight bucket on the shifted axis,
    # then the aggregated bar-start timestamp is shifted back to +08:00.
    resample_bars = bars
    if daily_shift != pd.Timedelta(0):
        resample_bars = bars.copy()
        resample_bars.index = resample_bars.index - daily_shift

    grouped = resample_bars.resample(
        rule=rule, label="left", closed="left", origin="start_day"
    )
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(min_count=1),
        "__source_1m_rows": grouped["close"].count(),
    })

    expected_rows = int(pd.Timedelta(rule) / pd.Timedelta(minutes=1))
    complete = out["__source_1m_rows"] >= expected_rows
    out = out.loc[complete, ["open", "high", "low", "close", "volume"]]
    if daily_shift != pd.Timedelta(0) and not out.empty:
        out.index = out.index + daily_shift
    return out.replace([np.inf, -np.inf], np.nan)



def build_fixed_bar_features(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Build scale-aware OHLCV descriptors on a fixed-timeframe bar axis."""

    bars = _prepare_bar_frame(frame)
    if bars.empty:
        return pd.DataFrame(index=bars.index)
    close = bars["close"].astype(float)
    log_close = np.log(close)
    ret1 = log_close.diff()
    open_ = bars["open"].astype(float)
    span = (bars["high"] - bars["low"]).clip(lower=0.0)

    out = pd.DataFrame(index=bars.index)
    for lag in (1, 3, 6, 12):
        out[f"{prefix}__logret_{lag}"] = log_close - log_close.shift(lag)
    out[f"{prefix}__range_frac"] = span / open_.replace(0.0, np.nan)
    out[f"{prefix}__body_frac"] = (bars["close"] - bars["open"]) / open_.replace(0.0, np.nan)
    out[f"{prefix}__upper_wick_frac"] = (
        bars["high"] - bars[["open", "close"]].max(axis=1)
    ) / open_.replace(0.0, np.nan)
    out[f"{prefix}__lower_wick_frac"] = (
        bars[["open", "close"]].min(axis=1) - bars["low"]
    ) / open_.replace(0.0, np.nan)
    out[f"{prefix}__close_location"] = (bars["close"] - bars["low"]) / span.replace(0.0, np.nan)

    for window in (12, 48):
        min_periods = max(3, window // 3)
        out[f"{prefix}__rv_{window}"] = ret1.rolling(window, min_periods=min_periods).std(ddof=0)
        rolling_low = bars["low"].rolling(window, min_periods=min_periods).min()
        rolling_high = bars["high"].rolling(window, min_periods=min_periods).max()
        out[f"{prefix}__position_{window}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)

    history_volume = bars["volume"].shift(1)
    volume_median = history_volume.rolling(48, min_periods=12).median()
    out[f"{prefix}__log_volume"] = np.log1p(bars["volume"].clip(lower=0.0))
    out[f"{prefix}__volume_vs_hist_median"] = _safe_ratio(bars["volume"], volume_median)
    return out.replace([np.inf, -np.inf], np.nan)


def _prepare_trade_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "open", "high", "low", "close", "notional", "delta_notional",
        "trades_count", "buy_notional", "sell_notional", "large_buy_notional",
        "large_sell_notional", "large_delta_notional", "large_trades_count",
        "max_trade_notional", "vwap",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trade-bar frame missing columns: {missing}")
    out = _numeric(frame, required)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def build_trade_bar_features(
    frame: pd.DataFrame,
    *,
    prefix: str,
    windows: Iterable[pd.Timedelta | str],
) -> pd.DataFrame:
    """Build rolling flow summaries on 1m or 5s trade-derived bars."""

    bars = _prepare_trade_bars(frame)
    if bars.empty:
        return pd.DataFrame(index=bars.index)
    notional = bars["notional"].abs().astype(float)
    large_abs = (bars["large_buy_notional"].abs() + bars["large_sell_notional"].abs()).astype(float)
    close = bars["close"].astype(float)
    log_close = np.log(close.where(close > 0))

    out = pd.DataFrame(index=bars.index)
    out[f"{prefix}__flow_ratio_now"] = _safe_ratio(bars["delta_notional"], notional)
    out[f"{prefix}__buy_ratio_now"] = _safe_ratio(bars["buy_notional"], notional)
    out[f"{prefix}__large_flow_ratio_now"] = _safe_ratio(bars["large_delta_notional"], notional)
    out[f"{prefix}__large_share_now"] = _safe_ratio(large_abs, notional)
    out[f"{prefix}__max_trade_share_now"] = _safe_ratio(bars["max_trade_notional"].abs(), notional)
    out[f"{prefix}__vwap_gap_now"] = _safe_ratio(close - bars["vwap"], close)
    out[f"{prefix}__log_notional_now"] = np.log1p(notional)
    out[f"{prefix}__log_trades_now"] = np.log1p(bars["trades_count"].clip(lower=0.0))
    out[f"{prefix}__log_max_trade_now"] = np.log1p(bars["max_trade_notional"].abs())

    for raw_window in windows:
        window = pd.Timedelta(raw_window)
        if window <= pd.Timedelta(0):
            raise ValueError("trade windows must be positive")
        key = _duration_key(window)
        roll_notional = notional.rolling(window, min_periods=1).sum()
        roll_delta = bars["delta_notional"].rolling(window, min_periods=1).sum()
        roll_large_delta = bars["large_delta_notional"].rolling(window, min_periods=1).sum()
        roll_large_abs = large_abs.rolling(window, min_periods=1).sum()
        roll_trades = bars["trades_count"].rolling(window, min_periods=1).sum()
        max_trade = bars["max_trade_notional"].abs().rolling(window, min_periods=1).max()
        first_log_close = log_close.shift(max(1, _approx_rows(window, bars.index)))

        out[f"{prefix}__{key}__flow_ratio"] = _safe_ratio(roll_delta, roll_notional)
        out[f"{prefix}__{key}__large_flow_ratio"] = _safe_ratio(roll_large_delta, roll_notional)
        out[f"{prefix}__{key}__large_share"] = _safe_ratio(roll_large_abs, roll_notional)
        out[f"{prefix}__{key}__log_notional"] = np.log1p(roll_notional.clip(lower=0.0))
        out[f"{prefix}__{key}__log_trades"] = np.log1p(roll_trades.clip(lower=0.0))
        out[f"{prefix}__{key}__log_max_trade"] = np.log1p(max_trade.clip(lower=0.0))
        out[f"{prefix}__{key}__logret"] = log_close - first_log_close
        out[f"{prefix}__{key}__rv"] = log_close.diff().rolling(window, min_periods=2).std(ddof=0)

    return out.replace([np.inf, -np.inf], np.nan)


def _approx_rows(window: pd.Timedelta, index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    positive = diffs[diffs > 0]
    if positive.empty:
        return 1
    seconds = float(positive.median())
    return max(1, int(round(window.total_seconds() / seconds)))


def _duration_key(window: pd.Timedelta) -> str:
    seconds = int(window.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def build_range_event_features(
    range_bars: pd.DataFrame,
    *,
    prefix: str,
    windows: Iterable[pd.Timedelta | str],
) -> pd.DataFrame:
    """Create event-time range-bar activity/flow summaries.

    The index of the returned frame is ``end_ts`` and therefore already means
    availability time.  No unfinished active range bar is observed.
    """

    if range_bars is None or range_bars.empty:
        return pd.DataFrame()
    required = {
        "bar_id", "end_ts", "duration_seconds", "direction", "notional",
        "delta_notional", "large_delta_notional", "trades_count",
        "taker_buy_ratio", "max_trade_notional",
    }
    missing = sorted(required - set(range_bars.columns))
    if missing:
        raise ValueError(f"range bars missing columns: {missing}")
    bars = _numeric(range_bars, required - {"end_ts"})
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], errors="coerce")
    bars = bars.dropna(subset=["end_ts"]).sort_values(["end_ts", "bar_id"], kind="mergesort")
    # RangeBarBuilder closes one bar at a trade and starts the next on a later
    # trade, so duplicate end_ts is unusual.  Aggregate deterministically if it
    # occurs to keep merge_asof semantics explicit.
    if bars["end_ts"].duplicated().any():
        bars = (
            bars.groupby("end_ts", as_index=False, sort=True)
            .agg(
                bar_id=("bar_id", "max"),
                duration_seconds=("duration_seconds", "mean"),
                direction=("direction", "sum"),
                notional=("notional", "sum"),
                delta_notional=("delta_notional", "sum"),
                large_delta_notional=("large_delta_notional", "sum"),
                trades_count=("trades_count", "sum"),
                taker_buy_ratio=("taker_buy_ratio", "mean"),
                max_trade_notional=("max_trade_notional", "max"),
            )
        )
    bars = bars.set_index("end_ts", drop=True).sort_index()
    abs_notional = bars["notional"].abs()

    out = pd.DataFrame(index=bars.index)
    out[f"{prefix}__last_direction"] = bars["direction"].clip(-1, 1)
    out[f"{prefix}__last_duration_log"] = np.log1p(bars["duration_seconds"].clip(lower=0.0))
    out[f"{prefix}__last_flow_ratio"] = _safe_ratio(bars["delta_notional"], abs_notional)
    out[f"{prefix}__last_large_flow_ratio"] = _safe_ratio(bars["large_delta_notional"], abs_notional)
    out[f"{prefix}__last_taker_buy_ratio"] = bars["taker_buy_ratio"]
    out[f"{prefix}__last_log_max_trade"] = np.log1p(bars["max_trade_notional"].abs())

    for raw_window in windows:
        window = pd.Timedelta(raw_window)
        key = _duration_key(window)
        count = bars["bar_id"].rolling(window, min_periods=1).count()
        nsum = abs_notional.rolling(window, min_periods=1).sum()
        dsum = bars["delta_notional"].rolling(window, min_periods=1).sum()
        ldsum = bars["large_delta_notional"].rolling(window, min_periods=1).sum()
        out[f"{prefix}__{key}__activity_per_min"] = count / max(window.total_seconds() / 60.0, _EPS)
        out[f"{prefix}__{key}__net_direction"] = bars["direction"].rolling(window, min_periods=1).mean()
        out[f"{prefix}__{key}__flow_ratio"] = _safe_ratio(dsum, nsum)
        out[f"{prefix}__{key}__large_flow_ratio"] = _safe_ratio(ldsum, nsum)
        out[f"{prefix}__{key}__mean_duration_log"] = np.log1p(
            bars["duration_seconds"].rolling(window, min_periods=1).mean().clip(lower=0.0)
        )
        out[f"{prefix}__{key}__log_notional"] = np.log1p(nsum.clip(lower=0.0))
        out[f"{prefix}__{key}__log_trades"] = np.log1p(
            bars["trades_count"].rolling(window, min_periods=1).sum().clip(lower=0.0)
        )
    return out.replace([np.inf, -np.inf], np.nan)


def summarize_footprint_bars(footprint: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Compress variable-length price buckets into per-range-bar descriptors."""

    if footprint is None or footprint.empty:
        return pd.DataFrame()
    required = {
        "bar_id", "end_ts", "price_bucket", "notional", "delta_notional",
        "buy_notional", "sell_notional", "large_delta_notional", "max_trade_notional",
    }
    missing = sorted(required - set(footprint.columns))
    if missing:
        raise ValueError(f"footprint frame missing columns: {missing}")
    fp = _numeric(footprint, required - {"end_ts"})
    fp["end_ts"] = pd.to_datetime(fp["end_ts"], errors="coerce")
    fp = fp.dropna(subset=["end_ts", "bar_id", "price_bucket"])
    if fp.empty:
        return pd.DataFrame()
    fp["abs_delta"] = fp["delta_notional"].abs()
    fp["abs_notional"] = fp["notional"].abs()
    fp["positive_delta"] = (fp["delta_notional"] > 0).astype(float)
    fp["negative_delta"] = (fp["delta_notional"] < 0).astype(float)

    group = fp.groupby("bar_id", sort=True)
    agg = group.agg(
        end_ts=("end_ts", "max"),
        bucket_count=("price_bucket", "count"),
        min_bucket=("price_bucket", "min"),
        max_bucket=("price_bucket", "max"),
        notional_sum=("abs_notional", "sum"),
        delta_sum=("delta_notional", "sum"),
        abs_delta_sum=("abs_delta", "sum"),
        max_abs_delta=("abs_delta", "max"),
        max_bucket_notional=("abs_notional", "max"),
        positive_delta_share=("positive_delta", "mean"),
        negative_delta_share=("negative_delta", "mean"),
        large_delta_sum=("large_delta_notional", "sum"),
        max_trade_notional=("max_trade_notional", "max"),
    )

    total_by_bar = fp.groupby("bar_id")["abs_notional"].transform("sum").replace(0.0, np.nan)
    fp["notional_share_sq"] = (fp["abs_notional"] / total_by_bar).pow(2)
    hhi = fp.groupby("bar_id")["notional_share_sq"].sum()
    poc_idx = fp.groupby("bar_id")["abs_notional"].idxmax()
    delta_idx = fp.groupby("bar_id")["abs_delta"].idxmax()
    poc_bucket = fp.loc[poc_idx].set_index("bar_id")["price_bucket"]
    delta_peak_bucket = fp.loc[delta_idx].set_index("bar_id")["price_bucket"]
    span = (agg["max_bucket"] - agg["min_bucket"]).replace(0.0, np.nan)

    out = pd.DataFrame(index=agg.index)
    out[f"{prefix}__bucket_count_log"] = np.log1p(agg["bucket_count"].clip(lower=0.0))
    out[f"{prefix}__flow_ratio"] = _safe_ratio(agg["delta_sum"], agg["notional_sum"])
    out[f"{prefix}__large_flow_ratio"] = _safe_ratio(agg["large_delta_sum"], agg["notional_sum"])
    out[f"{prefix}__delta_concentration"] = _safe_ratio(agg["max_abs_delta"], agg["abs_delta_sum"])
    out[f"{prefix}__notional_concentration"] = _safe_ratio(agg["max_bucket_notional"], agg["notional_sum"])
    out[f"{prefix}__notional_hhi"] = hhi.reindex(agg.index)
    out[f"{prefix}__positive_delta_bucket_share"] = agg["positive_delta_share"]
    out[f"{prefix}__negative_delta_bucket_share"] = agg["negative_delta_share"]
    out[f"{prefix}__poc_position"] = (poc_bucket.reindex(agg.index) - agg["min_bucket"]) / span
    out[f"{prefix}__delta_peak_position"] = (delta_peak_bucket.reindex(agg.index) - agg["min_bucket"]) / span
    out[f"{prefix}__log_notional"] = np.log1p(agg["notional_sum"].clip(lower=0.0))
    out[f"{prefix}__log_max_trade"] = np.log1p(agg["max_trade_notional"].abs())
    out["end_ts"] = pd.to_datetime(agg["end_ts"])
    out = out.reset_index().sort_values(["end_ts", "bar_id"], kind="mergesort")
    # One footprint summary per bar; if multiple bars share the same end time,
    # retain the highest bar_id as the latest fully closed event.
    out = out.drop_duplicates(subset=["end_ts"], keep="last").set_index("end_ts", drop=True)
    return out.replace([np.inf, -np.inf], np.nan)


def build_footprint_event_features(
    footprint_summary: pd.DataFrame,
    *,
    prefix: str,
    windows: Iterable[pd.Timedelta | str] = ("15min", "60min"),
) -> pd.DataFrame:
    if footprint_summary is None or footprint_summary.empty:
        return pd.DataFrame()
    base = footprint_summary.copy().sort_index()
    numeric_cols = [c for c in base.columns if c != "bar_id"]
    out = pd.DataFrame(index=base.index)
    for column in numeric_cols:
        out[f"{column}__last"] = pd.to_numeric(base[column], errors="coerce")
    for raw_window in windows:
        window = pd.Timedelta(raw_window)
        key = _duration_key(window)
        for column in numeric_cols:
            values = pd.to_numeric(base[column], errors="coerce")
            out[f"{prefix}__{key}__mean__{column.split('__')[-1]}"] = values.rolling(window, min_periods=1).mean()
    return out.replace([np.inf, -np.inf], np.nan)


def add_time_since_event_feature(
    aligned: pd.DataFrame,
    *,
    decision_index: pd.DatetimeIndex,
    source_available_time: pd.Series,
    name: str,
) -> pd.DataFrame:
    out = aligned.copy()
    available = pd.to_datetime(source_available_time, errors="coerce")
    seconds = pd.Series(np.nan, index=decision_index, dtype=float)
    valid = available.notna().to_numpy()
    if valid.any():
        seconds.iloc[np.flatnonzero(valid)] = (
            decision_index[valid].to_numpy(dtype="datetime64[ns]")
            - available.iloc[np.flatnonzero(valid)].to_numpy(dtype="datetime64[ns]")
        ).astype("timedelta64[ns]").astype(np.int64) / 1e9
    out[name] = np.log1p(seconds.clip(lower=0.0))
    return out


def fixed_bar_feature_names(prefix: str) -> list[str]:
    names = [f"{prefix}__logret_{lag}" for lag in (1, 3, 6, 12)]
    names += [
        f"{prefix}__range_frac",
        f"{prefix}__body_frac",
        f"{prefix}__upper_wick_frac",
        f"{prefix}__lower_wick_frac",
        f"{prefix}__close_location",
    ]
    for window in (12, 48):
        names += [f"{prefix}__rv_{window}", f"{prefix}__position_{window}"]
    names += [f"{prefix}__log_volume", f"{prefix}__volume_vs_hist_median"]
    return names


def trade_bar_feature_names(prefix: str, windows: Iterable[pd.Timedelta | str]) -> list[str]:
    names = [
        f"{prefix}__flow_ratio_now",
        f"{prefix}__buy_ratio_now",
        f"{prefix}__large_flow_ratio_now",
        f"{prefix}__large_share_now",
        f"{prefix}__max_trade_share_now",
        f"{prefix}__vwap_gap_now",
        f"{prefix}__log_notional_now",
        f"{prefix}__log_trades_now",
        f"{prefix}__log_max_trade_now",
    ]
    for window in windows:
        key = _duration_key(pd.Timedelta(window))
        names += [
            f"{prefix}__{key}__flow_ratio",
            f"{prefix}__{key}__large_flow_ratio",
            f"{prefix}__{key}__large_share",
            f"{prefix}__{key}__log_notional",
            f"{prefix}__{key}__log_trades",
            f"{prefix}__{key}__log_max_trade",
            f"{prefix}__{key}__logret",
            f"{prefix}__{key}__rv",
        ]
    return names


def range_event_feature_names(prefix: str, windows: Iterable[pd.Timedelta | str]) -> list[str]:
    names = [
        f"{prefix}__last_direction",
        f"{prefix}__last_duration_log",
        f"{prefix}__last_flow_ratio",
        f"{prefix}__last_large_flow_ratio",
        f"{prefix}__last_taker_buy_ratio",
        f"{prefix}__last_log_max_trade",
    ]
    for window in windows:
        key = _duration_key(pd.Timedelta(window))
        names += [
            f"{prefix}__{key}__activity_per_min",
            f"{prefix}__{key}__net_direction",
            f"{prefix}__{key}__flow_ratio",
            f"{prefix}__{key}__large_flow_ratio",
            f"{prefix}__{key}__mean_duration_log",
            f"{prefix}__{key}__log_notional",
            f"{prefix}__{key}__log_trades",
        ]
    names.append(f"{prefix}__time_since_last_event_log_seconds")
    return names


def footprint_summary_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}__bucket_count_log",
        f"{prefix}__flow_ratio",
        f"{prefix}__large_flow_ratio",
        f"{prefix}__delta_concentration",
        f"{prefix}__notional_concentration",
        f"{prefix}__notional_hhi",
        f"{prefix}__positive_delta_bucket_share",
        f"{prefix}__negative_delta_bucket_share",
        f"{prefix}__poc_position",
        f"{prefix}__delta_peak_position",
        f"{prefix}__log_notional",
        f"{prefix}__log_max_trade",
    ]


def footprint_event_feature_names(
    prefix: str,
    windows: Iterable[pd.Timedelta | str] = ("15min", "60min"),
) -> list[str]:
    summary = footprint_summary_feature_names(prefix)
    names = [f"{name}__last" for name in summary]
    for window in windows:
        key = _duration_key(pd.Timedelta(window))
        for name in summary:
            names.append(f"{prefix}__{key}__mean__{name.split('__')[-1]}")
    names.append(f"{prefix}__time_since_last_event_log_seconds")
    return names
