#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-timeframe features for R03 swing research.

All source bars are left-labelled.  A timeframe feature is indexed by the time
at which the completed bar becomes available, never by the bar's start time.
For example, the 4H bar starting at 00:00 is first visible at 04:00.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FLOW_SUM_COLUMNS = (
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
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_trades_count",
    "large_buy_trades_count",
    "large_sell_trades_count",
)
FLOW_MAX_COLUMNS = ("max_trade_notional", "max_trade_size")
REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")

TIMEFRAME_RULES: dict[str, tuple[str, pd.Timedelta]] = {
    "1m": ("1min", pd.Timedelta(minutes=1)),
    "5m": ("5min", pd.Timedelta(minutes=5)),
    "15m": ("15min", pd.Timedelta(minutes=15)),
    "30m": ("30min", pd.Timedelta(minutes=30)),
    "1h": ("1h", pd.Timedelta(hours=1)),
    "4h": ("4h", pd.Timedelta(hours=4)),
    "1d": ("1D", pd.Timedelta(days=1)),
}

BASE_FEATURE_WINDOWS: dict[str, tuple[int, ...]] = {
    "1m": (5, 15, 60, 240),
    "5m": (3, 6, 12, 48),
    "15m": (2, 4, 8, 32),
    "30m": (2, 4, 8, 24),
    "1h": (3, 6, 12, 24, 72),
    "4h": (2, 3, 6, 12, 30),
    "1d": (2, 3, 5, 10, 20, 50),
}


LONG_CONTEXT_PROFILE = "r03_long_context_v2"
BASE_FEATURE_PROFILE = "r03_multiframe_v1"

LONG_CONTEXT_FEATURE_WINDOWS: dict[str, tuple[int, ...]] = {
    "1m": (5, 15, 60, 240),
    "5m": (3, 6, 12, 48),
    "15m": (2, 4, 8, 32),
    "30m": (2, 4, 8, 24),
    "1h": (12, 24, 72, 168, 336, 720),
    "4h": (6, 12, 30, 90, 180, 360, 720),
    "1d": (5, 10, 20, 50, 90, 180, 365),
}


def feature_windows(timeframe: str, feature_profile: str) -> tuple[int, ...]:
    if feature_profile == BASE_FEATURE_PROFILE:
        return BASE_FEATURE_WINDOWS[timeframe]
    if feature_profile == LONG_CONTEXT_PROFILE:
        return LONG_CONTEXT_FEATURE_WINDOWS[timeframe]
    raise ValueError(f"unsupported swing feature profile: {feature_profile}")

HIGH_CONTEXT_PREFIXES = ("tf1d_", "tf4h_", "tf1h_")
ENTRY_CONTEXT_PREFIXES = ("tf30m_", "tf15m_", "tf5m_", "tf1m_")
CONTEXT_COLUMNS = (
    "ctx_recent_low_4h",
    "ctx_recent_high_4h",
    "ctx_atr_abs_4h",
    "ctx_atr_pct_4h",
    "ctx_atr_pct_15m",
    "ctx_close_1h",
    "ctx_ema20_1h",
    "ctx_close_4h",
    "ctx_ema20_4h",
)


def _ensure_naive_sorted_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index
    out = out.loc[~out.index.isna()]
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.sort_index(kind="stable")


def build_causal_minute_grid(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return a complete 1-minute grid using past-only fill for missing minutes."""
    source = _ensure_naive_sorted_index(bars)
    missing_prices = sorted(set(REQUIRED_PRICE_COLUMNS) - set(source.columns))
    if missing_prices:
        raise RuntimeError(f"public 1m trade-bar loader missing price columns: {missing_prices}")

    grid_index = pd.date_range(pd.Timestamp(start).floor("min"), pd.Timestamp(end).ceil("min"), freq="1min")
    source = source.loc[(source.index >= grid_index[0]) & (source.index <= grid_index[-1])]
    grid = source.reindex(grid_index)
    observed = grid["close"].notna()
    close = pd.to_numeric(grid["close"], errors="coerce").ffill()
    first_valid = close.first_valid_index()
    if first_valid is None:
        return pd.DataFrame(), {"observed_rows": 0, "grid_rows": int(len(grid)), "gap_ratio": 1.0}
    grid = grid.loc[first_valid:].copy()
    close = close.loc[first_valid:]
    for column in REQUIRED_PRICE_COLUMNS:
        numeric = pd.to_numeric(grid[column], errors="coerce") if column in grid.columns else pd.Series(index=grid.index, dtype=float)
        grid[column] = numeric.fillna(close)
    for column in (*FLOW_SUM_COLUMNS, *FLOW_MAX_COLUMNS):
        if column not in grid.columns:
            grid[column] = 0.0
        grid[column] = pd.to_numeric(grid[column], errors="coerce").fillna(0.0)
    grid.index.name = "timestamp"
    observed_rows = int(observed.loc[grid.index].sum())
    stats = {
        "observed_rows": observed_rows,
        "grid_rows": int(len(grid)),
        "gap_rows": int(len(grid) - observed_rows),
        "gap_ratio": float(1.0 - observed_rows / max(1, len(grid))),
    }
    return grid, stats


def aggregate_timeframe(minute_grid: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate the causal 1m grid into left-labelled bars."""
    key = timeframe.lower()
    if key not in TIMEFRAME_RULES:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if key == "1m":
        return minute_grid.copy()
    rule, _ = TIMEFRAME_RULES[key]
    aggregations: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    aggregations.update({column: "sum" for column in FLOW_SUM_COLUMNS})
    aggregations.update({column: "max" for column in FLOW_MAX_COLUMNS})
    bars = minute_grid.resample(rule, label="left", closed="left", origin="start_day").agg(aggregations)
    return bars.dropna(subset=["open", "high", "low", "close"])


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.astype(float)
    den = denominator.astype(float)
    valid_zero = num.notna() & den.notna() & (den.abs() <= 1e-12)
    out = num / den.where(den.abs() > 1e-12)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[valid_zero] = 0.0
    return out


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(2, window // 2)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return _safe_divide(series - mean, std)


def _true_range(bars: pd.DataFrame) -> pd.Series:
    previous_close = bars["close"].shift(1)
    return pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _causal_run_length(condition: pd.Series) -> pd.Series:
    valid = condition.fillna(False).astype(bool)
    groups = valid.ne(valid.shift(fill_value=False)).cumsum()
    run = valid.groupby(groups).cumcount().astype(float) + 1.0
    return run.where(valid, 0.0)


def _bars_since_extreme(series: pd.Series, window: int, *, kind: str) -> pd.Series:
    if kind not in {"high", "low"}:
        raise ValueError(kind)

    def distance(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        position = int(np.argmax(values) if kind == "high" else np.argmin(values))
        return float(len(values) - 1 - position)

    return series.rolling(window, min_periods=window).apply(distance, raw=True)


def _add_long_process_features(
    frame: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    key: str,
    prefix: str,
    windows: tuple[int, ...],
    ema20: pd.Series,
    ema50: pd.Series,
    atr_pct: pd.Series,
) -> None:
    if key not in {"1d", "4h", "1h"}:
        return
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    delta = bars["delta_notional"].astype(float)
    notional = bars["notional"].astype(float)
    short = windows[1]
    medium = windows[-2]
    long = windows[-1]

    long_high = high.rolling(long, min_periods=long).max()
    long_low = low.rolling(long, min_periods=long).min()
    medium_high = high.rolling(medium, min_periods=medium).max()
    medium_low = low.rolling(medium, min_periods=medium).min()
    frame[f"{prefix}drawdown_from_high_{long}"] = _safe_divide(close, long_high) - 1.0
    frame[f"{prefix}rebound_from_low_{long}"] = _safe_divide(close, long_low) - 1.0
    frame[f"{prefix}bars_since_high_{long}"] = _bars_since_extreme(high, long, kind="high") / float(long)
    frame[f"{prefix}bars_since_low_{long}"] = _bars_since_extreme(low, long, kind="low") / float(long)
    frame[f"{prefix}pullback_from_high_{medium}"] = _safe_divide(close, medium_high) - 1.0
    frame[f"{prefix}recovery_from_low_{medium}"] = _safe_divide(close, medium_low) - 1.0
    frame[f"{prefix}range_share_{medium}_{long}"] = _safe_divide(
        medium_high - medium_low, long_high - long_low
    )

    previous_high = high.shift(medium).rolling(medium, min_periods=medium).max()
    previous_low = low.shift(medium).rolling(medium, min_periods=medium).min()
    frame[f"{prefix}structure_high_change_{medium}"] = _safe_divide(medium_high, previous_high) - 1.0
    frame[f"{prefix}structure_low_change_{medium}"] = _safe_divide(medium_low, previous_low) - 1.0

    frame[f"{prefix}trend_age_above_ema50"] = _causal_run_length(close > ema50) / float(long)
    frame[f"{prefix}trend_age_below_ema50"] = _causal_run_length(close < ema50) / float(long)
    frame[f"{prefix}alignment_age_ema20_above_50"] = _causal_run_length(ema20 > ema50) / float(long)
    frame[f"{prefix}alignment_age_ema20_below_50"] = _causal_run_length(ema20 < ema50) / float(long)

    log_return = np.log(close.where(close > 0)).diff()
    rv_short = log_return.rolling(short, min_periods=max(2, short // 2)).std(ddof=0)
    rv_long = log_return.rolling(long, min_periods=long).std(ddof=0)
    frame[f"{prefix}vol_lifecycle_{short}_{long}"] = _safe_divide(rv_short, rv_long) - 1.0
    frame[f"{prefix}atr_lifecycle_{short}_{long}"] = _safe_divide(
        atr_pct.rolling(short, min_periods=max(2, short // 2)).mean(),
        atr_pct.rolling(long, min_periods=long).mean(),
    ) - 1.0
    signed_pressure = np.sign(delta) * np.minimum(_safe_divide(delta.abs(), notional.abs()).fillna(0.0), 1.0)
    frame[f"{prefix}flow_persistence_{medium}"] = signed_pressure.rolling(
        medium, min_periods=max(2, medium // 2)
    ).mean()
    frame[f"{prefix}flow_persistence_{long}"] = signed_pressure.rolling(long, min_periods=long).mean()

    if key in {"1d", "4h"}:
        ema100 = close.ewm(span=100, adjust=False, min_periods=100).mean()
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        frame[f"{prefix}close_rel_ema100"] = _safe_divide(close, ema100) - 1.0
        frame[f"{prefix}close_rel_ema200"] = _safe_divide(close, ema200) - 1.0
        frame[f"{prefix}ema50_100"] = _safe_divide(ema50, ema100) - 1.0
        frame[f"{prefix}ema100_200"] = _safe_divide(ema100, ema200) - 1.0
        frame[f"{prefix}ema200_slope10"] = ema200.pct_change(10, fill_method=None)


def _add_cross_timeframe_features(frame: pd.DataFrame) -> None:
    pairs = (
        ("tf1d_ret_180", "tf4h_ret_180", "cross_d1_180_4h_180_alignment"),
        ("tf1d_ret_365", "tf4h_ret_30", "cross_d1_365_4h_30_alignment"),
        ("tf4h_ret_180", "tf1h_ret_168", "cross_4h_180_1h_168_alignment"),
    )
    for left, right, name in pairs:
        if left in frame.columns and right in frame.columns:
            frame[name] = np.sign(frame[left]) * np.sign(frame[right])
    interactions = (
        ("tf1d_ret_180", "tf4h_pullback_from_high_360", "cross_d1_trend_x_4h_pullback"),
        ("tf4h_ret_180", "tf1h_recovery_from_low_336", "cross_4h_trend_x_1h_recovery"),
        ("tf1d_range_pos_365", "tf4h_range_pos_720", "cross_long_range_position"),
    )
    for left, right, name in interactions:
        if left in frame.columns and right in frame.columns:
            frame[name] = frame[left] * frame[right]


def build_timeframe_features(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    structural_swing_bars_4h: int,
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> pd.DataFrame:
    """Build completed-bar features and shift them to their availability time."""
    key = timeframe.lower()
    _, availability_delta = TIMEFRAME_RULES[key]
    prefix = f"tf{key}_"
    windows = feature_windows(key, feature_profile)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    open_ = bars["open"].astype(float)
    volume = bars["volume"].astype(float)
    notional = bars["notional"].astype(float)
    delta_notional = bars["delta_notional"].astype(float)
    large_delta = bars["large_delta_notional"].astype(float)

    frame = pd.DataFrame(index=bars.index)
    log_return = np.log(close.where(close > 0)).diff()
    frame[f"{prefix}ret_1"] = close.pct_change(1, fill_method=None)
    for window in windows:
        min_periods = max(2, window // 2)
        frame[f"{prefix}ret_{window}"] = close.pct_change(window, fill_method=None)
        frame[f"{prefix}rv_{window}"] = log_return.rolling(window, min_periods=min_periods).std(ddof=0)
        path = log_return.abs().rolling(window, min_periods=min_periods).sum()
        frame[f"{prefix}trend_eff_{window}"] = _safe_divide(log_return.rolling(window, min_periods=min_periods).sum().abs(), path)
        rolling_low = low.rolling(window, min_periods=min_periods).min()
        rolling_high = high.rolling(window, min_periods=min_periods).max()
        frame[f"{prefix}range_pos_{window}"] = _safe_divide(close - rolling_low, rolling_high - rolling_low)
        prior_high = high.shift(1).rolling(window, min_periods=min_periods).max()
        prior_low = low.shift(1).rolling(window, min_periods=min_periods).min()
        frame[f"{prefix}breakout_up_{window}"] = _safe_divide(close, prior_high) - 1.0
        frame[f"{prefix}breakout_down_{window}"] = _safe_divide(prior_low, close) - 1.0
        frame[f"{prefix}volume_z_{window}"] = _rolling_zscore(np.log1p(volume), window)
        frame[f"{prefix}flow_imb_{window}"] = _safe_divide(
            delta_notional.rolling(window, min_periods=min_periods).sum(),
            notional.rolling(window, min_periods=min_periods).sum(),
        )
        frame[f"{prefix}large_flow_imb_{window}"] = _safe_divide(
            large_delta.rolling(window, min_periods=min_periods).sum(),
            notional.rolling(window, min_periods=min_periods).sum(),
        )

    ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    frame[f"{prefix}close_rel_ema10"] = _safe_divide(close, ema10) - 1.0
    frame[f"{prefix}close_rel_ema20"] = _safe_divide(close, ema20) - 1.0
    frame[f"{prefix}close_rel_ema50"] = _safe_divide(close, ema50) - 1.0
    frame[f"{prefix}ema10_20"] = _safe_divide(ema10, ema20) - 1.0
    frame[f"{prefix}ema20_50"] = _safe_divide(ema20, ema50) - 1.0
    frame[f"{prefix}ema20_slope3"] = ema20.pct_change(3, fill_method=None)
    frame[f"{prefix}ema50_slope3"] = ema50.pct_change(3, fill_method=None)

    true_range = _true_range(bars)
    atr_abs = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    atr_pct = _safe_divide(atr_abs, close)
    frame[f"{prefix}atr_pct_14"] = atr_pct
    frame[f"{prefix}body_pct"] = _safe_divide(close - open_, open_)
    candle_range = (high - low).where((high - low).abs() > 1e-12)
    frame[f"{prefix}lower_wick_ratio"] = _safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, candle_range)
    frame[f"{prefix}upper_wick_ratio"] = _safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), candle_range)
    frame[f"{prefix}taker_buy_ratio"] = _safe_divide(bars["buy_notional"], notional)
    frame[f"{prefix}large_trade_share"] = _safe_divide(
        bars["large_buy_notional"] + bars["large_sell_notional"], notional
    )

    if key == "4h":
        frame["ctx_recent_low_4h"] = low.rolling(structural_swing_bars_4h, min_periods=structural_swing_bars_4h).min()
        frame["ctx_recent_high_4h"] = high.rolling(structural_swing_bars_4h, min_periods=structural_swing_bars_4h).max()
        frame["ctx_atr_abs_4h"] = atr_abs
        frame["ctx_atr_pct_4h"] = atr_pct
        frame["ctx_close_4h"] = close
        frame["ctx_ema20_4h"] = ema20
    elif key == "15m":
        frame["ctx_atr_pct_15m"] = atr_pct
    elif key == "1h":
        frame["ctx_close_1h"] = close
        frame["ctx_ema20_1h"] = ema20

    if feature_profile == LONG_CONTEXT_PROFILE:
        # Defragment after the wide rolling-stat block before appending process features.
        frame = frame.copy()
        _add_long_process_features(
            frame,
            bars,
            key=key,
            prefix=prefix,
            windows=windows,
            ema20=ema20,
            ema50=ema50,
            atr_pct=atr_pct,
        )

    frame.index = pd.DatetimeIndex(frame.index) + availability_delta
    frame.index.name = "available_time"
    return frame.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class FeatureBundle:
    frame: pd.DataFrame
    high_feature_columns: tuple[str, ...]
    full_feature_columns: tuple[str, ...]
    context_columns: tuple[str, ...]


def build_multitimeframe_feature_bundle(
    minute_grid: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    *,
    structural_swing_bars_4h: int,
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> FeatureBundle:
    """Align every completed timeframe onto the 15m decision axis."""
    aligned_parts: list[pd.DataFrame] = []
    for timeframe in ("1d", "4h", "1h", "30m", "15m", "5m", "1m"):
        bars = aggregate_timeframe(minute_grid, timeframe)
        features = build_timeframe_features(
            bars,
            timeframe,
            structural_swing_bars_4h=structural_swing_bars_4h,
            feature_profile=feature_profile,
        )
        aligned = features.reindex(decision_index, method="ffill")
        aligned_parts.append(aligned)
    frame = pd.concat(aligned_parts, axis=1)
    if feature_profile == LONG_CONTEXT_PROFILE:
        _add_cross_timeframe_features(frame)
    frame.index.name = "decision_time"
    all_feature_columns = tuple(
        column
        for column in frame.columns
        if column.startswith(HIGH_CONTEXT_PREFIXES + ENTRY_CONTEXT_PREFIXES)
        or (feature_profile == LONG_CONTEXT_PROFILE and column.startswith("cross_"))
    )
    high_columns = tuple(
        column
        for column in all_feature_columns
        if column.startswith(HIGH_CONTEXT_PREFIXES) or column.startswith("cross_")
    )
    context = tuple(column for column in CONTEXT_COLUMNS if column in frame.columns)
    return FeatureBundle(
        frame=frame,
        high_feature_columns=high_columns,
        full_feature_columns=all_feature_columns,
        context_columns=context,
    )


def assert_context_available_time(
    feature_frame: pd.DataFrame,
    decision_index: Iterable[pd.Timestamp],
) -> None:
    """Defensive contract used by tests and pipeline diagnostics."""
    decisions = pd.DatetimeIndex(decision_index)
    if not feature_frame.index.equals(decisions):
        raise AssertionError("feature frame must be indexed by the decision axis")
