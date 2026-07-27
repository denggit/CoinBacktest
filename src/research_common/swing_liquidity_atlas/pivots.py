#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-timeframe swing-low universe construction.

The universe intentionally starts with the widest reasonable definition: every
order-1 local low on 15m/30m/1H/4H/1D.  Higher pivot orders are recorded as
later causal confirmations of the same level, not as duplicate levels and not
as entry filters.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import AtlasConfig

EPS = 1e-12
_REQUIRED = ("open", "high", "low", "close")
_SUM_COLUMNS = (
    "volume",
    "notional",
    "trades_count",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)


def normalize_primary_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("primary bars are empty")
    if (
        bool(frame.attrs.get("_swing_liquidity_atlas_normalized", False))
        and isinstance(frame.index, pd.DatetimeIndex)
        and not frame.index.has_duplicates
        and all(name in frame.columns for name in _REQUIRED)
    ):
        return frame
    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"primary bars missing required columns: {missing}")
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" not in out.columns:
            raise ValueError("primary bars require DatetimeIndex or timestamp column")
        out.index = pd.to_datetime(out.pop("timestamp"), errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.loc[~out.index.isna()].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    for name in _REQUIRED:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=list(_REQUIRED))
    if len(out) < 3:
        raise ValueError("primary bars contain fewer than three valid rows")
    out.attrs["_swing_liquidity_atlas_normalized"] = True
    return out


def _rule(minutes: int) -> str:
    return "1D" if int(minutes) == 1440 else f"{int(minutes)}min"


def aggregate_timeframe(primary: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
    """Aggregate left-labelled project-time bars and drop incomplete HTF bars."""

    bars = normalize_primary_bars(primary)
    working = bars.copy()
    working["_source_bar_count"] = 1
    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "_source_bar_count": "sum",
    }
    for name in _SUM_COLUMNS:
        if name in working.columns:
            aggregation[name] = "sum"
    htf = working.resample(_rule(minutes), label="left", closed="left", origin="start_day").agg(aggregation)
    htf = htf.dropna(subset=list(_REQUIRED))
    # Partial HTF bars at either boundary can manufacture false pivots.  ETH
    # trade bars are expected every minute, so admit only fully observed bins.
    htf = htf.loc[pd.to_numeric(htf["_source_bar_count"], errors="coerce") >= int(minutes)].copy()
    source_delta = pd.Timedelta(minutes=1)
    source_available_end = bars.index[-1] + source_delta
    htf_delta = pd.Timedelta(minutes=int(minutes))
    htf = htf.loc[(htf.index + htf_delta) <= source_available_end].copy()
    htf["bar_end_time"] = htf.index + htf_delta
    return htf


def _pivot_mask(values: np.ndarray, order: int) -> np.ndarray:
    n = len(values)
    order = int(order)
    mask = np.ones(n, dtype=bool)
    if n < order * 2 + 1:
        return np.zeros(n, dtype=bool)
    mask[:order] = False
    mask[n - order :] = False
    for lag in range(1, order + 1):
        left = np.empty(n, dtype=float)
        left[:lag] = np.nan
        left[lag:] = values[:-lag]
        right = np.empty(n, dtype=float)
        right[-lag:] = np.nan
        right[:-lag] = values[lag:]
        mask &= values < left
        mask &= values <= right
    mask &= np.isfinite(values)
    return mask


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (np.abs(den) > EPS))


def _past_min(values: pd.Series, window: int) -> np.ndarray:
    return values.shift(1).rolling(int(window), min_periods=1).min().to_numpy(dtype=float)


def _past_max(values: pd.Series, window: int) -> np.ndarray:
    return values.shift(1).rolling(int(window), min_periods=1).max().to_numpy(dtype=float)


def build_timeframe_levels(
    primary: pd.DataFrame,
    *,
    timeframe: str,
    minutes: int,
    confirmation_orders: Iterable[int],
) -> pd.DataFrame:
    htf = aggregate_timeframe(primary, minutes=int(minutes))
    orders = tuple(sorted(set(int(v) for v in confirmation_orders)))
    max_order = max(orders)
    if len(htf) < max_order * 2 + 1:
        return pd.DataFrame()

    low = htf["low"].to_numpy(dtype=float)
    high = htf["high"].to_numpy(dtype=float)
    open_ = htf["open"].to_numpy(dtype=float)
    close = htf["close"].to_numpy(dtype=float)
    base_mask = _pivot_mask(low, 1)
    positions = np.flatnonzero(base_mask)
    if not len(positions):
        return pd.DataFrame()

    masks = {order: _pivot_mask(low, order) for order in orders}
    delta = pd.Timedelta(minutes=int(minutes))
    level = low[positions]
    bar_range = np.maximum(high[positions] - low[positions], EPS)
    lower_body = np.minimum(open_[positions], close[positions])
    close_location = _safe_ratio(close[positions] - low[positions], bar_range)
    lower_wick_fraction = _safe_ratio(lower_body - low[positions], bar_range)

    rows = pd.DataFrame(
        {
            "source_timeframe": str(timeframe),
            "source_timeframe_min": int(minutes),
            "pivot_pos_htf": positions.astype(np.int64),
            "pivot_time": htf.index[positions],
            "pivot_bar_end_time": htf.index[positions] + delta,
            "level_price": level,
            "initial_available_time": htf.index[positions] + 2 * delta,
            "pivot_range_bp": _safe_ratio(bar_range, close[positions]) * 10_000.0,
            "pivot_close_location": close_location,
            "pivot_lower_wick_fraction": lower_wick_fraction,
        }
    )

    for order in orders:
        qualified = masks[order][positions]
        available = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
        if np.any(qualified):
            available[qualified] = (htf.index[positions[qualified]] + (order + 1) * delta).to_numpy(dtype="datetime64[ns]")
        rows[f"order_{order}_available_time"] = pd.to_datetime(available)
        rows[f"future_eventual_order_{order}_label"] = qualified.astype(np.int8)

    max_eventual = np.ones(len(rows), dtype=np.int16)
    for order in orders:
        max_eventual = np.where(rows[f"future_eventual_order_{order}_label"].to_numpy(dtype=bool), order, max_eventual)
    rows["future_max_eventual_order_label"] = max_eventual

    low_s = htf["low"]
    high_s = htf["high"]
    for window in (3, 8, 20):
        prior_low = _past_min(low_s, window)[positions]
        prior_high = _past_max(high_s, window)[positions]
        rows[f"left_low_gap_{window}_bp"] = _safe_ratio(prior_low - level, level) * 10_000.0
        rows[f"left_high_range_{window}_bp"] = _safe_ratio(prior_high - level, level) * 10_000.0

    # At order-1 availability the next HTF bar is fully closed, therefore this
    # reaction is causal at initial_available_time.
    right_pos = positions + 1
    rows["confirmation_reaction_close_bp"] = _safe_ratio(close[right_pos] - level, level) * 10_000.0
    rows["confirmation_reaction_high_bp"] = _safe_ratio(high[right_pos] - level, level) * 10_000.0

    for name in _SUM_COLUMNS:
        if name not in htf.columns:
            rows[f"pivot_{name}"] = np.nan
            continue
        values = pd.to_numeric(htf[name], errors="coerce")
        rows[f"pivot_{name}"] = values.iloc[positions].to_numpy(dtype=float)
        baseline = values.shift(1).rolling(20, min_periods=5).median().iloc[positions].to_numpy(dtype=float)
        rows[f"pivot_{name}_vs_past20"] = _safe_ratio(values.iloc[positions].to_numpy(dtype=float), baseline)

    if "delta_notional" in htf.columns and "notional" in htf.columns:
        rows["pivot_delta_ratio"] = _safe_ratio(
            pd.to_numeric(htf["delta_notional"], errors="coerce").iloc[positions].to_numpy(dtype=float),
            pd.to_numeric(htf["notional"], errors="coerce").iloc[positions].to_numpy(dtype=float),
        )
    else:
        rows["pivot_delta_ratio"] = np.nan

    return rows.sort_values(["initial_available_time", "pivot_time", "level_price"], kind="mergesort").reset_index(drop=True)


def build_swing_low_universe(primary: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    parts: list[pd.DataFrame] = []
    for timeframe, minutes in cfg.timeframes:
        part = build_timeframe_levels(
            bars,
            timeframe=timeframe,
            minutes=int(minutes),
            confirmation_orders=cfg.confirmation_orders,
        )
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    levels = pd.concat(parts, ignore_index=True, sort=False)
    levels = levels.sort_values(
        ["initial_available_time", "source_timeframe_min", "pivot_time", "level_price"],
        kind="mergesort",
    ).reset_index(drop=True)
    levels.insert(0, "level_id", np.arange(1, len(levels) + 1, dtype=np.int64))
    if (pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).any():
        raise RuntimeError("swing level became available before the right confirmation bar closed")
    return levels
