#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-day liquidity/path context attached to 1-second events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LatentLiquidityPathAtlasConfig
from .time_axis import as_datetime_ns


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def normalize_minute_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    out = bars.copy()
    out.index = as_datetime_ns(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index(kind="mergesort")
    out = out.loc[~out.index.duplicated(keep="last")]
    full = pd.date_range(out.index.min().floor("min"), out.index.max().floor("min"), freq="1min")
    out = out.reindex(full)
    close = _numeric(out, "close", np.nan).replace(0.0, np.nan).ffill()
    out["close"] = close
    out["open"] = pd.to_numeric(out.get("open"), errors="coerce").fillna(close)
    out["high"] = pd.to_numeric(out.get("high"), errors="coerce").fillna(close)
    out["low"] = pd.to_numeric(out.get("low"), errors="coerce").fillna(close)
    for name in ("notional", "buy_notional", "sell_notional", "delta_notional", "trades_count"):
        out[name] = _numeric(out, name, 0.0)
    out.index.name = "bar_start_time"
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _rolling_overlap_ratio(bars: pd.DataFrame) -> pd.Series:
    prior_high = bars["high"].shift(1)
    prior_low = bars["low"].shift(1)
    overlap = (np.minimum(bars["high"], prior_high) - np.maximum(bars["low"], prior_low)).clip(lower=0.0)
    union = np.maximum(bars["high"], prior_high) - np.minimum(bars["low"], prior_low)
    return _safe_div(overlap, union)


def build_macro_path_context(
    minute_bars: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Build completed-1m multi-scale paths indexed by true available time.

    These features describe broad liquidity accumulation and price acceptance;
    they do not depend on any Swing definition.
    """
    bars = normalize_minute_bars(minute_bars)
    if bars.empty:
        return pd.DataFrame()
    close = bars["close"]
    ret1 = close.pct_change().fillna(0.0)
    abs_ret = ret1.abs()
    sign_change = np.sign(ret1).ne(np.sign(ret1.shift(1))).astype(float)
    overlap_ratio = _rolling_overlap_ratio(bars)
    additions: dict[str, pd.Series] = {}
    for window in config.macro_windows_minutes:
        w = int(window)
        start = close.shift(w)
        path_ret = _safe_div(close - start, start)
        high = bars["high"].rolling(w, min_periods=w).max()
        low = bars["low"].rolling(w, min_periods=w).min()
        buy = bars["buy_notional"].rolling(w, min_periods=w).sum()
        sell = bars["sell_notional"].rolling(w, min_periods=w).sum()
        notional_sum = bars["notional"].rolling(w, min_periods=w).sum()
        trades_sum = bars["trades_count"].rolling(w, min_periods=w).sum()
        delta_sum = bars["delta_notional"].rolling(w, min_periods=w).sum()
        previous_notional = bars["notional"].shift(w).rolling(w, min_periods=w).sum()
        previous_trades = bars["trades_count"].shift(w).rolling(w, min_periods=w).sum()
        range_bp = _safe_div(high - low, close) * 1e4
        travel = abs_ret.rolling(w, min_periods=w).sum()
        efficiency = _safe_div(path_ret.abs(), travel).clip(0.0, 1.0)
        delta_share = _safe_div(delta_sum, notional_sum)
        turnover_per_range = _safe_div(notional_sum, range_bp.clip(lower=0.1))
        previous_turnover_per_range = turnover_per_range.shift(w)
        notional_millions = notional_sum / 1_000_000.0
        impact_bp_per_million = _safe_div(path_ret.abs() * 1e4, notional_millions)
        additions.update(
            {
                f"macro_ret_{w}m": path_ret,
                f"macro_range_bp_{w}m": range_bp,
                f"macro_efficiency_{w}m": efficiency,
                f"macro_realized_vol_{w}m": ret1.rolling(w, min_periods=w).std(ddof=0),
                f"macro_drawdown_from_high_{w}m": _safe_div(close - high, high),
                f"macro_rally_from_low_{w}m": _safe_div(close - low, low),
                f"macro_notional_{w}m": notional_sum,
                f"macro_notional_intensity_{w}m": _safe_div(notional_sum, previous_notional),
                f"macro_delta_{w}m": delta_sum,
                f"macro_delta_share_{w}m": delta_share,
                f"macro_buy_share_{w}m": _safe_div(buy, buy + sell),
                f"macro_trades_{w}m": trades_sum,
                f"macro_trades_intensity_{w}m": _safe_div(trades_sum, previous_trades),
                # Liquidity-first path descriptors.
                f"macro_travel_bp_{w}m": travel * 1e4,
                f"macro_overlap_ratio_{w}m": overlap_ratio.rolling(w, min_periods=w).mean(),
                f"macro_sign_changes_{w}m": sign_change.rolling(w, min_periods=w).sum(),
                f"macro_turnover_per_range_{w}m": turnover_per_range,
                f"macro_turnover_per_range_intensity_{w}m": _safe_div(
                    turnover_per_range, previous_turnover_per_range
                ),
                f"macro_pressure_without_progress_{w}m": delta_share.abs() * (1.0 - efficiency),
                f"macro_impact_bp_per_million_{w}m": impact_bp_per_million,
                f"macro_price_residency_proxy_{w}m": (
                    overlap_ratio.rolling(w, min_periods=w).mean()
                    * _safe_div(notional_sum, previous_notional).clip(lower=0.0)
                ),
                f"macro_direction_flow_disagreement_{w}m": np.sign(path_ret).fillna(0.0)
                * np.sign(delta_share).fillna(0.0),
            }
        )

    context = pd.DataFrame(additions, index=bars.index)
    context["macro_bar_start_time"] = bars.index
    context["macro_available_time"] = bars.index + pd.Timedelta(minutes=1)
    context["macro_pre_event_close"] = bars["close"].to_numpy(dtype=float)
    return context.reset_index(drop=True).sort_values("macro_available_time", kind="mergesort")


def attach_macro_path_context(
    event_features: pd.DataFrame,
    macro_context: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    if event_features.empty:
        return event_features.copy()
    if macro_context.empty:
        out = event_features.copy()
        out["macro_available_time"] = pd.NaT
        out["macro_pre_event_close"] = np.nan
        return out
    left = event_features.copy()
    left["event_time"] = as_datetime_ns(left["event_time"])
    right = macro_context.copy()
    right["macro_available_time"] = as_datetime_ns(right["macro_available_time"])
    return pd.merge_asof(
        left.sort_values("event_time", kind="mergesort"),
        right.sort_values("macro_available_time", kind="mergesort"),
        left_on="event_time",
        right_on="macro_available_time",
        direction="backward",
        allow_exact_matches=True,
    )
