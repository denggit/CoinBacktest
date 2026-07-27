#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal pre-sweep price and volatility features for R03."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import ZoneStudyConfig

EPS = 1e-12


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float), where=np.isfinite(den) & (np.abs(den) > EPS))


def build_causal_market_feature_frame(primary: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    out = pd.DataFrame(index=bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    log_return = np.log(close.clip(lower=EPS)).diff()
    abs_return = log_return.abs()
    out["prior_close"] = prev_close
    for window in cfg.atr_windows:
        w = int(window)
        atr = tr.shift(1).rolling(w, min_periods=max(5, w // 4)).mean()
        out[f"pre_atr_{w}m_abs"] = atr
        out[f"pre_atr_{w}m_bp"] = atr / prev_close * 10_000.0
        out[f"pre_rv_{w}m_bp"] = log_return.shift(1).rolling(w, min_periods=max(5, w // 4)).std() * np.sqrt(w) * 10_000.0
    atr60 = out.get("pre_atr_60m_abs")
    if atr60 is not None:
        past_median = atr60.shift(1).rolling(10_080, min_periods=1_440).median()
        out["pre_atr_60m_vs_past7d"] = atr60 / past_median
    for window in cfg.pre_windows:
        w = int(window)
        prior_base = close.shift(w + 1)
        out[f"pre_return_{w}m"] = prev_close / prior_base - 1.0
        path_abs = abs_return.shift(1).rolling(w, min_periods=max(2, w // 4)).sum()
        out[f"pre_down_efficiency_{w}m"] = (-out[f"pre_return_{w}m"]).clip(lower=0.0) / path_abs.replace(0.0, np.nan)
        prior_high = high.shift(1).rolling(w, min_periods=max(2, w // 4)).max()
        prior_low = low.shift(1).rolling(w, min_periods=max(2, w // 4)).min()
        out[f"pre_range_{w}m_bp"] = (prior_high - prior_low) / prev_close * 10_000.0
        out[f"pre_drawdown_from_high_{w}m"] = prev_close / prior_high - 1.0
    bar_range = (high - low).clip(lower=EPS)
    out["current_bar_downside_from_prior_close_bp"] = (prev_close - low).clip(lower=0.0) / prev_close * 10_000.0
    for window in cfg.atr_windows:
        out[f"bar_downside_to_pre_atr_{int(window)}m"] = out["current_bar_downside_from_prior_close_bp"] / out[f"pre_atr_{int(window)}m_bp"].replace(0.0, np.nan)
    out["current_bar_range_bp"] = bar_range / prev_close * 10_000.0
    out["current_bar_close_location"] = (close - low) / bar_range
    out["current_bar_lower_wick_fraction"] = (np.minimum(open_, close) - low) / bar_range
    if "notional" in bars.columns:
        value = pd.to_numeric(bars["notional"], errors="coerce")
        baseline = value.shift(1).rolling(60, min_periods=20).median()
        out["current_notional_vs_prior60_median"] = value / baseline
    if "trades_count" in bars.columns:
        value = pd.to_numeric(bars["trades_count"], errors="coerce")
        baseline = value.shift(1).rolling(60, min_periods=20).median()
        out["current_trades_vs_prior60_median"] = value / baseline
    if "delta_notional" in bars.columns and "notional" in bars.columns:
        delta = pd.to_numeric(bars["delta_notional"], errors="coerce")
        notional = pd.to_numeric(bars["notional"], errors="coerce")
        out["current_delta_ratio"] = delta / notional.replace(0.0, np.nan)
    return out


def attach_causal_market_features(
    events: pd.DataFrame,
    primary: pd.DataFrame,
    config: ZoneStudyConfig,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    if feature_frame is None:
        feature_frame = build_causal_market_feature_frame(bars, cfg)
    elif not feature_frame.index.equals(bars.index):
        raise ValueError("feature_frame index must match primary bars")
    positions = pd.to_numeric(events["event_pos"], errors="raise").astype(np.int64).to_numpy()
    if np.any((positions < 0) | (positions >= len(feature_frame))):
        raise ValueError("event_pos outside primary frame")
    attached = feature_frame.iloc[positions].reset_index(drop=True)
    out = pd.concat([events.reset_index(drop=True), attached], axis=1)
    floor = pd.to_numeric(out.get("zone_floor_price", out.get("sweep_low")), errors="coerce").to_numpy(dtype=float)
    sweep_low = pd.to_numeric(out["sweep_low"], errors="coerce").to_numpy(dtype=float)
    depth_abs = np.maximum(floor - sweep_low, 0.0)
    zone_width_abs = np.maximum(
        pd.to_numeric(out.get("zone_ceiling_price", floor), errors="coerce").to_numpy(dtype=float) - floor,
        0.0,
    )
    for window in cfg.atr_windows:
        atr = pd.to_numeric(out[f"pre_atr_{int(window)}m_abs"], errors="coerce").to_numpy(dtype=float)
        out[f"sweep_depth_to_pre_atr_{int(window)}m"] = _safe_divide(depth_abs, atr)
        out[f"zone_width_to_pre_atr_{int(window)}m"] = _safe_divide(zone_width_abs, atr)
        downside = pd.to_numeric(out["current_bar_downside_from_prior_close_bp"], errors="coerce").to_numpy(dtype=float)
        atr_bp = pd.to_numeric(out[f"pre_atr_{int(window)}m_bp"], errors="coerce").to_numpy(dtype=float)
        out[f"bar_downside_to_pre_atr_{int(window)}m"] = _safe_divide(downside, atr_bp)
    return out
