#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal feature and signal builder for ETH Trend Breakout V1.

Design rule: the breakout event is deliberately broad.  Trend alignment,
breakout strength and candle quality become a continuous risk multiplier rather
than a chain of filters.  This keeps the event universe large enough for a
portfolio sleeve and makes funnel collapse visible instead of hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TrendBreakoutConfig


REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _clip01(values: pd.Series) -> pd.Series:
    return values.clip(lower=0.0, upper=1.0)


def build_features(bars: pd.DataFrame, cfg: TrendBreakoutConfig) -> pd.DataFrame:
    cfg.validate()
    missing = [col for col in REQUIRED_COLUMNS if col not in bars.columns]
    if missing:
        raise ValueError(f"missing required OHLC columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must use DatetimeIndex")

    out = bars.sort_index(kind="stable").copy()
    if out.index.has_duplicates:
        raise ValueError("duplicate timestamps are not allowed")

    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")

    out["prior_breakout_high"] = high.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)
    out["prior_breakout_low"] = low.rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).min().shift(1)
    out["atr"] = _atr(out, cfg.atr_period)
    out["ema_fast"] = close.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    out["ema_slow"] = close.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()

    long_now = close > out["prior_breakout_high"]
    short_now = close < out["prior_breakout_low"]
    # Only the first close through a currently available structure boundary is
    # an event.  This avoids counting every bar that remains outside the range.
    long_event = long_now & ~long_now.shift(1, fill_value=False)
    short_event = short_now & ~short_now.shift(1, fill_value=False)

    signal = pd.Series(0, index=out.index, dtype="int8")
    signal.loc[long_event] = 1
    signal.loc[short_event] = -1
    out["signal"] = signal
    out["structure_event"] = signal.ne(0)

    # Stop uses only bars known by signal close, including the fully closed
    # signal bar.  It is never calculated from the next execution bar.
    recent_low = low.rolling(cfg.stop_lookback, min_periods=cfg.stop_lookback).min()
    recent_high = high.rolling(cfg.stop_lookback, min_periods=cfg.stop_lookback).max()
    out["stop"] = np.where(signal > 0, recent_low, np.where(signal < 0, recent_high, np.nan))

    safe_atr = out["atr"].replace(0.0, np.nan)
    side = signal.astype(float)
    trend_gap = side * (out["ema_fast"] - out["ema_slow"]) / safe_atr
    trend_component = _clip01((trend_gap + 1.0) / 2.0).fillna(0.5)

    long_depth = (close - out["prior_breakout_high"]) / safe_atr
    short_depth = (out["prior_breakout_low"] - close) / safe_atr
    breakout_depth = pd.Series(np.where(signal > 0, long_depth, np.where(signal < 0, short_depth, np.nan)), index=out.index)
    depth_component = _clip01(breakout_depth / 0.50).fillna(0.0)

    candle_range = (high - low).replace(0.0, np.nan)
    directional_body = side * (close - open_) / candle_range
    body_component = _clip01((directional_body + 0.25) / 1.25).fillna(0.0)

    close_location_long = (close - low) / candle_range
    close_location_short = (high - close) / candle_range
    close_location = pd.Series(
        np.where(signal > 0, close_location_long, np.where(signal < 0, close_location_short, np.nan)),
        index=out.index,
    )
    location_component = _clip01(close_location).fillna(0.0)

    quality = (
        0.35
        + 0.25 * trend_component
        + 0.15 * depth_component
        + 0.15 * body_component
        + 0.10 * location_component
    )
    quality = quality.clip(lower=cfg.min_risk_mult, upper=cfg.max_risk_mult).where(signal.ne(0), 0.0)
    out["trend_component"] = trend_component.where(signal.ne(0))
    out["breakout_depth_atr"] = breakout_depth.where(signal.ne(0))
    out["body_component"] = body_component.where(signal.ne(0))
    out["close_location_component"] = location_component.where(signal.ne(0))
    out["risk_mult"] = quality
    out["signal_reason"] = np.where(signal > 0, "STRUCTURE_BREAK_LONG", np.where(signal < 0, "STRUCTURE_BREAK_SHORT", ""))

    # Feature completeness is diagnostics only; it does not look at the next
    # bar and does not silently delete imperfect-but-tradable events.
    out["feature_complete"] = (
        out["structure_event"]
        & out["stop"].notna()
        & out["atr"].notna()
        & out["ema_fast"].notna()
        & out["ema_slow"].notna()
        & out["risk_mult"].notna()
    )
    return out
