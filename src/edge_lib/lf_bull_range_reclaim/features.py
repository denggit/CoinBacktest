#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature and signal generation for ETH LF Bull Range Reclaim V2."""

from __future__ import annotations

import pandas as pd

from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv, rsi
from src.edge_lib.lf_bull_range_reclaim.config import BullRangeConfig


def build_daily_regime(base_4h: pd.DataFrame, cfg: BullRangeConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_close"] = d1["close"]
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_mid"] = ema(d1["close"], cfg.d1_ema_mid)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_mid_slope"] = d1["d1_ema_mid"] / d1["d1_ema_mid"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_dist_mid"] = d1["close"] / d1["d1_ema_mid"] - 1.0
    d1["d1_not_bear"] = (
        (d1["close"] > d1["d1_ema_mid"] * cfg.d1_close_mult)
        & (d1["d1_ema_fast"] > d1["d1_ema_mid"] * cfg.d1_fast_mult)
        & (d1["d1_mid_slope"] > cfg.d1_slope_min)
        & (d1["d1_dist_mid"].between(cfg.d1_min_dist, cfg.d1_max_dist))
    )
    cols = ["d1_close", "d1_ema_fast", "d1_ema_mid", "d1_ema_slow", "d1_mid_slope", "d1_slow_slope", "d1_dist_mid", "d1_not_bear"]
    return pd.DataFrame({c: d1[c].shift(1) for c in cols}, index=d1.index)


def build_features(base_4h: pd.DataFrame, cfg: BullRangeConfig) -> pd.DataFrame:
    out = base_4h.copy()
    out["ema20"] = ema(out["close"], cfg.ema_fast)
    out["ema50"] = ema(out["close"], cfg.ema_mid)
    out["ema100"] = ema(out["close"], cfg.ema_slow)
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["volume_med"] = out["volume"].rolling(30, min_periods=10).median().shift(1)
    out = out.join(build_daily_regime(base_4h, cfg).reindex(out.index, method="ffill"))

    out["prev_close_below_ema20"] = out["close"].shift(1) < out["ema20"].shift(1)
    out["pb_min_dist50"] = (out["low"] / out["ema50"] - 1.0).rolling(cfg.pullback_lookback, min_periods=1).min().shift(1)
    out["pb_min_dist100"] = (out["low"] / out["ema100"] - 1.0).rolling(cfg.pullback_lookback, min_periods=1).min().shift(1)
    out["recent_pullback"] = (out["pb_min_dist50"] < cfg.pb_dist50) | (out["pb_min_dist100"] < cfg.pb_dist100) | out["prev_close_below_ema20"]
    out["reclaim"] = (out["close"] > out["ema20"] * cfg.reclaim_mult) & (out["close"] > out["open"]) & (out["close"] > out["close"].shift(1)) & (out["rsi"] > cfg.rsi_min)
    out["range_ok"] = out["adx"].between(cfg.adx_min, cfg.adx_max) & out["atr_pct"].between(cfg.atr_min, cfg.atr_max)
    out["volume_ok"] = out["volume"] > out["volume_med"] * cfg.vol_mult
    out["h4_dist50"] = out["close"] / out["ema50"] - 1.0
    out["not_extended"] = out["h4_dist50"] < cfg.h4_max_dist50
    out["daily_ok"] = out["d1_not_bear"].astype("boolean").fillna(False).astype(bool)
    out["macro_bull_ok"] = out["daily_ok"] & ((out["d1_ema_mid"] / out["d1_ema_slow"]) > cfg.d1_mid_vs_slow_min)
    out["quality_bucket_a"] = out["macro_bull_ok"] & out["recent_pullback"] & out["reclaim"] & out["range_ok"] & out["volume_ok"] & out["not_extended"]
    out["secondary_reclaim"] = (
        out["macro_bull_ok"]
        & out["recent_pullback"]
        & (out["close"] > out["ema20"])
        & (out["close"] > out["open"])
        & (out["close"] > out["close"].shift(1))
        & (out["rsi"] > cfg.secondary_rsi_min)
        & out["adx"].between(cfg.adx_min, cfg.secondary_adx_max)
        & out["atr_pct"].between(cfg.atr_min, cfg.atr_max)
        & out["volume_ok"]
        & out["not_extended"]
        & (~out["quality_bucket_a"])
    )
    out["quality_bucket_b"] = out["secondary_reclaim"]
    out["long_signal"] = out["quality_bucket_a"] | out["quality_bucket_b"]
    out["short_signal"] = False
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out["exit_low"] = out["low"].rolling(16, min_periods=4).min().shift(1)
    out["long_exit_channel"] = (out["close"] < out["ema50"] * cfg.exit_ema50_mult) | (out["close"] < out["exit_low"])
    out["short_exit_channel"] = False
    out["risk_mult"] = 1.0
    out.loc[out["adx"].between(10.0, 18.0), "risk_mult"] += 0.15
    out.loc[out["atr_pct"].between(0.004, 0.030), "risk_mult"] += 0.15
    out.loc[out["atr_pct"] > 0.040, "risk_mult"] -= 0.25
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)
    out["quality_mult"] = 0.0
    out.loc[out["quality_bucket_a"], "quality_mult"] = 1.00
    out.loc[out["quality_bucket_b"], "quality_mult"] = cfg.secondary_quality_mult
    out["quality_mult"] = out["quality_mult"].clip(0.20, 1.20)
    return out.dropna().copy()

