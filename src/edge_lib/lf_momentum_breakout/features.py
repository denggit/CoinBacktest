#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature and signal generation for ETH LF Momentum Breakout V3."""

from __future__ import annotations

import pandas as pd

from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv
from src.edge_lib.lf_momentum_breakout.config import MomentumConfig


def build_daily_regime(base_4h: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_ema_fast"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema_slow"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_bull"] = (
        (d1["close"] > d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995)
        & (d1["d1_slow_slope"] > cfg.bull_slope_min)
    )
    d1["d1_bear"] = (
        (d1["close"] < d1["d1_ema_slow"])
        & (d1["d1_ema_fast"] < d1["d1_ema_slow"])
        & (d1["d1_slow_slope"] < cfg.bear_slope_max)
    )
    cols = ["d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]
    return d1[cols].shift(1)


def build_weekly_regime(base_4h: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    wk = resample_ohlcv(base_4h, "1W")
    wk["w_close"] = wk["close"]
    wk["w_ema_fast"] = ema(wk["close"], cfg.w_ema_fast)
    wk["w_ema_mid"] = ema(wk["close"], cfg.w_ema_mid)
    wk["w_slope_mid"] = wk["w_ema_mid"] / wk["w_ema_mid"].shift(cfg.w_slope_lookback) - 1.0
    return wk[["w_close", "w_ema_fast", "w_ema_mid", "w_slope_mid"]].shift(1)


def build_features(base_4h: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    out = base_4h.copy()
    out["atr"] = atr(out, cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["adx"] = adx(out, cfg.adx_period)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema100"] = ema(out["close"], 100)
    out["ema200"] = ema(out["close"], 200)

    out = out.join(build_daily_regime(base_4h, cfg).reindex(out.index, method="ffill"))
    out = out.join(build_weekly_regime(base_4h, cfg).reindex(out.index, method="ffill"))

    out["entry_high"] = out["high"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).max().shift(1)
    out["entry_low"] = out["low"].rolling(cfg.entry_lookback, min_periods=cfg.entry_lookback).min().shift(1)
    out["exit_low"] = out["low"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).min().shift(1)
    out["exit_high"] = out["high"].rolling(cfg.exit_lookback, min_periods=cfg.exit_lookback).max().shift(1)
    out["volume_median"] = out["volume"].rolling(cfg.volume_window, min_periods=30).median().shift(1)

    d1_bull = out["d1_bull"].astype("boolean").fillna(False).astype(bool)
    d1_bear = out["d1_bear"].astype("boolean").fillna(False).astype(bool)
    d1_distance = out["close"] / out["d1_ema_slow"] - 1.0
    out["d1_distance"] = d1_distance

    vol_ok = out["volume"] > out["volume_median"] * cfg.volume_mult
    atr_ok = out["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
    long_filter = d1_bull & atr_ok & out["adx"].between(cfg.min_adx_long, cfg.max_adx_long) & (d1_distance.abs() < cfg.max_d1_distance_long)
    short_filter = d1_bear & atr_ok & out["adx"].between(cfg.min_adx_short, cfg.max_adx_short) & (d1_distance.abs() < cfg.max_d1_distance_short) & cfg.enable_short

    out["long_breakout_setup"] = (
        (out["close"] > out["entry_high"])
        & (out["close"] > out["open"])
        & (out["close"] > out["ema50"])
        & (out["ema20"] > out["ema50"])
        & vol_ok
    )
    out["short_breakout_setup"] = (
        (out["close"] < out["entry_low"])
        & (out["close"] < out["open"])
        & (out["close"] < out["ema50"])
        & (out["ema20"] < out["ema50"])
        & vol_ok
    )

    out["long_signal"] = long_filter & out["long_breakout_setup"]
    out["short_signal"] = short_filter & out["short_breakout_setup"]
    weekly_bull = (out["w_close"] > out["w_ema_mid"]) | (out["w_slope_mid"] > 0)
    out["long_quality_full"] = (
        out["long_signal"]
        & weekly_bull.fillna(False).astype(bool)
        & (out["d1_slow_slope"] > 0.004)
        & (out["adx"] < 32.0)
        & (d1_distance.abs() < 0.110)
    )
    out["long_quality_weak"] = out["long_signal"] & ~out["long_quality_full"]
    out["long_mature_breakout"] = out["long_signal"] & (out["adx"] > cfg.mature_long_adx_threshold)

    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    out["long_exit_channel"] = out["close"] < out["exit_low"]
    out["short_exit_channel"] = out["close"] > out["exit_high"]

    out["risk_mult"] = 1.0
    out.loc[out["adx"].between(14.0, 30.0), "risk_mult"] += 0.20
    out.loc[out["atr_pct"] > 0.040, "risk_mult"] -= 0.25
    out.loc[d1_distance.abs() > 0.100, "risk_mult"] -= 0.20
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)

    out["quality_mult"] = 1.0
    out.loc[out["long_quality_weak"], "quality_mult"] *= cfg.weak_long_quality_mult
    out.loc[out["long_mature_breakout"], "quality_mult"] *= cfg.mature_long_quality_mult
    out.loc[out["volume"] > out["volume_median"] * 1.50, "quality_mult"] *= 1.10
    return out.dropna().copy()

