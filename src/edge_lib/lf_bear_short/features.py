#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature and signal generation for ETH LF Bear Short V3."""

from __future__ import annotations

import pandas as pd

from src.backtest_common.indicators import adx, atr, ema, resample_ohlcv
from src.edge_lib.lf_bear_short.config import BearConfig


def _build_v8_daily_regime(base_4h: pd.DataFrame) -> pd.DataFrame:
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_ema_fast"] = ema(d1["close"], 8)
    d1["d1_ema_slow"] = ema(d1["close"], 30)
    d1["d1_slow_slope"] = d1["d1_ema_slow"] / d1["d1_ema_slow"].shift(10) - 1.0
    d1["d1_bull"] = (d1["close"] > d1["d1_ema_slow"]) & (d1["d1_ema_fast"] > d1["d1_ema_slow"] * 0.995) & (d1["d1_slow_slope"] > -0.0300)
    d1["d1_bear"] = (d1["close"] < d1["d1_ema_slow"]) & (d1["d1_ema_fast"] < d1["d1_ema_slow"]) & (d1["d1_slow_slope"] <= -0.0030)
    for col in ["d1_ema_fast", "d1_ema_slow", "d1_slow_slope", "d1_bull", "d1_bear"]:
        d1[f"{col}_available"] = d1[col].shift(1)
    return d1


def _build_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d1 = _build_v8_daily_regime(out)
    out = out.join(
        d1[["d1_ema_fast_available", "d1_ema_slow_available", "d1_slow_slope_available", "d1_bull_available", "d1_bear_available"]].reindex(out.index, method="ffill")
    )
    out = out.rename(
        columns={
            "d1_ema_fast_available": "d1_ema_fast",
            "d1_ema_slow_available": "d1_ema_slow",
            "d1_slow_slope_available": "d1_slow_slope",
            "d1_bull_available": "d1_bull",
            "d1_bear_available": "d1_bear",
        }
    )
    out["atr"] = atr(out, 20)
    out["atr_pct"] = out["atr"] / out["close"]
    out["atr_ok"] = out["atr_pct"].between(0.0030, 0.0800)
    out["adx"] = adx(out, 14)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema89"] = ema(out["close"], 89)
    out["ema100"] = ema(out["close"], 100)
    out["ema200"] = ema(out["close"], 200)
    out["entry_high"] = out["high"].rolling(40, min_periods=40).max().shift(1)
    out["entry_low"] = out["low"].rolling(40, min_periods=40).min().shift(1)
    out["exit_high"] = out["high"].rolling(36, min_periods=36).max().shift(1)
    out["exit_low"] = out["low"].rolling(36, min_periods=36).min().shift(1)
    d1_bull = out["d1_bull"].astype("boolean").fillna(False).astype(bool)
    d1_bear = out["d1_bear"].astype("boolean").fillna(False).astype(bool)
    long_filter = d1_bull & out["atr_ok"] & (out["adx"] >= 8.0)
    short_filter = d1_bear & out["atr_ok"] & (out["adx"] >= 18.0)
    breakout_long = (out["close"] > out["entry_high"]) & (out["close"] > out["ema100"])
    trend_long = (out["close"] > out["ema50"]) & (out["ema20"] > out["ema50"]) & (out["close"] > out["open"])
    breakout_short = (out["close"] < out["entry_low"]) & (out["close"] < out["ema100"])
    trend_short = (out["close"] < out["ema50"]) & (out["ema20"] < out["ema50"]) & (out["close"] < out["open"])
    ema_spread = (out["ema20"] / out["ema50"] - 1.0).abs()
    price_distance = (out["close"] / out["d1_ema_slow"] - 1.0).abs()
    d1_slope_abs = out["d1_slow_slope"].abs()
    out["risk_mult"] = 0.85
    out.loc[out["adx"].between(14.0, 30.0), "risk_mult"] += 0.25
    out.loc[d1_slope_abs.between(0.004, 0.060), "risk_mult"] += 0.20
    out.loc[price_distance.between(0.010, 0.085), "risk_mult"] += 0.20
    out.loc[ema_spread.between(0.004, 0.025), "risk_mult"] += 0.15
    out.loc[out["adx"] < 12.0, "risk_mult"] -= 0.25
    out.loc[out["adx"] > 36.0, "risk_mult"] -= 0.30
    out.loc[out["atr_pct"] >= 0.030, "risk_mult"] -= 0.25
    out.loc[price_distance > 0.100, "risk_mult"] -= 0.30
    out.loc[ema_spread > 0.035, "risk_mult"] -= 0.20
    out.loc[d1_slope_abs > 0.090, "risk_mult"] -= 0.20
    out["risk_mult"] = out["risk_mult"].clip(0.35, 2.3)
    out["long_breakout_setup"] = breakout_long
    out["short_breakout_setup"] = breakout_short
    out["quality_mult"] = 1.0
    out.loc[long_filter & breakout_long, "quality_mult"] *= 1.45
    out.loc[long_filter & (~breakout_long), "quality_mult"] *= 0.65
    out.loc[short_filter & breakout_short, "quality_mult"] *= 0.45
    out.loc[short_filter & (~breakout_short), "quality_mult"] *= 1.20
    out.loc[out["risk_mult"] < 1.0, "quality_mult"] *= 0.45
    out.loc[out["atr_pct"] > 0.022, "quality_mult"] *= 0.55
    out.loc[(out["adx"] > 30.0) & short_filter, "quality_mult"] *= 0.50
    out.loc[(out["adx"] > 30.0) & long_filter & (~breakout_long), "quality_mult"] *= 0.60
    out["quality_mult"] = out["quality_mult"].clip(0.20, 1.80)
    out["long_signal"] = long_filter & (breakout_long | trend_long)
    out["short_signal"] = short_filter & (breakout_short | trend_short)
    out["long_exit_channel"] = (out["close"] < out["exit_low"]) | ((out["close"] < out["ema89"]) & (out["ema20"] < out["ema50"])) | ((~d1_bull) & (out["close"] < out["ema50"]))
    out["short_exit_channel"] = (out["close"] > out["exit_high"]) | ((out["close"] > out["ema89"]) & (out["ema20"] > out["ema50"])) | ((~d1_bear) & (out["close"] > out["ema50"]))
    out["signal"] = 0
    out.loc[out["long_signal"], "signal"] = 1
    out.loc[out["short_signal"], "signal"] = -1
    return out.dropna().copy()


def add_shifted_higher_tf_features(base_4h: pd.DataFrame, features: pd.DataFrame, cfg: BearConfig) -> pd.DataFrame:
    out = features.copy()
    d1 = resample_ohlcv(base_4h, "1D")
    d1["d1_close"] = d1["close"]
    d1["d1_ema20"] = ema(d1["close"], cfg.d1_ema_fast)
    d1["d1_ema50"] = ema(d1["close"], cfg.d1_ema_mid)
    d1["d1_ema100"] = ema(d1["close"], cfg.d1_ema_slow)
    d1["d1_ema200"] = ema(d1["close"], cfg.d1_ema_major)
    d1["d1_ema50_slope"] = d1["d1_ema50"] / d1["d1_ema50"].shift(cfg.d1_slope_lookback) - 1.0
    d1["d1_ema100_slope"] = d1["d1_ema100"] / d1["d1_ema100"].shift(cfg.d1_slope_lookback) - 1.0
    d1_cols = ["d1_close", "d1_ema20", "d1_ema50", "d1_ema100", "d1_ema200", "d1_ema50_slope", "d1_ema100_slope"]
    out = out.join(pd.DataFrame({col: d1[col].shift(1) for col in d1_cols}, index=d1.index).reindex(out.index, method="ffill"))

    wk = resample_ohlcv(base_4h, "1W")
    wk["w_close"] = wk["close"]
    wk["w_ema10"] = ema(wk["close"], cfg.w_ema_fast)
    wk["w_ema20"] = ema(wk["close"], cfg.w_ema_mid)
    wk["w_ema40"] = ema(wk["close"], cfg.w_ema_slow)
    wk["w_ema20_slope"] = wk["w_ema20"] / wk["w_ema20"].shift(cfg.w_slope_lookback) - 1.0
    w_cols = ["w_close", "w_ema10", "w_ema20", "w_ema40", "w_ema20_slope"]
    out = out.join(pd.DataFrame({col: wk[col].shift(1) for col in w_cols}, index=wk.index).reindex(out.index, method="ffill"))

    out["ret_6"] = out["close"] / out["close"].shift(6) - 1.0
    out["ret_12"] = out["close"] / out["close"].shift(12) - 1.0
    out["ret_30"] = out["close"] / out["close"].shift(30) - 1.0
    return out


def build_bear_features(base_4h: pd.DataFrame, cfg: BearConfig) -> pd.DataFrame:
    out = _build_v8_features(base_4h)
    out = add_shifted_higher_tf_features(base_4h, out, cfg)
    out["long_signal"] = False

    weekly_bear = (out["w_close"] < out["w_ema20"]) & (out["w_ema20_slope"] < 0)
    d1_major_bear = (out["d1_close"] < out["d1_ema100"]) & (out["d1_ema50_slope"] < -0.008)
    bear_permission_v2 = (
        d1_major_bear
        & (out["d1_ema100_slope"] > -0.025)
        & (out["ret_12"] < 0.005)
        & ((out["close"] / out["d1_ema100"] - 1.0) > -0.110)
    )
    four_h_bear = (
        (out["ema20"] < out["ema50"])
        & (out["close"] < out["ema20"])
        & (out["close"] < out["open"])
        & out["adx"].between(12.0, 32.0)
        & out["atr_pct"].between(0.006, 0.030)
        & ((out["close"] / out["d1_ema100"] - 1.0).between(-0.18, 0.02))
    )
    breakdown = (
        weekly_bear
        & (out["d1_close"] < out["d1_ema100"])
        & (out["ema20"] < out["ema50"])
        & (out["close"] < out["entry_low"])
        & (out["close"] < out["open"])
        & out["adx"].between(10.0, 30.0)
        & out["atr_pct"].between(0.004, 0.032)
    )
    crash_continuation = d1_major_bear & four_h_bear
    permission_continuation = bear_permission_v2 & four_h_bear
    if cfg.style == "breakdown":
        short_signal = breakdown
    elif cfg.style == "crash_continuation":
        short_signal = crash_continuation
    elif cfg.style in {"bear_permission_v2", "bear_permission_v3"}:
        short_signal = permission_continuation
    elif cfg.style == "combo":
        short_signal = breakdown | permission_continuation
    else:
        raise ValueError(f"Unsupported style: {cfg.style}")

    out["weekly_bear"] = weekly_bear.fillna(False).astype(bool)
    out["bear_permission_v3"] = bear_permission_v2.fillna(False).astype(bool)
    out["bear_permission_v2"] = out["bear_permission_v3"]
    out["short_signal"] = short_signal.fillna(False).astype(bool)
    out["signal"] = 0
    out.loc[out["short_signal"], "signal"] = -1
    out["short_exit_channel"] = (out["close"] > out["ema50"]) | ((out["close"] > out["ema89"]) & (out["ema20"] > out["ema50"])) | (out["close"] > out["exit_high"])
    out["long_exit_channel"] = False
    out["risk_mult"] = 0.60
    out.loc[out["adx"].between(14.0, 28.0), "risk_mult"] += 0.30
    out.loc[out["d1_ema100_slope"] < -0.006, "risk_mult"] += 0.30
    out.loc[out["atr_pct"].between(0.006, 0.026), "risk_mult"] += 0.20
    out.loc[out["atr_pct"] > 0.030, "risk_mult"] -= 0.35
    out["risk_mult"] = out["risk_mult"].clip(cfg.min_risk_mult, cfg.max_risk_mult)
    out["quality_mult"] = 1.0
    trend_cont = (out["close"] < out["ema20"]) & (out["ema20"] < out["ema50"])
    out.loc[trend_cont, "quality_mult"] *= 1.35
    out.loc[out["close"] < out["entry_low"], "quality_mult"] *= 0.75
    out.loc[out["adx"] > 32.0, "quality_mult"] *= 0.60
    out.loc[out["atr_pct"] > 0.025, "quality_mult"] *= 0.70
    out["quality_mult"] = out["quality_mult"].clip(0.20, 1.70)
    return out.dropna().copy()

