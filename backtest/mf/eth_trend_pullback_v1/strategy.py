#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-timeframe feature builder for ETH Trend Pullback V1.

Design:
- 4H: established directional regime only.
- 1H: healthy pullback into EMA20, still on the trend side of EMA50, then reclaim.
- 15m: re-acceleration breakout after the completed 1H reclaim.

All higher-timeframe rows are aligned by ``available_time`` rather than bar
start time.  A 15m signal is generated only after its own candle is closed and
is executed by the engine at the next 15m open.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    signal_timeframe: str = "15m"
    signal_delta: pd.Timedelta = pd.Timedelta(minutes=15)

    # 4H trend regime.
    h4_fast_ema: int = 50
    h4_slow_ema: int = 200
    h4_slope_lookback: int = 6  # 24h

    # 1H pullback/reclaim.
    h1_fast_ema: int = 20
    h1_slow_ema: int = 50
    h1_atr_period: int = 14
    setup_active_hours: int = 3
    h1_pullback_floor_atr: float = 0.50

    # 15m re-acceleration and stop.
    signal_ema: int = 20
    signal_atr_period: int = 14
    trigger_lookback_bars: int = 4  # 1h on 15m
    stop_lookback_bars: int = 12    # 3h on 15m
    stop_buffer_atr: float = 0.25
    max_reaccel_distance_h1_atr: float = 1.50


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _resample_complete_ohlcv(df: pd.DataFrame, rule: str, expected_bars: int) -> pd.DataFrame:
    agg = df.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        source_bars=("close", "count"),
    )
    agg = agg.loc[agg["source_bars"] == int(expected_bars)].copy()
    return agg.dropna(subset=["open", "high", "low", "close"])


def _build_h4(base: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    h4 = _resample_complete_ohlcv(base, "4h", 16)
    h4["ema_fast"] = h4["close"].ewm(span=cfg.h4_fast_ema, adjust=False, min_periods=cfg.h4_fast_ema).mean()
    h4["ema_slow"] = h4["close"].ewm(span=cfg.h4_slow_ema, adjust=False, min_periods=cfg.h4_slow_ema).mean()
    h4["ema_fast_slope"] = h4["ema_fast"] / h4["ema_fast"].shift(cfg.h4_slope_lookback) - 1.0
    h4["regime_long"] = (
        (h4["ema_fast"] > h4["ema_slow"])
        & (h4["close"] > h4["ema_fast"])
        & (h4["ema_fast_slope"] > 0)
    )
    h4["regime_short"] = (
        (h4["ema_fast"] < h4["ema_slow"])
        & (h4["close"] < h4["ema_fast"])
        & (h4["ema_fast_slope"] < 0)
    )
    h4["h4_bar_time"] = h4.index
    h4["h4_available_time"] = h4.index + pd.Timedelta(hours=4)
    return h4


def _build_h1(base: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    h1 = _resample_complete_ohlcv(base, "1h", 4)
    h1["ema20"] = h1["close"].ewm(span=cfg.h1_fast_ema, adjust=False, min_periods=cfg.h1_fast_ema).mean()
    h1["ema50"] = h1["close"].ewm(span=cfg.h1_slow_ema, adjust=False, min_periods=cfg.h1_slow_ema).mean()
    h1["atr14"] = atr(h1, cfg.h1_atr_period)

    # Healthy pullback: previous completed 1H candle was on/through EMA20 but
    # the pullback did not meaningfully break the slower 1H trend anchor.
    prior_3h_low = h1["low"].rolling(3, min_periods=3).min()
    prior_3h_high = h1["high"].rolling(3, min_periods=3).max()
    long_floor_ok = prior_3h_low > (h1["ema50"] - cfg.h1_pullback_floor_atr * h1["atr14"])
    short_ceiling_ok = prior_3h_high < (h1["ema50"] + cfg.h1_pullback_floor_atr * h1["atr14"])

    h1["reclaim_long"] = (
        (h1["ema20"] > h1["ema50"])
        & (h1["close"].shift(1) <= h1["ema20"].shift(1))
        & (h1["close"] > h1["ema20"])
        & (h1["close"] > h1["ema50"])
        & long_floor_ok
    )
    h1["reclaim_short"] = (
        (h1["ema20"] < h1["ema50"])
        & (h1["close"].shift(1) >= h1["ema20"].shift(1))
        & (h1["close"] < h1["ema20"])
        & (h1["close"] < h1["ema50"])
        & short_ceiling_ok
    )

    window = max(1, int(cfg.setup_active_hours))
    h1["setup_long_active"] = h1["reclaim_long"].rolling(window, min_periods=1).max().fillna(0).astype(bool)
    h1["setup_short_active"] = h1["reclaim_short"].rolling(window, min_periods=1).max().fillna(0).astype(bool)
    h1["pullback_low_3h"] = prior_3h_low
    h1["pullback_high_3h"] = prior_3h_high
    h1["h1_bar_time"] = h1.index
    h1["h1_available_time"] = h1.index + pd.Timedelta(hours=1)
    return h1


def _align_context(
    signal_frame: pd.DataFrame,
    context: pd.DataFrame,
    *,
    available_col: str,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    left = signal_frame[["signal_available_time"]].reset_index(names="bar_time")
    keep = [available_col] + columns
    right = context[keep].dropna(subset=[available_col]).sort_values(available_col).copy()
    right = right.rename(columns={c: f"{prefix}{c}" for c in columns})
    merged = pd.merge_asof(
        left.sort_values("signal_available_time"),
        right,
        left_on="signal_available_time",
        right_on=available_col,
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.set_index("bar_time").reindex(signal_frame.index)
    return merged.drop(columns=["signal_available_time"], errors="ignore")


def build_features(base: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StrategyConfig()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(base.columns)
    if missing:
        raise RuntimeError(f"base bars missing required columns: {sorted(missing)}")

    out = base.sort_index().copy()
    out.index = pd.to_datetime(out.index)
    out["signal_available_time"] = out.index + cfg.signal_delta
    out["atr14"] = atr(out, cfg.signal_atr_period)
    out["ema20_15m"] = out["close"].ewm(span=cfg.signal_ema, adjust=False, min_periods=cfg.signal_ema).mean()
    out["trigger_high"] = out["close"].rolling(cfg.trigger_lookback_bars, min_periods=cfg.trigger_lookback_bars).max().shift(1)
    out["trigger_low"] = out["close"].rolling(cfg.trigger_lookback_bars, min_periods=cfg.trigger_lookback_bars).min().shift(1)
    out["recent_low"] = out["low"].rolling(cfg.stop_lookback_bars, min_periods=cfg.stop_lookback_bars).min().shift(1)
    out["recent_high"] = out["high"].rolling(cfg.stop_lookback_bars, min_periods=cfg.stop_lookback_bars).max().shift(1)

    h4 = _build_h4(out[["open", "high", "low", "close", "volume"]], cfg)
    h1 = _build_h1(out[["open", "high", "low", "close", "volume"]], cfg)

    h4_aligned = _align_context(
        out,
        h4,
        available_col="h4_available_time",
        prefix="h4_",
        columns=["h4_bar_time", "ema_fast", "ema_slow", "ema_fast_slope", "regime_long", "regime_short", "close"],
    )
    h1_aligned = _align_context(
        out,
        h1,
        available_col="h1_available_time",
        prefix="h1_",
        columns=[
            "h1_bar_time", "ema20", "ema50", "atr14", "reclaim_long", "reclaim_short",
            "setup_long_active", "setup_short_active", "pullback_low_3h", "pullback_high_3h", "close",
        ],
    )
    out = out.join(h4_aligned).join(h1_aligned)

    # Rename duplicated audit time fields generated by _align_context prefixing.
    out = out.rename(
        columns={
            "h4_h4_bar_time": "used_h4_timestamp",
            "h1_h1_bar_time": "used_h1_timestamp",
            "h4_available_time": "used_h4_available_time",
            "h1_available_time": "used_h1_available_time",
        }
    )
    # merge_asof leaves the right key un-prefixed; recover it explicitly from
    # available time using the selected completed bar timestamps.
    out["used_h4_available_time"] = pd.to_datetime(out["used_h4_timestamp"]) + pd.Timedelta(hours=4)
    out["used_h1_available_time"] = pd.to_datetime(out["used_h1_timestamp"]) + pd.Timedelta(hours=1)

    long_distance_ok = (out["close"] - out["h1_ema20"]).abs() <= cfg.max_reaccel_distance_h1_atr * out["h1_atr14"]
    short_distance_ok = long_distance_ok

    long_fire = (
        out["h4_regime_long"].astype("boolean").fillna(False).astype(bool)
        & out["h1_setup_long_active"].astype("boolean").fillna(False).astype(bool)
        & long_distance_ok.fillna(False)
        & (out["close"] > out["trigger_high"])
        & (out["close"] > out["ema20_15m"])
        & (out["close"] > out["open"])
    )
    short_fire = (
        out["h4_regime_short"].astype("boolean").fillna(False).astype(bool)
        & out["h1_setup_short_active"].astype("boolean").fillna(False).astype(bool)
        & short_distance_ok.fillna(False)
        & (out["close"] < out["trigger_low"])
        & (out["close"] < out["ema20_15m"])
        & (out["close"] < out["open"])
    )

    # Fire only on the first bar of a contiguous raw-trigger state.  This avoids
    # repeated entries from the same one-way breakout without introducing a
    # learned threshold or future-dependent de-duplication.
    long_first = long_fire & ~long_fire.shift(1, fill_value=False)
    short_first = short_fire & ~short_fire.shift(1, fill_value=False)
    out["signal"] = np.select([long_first, short_first], [1, -1], default=0).astype("int8")
    out["stop"] = np.where(
        out["signal"] > 0,
        out["recent_low"] - cfg.stop_buffer_atr * out["atr14"],
        np.where(out["signal"] < 0, out["recent_high"] + cfg.stop_buffer_atr * out["atr14"], np.nan),
    )

    out["context_available_time_flag"] = (
        out["used_h1_available_time"].notna()
        & out["used_h4_available_time"].notna()
        & (out["used_h1_available_time"] <= out["signal_available_time"])
        & (out["used_h4_available_time"] <= out["signal_available_time"])
    )
    return out


def robustness_configs(base: StrategyConfig | None = None) -> list[tuple[str, StrategyConfig]]:
    """Small predeclared neighborhood used only as a robustness diagnostic.

    The caller must not select the best row and call it the new baseline.
    """
    cfg = base or StrategyConfig()
    return [
        ("BASE", cfg),
        ("TRIGGER_3", replace(cfg, trigger_lookback_bars=3)),
        ("TRIGGER_6", replace(cfg, trigger_lookback_bars=6)),
        ("SETUP_2H", replace(cfg, setup_active_hours=2)),
        ("SETUP_4H", replace(cfg, setup_active_hours=4)),
    ]
