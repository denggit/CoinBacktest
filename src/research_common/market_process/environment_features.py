#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal environment features and candidate construction for R02.

All range-bar context is aligned by completed ``end_ts`` with vectorized
``numpy.searchsorted``.  This module contains research-specific transforms; it
does not access files or exchange APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.trade_bar_orderflow import build_trade_bar_orderflow_features

FAMILY_ORDER = (
    "compression_breakout",
    "expansion_exhaustion",
    "balance_failed_auction",
)
DEFINITION_ORDER = ("loose", "base", "strict")
SCENARIO_ORDER = ("base", "fee_2x", "delay_1m", "delay_3m", "slip_2bps")


@dataclass(frozen=True)
class Definition:
    name: str
    compression_vol_max: float
    compression_speed_max: float
    compression_efficiency_max: float
    breakout_flow_min: float
    breakout_activity_min: float
    expansion_vol_min: float
    expansion_speed_min: float
    expansion_efficiency_min: float
    expansion_move_min: float
    exhaustion_flow_min: float
    balance_efficiency_max: float
    balance_vol_max: float
    failed_auction_flow_min: float


DEFINITIONS: tuple[Definition, ...] = (
    Definition(
        name="loose",
        compression_vol_max=0.88,
        compression_speed_max=0.90,
        compression_efficiency_max=0.52,
        breakout_flow_min=0.12,
        breakout_activity_min=1.25,
        expansion_vol_min=1.20,
        expansion_speed_min=1.15,
        expansion_efficiency_min=0.38,
        expansion_move_min=0.0050,
        exhaustion_flow_min=0.14,
        balance_efficiency_max=0.38,
        balance_vol_max=1.25,
        failed_auction_flow_min=0.05,
    ),
    Definition(
        name="base",
        compression_vol_max=0.78,
        compression_speed_max=0.80,
        compression_efficiency_max=0.45,
        breakout_flow_min=0.16,
        breakout_activity_min=1.45,
        expansion_vol_min=1.35,
        expansion_speed_min=1.30,
        expansion_efficiency_min=0.45,
        expansion_move_min=0.0065,
        exhaustion_flow_min=0.18,
        balance_efficiency_max=0.32,
        balance_vol_max=1.12,
        failed_auction_flow_min=0.08,
    ),
    Definition(
        name="strict",
        compression_vol_max=0.68,
        compression_speed_max=0.70,
        compression_efficiency_max=0.38,
        breakout_flow_min=0.21,
        breakout_activity_min=1.70,
        expansion_vol_min=1.55,
        expansion_speed_min=1.50,
        expansion_efficiency_min=0.52,
        expansion_move_min=0.0080,
        exhaustion_flow_min=0.23,
        balance_efficiency_max=0.27,
        balance_vol_max=1.00,
        failed_auction_flow_min=0.12,
    ),
)



def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    den = pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return (pd.to_numeric(a, errors="coerce") / den).replace([np.inf, -np.inf], np.nan)


def _rolling_efficiency(close: pd.Series, ret_abs: pd.Series, window: int) -> pd.Series:
    displacement = (close / close.shift(window) - 1.0).abs()
    path = ret_abs.rolling(window, min_periods=window).sum()
    return _safe_divide(displacement, path).clip(0.0, 1.0)


def _prior_ewm(series: pd.Series, span: int, min_periods: int) -> pd.Series:
    return series.shift(1).ewm(span=span, adjust=False, min_periods=min_periods).mean()


def build_range_context(minute_index: pd.DatetimeIndex, range_bars: pd.DataFrame) -> pd.DataFrame:
    """Align completed range-bar context to closed 1m bars in O(N log M).

    ``range_bars.end_ts`` is the availability timestamp.  For every minute bar,
    the selected range bar is the last one whose ``end_ts <= signal_time``.
    Rolling window counts are computed with two vectorized ``searchsorted``
    calls rather than a full as-of merge or per-row scans.
    """
    out = pd.DataFrame(index=minute_index)
    if range_bars.empty:
        for col in (
            "range_available_time",
            "range_duration_seconds",
            "range_speed_ratio",
            "range_delta_ratio",
            "range_direction",
            "range_count_15m",
            "range_count_60m",
            "range_count_ratio_15m",
            "range_context_causal",
        ):
            out[col] = pd.NaT if col == "range_available_time" else np.nan
        out["range_context_causal"] = False
        return out

    rb = range_bars.reset_index(drop=True).sort_values(["end_ts", "bar_id"], kind="stable").copy()
    end_time = pd.DatetimeIndex(pd.to_datetime(rb["end_ts"], errors="coerce"))
    valid = ~end_time.isna()
    rb = rb.loc[valid].copy()
    end_time = end_time[valid]
    if rb.empty:
        return build_range_context(minute_index, rb)

    end_ns = end_time.asi8
    minute_ns = minute_index.asi8
    right = np.searchsorted(end_ns, minute_ns, side="right")
    last = right - 1
    valid_last = last >= 0

    duration = pd.to_numeric(rb["duration_seconds"], errors="coerce").clip(lower=1.0)
    speed = 60.0 / duration
    speed_base = speed.shift(1).ewm(span=240, adjust=False, min_periods=60).mean()
    speed_ratio = _safe_divide(speed, speed_base).clip(0.0, 20.0)
    delta_ratio = _safe_divide(rb["delta_notional"], rb["notional"]).clip(-1.0, 1.0)
    direction = pd.to_numeric(rb["direction"], errors="coerce").clip(-1.0, 1.0)

    def align(values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        result = np.full(len(minute_index), fill, dtype=float)
        result[valid_last] = values[last[valid_last]]
        return result

    available = np.full(len(minute_index), np.datetime64("NaT"), dtype="datetime64[ns]")
    available[valid_last] = end_time.to_numpy(dtype="datetime64[ns]")[last[valid_last]]
    out["range_available_time"] = available
    out["range_duration_seconds"] = align(duration.to_numpy(float))
    out["range_speed_ratio"] = align(speed_ratio.to_numpy(float))
    out["range_delta_ratio"] = align(delta_ratio.to_numpy(float))
    out["range_direction"] = align(direction.to_numpy(float))

    left_15 = np.searchsorted(end_ns, minute_ns - pd.Timedelta(minutes=15).value, side="right")
    left_60 = np.searchsorted(end_ns, minute_ns - pd.Timedelta(minutes=60).value, side="right")
    count_15 = (right - left_15).astype(float)
    count_60 = (right - left_60).astype(float)
    out["range_count_15m"] = count_15
    out["range_count_60m"] = count_60
    count_base = _prior_ewm(pd.Series(count_15, index=minute_index), span=1440, min_periods=360)
    out["range_count_ratio_15m"] = _safe_divide(out["range_count_15m"], count_base).clip(0.0, 20.0)
    out["range_context_causal"] = pd.to_datetime(out["range_available_time"]) <= minute_index
    return out


def build_market_features(
    trade_bars: pd.DataFrame,
    range_bars: pd.DataFrame,
    *,
    baseline_window: int,
) -> pd.DataFrame:
    """Build causal market-state and order-flow features."""
    if trade_bars.empty:
        return pd.DataFrame()
    bars = trade_bars.sort_index(kind="stable")
    if not bars.index.is_unique:
        bars = bars[~bars.index.duplicated(keep="last")]
    out = build_trade_bar_orderflow_features(bars, baseline_window=baseline_window)

    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    ret_1 = close.pct_change()
    ret_abs = ret_1.abs()

    rv_30 = np.sqrt(ret_1.pow(2).rolling(30, min_periods=30).sum())
    rv_base = _prior_ewm(rv_30, span=1440, min_periods=360)
    out["vol_ratio_30"] = _safe_divide(rv_30, rv_base).clip(0.0, 20.0)
    out["efficiency_15"] = _rolling_efficiency(close, ret_abs, 15)
    out["efficiency_30"] = _rolling_efficiency(close, ret_abs, 30)
    out["efficiency_60"] = _rolling_efficiency(close, ret_abs, 60)
    out["return_15"] = close / close.shift(15) - 1.0
    out["return_30"] = close / close.shift(30) - 1.0

    for window in (15, 30, 60):
        out[f"prior_high_{window}"] = high.shift(1).rolling(window, min_periods=window).max()
        out[f"prior_low_{window}"] = low.shift(1).rolling(window, min_periods=window).min()
    out["width_30"] = _safe_divide(out["prior_high_30"] - out["prior_low_30"], close)
    out["width_60"] = _safe_divide(out["prior_high_60"] - out["prior_low_60"], close)
    out["mid_60"] = (out["prior_high_60"] + out["prior_low_60"]) / 2.0
    out["ema_60"] = close.ewm(span=60, adjust=False, min_periods=60).mean()
    out["upper_wick_frac"] = (
        (high - out[["open", "close"]].max(axis=1)).clip(lower=0.0)
        / (high - low).clip(lower=close.abs() * 1e-9)
    ).clip(0.0, 1.0)

    range_ctx = build_range_context(pd.DatetimeIndex(out.index), range_bars)
    for col in range_ctx.columns:
        out[col] = range_ctx[col]
    return out.replace([np.inf, -np.inf], np.nan)


def build_environment_masks(features: pd.DataFrame, definition: Definition) -> dict[str, pd.Series]:
    f = features
    compression = (
        (f["vol_ratio_30"] <= definition.compression_vol_max)
        & (f["range_count_ratio_15m"] <= definition.compression_speed_max)
        & (f["efficiency_30"] <= definition.compression_efficiency_max)
        & (f["width_30"].between(0.0025, 0.0100))
        & f["range_context_causal"].fillna(False)
    )
    expansion_up = (
        (f["vol_ratio_30"] >= definition.expansion_vol_min)
        & (f["range_count_ratio_15m"] >= definition.expansion_speed_min)
        & (f["efficiency_15"] >= definition.expansion_efficiency_min)
        & (f["return_15"] >= definition.expansion_move_min)
    )
    expansion_down = (
        (f["vol_ratio_30"] >= definition.expansion_vol_min)
        & (f["range_count_ratio_15m"] >= definition.expansion_speed_min)
        & (f["efficiency_15"] >= definition.expansion_efficiency_min)
        & (f["return_15"] <= -definition.expansion_move_min)
    )
    balance = (
        (f["efficiency_60"] <= definition.balance_efficiency_max)
        & (f["vol_ratio_30"] <= definition.balance_vol_max)
        & (f["width_60"].between(0.0040, 0.0200))
        & (f["range_count_ratio_15m"] <= 1.35)
        & f["range_context_causal"].fillna(False)
    )
    return {
        "compression": compression.fillna(False),
        "expansion_up": expansion_up.fillna(False),
        "expansion_down": expansion_down.fillna(False),
        "balance": balance.fillna(False),
    }


def _recent_prior(mask: pd.Series, window: int) -> pd.Series:
    return mask.shift(1).rolling(window, min_periods=1).max().fillna(0.0).astype(bool)


def build_strategy_candidates(features: pd.DataFrame, definition: Definition) -> pd.DataFrame:
    """Build pre-declared strategy candidates from environment + trigger."""
    if features.empty:
        return pd.DataFrame()
    f = features
    env = build_environment_masks(f, definition)
    rows: list[pd.DataFrame] = []

    compression_armed = _recent_prior(env["compression"], 20)
    break_long = (
        compression_armed
        & (f["close"] > f["prior_high_30"] * 1.0002)
        & (f["close"] <= f["prior_high_30"] * 1.0045)
        & (f["delta_ratio_3"] >= definition.breakout_flow_min)
        & (f["large_delta_ratio_3"] >= 0.02)
        & (f["notional_ratio_base"] >= definition.breakout_activity_min)
        & (f["price_return_3"] > 0.0)
        & (f["range_delta_ratio"] >= -0.05)
    )
    break_short = (
        compression_armed
        & (f["close"] < f["prior_low_30"] * 0.9998)
        & (f["close"] >= f["prior_low_30"] * 0.9955)
        & (f["delta_ratio_3"] <= -definition.breakout_flow_min)
        & (f["large_delta_ratio_3"] <= -0.02)
        & (f["notional_ratio_base"] >= definition.breakout_activity_min)
        & (f["price_return_3"] < 0.0)
        & (f["range_delta_ratio"] <= 0.05)
    )
    rows.extend(
        _candidate_parts(
            f,
            definition.name,
            "compression_breakout",
            break_long,
            break_short,
            long_stop=f["prior_low_30"] * 0.9997,
            short_stop=f["prior_high_30"] * 1.0003,
            long_target_ref=f["prior_high_30"],
            short_target_ref=f["prior_low_30"],
            environment="compression",
        )
    )

    expansion_down_recent = _recent_prior(env["expansion_down"], 10)
    expansion_up_recent = _recent_prior(env["expansion_up"], 10)
    exhaust_long = (
        expansion_down_recent
        & (f["low"] < f["prior_low_15"] * 0.9998)
        & (f["sell_notional_ratio_base"] >= 1.8)
        & (f["delta_ratio_3"] <= -definition.exhaustion_flow_min)
        & (f["close_pos"] >= 0.58)
        & (f["lower_wick_frac"] >= 0.16)
        & (f["down_move_norm"] <= 1.45)
        & (f["delta_reversal_short"] >= 0.02)
    )
    exhaust_short = (
        expansion_up_recent
        & (f["high"] > f["prior_high_15"] * 1.0002)
        & (f["buy_notional_ratio_base"] >= 1.8)
        & (f["delta_ratio_3"] >= definition.exhaustion_flow_min)
        & (f["close_pos"] <= 0.42)
        & (f["upper_wick_frac"] >= 0.16)
        & (f["up_move_norm"] <= 1.45)
        & (f["delta_reversal_short"] <= -0.02)
    )
    rows.extend(
        _candidate_parts(
            f,
            definition.name,
            "expansion_exhaustion",
            exhaust_long,
            exhaust_short,
            long_stop=f["low"] * 0.9996,
            short_stop=f["high"] * 1.0004,
            long_target_ref=f["ema_60"],
            short_target_ref=f["ema_60"],
            environment="expansion",
        )
    )

    balance_recent = _recent_prior(env["balance"], 10)
    failed_long = (
        balance_recent
        & (f["low"] < f["prior_low_60"] * 0.9998)
        & (f["close"] > f["prior_low_60"])
        & (f["close_pos"] >= 0.55)
        & (f["delta_ratio_2"] >= definition.failed_auction_flow_min)
        & (f["delta_reversal_short"] >= 0.08)
        & (f["notional_ratio_base"] >= 1.15)
    )
    failed_short = (
        balance_recent
        & (f["high"] > f["prior_high_60"] * 1.0002)
        & (f["close"] < f["prior_high_60"])
        & (f["close_pos"] <= 0.45)
        & (f["delta_ratio_2"] <= -definition.failed_auction_flow_min)
        & (f["delta_reversal_short"] <= -0.08)
        & (f["notional_ratio_base"] >= 1.15)
    )
    rows.extend(
        _candidate_parts(
            f,
            definition.name,
            "balance_failed_auction",
            failed_long,
            failed_short,
            long_stop=f["low"] * 0.9996,
            short_stop=f["high"] * 1.0004,
            long_target_ref=f["mid_60"],
            short_target_ref=f["mid_60"],
            environment="balance",
        )
    )

    parts = [part for part in rows if not part.empty]
    if not parts:
        return pd.DataFrame()
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates.sort_values(["signal_time", "family", "side"], kind="stable").reset_index(drop=True)
    return candidates


def _candidate_parts(
    features: pd.DataFrame,
    definition: str,
    family: str,
    long_mask: pd.Series,
    short_mask: pd.Series,
    *,
    long_stop: pd.Series,
    short_stop: pd.Series,
    long_target_ref: pd.Series,
    short_target_ref: pd.Series,
    environment: str,
) -> list[pd.DataFrame]:
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "vol_ratio_30",
        "range_count_ratio_15m",
        "range_available_time",
        "range_context_causal",
        "delta_ratio_2",
        "delta_ratio_3",
        "large_delta_ratio_3",
        "notional_ratio_base",
        "efficiency_15",
        "efficiency_30",
        "efficiency_60",
        "width_30",
        "width_60",
        "return_15",
    ]
    out: list[pd.DataFrame] = []
    for side, mask, stop, target in (
        (1, long_mask, long_stop, long_target_ref),
        (-1, short_mask, short_stop, short_target_ref),
    ):
        selected = mask.fillna(False)
        if not selected.any():
            continue
        part = features.loc[selected, [c for c in feature_cols if c in features.columns]].copy()
        part.insert(0, "signal_time", part.index)
        part.insert(1, "definition", definition)
        part.insert(2, "family", family)
        part.insert(3, "environment", environment)
        part.insert(4, "side", side)
        part["structural_stop"] = pd.to_numeric(stop.loc[selected], errors="coerce").to_numpy()
        part["target_reference"] = pd.to_numeric(target.loc[selected], errors="coerce").to_numpy()
        out.append(part.reset_index(drop=True))
    return out


def apply_cooldown(candidates: pd.DataFrame, cooldown_minutes: int) -> pd.DataFrame:
    """Remove simultaneous side conflicts, then keep first event per cooldown."""
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy()
    conflict = (
        work.groupby(["definition", "family", "signal_time"])["side"]
        .transform("nunique")
        .gt(1)
    )
    work = work.loc[~conflict].sort_values(["signal_time", "definition", "family", "side"], kind="stable")
    kept: list[int] = []
    cooldown = pd.Timedelta(minutes=max(0, int(cooldown_minutes)))
    for _, group in work.groupby(["definition", "family", "side"], sort=False):
        next_allowed = pd.Timestamp.min
        for idx, ts in zip(group.index, pd.to_datetime(group["signal_time"])):
            if ts >= next_allowed:
                kept.append(int(idx))
                next_allowed = ts + cooldown
    return work.loc[sorted(kept)].reset_index(drop=True)


