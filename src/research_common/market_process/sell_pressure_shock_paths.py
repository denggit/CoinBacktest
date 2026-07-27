#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal sell-pressure shock and price-response path utilities.

The module studies a deliberately broad hypothesis:

    abrupt aggressive selling can lead either to continuation or reversal;
    the differentiator may be the contemporaneous/subsequent price response.

No file or database access belongs here. Research scripts must obtain data via
``src.data_feed``. Every feature at row ``t`` uses only bars closed by ``t``.
Post-shock reclaim/acceptance events are emitted when they become observable and
are tradable only from the next bar open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.research_common.market_process.broad_order_flow_paths import (
    BAND_NAMES,
    OutcomeArrays,
    build_outcome_arrays,
    directional_outcomes,
    pressure_band_codes,
    rolling_pressure_ratio,
)

FLOW_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 15, 30)
HORIZONS: tuple[int, ...] = (5, 15, 30, 60, 120)
SHOCK_TYPES: tuple[str, ...] = (
    "sell_band_entry",
    "sell_strengthening",
    "buy_to_sell_reversal",
)
RECLAIM_WAITS: tuple[int, ...] = (3, 5, 10, 15)
ACCEPTANCE_BARS: tuple[int, ...] = (2, 3)
ACTIVITY_THRESHOLDS: tuple[float, ...] = (1.5, 2.5)


@dataclass(frozen=True)
class SellShockArrays:
    event_mask: np.ndarray
    current_pressure: np.ndarray
    prior_pressure: np.ndarray
    pressure_change: np.ndarray
    band_code: np.ndarray


@dataclass(frozen=True)
class SellShockPA:
    window_open: np.ndarray
    window_high: np.ndarray
    window_low: np.ndarray
    window_close: np.ndarray
    prior_close: np.ndarray
    prior_low_30: np.ndarray
    window_return: np.ndarray
    downside_excursion: np.ndarray
    lower_wick_fraction: np.ndarray
    close_recovery_fraction: np.ndarray
    downside_impulse: np.ndarray
    lower_wick: np.ndarray
    deep_lower_wick: np.ndarray
    prior_low_sweep: np.ndarray
    same_window_sweep_reclaim: np.ndarray
    sweep_without_reclaim: np.ndarray


@dataclass(frozen=True)
class PostShockEvents:
    delayed_reclaim: Mapping[int, np.ndarray]
    breakdown_acceptance: Mapping[int, np.ndarray]
    source_shock_index: np.ndarray
    reference_level: np.ndarray


def _float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _rolling_sum(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return trailing sum and finite count in O(N)."""
    x = np.asarray(values, dtype=float)
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    n = len(x)
    total = np.full(n, np.nan, dtype=float)
    count_out = np.zeros(n, dtype=np.int64)
    if n < w:
        return total, count_out
    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(safe, dtype=float)))
    ccount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    sums = csum[w:] - csum[:-w]
    counts = ccount[w:] - ccount[:-w]
    total[w - 1 :] = sums
    count_out[w - 1 :] = counts
    return total, count_out


def rolling_activity_ratio(
    notional: pd.Series | np.ndarray,
    window: int,
    baseline_minutes: int = 240,
) -> np.ndarray:
    """Current-window average notional / preceding baseline average.

    The baseline ends immediately before the current equal-length shock window,
    so it never includes the candidate window itself.
    """
    x = _float_array(notional)
    w = int(window)
    b = int(baseline_minutes)
    if w <= 0 or b <= 0:
        raise ValueError("window and baseline_minutes must be positive")
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    current_sum, current_count = _rolling_sum(x, w)
    baseline_sum, baseline_count = _rolling_sum(x, b)
    shifted_sum = np.full(n, np.nan, dtype=float)
    shifted_count = np.zeros(n, dtype=np.int64)
    if w < n:
        shifted_sum[w:] = baseline_sum[:-w]
        shifted_count[w:] = baseline_count[:-w]
    good = (
        (current_count == w)
        & (shifted_count == b)
        & np.isfinite(current_sum)
        & np.isfinite(shifted_sum)
        & (shifted_sum > 0.0)
    )
    out[good] = (current_sum[good] / float(w)) / (shifted_sum[good] / float(b))
    return out


def _rising_edge(condition: np.ndarray) -> np.ndarray:
    cond = np.asarray(condition, dtype=bool)
    out = cond.copy()
    if len(out):
        out[0] = bool(cond[0])
        out[1:] &= ~cond[:-1]
    return out


def build_sell_shock_arrays(pressure: np.ndarray, window: int) -> dict[str, SellShockArrays]:
    """Build broad adjacent-window sell-pressure shock events.

    Current pressure is compared with the immediately preceding non-overlapping
    equal-length window. Only rising edges are emitted; no arbitrary cooldown or
    market-state filter is applied.
    """
    p = np.asarray(pressure, dtype=float)
    w = int(window)
    prior = np.full(len(p), np.nan, dtype=float)
    if w < len(p):
        prior[w:] = p[:-w]
    current_band = pressure_band_codes(p)
    prior_band = pressure_band_codes(prior)
    valid = np.isfinite(p) & np.isfinite(prior)
    current_sell = current_band < 0
    prior_sell = prior_band < 0
    prior_buy = prior_band > 0

    previous_bar_band = np.roll(current_band, 1)
    if len(previous_bar_band):
        previous_bar_band[0] = 0

    definitions = {
        "sell_band_entry": valid & current_sell & (previous_bar_band >= 0),
        "sell_strengthening": valid & current_sell & prior_sell & (np.abs(current_band) > np.abs(prior_band)),
        "buy_to_sell_reversal": valid & current_sell & prior_buy,
    }
    out: dict[str, SellShockArrays] = {}
    for name, condition in definitions.items():
        mask = _rising_edge(condition)
        out[name] = SellShockArrays(
            event_mask=mask,
            current_pressure=p,
            prior_pressure=prior,
            pressure_change=p - prior,
            band_code=current_band,
        )
    return out


def build_sell_shock_pa(bars: pd.DataFrame, window: int) -> SellShockPA:
    """Build aggregated shock-window PA and a pre-window prior-low reference."""
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"missing OHLC fields: {missing}")
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")

    open_ = pd.to_numeric(bars["open"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")

    window_open = open_.shift(w - 1)
    window_high = high.rolling(w, min_periods=w).max()
    window_low = low.rolling(w, min_periods=w).min()
    prior_close = close.shift(w)
    # At t, this reference ends at t-w and therefore excludes the entire shock window.
    prior_low_30 = low.shift(w).rolling(30, min_periods=30).min()

    total_range = (window_high - window_low).replace(0.0, np.nan)
    lower_body = pd.concat([window_open, close], axis=1).min(axis=1)
    lower_wick_fraction = (lower_body - window_low) / total_range
    close_recovery_fraction = (close - window_low) / total_range
    window_return = close / prior_close - 1.0
    downside_excursion = window_low / prior_close - 1.0

    sweep = window_low < prior_low_30
    reclaim = sweep & (close > prior_low_30)
    no_reclaim = sweep & (close <= prior_low_30)
    downside_impulse = (window_return < 0.0) & (downside_excursion < 0.0)
    lower_wick = (lower_wick_fraction >= 0.35) & (close_recovery_fraction >= 0.55)
    deep_lower_wick = (lower_wick_fraction >= 0.50) & (close_recovery_fraction >= 0.65)

    def arr(series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    return SellShockPA(
        window_open=arr(window_open),
        window_high=arr(window_high),
        window_low=arr(window_low),
        window_close=arr(close),
        prior_close=arr(prior_close),
        prior_low_30=arr(prior_low_30),
        window_return=arr(window_return),
        downside_excursion=arr(downside_excursion),
        lower_wick_fraction=arr(lower_wick_fraction),
        close_recovery_fraction=arr(close_recovery_fraction),
        downside_impulse=downside_impulse.fillna(False).to_numpy(dtype=bool),
        lower_wick=lower_wick.fillna(False).to_numpy(dtype=bool),
        deep_lower_wick=deep_lower_wick.fillna(False).to_numpy(dtype=bool),
        prior_low_sweep=sweep.fillna(False).to_numpy(dtype=bool),
        same_window_sweep_reclaim=reclaim.fillna(False).to_numpy(dtype=bool),
        sweep_without_reclaim=no_reclaim.fillna(False).to_numpy(dtype=bool),
    )


def build_post_shock_events(
    close: pd.Series | np.ndarray,
    shock_mask: np.ndarray,
    sweep_without_reclaim: np.ndarray,
    reference_level: np.ndarray,
    reclaim_waits: Iterable[int] = RECLAIM_WAITS,
    acceptance_bars: Iterable[int] = ACCEPTANCE_BARS,
) -> PostShockEvents:
    """Emit causal delayed-reclaim and breakdown-acceptance signal bars.

    A single active unresolved shock is maintained. A newer qualifying shock
    supersedes the older one, preventing multiple historical shocks from being
    credited to one later reclaim. Signals are stamped on the confirmation bar.
    """
    close_v = _float_array(close)
    shock = np.asarray(shock_mask, dtype=bool)
    unresolved = np.asarray(sweep_without_reclaim, dtype=bool)
    ref = np.asarray(reference_level, dtype=float)
    if not (len(close_v) == len(shock) == len(unresolved) == len(ref)):
        raise ValueError("all arrays must have the same length")

    waits = tuple(sorted({int(x) for x in reclaim_waits if int(x) > 0}))
    accepts = tuple(sorted({int(x) for x in acceptance_bars if int(x) > 0}))
    if not waits or not accepts:
        raise ValueError("reclaim_waits and acceptance_bars must be non-empty positive integers")
    max_wait = max(waits)
    max_accept = max(accepts)

    reclaim_events = {wait: np.zeros(len(close_v), dtype=bool) for wait in waits}
    acceptance_events = {count: np.zeros(len(close_v), dtype=bool) for count in accepts}
    source_index = np.full(len(close_v), -1, dtype=np.int64)
    source_ref = np.full(len(close_v), np.nan, dtype=float)

    active_start = -1
    active_ref = np.nan
    below_count = 0
    accepted_counts: set[int] = set()

    for i in range(len(close_v)):
        # Newer unresolved shock supersedes an older one by design.
        if shock[i] and unresolved[i] and np.isfinite(ref[i]) and np.isfinite(close_v[i]):
            active_start = i
            active_ref = float(ref[i])
            below_count = 1 if close_v[i] <= active_ref else 0
            accepted_counts.clear()

        if active_start < 0:
            continue
        elapsed = i - active_start
        if elapsed > max_wait:
            active_start = -1
            active_ref = np.nan
            below_count = 0
            accepted_counts.clear()
            continue
        if not np.isfinite(close_v[i]):
            continue

        if i > active_start:
            if close_v[i] > active_ref:
                for wait in waits:
                    if elapsed <= wait:
                        reclaim_events[wait][i] = True
                source_index[i] = active_start
                source_ref[i] = active_ref
                active_start = -1
                active_ref = np.nan
                below_count = 0
                accepted_counts.clear()
                continue
            below_count = below_count + 1 if close_v[i] <= active_ref else 0

        for count in accepts:
            if below_count >= count and count not in accepted_counts and elapsed <= max(max_wait, max_accept):
                acceptance_events[count][i] = True
                source_index[i] = active_start
                source_ref[i] = active_ref
                accepted_counts.add(count)

    return PostShockEvents(
        delayed_reclaim=reclaim_events,
        breakdown_acceptance=acceptance_events,
        source_shock_index=source_index,
        reference_level=source_ref,
    )


def fixed_side_array(length: int, side: int) -> np.ndarray:
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    return np.full(int(length), int(side), dtype=np.int8)


__all__ = [
    "ACCEPTANCE_BARS",
    "ACTIVITY_THRESHOLDS",
    "BAND_NAMES",
    "FLOW_WINDOWS",
    "HORIZONS",
    "RECLAIM_WAITS",
    "SHOCK_TYPES",
    "OutcomeArrays",
    "PostShockEvents",
    "SellShockArrays",
    "SellShockPA",
    "build_outcome_arrays",
    "build_post_shock_events",
    "build_sell_shock_arrays",
    "build_sell_shock_pa",
    "directional_outcomes",
    "fixed_side_array",
    "pressure_band_codes",
    "rolling_activity_ratio",
    "rolling_pressure_ratio",
]
