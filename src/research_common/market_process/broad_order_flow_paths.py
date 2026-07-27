#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Broad, causal order-flow path atlas utilities.

The module deliberately starts from high-coverage order-flow observations and
adds one price-action context at a time.  It does not access files, databases or
exchange APIs; research scripts must obtain data through ``src.data_feed``.

Timing contract
---------------
- Every feature at row ``t`` uses data available when the 1m bar at ``t`` closes.
- Prior-trend and prior-extreme context exclude the current bar via ``shift(1)``.
- A signal at ``t`` is evaluated from the next bar open.
- Forward highs/lows/closes are labels only and never enter event construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

FLOW_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 15, 30, 60)
HORIZONS: tuple[int, ...] = (5, 15, 30, 60, 120)
PA_CONTEXTS: tuple[str, ...] = (
    "all",
    "trend_aligned",
    "trend_opposed",
    "sweep_reclaim",
    "breakout_acceptance",
)
TRANSITION_TYPES: tuple[str, ...] = (
    "band_entry_follow",
    "strengthening_follow",
    "weakening_follow",
    "weakening_fade",
    "reversal_follow",
)

# Fixed, intentionally coarse bins.  They are not estimated on the research or
# holdout sample, so the event definitions do not look into the future.
PRESSURE_EDGES: tuple[float, ...] = (0.03, 0.08, 0.16)
BAND_CODES: tuple[int, ...] = (-4, -3, -2, -1, 1, 2, 3, 4)
BAND_NAMES: Mapping[int, str] = {
    -4: "strong_sell",
    -3: "moderate_sell",
    -2: "mild_sell",
    -1: "weak_sell",
    1: "weak_buy",
    2: "mild_buy",
    3: "moderate_buy",
    4: "strong_buy",
}


@dataclass(frozen=True)
class OutcomeArrays:
    gross_long: np.ndarray
    high_from_entry: np.ndarray
    low_from_entry: np.ndarray
    entry_open: np.ndarray


@dataclass(frozen=True)
class TransitionArrays:
    event_mask: np.ndarray
    trade_side: np.ndarray
    flow_side: np.ndarray
    band_code: np.ndarray
    prior_pressure: np.ndarray
    pressure_change: np.ndarray


def _as_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def rolling_pressure_ratio(
    delta_notional: pd.Series | np.ndarray,
    notional: pd.Series | np.ndarray,
    window: int,
) -> np.ndarray:
    """Return rolling ``sum(delta) / sum(notional)`` in O(N).

    The implementation uses cumulative sums instead of repeated pandas rolling
    objects.  The first ``window - 1`` rows are NaN and no forward information
    is used.
    """
    w = int(window)
    if w <= 0:
        raise ValueError("window must be positive")
    delta = _as_float_array(delta_notional)
    total = _as_float_array(notional)
    if delta.shape != total.shape:
        raise ValueError("delta_notional and notional must have the same shape")

    n = len(delta)
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out

    finite = np.isfinite(delta) & np.isfinite(total)
    d = np.where(finite, delta, 0.0)
    t = np.where(finite, total, 0.0)
    count = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    dsum = np.concatenate(([0.0], np.cumsum(d, dtype=float)))
    tsum = np.concatenate(([0.0], np.cumsum(t, dtype=float)))

    valid_count = count[w:] - count[:-w]
    numer = dsum[w:] - dsum[:-w]
    denom = tsum[w:] - tsum[:-w]
    good = (valid_count == w) & np.isfinite(denom) & (denom > 0.0)
    values = np.full(n - w + 1, np.nan, dtype=float)
    values[good] = numer[good] / denom[good]
    out[w - 1 :] = np.clip(values, -1.0, 1.0)
    return out


def pressure_band_codes(pressure: pd.Series | np.ndarray) -> np.ndarray:
    """Map pressure to signed fixed bands ``-3..-1, 0, 1..3``."""
    x = _as_float_array(pressure)
    out = np.zeros(len(x), dtype=np.int8)
    finite = np.isfinite(x)
    abs_x = np.abs(x)
    level = np.zeros(len(x), dtype=np.int8)
    level[finite & (abs_x > 0.0)] = 1
    level[finite & (abs_x >= PRESSURE_EDGES[0])] = 2
    level[finite & (abs_x >= PRESSURE_EDGES[1])] = 3
    level[finite & (abs_x >= PRESSURE_EDGES[2])] = 4
    sign = np.zeros(len(x), dtype=np.int8)
    sign[finite] = np.sign(x[finite]).astype(np.int8)
    out = (sign * level).astype(np.int8)
    return out


def build_pressure_paths(
    bars: pd.DataFrame,
    windows: Iterable[int] = FLOW_WINDOWS,
) -> dict[int, np.ndarray]:
    """Build causal rolling pressure arrays for all requested windows."""
    if "delta_notional" not in bars.columns or "notional" not in bars.columns:
        raise ValueError("trade bars require delta_notional and notional")
    return {
        int(window): rolling_pressure_ratio(bars["delta_notional"], bars["notional"], int(window))
        for window in windows
    }


def build_pa_context_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build simple causal PA arrays without order-flow conditions.

    ``prior_trend_return_60`` and prior 30m highs/lows exclude the signal bar.
    Sweep/reclaim and breakout/acceptance use the just-closed signal bar and are
    therefore tradable only from the next bar open.
    """
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"missing OHLC fields: {missing}")

    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    prior_trend = close.shift(1) / close.shift(61) - 1.0
    prior_high_30 = high.shift(1).rolling(30, min_periods=30).max()
    prior_low_30 = low.shift(1).rolling(30, min_periods=30).min()

    sweep_long = (low < prior_low_30) & (close > prior_low_30)
    sweep_short = (high > prior_high_30) & (close < prior_high_30)
    breakout_long = close > prior_high_30
    breakout_short = close < prior_low_30

    return {
        "prior_trend_return_60": prior_trend.to_numpy(dtype=float),
        "prior_high_30": prior_high_30.to_numpy(dtype=float),
        "prior_low_30": prior_low_30.to_numpy(dtype=float),
        "sweep_long": sweep_long.fillna(False).to_numpy(dtype=bool),
        "sweep_short": sweep_short.fillna(False).to_numpy(dtype=bool),
        "breakout_long": breakout_long.fillna(False).to_numpy(dtype=bool),
        "breakout_short": breakout_short.fillna(False).to_numpy(dtype=bool),
    }


def pa_context_mask(
    context: str,
    trade_side: np.ndarray,
    pa: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return one independent PA overlay for the supplied trade direction."""
    side = np.asarray(trade_side, dtype=np.int8)
    if context == "all":
        return side != 0
    trend = np.asarray(pa["prior_trend_return_60"], dtype=float)
    if context == "trend_aligned":
        return (side != 0) & np.isfinite(trend) & (side * trend > 0.0)
    if context == "trend_opposed":
        return (side != 0) & np.isfinite(trend) & (side * trend < 0.0)
    if context == "sweep_reclaim":
        return np.where(side > 0, pa["sweep_long"], np.where(side < 0, pa["sweep_short"], False))
    if context == "breakout_acceptance":
        return np.where(
            side > 0,
            pa["breakout_long"],
            np.where(side < 0, pa["breakout_short"], False),
        )
    raise ValueError(f"unknown PA context: {context}")


def _rising_edge(condition: np.ndarray) -> np.ndarray:
    cond = np.asarray(condition, dtype=bool)
    out = cond.copy()
    if len(out):
        out[1:] &= ~cond[:-1]
    return out


def build_transition_arrays(pressure: np.ndarray, window: int) -> dict[str, TransitionArrays]:
    """Build broad adjacent-window pressure path events.

    The current rolling window is compared with the immediately preceding
    equal-length window via ``pressure.shift(window)``.  Conditions are reduced
    to rising edges only; no arbitrary cooldown or extra market filter is used.
    """
    p = np.asarray(pressure, dtype=float)
    w = int(window)
    prior = np.full(len(p), np.nan, dtype=float)
    if w < len(p):
        prior[w:] = p[:-w]

    current_band = pressure_band_codes(p)
    prior_band = pressure_band_codes(prior)
    current_sign = np.sign(current_band).astype(np.int8)
    prior_sign = np.sign(prior_band).astype(np.int8)
    current_abs = np.abs(current_band)
    prior_abs = np.abs(prior_band)
    valid_pair = (current_band != 0) & (prior_band != 0)
    same_sign = valid_pair & (current_sign == prior_sign)
    opposite_sign = valid_pair & (current_sign != prior_sign)
    change = p - prior

    band_entry_cond = (current_band != 0) & (current_band != np.roll(current_band, 1))
    if len(band_entry_cond):
        band_entry_cond[0] = False
    strengthening_cond = same_sign & (current_abs > prior_abs)
    weakening_cond = same_sign & (current_abs < prior_abs)
    reversal_cond = opposite_sign

    definitions = {
        "band_entry_follow": (band_entry_cond, current_sign),
        "strengthening_follow": (_rising_edge(strengthening_cond), current_sign),
        "weakening_follow": (_rising_edge(weakening_cond), current_sign),
        "weakening_fade": (_rising_edge(weakening_cond), -current_sign),
        "reversal_follow": (_rising_edge(reversal_cond), current_sign),
    }
    out: dict[str, TransitionArrays] = {}
    for name, (mask, trade_side) in definitions.items():
        out[name] = TransitionArrays(
            event_mask=mask.astype(bool),
            trade_side=np.asarray(trade_side, dtype=np.int8),
            flow_side=current_sign,
            band_code=current_band,
            prior_pressure=prior,
            pressure_change=change,
        )
    return out


def build_outcome_arrays(bars: pd.DataFrame, horizon: int) -> OutcomeArrays:
    """Build vectorized next-open path labels for one horizon."""
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive")
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")

    entry_open = open_.shift(-1).to_numpy(dtype=float)
    future_close = close.shift(-h).to_numpy(dtype=float)
    gross_long = future_close / entry_open - 1.0

    indexer = FixedForwardWindowIndexer(window_size=h)
    high_from_entry = high.shift(-1).rolling(indexer, min_periods=h).max().to_numpy(dtype=float)
    low_from_entry = low.shift(-1).rolling(indexer, min_periods=h).min().to_numpy(dtype=float)
    return OutcomeArrays(
        gross_long=gross_long,
        high_from_entry=high_from_entry,
        low_from_entry=low_from_entry,
        entry_open=entry_open,
    )


def directional_outcomes(
    labels: OutcomeArrays,
    trade_side: np.ndarray,
    round_trip_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gross, net, MFE and MAE arrays for long/short directions."""
    side = np.asarray(trade_side, dtype=np.int8)
    gross = labels.gross_long * side
    net = gross - float(round_trip_cost)

    entry = labels.entry_open
    long_mfe = labels.high_from_entry / entry - 1.0
    long_mae = labels.low_from_entry / entry - 1.0
    short_mfe = entry / labels.low_from_entry - 1.0
    short_mae = entry / labels.high_from_entry - 1.0
    mfe = np.where(side > 0, long_mfe, np.where(side < 0, short_mfe, np.nan))
    mae = np.where(side > 0, long_mae, np.where(side < 0, short_mae, np.nan))
    return gross, net, mfe, mae


def band_index(codes: np.ndarray) -> np.ndarray:
    """Map signed band codes to compact indexes 0..5; invalid/neutral -> -1."""
    code = np.asarray(codes, dtype=np.int8)
    out = np.full(len(code), -1, dtype=np.int8)
    neg = code < 0
    pos = code > 0
    out[neg] = code[neg] + 4  # -4,-3,-2,-1 -> 0,1,2,3
    out[pos] = code[pos] + 3  # 1,2,3,4 -> 4,5,6,7
    return out


def sufficient_stats_by_band(
    band_codes: np.ndarray,
    selection: np.ndarray,
    gross: np.ndarray,
    net: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Aggregate mergeable statistics by signed pressure band with bincount."""
    group = band_index(band_codes)
    select = (
        np.asarray(selection, dtype=bool)
        & (group >= 0)
        & np.isfinite(gross)
        & np.isfinite(net)
        & np.isfinite(mfe)
        & np.isfinite(mae)
    )
    if not select.any():
        return []

    g = group[select].astype(np.int64)
    gross_v = np.asarray(gross, dtype=float)[select]
    net_v = np.asarray(net, dtype=float)[select]
    mfe_v = np.asarray(mfe, dtype=float)[select]
    mae_v = np.asarray(mae, dtype=float)[select]
    minlength = len(BAND_CODES)

    def bc(weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(g, weights=weights, minlength=minlength).astype(float)

    count = bc()
    sum_gross = bc(gross_v)
    sum_net = bc(net_v)
    sum_sq_net = bc(net_v * net_v)
    wins = bc((net_v > 0.0).astype(float))
    gross_gains = bc(np.where(gross_v > 0.0, gross_v, 0.0))
    gross_losses = bc(np.where(gross_v < 0.0, -gross_v, 0.0))
    net_gains = bc(np.where(net_v > 0.0, net_v, 0.0))
    net_losses = bc(np.where(net_v < 0.0, -net_v, 0.0))
    sum_mfe = bc(mfe_v)
    sum_mae = bc(mae_v)

    rows: list[dict[str, float | int | str]] = []
    for i, code in enumerate(BAND_CODES):
        if count[i] <= 0:
            continue
        rows.append(
            {
                "pressure_band": BAND_NAMES[code],
                "band_code": int(code),
                "flow_side": "BUY" if code > 0 else "SELL",
                "events": int(count[i]),
                "sum_gross": float(sum_gross[i]),
                "sum_net": float(sum_net[i]),
                "sum_sq_net": float(sum_sq_net[i]),
                "wins": int(wins[i]),
                "gross_gains": float(gross_gains[i]),
                "gross_losses": float(gross_losses[i]),
                "net_gains": float(net_gains[i]),
                "net_losses": float(net_losses[i]),
                "sum_mfe": float(sum_mfe[i]),
                "sum_mae": float(sum_mae[i]),
            }
        )
    return rows


def finalize_sufficient_stats(frame: pd.DataFrame, months_in_window: int) -> pd.DataFrame:
    """Convert mergeable components to readable performance statistics."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    n = pd.to_numeric(out["events"], errors="coerce").clip(lower=1.0)
    out["events_per_month"] = n / max(1, int(months_in_window))
    out["mean_gross"] = out["sum_gross"] / n
    out["mean_net"] = out["sum_net"] / n
    out["win_rate_net"] = out["wins"] / n
    out["profit_factor_gross"] = out["gross_gains"] / out["gross_losses"].replace(0.0, np.nan)
    out["profit_factor_net"] = out["net_gains"] / out["net_losses"].replace(0.0, np.nan)
    out["mean_mfe"] = out["sum_mfe"] / n
    out["mean_mae"] = out["sum_mae"] / n
    variance = (out["sum_sq_net"] - (out["sum_net"] ** 2) / n) / (n - 1.0).replace(0.0, np.nan)
    std = np.sqrt(variance.clip(lower=0.0))
    out["t_stat_naive"] = out["mean_net"] / (std / np.sqrt(n)).replace(0.0, np.nan)
    return out


def combine_sufficient_stats(
    yearly_components: pd.DataFrame,
    group_cols: list[str],
    months_in_window: int,
) -> pd.DataFrame:
    """Combine per-chunk sufficient statistics without retaining all events."""
    if yearly_components.empty:
        return pd.DataFrame()
    sum_cols = [
        "events",
        "sum_gross",
        "sum_net",
        "sum_sq_net",
        "wins",
        "gross_gains",
        "gross_losses",
        "net_gains",
        "net_losses",
        "sum_mfe",
        "sum_mae",
    ]
    combined = yearly_components.groupby(group_cols, dropna=False, sort=True)[sum_cols].sum().reset_index()
    combined["events"] = combined["events"].astype(int)
    combined["wins"] = combined["wins"].astype(int)
    return finalize_sufficient_stats(combined, months_in_window)


def add_cross_year_diagnostics(overall: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """Attach minimum yearly sample and direction consistency diagnostics."""
    if overall.empty:
        return overall.copy()
    keys = [
        "atlas_type",
        "event_type",
        "flow_window",
        "pressure_band",
        "band_code",
        "trade_side",
        "pa_context",
        "horizon",
    ]
    year_diag = (
        yearly.groupby(keys, dropna=False, sort=False)
        .agg(
            years_present=("year", "nunique"),
            positive_net_years=("mean_net", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            positive_gross_years=("mean_gross", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            min_year_events=("events", "min"),
        )
        .reset_index()
    )
    return overall.merge(year_diag, on=keys, how="left", validate="one_to_one")


def build_incremental_pa_table(overall: pd.DataFrame) -> pd.DataFrame:
    """Compare every single PA overlay with its pressure-only parent."""
    if overall.empty:
        return pd.DataFrame()
    keys = [
        "atlas_type",
        "event_type",
        "flow_window",
        "pressure_band",
        "band_code",
        "trade_side",
        "horizon",
    ]
    parent_cols = keys + [
        "events",
        "mean_gross",
        "mean_net",
        "profit_factor_net",
        "win_rate_net",
        "mean_mfe",
        "mean_mae",
        "positive_net_years",
        "min_year_events",
    ]
    parent = overall[overall["pa_context"] == "all"][parent_cols].copy()
    parent = parent.rename(columns={c: f"parent_{c}" for c in parent_cols if c not in keys})
    child = overall[overall["pa_context"] != "all"].copy()
    merged = child.merge(parent, on=keys, how="left", validate="many_to_one")
    merged["retention_vs_parent"] = merged["events"] / merged["parent_events"].replace(0.0, np.nan)
    merged["delta_mean_gross"] = merged["mean_gross"] - merged["parent_mean_gross"]
    merged["delta_mean_net"] = merged["mean_net"] - merged["parent_mean_net"]
    merged["delta_pf_net"] = merged["profit_factor_net"] - merged["parent_profit_factor_net"]
    merged["delta_win_rate_net"] = merged["win_rate_net"] - merged["parent_win_rate_net"]
    return merged
