#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-scale absorption and repeated-defense research primitives.

The module models absorption as a *process*, not as one finished candle.  It is
research-facing and deliberately symmetric for sell/buy pressure:

* active taker pressure is measured from trade-bar buy/sell notional;
* price response is measured across rolling multi-bar windows;
* impact-decay asks whether similar/stronger pressure produces less progress;
* floor/ceiling defense counts distinct tests of a previously-known extreme;
* spring / upthrust paths require a break and causal reclaim before the signal.

Every feature at row ``t`` uses only rows ``<= t``.  A left-labelled bar is
therefore usable only after that bar closes; callers should execute no earlier
than the next bar open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

_EPS = 1e-12


@dataclass(frozen=True)
class AbsorptionFeatureConfig:
    """One causal feature specification on a fixed bar axis."""

    process_window: int
    baseline_bars: int
    baseline_min_periods: int
    floor_lookback: int
    defense_lookback: int
    reclaim_bars: int = 3
    atr_lookback: int = 48

    def validate(self) -> None:
        values = {
            "process_window": self.process_window,
            "baseline_bars": self.baseline_bars,
            "baseline_min_periods": self.baseline_min_periods,
            "floor_lookback": self.floor_lookback,
            "defense_lookback": self.defense_lookback,
            "reclaim_bars": self.reclaim_bars,
            "atr_lookback": self.atr_lookback,
        }
        if any(int(v) <= 0 for v in values.values()):
            raise ValueError(f"all window lengths must be positive: {values}")
        if self.baseline_min_periods > self.baseline_bars:
            raise ValueError("baseline_min_periods must be <= baseline_bars")
        if self.reclaim_bars > self.floor_lookback:
            raise ValueError("reclaim_bars must be <= floor_lookback")


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def validate_absorption_input(frame: pd.DataFrame) -> None:
    required = {
        "open",
        "high",
        "low",
        "close",
        "notional",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "trades_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"absorption research requires trade-bar fields: missing={missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("absorption input must use DatetimeIndex")


def resample_trade_bars(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate a causal trade-bar frame to a larger left-labelled bar axis."""
    validate_absorption_input(frame)
    source = frame.sort_index().copy()
    source.index = pd.to_datetime(source.index)
    source = source[~source.index.duplicated(keep="last")]

    sum_columns = [
        "volume",
        "notional",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "trades_count",
        "buy_trades_count",
        "sell_trades_count",
        "buy_volume",
        "sell_volume",
        "delta_volume",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
        "large_trades_count",
        "large_buy_trades_count",
        "large_sell_trades_count",
    ]
    max_columns = ["max_trade_notional", "max_trade_size"]
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for column in sum_columns:
        if column in source.columns:
            agg[column] = "sum"
    for column in max_columns:
        if column in source.columns:
            agg[column] = "max"
    if "vwap" in source.columns:
        # Recomputed below from quote/base when possible; last is only a fallback.
        agg["vwap"] = "last"
    out = source.resample(rule, label="left", closed="left").agg(agg)
    out = out[out["close"].notna()].copy()

    if "volume" in out.columns:
        out["vwap"] = _safe_div(out["notional"], out["volume"])
    if "notional" in out.columns:
        out["taker_buy_ratio"] = _safe_div(out["buy_notional"], out["notional"])
        out["delta_notional"] = out["buy_notional"] - out["sell_notional"]
    if "trades_count" in out.columns:
        out["avg_trade_size"] = _safe_div(out.get("volume", pd.Series(np.nan, index=out.index)), out["trades_count"])
    if "delta_volume" not in out.columns and {"buy_volume", "sell_volume"}.issubset(out.columns):
        out["delta_volume"] = out["buy_volume"] - out["sell_volume"]
    return out


def _rolling_distinct_episode_count(condition: pd.Series, window: int) -> pd.Series:
    cond = condition.fillna(False).astype(bool)
    start = cond & ~cond.shift(1, fill_value=False)
    return start.astype(np.int16).shift(1, fill_value=0).rolling(int(window), min_periods=1).sum()


def _true_range_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _safe_div(tr, prev_close.abs())


def build_absorption_features(frame: pd.DataFrame, config: AbsorptionFeatureConfig) -> pd.DataFrame:
    """Build causal pressure-efficiency, defense and spring/upthrust features."""
    config.validate()
    validate_absorption_input(frame)
    out = frame.sort_index().copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]

    w = int(config.process_window)
    baseline = int(config.baseline_bars)
    baseline_min = int(config.baseline_min_periods)
    floor_lb = int(config.floor_lookback)
    defense_lb = int(config.defense_lookback)
    reclaim = int(config.reclaim_bars)

    open_ = _num(out, "open")
    high = _num(out, "high")
    low = _num(out, "low")
    close = _num(out, "close")
    buy = _num(out, "buy_notional", 0.0).clip(lower=0.0)
    sell = _num(out, "sell_notional", 0.0).clip(lower=0.0)
    total = _num(out, "notional", 0.0).clip(lower=0.0)
    delta = _num(out, "delta_notional")
    delta = delta.where(delta.notna(), buy - sell)
    total = total.where(total > 0.0, buy + sell)

    delta_sum = delta.rolling(w, min_periods=w).sum()
    total_sum = total.rolling(w, min_periods=w).sum()
    pressure = _safe_div(delta_sum, total_sum).clip(-1.0, 1.0)
    flow_side = np.sign(delta_sum).astype(float)
    pressure_mag = pressure.abs()

    historical_mag = pressure_mag.shift(w)
    p_mean = historical_mag.rolling(baseline, min_periods=baseline_min).mean()
    p_std = historical_mag.rolling(baseline, min_periods=baseline_min).std(ddof=0)
    pressure_z = _safe_div(pressure_mag - p_mean, p_std).clip(-8.0, 12.0)

    positive_fraction = delta.gt(0.0).astype(float).rolling(w, min_periods=w).mean()
    negative_fraction = delta.lt(0.0).astype(float).rolling(w, min_periods=w).mean()
    persistence = pd.Series(
        np.where(flow_side > 0.0, positive_fraction, np.where(flow_side < 0.0, negative_fraction, np.nan)),
        index=out.index,
        dtype=float,
    )

    window_open = open_.shift(w - 1)
    raw_ret = _safe_div(close, window_open) - 1.0
    directional_ret = raw_ret * flow_side

    rolling_high = high.rolling(w, min_periods=w).max()
    rolling_low = low.rolling(w, min_periods=w).min()
    buy_progress = _safe_div(rolling_high, window_open) - 1.0
    sell_progress = _safe_div(window_open, rolling_low) - 1.0
    directional_excursion = pd.Series(
        np.where(flow_side > 0.0, buy_progress, np.where(flow_side < 0.0, sell_progress, np.nan)),
        index=out.index,
        dtype=float,
    )

    ret1 = close.pct_change()
    prior_vol = ret1.shift(w).rolling(baseline, min_periods=baseline_min).std(ddof=0) * np.sqrt(float(w))
    prior_vol = prior_vol.clip(lower=1e-7)
    response_norm = _safe_div(directional_ret, prior_vol)
    excursion_norm = _safe_div(directional_excursion, prior_vol)
    impact_efficiency = _safe_div(response_norm, pressure_mag.clip(lower=1e-4))

    prior_pressure = pressure.shift(w)
    prior_pressure_mag = pressure_mag.shift(w)
    prior_response_norm = response_norm.shift(w)
    same_side_adjacent = (np.sign(prior_pressure) == np.sign(pressure)) & pressure.notna() & prior_pressure.notna()
    pressure_retention = _safe_div(pressure_mag, prior_pressure_mag)
    response_retention = _safe_div(response_norm.clip(lower=0.0), prior_response_norm.clip(lower=0.05))

    tr_pct = _true_range_pct(high, low, close)
    atr_pct = tr_pct.shift(1).rolling(int(config.atr_lookback), min_periods=max(3, int(config.atr_lookback) // 3)).median()
    atr_pct = atr_pct.clip(lower=1e-6)

    prior_floor = low.shift(1).rolling(floor_lb, min_periods=floor_lb).min()
    prior_ceiling = high.shift(1).rolling(floor_lb, min_periods=floor_lb).max()
    # Zone tolerance adapts to recent volatility; no future bar participates.
    floor_zone_top = prior_floor * (1.0 + 0.75 * atr_pct)
    floor_accept_low = prior_floor * (1.0 - 0.25 * atr_pct)
    ceiling_zone_bottom = prior_ceiling * (1.0 - 0.75 * atr_pct)
    ceiling_accept_high = prior_ceiling * (1.0 + 0.25 * atr_pct)

    near_floor = (low <= floor_zone_top) & (close >= floor_accept_low)
    near_ceiling = (high >= ceiling_zone_bottom) & (close <= ceiling_accept_high)
    defense_count_long = _rolling_distinct_episode_count(near_floor, defense_lb)
    defense_count_short = _rolling_distinct_episode_count(near_ceiling, defense_lb)

    long_hold = (close >= floor_accept_low).astype(float).where(prior_floor.notna())
    short_hold = (close <= ceiling_accept_high).astype(float).where(prior_ceiling.notna())
    hold_ratio_long = long_hold.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).mean()
    hold_ratio_short = short_hold.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).mean()

    floor_hist_max = prior_floor.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).max()
    floor_hist_min = prior_floor.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).min()
    ceiling_hist_max = prior_ceiling.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).max()
    ceiling_hist_min = prior_ceiling.shift(1).rolling(defense_lb, min_periods=max(2, defense_lb // 4)).min()
    floor_stability_atr = _safe_div((floor_hist_max - floor_hist_min) / close.abs(), atr_pct)
    ceiling_stability_atr = _safe_div((ceiling_hist_max - ceiling_hist_min) / close.abs(), atr_pct)

    # Same-bar spring/upthrust is only known after the bar closes.
    spring_same_bar_long = (low < prior_floor * (1.0 - 0.10 * atr_pct)) & (close > prior_floor)
    spring_same_bar_short = (high > prior_ceiling * (1.0 + 0.10 * atr_pct)) & (close < prior_ceiling)

    # Multi-bar reclaim freezes its reference before the reclaim window begins.
    frozen_floor = low.shift(reclaim).rolling(floor_lb, min_periods=floor_lb).min()
    frozen_ceiling = high.shift(reclaim).rolling(floor_lb, min_periods=floor_lb).max()
    frozen_atr = atr_pct.shift(reclaim - 1)
    recent_low = low.rolling(reclaim, min_periods=reclaim).min()
    recent_high = high.rolling(reclaim, min_periods=reclaim).max()
    spring_reclaim_long_raw = (recent_low < frozen_floor * (1.0 - 0.10 * frozen_atr)) & (close > frozen_floor)
    spring_reclaim_short_raw = (recent_high > frozen_ceiling * (1.0 + 0.10 * frozen_atr)) & (close < frozen_ceiling)
    spring_reclaim_long = spring_reclaim_long_raw & ~spring_reclaim_long_raw.shift(1, fill_value=False)
    spring_reclaim_short = spring_reclaim_short_raw & ~spring_reclaim_short_raw.shift(1, fill_value=False)

    out["process_window"] = w
    out["flow_side"] = flow_side
    out["pressure"] = pressure
    out["pressure_mag"] = pressure_mag
    out["pressure_z"] = pressure_z
    out["flow_persistence"] = persistence
    out["directional_return"] = directional_ret
    out["directional_excursion"] = directional_excursion
    out["prior_volatility"] = prior_vol
    out["price_response_norm"] = response_norm
    out["price_excursion_norm"] = excursion_norm
    out["impact_efficiency"] = impact_efficiency
    out["same_side_adjacent_window"] = same_side_adjacent.fillna(False)
    out["pressure_retention"] = pressure_retention
    out["response_retention"] = response_retention
    out["atr_pct_prior"] = atr_pct
    out["prior_floor"] = prior_floor
    out["prior_ceiling"] = prior_ceiling
    out["near_floor"] = near_floor.fillna(False)
    out["near_ceiling"] = near_ceiling.fillna(False)
    out["prior_defense_count_long"] = defense_count_long
    out["prior_defense_count_short"] = defense_count_short
    out["hold_ratio_long"] = hold_ratio_long
    out["hold_ratio_short"] = hold_ratio_short
    out["floor_stability_atr"] = floor_stability_atr
    out["ceiling_stability_atr"] = ceiling_stability_atr
    out["spring_same_bar_long"] = spring_same_bar_long.fillna(False)
    out["spring_same_bar_short"] = spring_same_bar_short.fillna(False)
    out["spring_reclaim_long"] = spring_reclaim_long.fillna(False)
    out["spring_reclaim_short"] = spring_reclaim_short.fillna(False)
    out["frozen_floor"] = frozen_floor
    out["frozen_ceiling"] = frozen_ceiling

    out["feature_ready"] = (
        pressure_z.notna()
        & persistence.notna()
        & response_norm.notna()
        & atr_pct.notna()
        & prior_floor.notna()
        & prior_ceiling.notna()
    )
    return out


def semantic_absorption_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Return fixed, predeclared process masks without outcome-based tuning."""
    ready = features["feature_ready"].fillna(False).astype(bool)
    z = _num(features, "pressure_z")
    persistence = _num(features, "flow_persistence")
    response = _num(features, "price_response_norm")
    pressure_retention = _num(features, "pressure_retention")
    response_retention = _num(features, "response_retention")
    same_side = features["same_side_adjacent_window"].fillna(False).astype(bool)

    strong = ready & z.ge(1.5) & persistence.ge(0.60)
    stall = strong & response.le(0.25)
    rejection = strong & response.lt(0.0)
    impact_decay = (
        ready
        & z.ge(1.0)
        & persistence.ge(0.55)
        & same_side
        & pressure_retention.ge(0.80)
        & response.le(0.50)
        & response_retention.le(0.50)
    )
    return {
        "strong_pressure_control": strong,
        "pressure_stall": stall,
        "pressure_rejection": rejection,
        "impact_decay": impact_decay,
    }


def response_state(response_norm: pd.Series) -> pd.Series:
    """Fixed semantic bins used to compare absorption vs efficient pressure."""
    x = pd.to_numeric(response_norm, errors="coerce")
    labels = pd.Series("NA", index=x.index, dtype=object)
    labels.loc[x < 0.0] = "rejected_lt0"
    labels.loc[(x >= 0.0) & (x < 0.25)] = "stalled_0_0.25"
    labels.loc[(x >= 0.25) & (x < 0.75)] = "partial_0.25_0.75"
    labels.loc[x >= 0.75] = "effective_ge0.75"
    return labels


def defense_count_bucket(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("NA", index=x.index, dtype=object)
    out.loc[x < 1] = "0"
    out.loc[(x >= 1) & (x < 2)] = "1"
    out.loc[(x >= 2) & (x < 3)] = "2"
    out.loc[x >= 3] = "3+"
    return out


def stability_bucket(values: pd.Series) -> pd.Series:
    """How stationary the defended floor/ceiling was, in prior ATR units."""
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("NA", index=x.index, dtype=object)
    out.loc[x <= 0.50] = "stable_le0.5atr"
    out.loc[(x > 0.50) & (x <= 1.50)] = "moderate_0.5_1.5atr"
    out.loc[x > 1.50] = "drifting_gt1.5atr"
    return out


def _rising_edge(mask: pd.Series) -> pd.Series:
    x = mask.fillna(False).astype(bool)
    return x & ~x.shift(1, fill_value=False)


def extract_events(
    features: pd.DataFrame,
    *,
    scale: str,
    bar_delta: pd.Timedelta,
    floor_lookback_label: str,
) -> pd.DataFrame:
    """Extract absorption, repeated-defense and spring/upthrust events."""
    masks = semantic_absorption_masks(features)
    rows: list[pd.DataFrame] = []
    response_labels = response_state(features["price_response_norm"])

    def collect(mask: pd.Series, pattern: str, trade_side: pd.Series | np.ndarray) -> None:
        idx = mask.fillna(False).to_numpy(dtype=bool)
        if not idx.any():
            return
        part = features.loc[idx].copy()
        side_values = np.asarray(trade_side)
        if len(side_values) == len(features):
            side_values = side_values[idx]
        part["pattern"] = pattern
        part["trade_side"] = np.asarray(side_values, dtype=np.int8)
        rows.append(part)

    flow_side = np.sign(_num(features, "flow_side").fillna(0.0)).astype(np.int8)
    reversal_side = -flow_side
    for pattern, mask in masks.items():
        if pattern == "strong_pressure_control":
            collect(_rising_edge(mask), pattern + "_fade", reversal_side)
            collect(_rising_edge(mask), pattern + "_follow", flow_side)
        else:
            collect(_rising_edge(mask), pattern, reversal_side)

    # Repeated test itself: count each distinct touch episode once.
    floor_touch = _rising_edge(features["near_floor"].fillna(False).astype(bool)) & features["feature_ready"].fillna(False)
    ceiling_touch = _rising_edge(features["near_ceiling"].fillna(False).astype(bool)) & features["feature_ready"].fillna(False)
    collect(floor_touch, "floor_retest", np.ones(len(features), dtype=np.int8))
    collect(ceiling_touch, "ceiling_retest", -np.ones(len(features), dtype=np.int8))
    collect(features["spring_same_bar_long"].astype(bool), "spring_same_bar", np.ones(len(features), dtype=np.int8))
    collect(features["spring_same_bar_short"].astype(bool), "upthrust_same_bar", -np.ones(len(features), dtype=np.int8))
    collect(features["spring_reclaim_long"].astype(bool), "spring_reclaim", np.ones(len(features), dtype=np.int8))
    collect(features["spring_reclaim_short"].astype(bool), "upthrust_reclaim", -np.ones(len(features), dtype=np.int8))

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, axis=0, ignore_index=False).sort_index(kind="stable")
    events["signal_bar_start"] = events.index
    events["signal_time"] = events.index + bar_delta
    events["entry_time"] = events.index + bar_delta
    events["scale"] = scale
    events["floor_lookback_label"] = floor_lookback_label
    events["response_state"] = response_labels.reindex(events.index).to_numpy()

    long = events["trade_side"].astype(int) > 0
    defense_values = pd.Series(
        np.where(long, events["prior_defense_count_long"], events["prior_defense_count_short"]),
        index=events.index,
        dtype=float,
    )
    stability_values = pd.Series(
        np.where(long, events["floor_stability_atr"], events["ceiling_stability_atr"]),
        index=events.index,
        dtype=float,
    )
    hold_values = pd.Series(
        np.where(long, events["hold_ratio_long"], events["hold_ratio_short"]),
        index=events.index,
        dtype=float,
    )
    events["prior_defense_count"] = defense_values.to_numpy()
    events["defense_count_bucket"] = defense_count_bucket(defense_values).to_numpy()
    events["zone_stability_bucket"] = stability_bucket(stability_values).to_numpy()
    events["hold_ratio"] = hold_values.to_numpy()
    return events.reset_index(drop=True)


def attach_forward_outcomes(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizons: Iterable[int],
    round_trip_cost: float,
) -> pd.DataFrame:
    """Attach next-open fixed-horizon return/MFE/MAE outcomes causally.

    Forward path arrays are built once per horizon for the whole bar frame and
    event rows are gathered by position. This keeps multi-year 5s/1m research
    vectorized instead of looping over every event.
    """
    if events.empty:
        return events.copy()
    out = events.copy()
    index = pd.DatetimeIndex(pd.to_datetime(bars.index))
    open_s = _num(bars, "open")
    high_s = _num(bars, "high")
    low_s = _num(bars, "low")
    close_s = _num(bars, "close")
    signal_starts = pd.to_datetime(out["signal_bar_start"])
    positions = index.get_indexer(signal_starts)
    side = pd.to_numeric(out["trade_side"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

    entry_all = open_s.shift(-1).to_numpy(dtype=float)
    valid_pos = positions >= 0
    entry_price = np.full(len(out), np.nan, dtype=float)
    entry_price[valid_pos] = entry_all[positions[valid_pos]]
    out["entry_price"] = entry_price

    for horizon in sorted({int(v) for v in horizons if int(v) > 0}):
        future_close = close_s.shift(-horizon).to_numpy(dtype=float)
        indexer = FixedForwardWindowIndexer(window_size=horizon)
        high_from_entry = high_s.shift(-1).rolling(indexer, min_periods=horizon).max().to_numpy(dtype=float)
        low_from_entry = low_s.shift(-1).rolling(indexer, min_periods=horizon).min().to_numpy(dtype=float)

        gross_long_all = future_close / entry_all - 1.0
        long_mfe_all = high_from_entry / entry_all - 1.0
        long_mae_all = low_from_entry / entry_all - 1.0
        short_mfe_all = -(low_from_entry / entry_all - 1.0)
        short_mae_all = -(high_from_entry / entry_all - 1.0)

        gross = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)
        if valid_pos.any():
            pos = positions[valid_pos]
            s = side[valid_pos].astype(float)
            gross[valid_pos] = gross_long_all[pos] * s
            mfe[valid_pos] = np.where(
                s > 0.0, long_mfe_all[pos], np.where(s < 0.0, short_mfe_all[pos], np.nan)
            )
            mae[valid_pos] = np.where(
                s > 0.0, long_mae_all[pos], np.where(s < 0.0, short_mae_all[pos], np.nan)
            )

        out[f"gross_h{horizon}"] = gross
        out[f"net_h{horizon}"] = gross - float(round_trip_cost)
        out[f"mfe_h{horizon}"] = mfe
        out[f"mae_h{horizon}"] = mae
    return out

