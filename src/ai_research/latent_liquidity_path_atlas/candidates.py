#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal broad candidate generation for micro liquidity-release events.

Admission is the union of several high-recall mechanisms.  None of them requires
an existing Swing High/Low, so the downstream path learner can discover both
Swing-related and non-Swing stop-pool mechanisms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LatentLiquidityPathAtlasConfig
from .time_axis import as_datetime_ns

FLOW_COLUMNS = (
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "trades_count",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "max_trade_notional",
)


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def normalize_second_bars(
    bars: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Return a regular 1-second causal frame and mark unsafe long gaps.

    A missing second is interpreted as no trade only for short gaps.  Long gaps
    are still filled for vectorized calculations but marked through
    ``unsafe_gap`` so candidates whose context/label window crosses them can be
    rejected later.
    """
    if bars.empty:
        return pd.DataFrame()
    out = bars.copy()
    out.index = as_datetime_ns(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index(kind="mergesort")
    out = out.loc[~out.index.duplicated(keep="last")]
    if out.empty:
        return out
    full_index = pd.date_range(out.index.min().floor("s"), out.index.max().floor("s"), freq="1s")
    observed = pd.Series(True, index=out.index)
    out = out.reindex(full_index)
    out["observed_bar"] = observed.reindex(full_index, fill_value=False).astype(bool)

    close = pd.to_numeric(out.get("close"), errors="coerce").ffill()
    open_ = pd.to_numeric(out.get("open"), errors="coerce").fillna(close)
    high = pd.to_numeric(out.get("high"), errors="coerce").fillna(close)
    low = pd.to_numeric(out.get("low"), errors="coerce").fillna(close)
    out["close"] = close
    out["open"] = open_
    out["high"] = high
    out["low"] = low
    for name in FLOW_COLUMNS:
        out[name] = _numeric(out, name, 0.0)

    missing = (~out["observed_bar"]).astype(np.int8)
    # Consecutive-run size, vectorized through run ids.
    run_id = (missing != missing.shift(fill_value=0)).cumsum()
    missing_run = missing.groupby(run_id).transform("sum")
    out["unsafe_gap"] = (missing.eq(1) & missing_run.gt(config.max_fill_gap_seconds)).astype(np.int8)
    out.index.name = "timestamp"
    return out


def _prior_zscore(series: pd.Series, window: int) -> pd.Series:
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=max(30, window // 3)).mean()
    std = prior.rolling(window, min_periods=max(30, window // 3)).std(ddof=0)
    fallback = mean.abs().mul(0.01).clip(lower=1e-9)
    scale = std.where(std.gt(1e-12), fallback)
    return (series - mean) / scale


def _signed_return_z(close: pd.Series, seconds: int, baseline: int) -> pd.Series:
    ret = np.log(close / close.shift(seconds))
    prior_vol = ret.shift(1).rolling(baseline, min_periods=max(30, baseline // 3)).std(ddof=0)
    scale = prior_vol.where(prior_vol.gt(1e-12), 1e-9)
    return ret / scale


def build_candidate_frame(
    bars: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Build causal event scores and broad liquidity-release source flags."""
    if bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    close = frame["close"].astype(float)
    notional = frame["notional"].astype(float)
    trades = frame["trades_count"].astype(float)
    sell = frame["sell_notional"].astype(float)
    buy = frame["buy_notional"].astype(float)
    delta = frame["delta_notional"].astype(float)
    max_trade = frame["max_trade_notional"].astype(float)
    large_sell = frame["large_sell_notional"].astype(float)
    large_buy = frame["large_buy_notional"].astype(float)
    baseline = int(config.baseline_seconds)

    frame["z_notional"] = _prior_zscore(np.log1p(notional), baseline)
    frame["z_trades"] = _prior_zscore(np.log1p(trades), baseline)
    frame["z_max_trade"] = _prior_zscore(np.log1p(max_trade), baseline)
    frame["z_sell"] = _prior_zscore(np.log1p(sell), baseline)
    frame["z_buy"] = _prior_zscore(np.log1p(buy), baseline)
    frame["z_large_sell"] = _prior_zscore(np.log1p(large_sell), baseline)
    frame["z_large_buy"] = _prior_zscore(np.log1p(large_buy), baseline)
    frame["z_neg_delta"] = _prior_zscore((-delta).clip(lower=0.0), baseline)
    frame["z_pos_delta"] = _prior_zscore(delta.clip(lower=0.0), baseline)

    for seconds in (1, 5, 15):
        frame[f"return_z_{seconds}s"] = _signed_return_z(close, seconds, baseline)

    prior_low_60 = frame["low"].shift(1).rolling(60, min_periods=30).min()
    prior_high_60 = frame["high"].shift(1).rolling(60, min_periods=30).max()
    prior_low_300 = frame["low"].shift(1).rolling(300, min_periods=100).min()
    prior_high_300 = frame["high"].shift(1).rolling(300, min_periods=100).max()
    vol_bp = close.pct_change().shift(1).rolling(baseline, min_periods=100).std(ddof=0) * 1e4
    vol_bp = vol_bp.clip(lower=0.1)
    frame["lower_break_score"] = ((prior_low_60 - frame["low"]) / close * 1e4 / vol_bp).clip(lower=0.0)
    frame["upper_break_score"] = ((frame["high"] - prior_high_60) / close * 1e4 / vol_bp).clip(lower=0.0)
    frame["lower_break_300_score"] = ((prior_low_300 - frame["low"]) / close * 1e4 / vol_bp).clip(lower=0.0)
    frame["upper_break_300_score"] = ((frame["high"] - prior_high_300) / close * 1e4 / vol_bp).clip(lower=0.0)

    second_range_bp = (frame["high"] - frame["low"]) / close * 1e4
    prior_range = second_range_bp.shift(1).rolling(baseline, min_periods=100)
    frame["z_range"] = (second_range_bp - prior_range.mean()) / prior_range.std(ddof=0).replace(0.0, np.nan)

    down_components = pd.concat(
        [
            frame["z_sell"], frame["z_large_sell"], frame["z_neg_delta"],
            (-frame["return_z_1s"]), (-frame["return_z_5s"]), (-frame["return_z_15s"]),
            frame["lower_break_score"], frame["lower_break_300_score"], frame["z_range"],
            frame["z_notional"], frame["z_trades"], frame["z_max_trade"],
        ], axis=1,
    )
    up_components = pd.concat(
        [
            frame["z_buy"], frame["z_large_buy"], frame["z_pos_delta"],
            frame["return_z_1s"], frame["return_z_5s"], frame["return_z_15s"],
            frame["upper_break_score"], frame["upper_break_300_score"], frame["z_range"],
            frame["z_notional"], frame["z_trades"], frame["z_max_trade"],
        ], axis=1,
    )
    frame["down_event_score"] = down_components.max(axis=1, skipna=True)
    frame["up_event_score"] = up_components.max(axis=1, skipna=True)

    frame["source_flow_burst_down"] = (
        pd.concat([frame["z_sell"], frame["z_large_sell"], frame["z_neg_delta"], frame["z_notional"], frame["z_trades"]], axis=1)
        .max(axis=1).ge(config.event_score_threshold)
    )
    frame["source_flow_burst_up"] = (
        pd.concat([frame["z_buy"], frame["z_large_buy"], frame["z_pos_delta"], frame["z_notional"], frame["z_trades"]], axis=1)
        .max(axis=1).ge(config.event_score_threshold)
    )
    frame["source_price_shock_down"] = pd.concat(
        [-frame["return_z_1s"], -frame["return_z_5s"], -frame["return_z_15s"]], axis=1
    ).max(axis=1).ge(config.event_score_threshold)
    frame["source_price_shock_up"] = pd.concat(
        [frame["return_z_1s"], frame["return_z_5s"], frame["return_z_15s"]], axis=1
    ).max(axis=1).ge(config.event_score_threshold)
    frame["source_boundary_down"] = pd.concat(
        [frame["lower_break_score"], frame["lower_break_300_score"]], axis=1
    ).max(axis=1).ge(config.boundary_score_threshold)
    frame["source_boundary_up"] = pd.concat(
        [frame["upper_break_score"], frame["upper_break_300_score"]], axis=1
    ).max(axis=1).ge(config.boundary_score_threshold)
    frame["source_range_expansion"] = frame["z_range"].ge(config.event_score_threshold)

    frame["pre_context_gap_flag"] = frame["unsafe_gap"].rolling(config.pre_context_seconds, min_periods=1).max().fillna(0).astype(np.int8)

    return frame


def _debounce(frame: pd.DataFrame, side: str, seconds: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    score_col = "down_event_score" if side == "DOWN" else "up_event_score"
    work = frame.sort_index(kind="mergesort").copy()
    keep = np.zeros(len(work), dtype=bool)
    best_pos: int | None = None
    best_score = -np.inf
    previous_time: pd.Timestamp | None = None
    scores = pd.to_numeric(work[score_col], errors="coerce").fillna(-np.inf).to_numpy()
    for pos, (ts, score) in enumerate(zip(work.index, scores)):
        if previous_time is None or (ts - previous_time).total_seconds() > seconds:
            if best_pos is not None:
                keep[best_pos] = True
            best_pos = pos
            best_score = score
        elif score > best_score:
            best_pos = pos
            best_score = score
        previous_time = ts
    if best_pos is not None:
        keep[best_pos] = True
    return work.loc[keep]


def _assign_release_episodes(events: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    """Group nearby same-side impulses into one liquidity-release episode."""
    if events.empty:
        return events.copy()
    out = events.sort_index(kind="mergesort").copy()
    episode_number = np.zeros(len(out), dtype=np.int64)
    serial = 0
    last_by_side: dict[str, pd.Timestamp] = {}
    active_episode_by_side: dict[str, int] = {}
    for pos, (ts, side) in enumerate(zip(out.index, out["event_side"].astype(str))):
        previous = last_by_side.get(side)
        if previous is None or (ts - previous).total_seconds() > int(gap_seconds):
            serial += 1
            active_episode_by_side[side] = serial
        episode_number[pos] = active_episode_by_side[side]
        last_by_side[side] = ts
    out["release_episode_number"] = episode_number
    first_time = pd.Series(out.index, index=out.index).groupby(episode_number, sort=False).transform("first")
    first_side = out["event_side"].astype(str).groupby(episode_number, sort=False).transform("first")
    out["release_episode_id"] = [
        f"LLE_{pd.Timestamp(ts):%Y%m%d_%H%M%S}_{side}"
        for ts, side in zip(first_time, first_side)
    ]
    out["release_episode_ordinal"] = out.groupby("release_episode_id", sort=False).cumcount() + 1
    sizes = out.groupby("release_episode_id", sort=False)["release_episode_id"].transform("size")
    out["release_episode_size"] = sizes.astype(np.int32)
    out["release_episode_weight"] = 1.0 / sizes.astype(float)
    return out


def select_candidates(
    frame: pd.DataFrame,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Select high-recall candidates in the core interval and debounce impulses."""
    if frame.empty:
        return pd.DataFrame()
    core = frame.loc[(frame.index >= core_start) & (frame.index <= core_end)].copy()
    if core.empty:
        return core
    down_flag = (
        core["source_flow_burst_down"] | core["source_price_shock_down"] |
        core["source_boundary_down"] | core["source_range_expansion"]
    )
    up_flag = (
        core["source_flow_burst_up"] | core["source_price_shock_up"] |
        core["source_boundary_up"] | core["source_range_expansion"]
    )
    safe = core["pre_context_gap_flag"].eq(0) & core["close"].notna()
    down = core.loc[safe & down_flag & core["down_event_score"].ge(core["up_event_score"])].copy()
    up = core.loc[safe & up_flag & core["up_event_score"].gt(core["down_event_score"])].copy()
    down["event_side"] = "DOWN"
    up["event_side"] = "UP"
    down = _debounce(down, "DOWN", config.debounce_seconds)
    up = _debounce(up, "UP", config.debounce_seconds)
    events = pd.concat([down, up], axis=0).sort_index(kind="mergesort")
    if events.empty:
        return events
    events = _assign_release_episodes(events, config.release_episode_gap_seconds)
    events["event_time"] = events.index
    events["event_id"] = [f"LLP_{ts:%Y%m%d_%H%M%S}_{side}" for ts, side in zip(events.index, events["event_side"])]
    return events
