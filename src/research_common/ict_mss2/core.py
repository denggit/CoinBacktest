#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal ICT MSS research primitives for ETH perpetuals.

This module intentionally separates three questions that are often conflated in
visual ICT/SMC analysis:

1. Is a local high/low a *swing candidate*?
2. At a later timestamp, what evidence was already available that the swing was
   a meaningful liquidity pool rather than a weak micro pivot?
3. After the first true sweep of that still-unconsumed pool, does a lower
   execution timeframe produce displacement -> close-confirmed MSS -> FVG and a
   tradable pullback?

All bars are left-labelled.  A bar at timestamp ``t`` is only usable after
``t + timeframe``.  Pivot order k becomes available only after the kth right
confirmation bar has closed.  Future eventual pivot order is never used as an
entry-time feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import FenwickTree, SegmentThresholdIndex

EPS = 1e-12
REQUIRED_OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class MSS2Config:
    liquidity_timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
    )
    liquidity_confirmation_orders: tuple[int, ...] = (1, 2, 3, 5)
    execution_confirmation_orders: tuple[int, ...] = (1, 2, 3)
    confluence_tolerance_bps: float = 10.0
    touch_tolerance_bps: float = 5.0
    approach_tolerance_bps: float = 25.0
    sweep_epsilon_bps: float = 0.01
    mss_break_epsilon_bps: float = 0.01
    max_mss_minutes: int = 60
    max_entry_wait_minutes: int = 60
    max_outcome_minutes: int = 180
    atr_window: int = 20
    stop_buffer_bps: float = 2.0
    fixed_r_targets: tuple[float, ...] = (1.0, 2.0, 3.0)
    outcome_horizons_minutes: tuple[int, ...] = (5, 15, 30, 60, 120, 180)

    def validate(self) -> "MSS2Config":
        if not self.liquidity_timeframes:
            raise ValueError("liquidity_timeframes cannot be empty")
        if any(int(minutes) <= 0 for _, minutes in self.liquidity_timeframes):
            raise ValueError("liquidity timeframe minutes must be positive")
        if tuple(sorted(set(self.liquidity_confirmation_orders))) != self.liquidity_confirmation_orders:
            raise ValueError("liquidity_confirmation_orders must be sorted unique")
        if tuple(sorted(set(self.execution_confirmation_orders))) != self.execution_confirmation_orders:
            raise ValueError("execution_confirmation_orders must be sorted unique")
        if self.liquidity_confirmation_orders[0] != 1 or self.execution_confirmation_orders[0] != 1:
            raise ValueError("confirmation orders must include order 1")
        if self.atr_window < 2:
            raise ValueError("atr_window must be >= 2")
        if self.max_mss_minutes <= 0 or self.max_entry_wait_minutes <= 0 or self.max_outcome_minutes <= 0:
            raise ValueError("time limits must be positive")
        if self.confluence_tolerance_bps <= 0:
            raise ValueError("confluence_tolerance_bps must be positive")
        if self.touch_tolerance_bps <= 0 or self.approach_tolerance_bps <= self.touch_tolerance_bps:
            raise ValueError("approach_tolerance_bps must exceed positive touch_tolerance_bps")
        return self


def normalize_1m_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize official 1m OHLC(V) without creating synthetic market data."""
    if frame.empty:
        raise ValueError("1m bars are empty")
    missing = [name for name in REQUIRED_OHLC if name not in frame.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" not in out.columns:
            raise ValueError("bars require DatetimeIndex or timestamp column")
        out.index = pd.to_datetime(out.pop("timestamp"), errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.loc[~out.index.isna()].sort_index(kind="stable")
    out = out.loc[~out.index.duplicated(keep="last")]
    for name in REQUIRED_OHLC:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_OHLC))
    if len(out) < 3:
        raise ValueError("insufficient valid 1m bars")
    return out


def aggregate_bars(primary_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate complete left-labelled bars and expose explicit close time."""
    bars = normalize_1m_bars(primary_1m)
    minutes = int(minutes)
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if minutes == 1:
        out = bars.copy()
        out["_source_bar_count"] = 1
        out["bar_end_time"] = out.index + pd.Timedelta(minutes=1)
        return out
    working = bars.copy()
    working["_source_bar_count"] = 1
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "_source_bar_count": "sum",
    }
    if "volume" in working.columns:
        agg["volume"] = "sum"
    out = working.resample(
        f"{minutes}min",
        label="left",
        closed="left",
        origin="start_day",
    ).agg(agg)
    out = out.dropna(subset=list(REQUIRED_OHLC))
    out = out.loc[pd.to_numeric(out["_source_bar_count"], errors="coerce").eq(minutes)].copy()
    source_available_end = bars.index[-1] + pd.Timedelta(minutes=1)
    delta = pd.Timedelta(minutes=minutes)
    out = out.loc[(out.index + delta) <= source_available_end].copy()
    out["bar_end_time"] = out.index + delta
    return out


def _pivot_mask(values: np.ndarray, order: int, side: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = int(order)
    if order <= 0:
        raise ValueError("pivot order must be positive")
    if n < 2 * order + 1:
        return np.zeros(n, dtype=bool)
    mask = np.isfinite(values)
    mask[:order] = False
    mask[n - order :] = False
    for lag in range(1, order + 1):
        left = np.empty(n, dtype=float)
        right = np.empty(n, dtype=float)
        left[:lag] = np.nan
        left[lag:] = values[:-lag]
        right[-lag:] = np.nan
        right[:-lag] = values[lag:]
        if side == "low":
            mask &= values < left
            mask &= values <= right
        elif side == "high":
            mask &= values > left
            mask &= values >= right
        else:
            raise ValueError("side must be low or high")
    return mask


def _safe_div(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    return np.divide(
        num_arr,
        den_arr,
        out=np.full(np.broadcast_shapes(num_arr.shape, den_arr.shape), np.nan, dtype=float),
        where=np.isfinite(den_arr) & (np.abs(den_arr) > EPS),
    )


def _true_range(frame: pd.DataFrame) -> pd.Series:
    close_prev = pd.to_numeric(frame["close"], errors="coerce").shift(1)
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    parts = pd.concat([(high - low).abs(), (high - close_prev).abs(), (low - close_prev).abs()], axis=1)
    return parts.max(axis=1)


def _timeframe_pivots(
    htf: pd.DataFrame,
    *,
    timeframe: str,
    minutes: int,
    orders: tuple[int, ...],
    include_liquidity_features: bool,
) -> pd.DataFrame:
    if htf.empty:
        return pd.DataFrame()
    max_order = max(orders)
    if len(htf) < 2 * max_order + 1:
        return pd.DataFrame()
    delta = pd.Timedelta(minutes=int(minutes))
    high = htf["high"].to_numpy(dtype=float)
    low = htf["low"].to_numpy(dtype=float)
    open_ = htf["open"].to_numpy(dtype=float)
    close = htf["close"].to_numpy(dtype=float)
    atr_prev = _true_range(htf).shift(1).rolling(20, min_periods=5).mean().to_numpy(dtype=float)
    parts: list[pd.DataFrame] = []
    for side, values in (("low", low), ("high", high)):
        masks = {order: _pivot_mask(values, order, side) for order in orders}
        positions = np.flatnonzero(masks[1])
        if not len(positions):
            continue
        levels = values[positions]
        rows = pd.DataFrame(
            {
                "pivot_side": side,
                "source_timeframe": str(timeframe),
                "source_timeframe_min": int(minutes),
                "pivot_pos_htf": positions.astype(np.int64),
                "pivot_time": htf.index[positions],
                "pivot_bar_end_time": htf.index[positions] + delta,
                "level_price": levels,
                # order-1 pivot needs one right bar; it is known after that bar closes.
                "initial_available_time": htf.index[positions] + 2 * delta,
            }
        )
        for order in orders:
            qualified = masks[order][positions]
            available = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
            if np.any(qualified):
                available[qualified] = (
                    htf.index[positions[qualified]] + (int(order) + 1) * delta
                ).to_numpy(dtype="datetime64[ns]")
            rows[f"order_{order}_available_time"] = pd.to_datetime(available)
            # Explicitly named FUTURE label: useful only for ex-post audits, never feature selection.
            rows[f"future_eventual_order_{order}_label"] = qualified.astype(np.int8)

        if include_liquidity_features:
            bar_range = np.maximum(high[positions] - low[positions], EPS)
            body_high = np.maximum(open_[positions], close[positions])
            body_low = np.minimum(open_[positions], close[positions])
            if side == "low":
                wick = body_low - low[positions]
                close_loc = (close[positions] - low[positions]) / bar_range
                reaction = np.maximum(high[positions + 1], close[positions + 1]) - levels
                prior_20 = pd.Series(low, index=htf.index).shift(1).rolling(20, min_periods=3).min().to_numpy()[positions]
                prior_50 = pd.Series(low, index=htf.index).shift(1).rolling(50, min_periods=5).min().to_numpy()[positions]
                external20 = levels < prior_20
                external50 = levels < prior_50
                prior_opposite = pd.Series(high, index=htf.index).shift(1).rolling(20, min_periods=3).max().to_numpy()[positions]
                left_excursion = prior_opposite - levels
            else:
                wick = high[positions] - body_high
                close_loc = (high[positions] - close[positions]) / bar_range
                reaction = levels - np.minimum(low[positions + 1], close[positions + 1])
                prior_20 = pd.Series(high, index=htf.index).shift(1).rolling(20, min_periods=3).max().to_numpy()[positions]
                prior_50 = pd.Series(high, index=htf.index).shift(1).rolling(50, min_periods=5).max().to_numpy()[positions]
                external20 = levels > prior_20
                external50 = levels > prior_50
                prior_opposite = pd.Series(low, index=htf.index).shift(1).rolling(20, min_periods=3).min().to_numpy()[positions]
                left_excursion = levels - prior_opposite
            rows["pivot_wick_fraction"] = _safe_div(wick, bar_range)
            rows["pivot_close_rejection_fraction"] = close_loc
            rows["left_excursion_bp"] = _safe_div(left_excursion, levels) * 10_000.0
            rows["confirmation_reaction_bp"] = _safe_div(reaction, levels) * 10_000.0
            rows["confirmation_reaction_atr"] = _safe_div(reaction, atr_prev[positions])
            rows["external_20_flag"] = np.asarray(external20, dtype=np.int8)
            rows["external_50_flag"] = np.asarray(external50, dtype=np.int8)
        parts.append(rows)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["initial_available_time", "pivot_time", "pivot_side", "level_price"], kind="stable"
    ).reset_index(drop=True)


def build_liquidity_levels(primary_1m: pd.DataFrame, config: MSS2Config | None = None) -> pd.DataFrame:
    """Build symmetric HTF swing candidates; *candidate* is not synonymous with liquidity."""
    cfg = (config or MSS2Config()).validate()
    bars = normalize_1m_bars(primary_1m)
    parts: list[pd.DataFrame] = []
    for timeframe, minutes in cfg.liquidity_timeframes:
        htf = aggregate_bars(bars, int(minutes))
        part = _timeframe_pivots(
            htf,
            timeframe=str(timeframe),
            minutes=int(minutes),
            orders=cfg.liquidity_confirmation_orders,
            include_liquidity_features=True,
        )
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    levels = pd.concat(parts, ignore_index=True, sort=False)
    levels = levels.sort_values(
        ["initial_available_time", "source_timeframe_min", "pivot_time", "pivot_side", "level_price"],
        kind="stable",
    ).reset_index(drop=True)
    levels.insert(0, "level_id", np.arange(1, len(levels) + 1, dtype=np.int64))
    levels["liquidity_side"] = np.where(levels["pivot_side"].eq("low"), "sell_side", "buy_side")
    levels["trade_direction"] = np.where(levels["pivot_side"].eq("low"), 1, -1).astype(np.int8)
    if (pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).any():
        raise RuntimeError("liquidity pivot became usable before right confirmation bar closed")
    return levels


def build_execution_pivots(exec_bars: pd.DataFrame, minutes: int, config: MSS2Config | None = None) -> pd.DataFrame:
    cfg = (config or MSS2Config()).validate()
    return _timeframe_pivots(
        exec_bars,
        timeframe=f"{int(minutes)}m",
        minutes=int(minutes),
        orders=cfg.execution_confirmation_orders,
        include_liquidity_features=False,
    )


def _confirmed_order(row: pd.Series, at_time: pd.Timestamp, orders: Iterable[int]) -> int:
    confirmed = 0
    for order in orders:
        value = pd.to_datetime(row.get(f"order_{int(order)}_available_time"), errors="coerce")
        if pd.notna(value) and value <= at_time:
            confirmed = max(confirmed, int(order))
    return int(confirmed)


def build_first_sweep_lifecycle(
    primary_1m: pd.DataFrame,
    levels: pd.DataFrame,
    config: MSS2Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Find the first true 1m penetration after each level is causally available."""
    cfg = (config or MSS2Config()).validate()
    bars = normalize_1m_bars(primary_1m)
    if levels.empty:
        return levels.copy()
    index = bars.index
    low_index = SegmentThresholdIndex(bars["low"].to_numpy(dtype=float))
    high_index = SegmentThresholdIndex(bars["high"].to_numpy(dtype=float))
    end = len(bars) - 1
    eps = float(cfg.sweep_epsilon_bps) / 10_000.0
    touch = float(cfg.touch_tolerance_bps) / 10_000.0
    approach = float(cfg.approach_tolerance_bps) / 10_000.0
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter("[liquidity-sweeps]", total=len(levels), every=max(1, len(levels) // 100), enabled=show_progress)
    for loop_i, source in enumerate(levels.itertuples(index=False), start=1):
        reporter.update(loop_i)
        row = source._asdict()
        active_time = pd.Timestamp(row["initial_available_time"])
        active_pos = int(index.searchsorted(active_time, side="left"))
        if active_pos >= len(index):
            continue
        level = float(row["level_price"])
        if str(row["pivot_side"]) == "low":
            approach_pos = low_index.first_leq(active_pos, end, level * (1.0 + approach))
            touch_pos = low_index.first_leq(active_pos, end, level * (1.0 + touch))
            sweep_pos = low_index.first_leq(active_pos, end, level * (1.0 - eps))
        else:
            approach_pos = high_index.first_geq(active_pos, end, level * (1.0 - approach))
            touch_pos = high_index.first_geq(active_pos, end, level * (1.0 - touch))
            sweep_pos = high_index.first_geq(active_pos, end, level * (1.0 + eps))
        row["active_pos_1m"] = active_pos
        row["first_approach_pos_1m"] = int(approach_pos)
        row["first_touch_pos_1m"] = int(touch_pos)
        row["sweep_pos_1m"] = int(sweep_pos)
        if sweep_pos >= 0:
            row["sweep_bar_time_1m"] = index[sweep_pos]
            row["sweep_available_time_1m"] = index[sweep_pos] + pd.Timedelta(minutes=1)
            row["age_minutes_active_at_sweep"] = float((index[sweep_pos] - active_time) / pd.Timedelta(minutes=1))
            row["age_minutes_since_pivot_at_sweep"] = float((index[sweep_pos] - pd.Timestamp(row["pivot_time"])) / pd.Timedelta(minutes=1))
            row["age_minutes_at_sweep"] = row["age_minutes_since_pivot_at_sweep"]
            row["first_approach_time_1m"] = index[approach_pos] if approach_pos >= 0 else pd.NaT
            row["first_touch_time_1m"] = index[touch_pos] if touch_pos >= 0 else pd.NaT
            row["minutes_first_approach_to_sweep"] = float(sweep_pos - approach_pos) if approach_pos >= 0 else np.nan
            row["minutes_first_touch_to_sweep"] = float(sweep_pos - touch_pos) if touch_pos >= 0 else np.nan
            row["clean_sweep_no_prior_touch_flag"] = int(touch_pos < 0 or touch_pos == sweep_pos)
            row["pretested_before_sweep_flag"] = int(touch_pos >= 0 and touch_pos < sweep_pos)
            if str(row["pivot_side"]) == "low":
                row["sweep_depth_bp"] = (level / float(bars["low"].iloc[sweep_pos]) - 1.0) * 10_000.0
            else:
                row["sweep_depth_bp"] = (float(bars["high"].iloc[sweep_pos]) / level - 1.0) * 10_000.0
            row["confirmed_order_at_sweep"] = _confirmed_order(
                pd.Series(row), pd.Timestamp(row["sweep_available_time_1m"]), cfg.liquidity_confirmation_orders
            )
        else:
            row["sweep_bar_time_1m"] = pd.NaT
            row["sweep_available_time_1m"] = pd.NaT
            row["age_minutes_active_at_sweep"] = np.nan
            row["age_minutes_since_pivot_at_sweep"] = np.nan
            row["age_minutes_at_sweep"] = np.nan
            row["first_approach_time_1m"] = index[approach_pos] if approach_pos >= 0 else pd.NaT
            row["first_touch_time_1m"] = index[touch_pos] if touch_pos >= 0 else pd.NaT
            row["minutes_first_approach_to_sweep"] = np.nan
            row["minutes_first_touch_to_sweep"] = np.nan
            row["clean_sweep_no_prior_touch_flag"] = 0
            row["pretested_before_sweep_flag"] = 0
            row["sweep_depth_bp"] = np.nan
            row["confirmed_order_at_sweep"] = 0
        rows.append(row)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["active_pos_1m", "level_id"], kind="stable").reset_index(drop=True)


def _active_confluence_for_side(
    side_levels: pd.DataFrame,
    *,
    tolerance_bps: float,
) -> pd.DataFrame:
    """Count same-side active unconsumed levels around each first-sweep price."""
    if side_levels.empty:
        return side_levels.copy()
    out = side_levels.copy().sort_values(["sweep_pos_1m", "level_id"], kind="stable").reset_index(drop=True)
    prices = np.sort(out["level_price"].dropna().astype(float).unique())
    rank = {float(price): int(i) for i, price in enumerate(prices)}
    timeframes = sorted(out["source_timeframe"].astype(str).unique())
    total = FenwickTree(len(prices))
    tf_trees = {tf: FenwickTree(len(prices)) for tf in timeframes}
    additions: dict[int, list[tuple[float, str]]] = {}
    removals: dict[int, list[tuple[float, str]]] = {}
    for row in out.itertuples(index=False):
        additions.setdefault(int(row.active_pos_1m), []).append((float(row.level_price), str(row.source_timeframe)))
        if int(row.sweep_pos_1m) >= 0:
            removals.setdefault(int(row.sweep_pos_1m) + 1, []).append((float(row.level_price), str(row.source_timeframe)))
    updates = sorted(set(additions) | set(removals))
    pointer = 0
    counts = np.zeros(len(out), dtype=np.int32)
    tf_counts = np.zeros(len(out), dtype=np.int8)
    tol = float(tolerance_bps) / 10_000.0
    for i, row in enumerate(out.itertuples(index=False)):
        event_pos = int(row.sweep_pos_1m)
        if event_pos < 0:
            continue
        while pointer < len(updates) and updates[pointer] <= event_pos:
            pos = updates[pointer]
            for price, tf in removals.get(pos, []):
                idx = rank[price]
                total.add(idx, -1)
                tf_trees[tf].add(idx, -1)
            for price, tf in additions.get(pos, []):
                idx = rank[price]
                total.add(idx, 1)
                tf_trees[tf].add(idx, 1)
            pointer += 1
        center = float(row.level_price)
        left = int(np.searchsorted(prices, center * (1.0 - tol), side="left"))
        right = int(np.searchsorted(prices, center * (1.0 + tol), side="right"))
        counts[i] = total.range_sum(left, right)
        tf_counts[i] = sum(tree.range_sum(left, right) > 0 for tree in tf_trees.values())
    token = str(float(tolerance_bps)).replace(".", "p")
    out[f"active_same_side_level_count_{token}bp"] = counts
    out[f"active_same_side_timeframe_count_{token}bp"] = tf_counts
    return out


def classify_liquidity(lifecycle: pd.DataFrame, config: MSS2Config | None = None) -> pd.DataFrame:
    """Attach an interpretable *structural taxonomy*, not a future-derived truth label.

    The classification is intentionally broad.  It lets the research answer
    which categories actually behave like liquidity in ETH, instead of assuming
    every swing is equal or hardcoding a recent-swing rule.
    """
    cfg = (config or MSS2Config()).validate()
    if lifecycle.empty:
        return lifecycle.copy()
    parts = [
        _active_confluence_for_side(part, tolerance_bps=cfg.confluence_tolerance_bps)
        for _, part in lifecycle.groupby("pivot_side", sort=False)
    ]
    out = pd.concat(parts, ignore_index=True, sort=False).sort_values("level_id", kind="stable").reset_index(drop=True)
    token = str(float(cfg.confluence_tolerance_bps)).replace(".", "p")
    near_count = pd.to_numeric(out[f"active_same_side_level_count_{token}bp"], errors="coerce").fillna(0).astype(int)
    tf_count = pd.to_numeric(out[f"active_same_side_timeframe_count_{token}bp"], errors="coerce").fillna(0).astype(int)
    order = pd.to_numeric(out["confirmed_order_at_sweep"], errors="coerce").fillna(0).astype(int)
    external20 = pd.to_numeric(out.get("external_20_flag", 0), errors="coerce").fillna(0).astype(int)
    external50 = pd.to_numeric(out.get("external_50_flag", 0), errors="coerce").fillna(0).astype(int)
    age = pd.to_numeric(out["age_minutes_at_sweep"], errors="coerce")
    score = (
        1
        + (order >= 2).astype(int)
        + (order >= 3).astype(int)
        + (order >= 5).astype(int)
        + external20
        + external50
        + (near_count >= 2).astype(int)
        + (tf_count >= 2).astype(int)
        + (age >= 360).astype(int)
        + (age >= 1440).astype(int)
    )
    out["liquidity_structural_score"] = score.astype(np.int16)
    out["old_remote_flag_6h"] = (age >= 360).astype(np.int8)
    out["old_remote_flag_24h"] = (age >= 1440).astype(np.int8)
    out["old_remote_flag_72h"] = (age >= 4320).astype(np.int8)
    out["age_bucket"] = pd.cut(
        age,
        bins=[-np.inf, 60, 360, 1440, 4320, 10080, np.inf],
        labels=["<1h", "1-6h", "6-24h", "1-3d", "3-7d", ">=7d"],
        right=False,
    ).astype("string")

    pool = near_count >= 2
    multi_tf = tf_count >= 2
    major = order >= 3
    structural = order >= 2
    external = external20.astype(bool)
    out["liquidity_class"] = np.select(
        [pool & multi_tf, pool, major & external, major, structural & external, structural],
        ["multi_tf_pool", "same_price_pool", "major_external", "major_swing", "structural_external", "structural_swing"],
        default="minor_swing_candidate",
    )
    out["quality_tier"] = pd.cut(
        score,
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=["D", "C", "B", "A"],
        right=False,
    ).astype("string")
    return out


def _project_offset_hours(timezone_value: str | int | float | None) -> float:
    if timezone_value is None:
        return 8.0
    if isinstance(timezone_value, (int, float)):
        return float(timezone_value)
    token = str(timezone_value).strip().upper().replace("UTC", "")
    if token in {"", "Z"}:
        return 0.0
    try:
        return float(token)
    except ValueError:
        return 8.0


def attach_session_context(
    frame: pd.DataFrame,
    time_column: str,
    *,
    project_timezone: str | int | float | None = "+8",
) -> pd.DataFrame:
    """Attach transparent clock/session fields with DST-aware target zones."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    ts = pd.to_datetime(out[time_column], errors="coerce")
    offset = _project_offset_hours(project_timezone)
    aware = ts.dt.tz_localize(timezone(timedelta(hours=offset)), nonexistent="shift_forward", ambiguous="NaT")
    utc = aware.dt.tz_convert("UTC")
    ny = aware.dt.tz_convert("America/New_York")
    london = aware.dt.tz_convert("Europe/London")
    shanghai = aware.dt.tz_convert("Asia/Shanghai")
    out["utc_hour"] = utc.dt.hour.astype("Int8")
    out["ny_hour"] = ny.dt.hour.astype("Int8")
    out["london_hour"] = london.dt.hour.astype("Int8")
    out["shanghai_hour"] = shanghai.dt.hour.astype("Int8")
    out["utc_weekday"] = utc.dt.dayofweek.astype("Int8")
    out["ny_weekday"] = ny.dt.dayofweek.astype("Int8")
    out["is_weekend_utc"] = (utc.dt.dayofweek >= 5).astype("Int8")
    out["is_weekend_ny"] = (ny.dt.dayofweek >= 5).astype("Int8")
    ny_minute = ny.dt.hour * 60 + ny.dt.minute
    london_minute = london.dt.hour * 60 + london.dt.minute
    sh_minute = shanghai.dt.hour * 60 + shanghai.dt.minute
    out["ny_cash_open_30m"] = ((ny_minute >= 9 * 60 + 30) & (ny_minute < 10 * 60)).astype("Int8")
    out["ny_open_90m"] = ((ny_minute >= 9 * 60) & (ny_minute < 10 * 60 + 30)).astype("Int8")
    out["ny_am_0800_1200"] = ((ny_minute >= 8 * 60) & (ny_minute < 12 * 60)).astype("Int8")
    out["london_am_0700_1100"] = ((london_minute >= 7 * 60) & (london_minute < 11 * 60)).astype("Int8")
    out["asia_0800_1600_shanghai"] = ((sh_minute >= 8 * 60) & (sh_minute < 16 * 60)).astype("Int8")
    out["session_primary"] = np.select(
        [
            out["ny_open_90m"].eq(1),
            out["london_am_0700_1100"].eq(1),
            out["asia_0800_1600_shanghai"].eq(1),
        ],
        ["new_york_open", "london_am", "asia_day"],
        default="other",
    )
    return out


class _PivotReferenceIndex:
    def __init__(self, pivots: pd.DataFrame, side: str, orders: tuple[int, ...]):
        part = pivots.loc[pivots["pivot_side"].eq(side)].sort_values("pivot_pos_htf", kind="stable").reset_index(drop=True)
        self.part = part
        self.positions = pd.to_numeric(part.get("pivot_pos_htf", pd.Series(dtype=float)), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        self.levels = pd.to_numeric(part.get("level_price", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        self.orders = orders

    def latest_before(
        self, *, sweep_pos: int, known_time: pd.Timestamp, min_order: int
    ) -> tuple[float, int, int, pd.Timestamp]:
        if self.part.empty:
            return np.nan, -1, 0, pd.NaT
        hi = int(np.searchsorted(self.positions, int(sweep_pos), side="left"))
        if hi <= 0:
            return np.nan, -1, 0, pd.NaT
        for i in range(hi - 1, -1, -1):
            row = self.part.iloc[i]
            confirmed = _confirmed_order(row, known_time, self.orders)
            if confirmed >= int(min_order):
                min_order_available = pd.to_datetime(row.get(f"order_{int(min_order)}_available_time"), errors="coerce")
                return float(self.levels[i]), int(self.positions[i]), int(confirmed), min_order_available
        return np.nan, -1, 0, pd.NaT


def _first_fvg_in_range(exec_bars: pd.DataFrame, direction: int, start: int, end: int) -> tuple[int, float, float, float]:
    if end < start or len(exec_bars) < 3:
        return -1, np.nan, np.nan, np.nan
    high = exec_bars["high"].to_numpy(dtype=float)
    low = exec_bars["low"].to_numpy(dtype=float)
    left = max(2, int(start))
    right = min(int(end), len(exec_bars) - 1)
    for pos in range(left, right + 1):
        if direction > 0 and low[pos] > high[pos - 2]:
            lower = float(high[pos - 2])
            upper = float(low[pos])
            return pos, lower, upper, upper
        if direction < 0 and high[pos] < low[pos - 2]:
            lower = float(high[pos])
            upper = float(low[pos - 2])
            return pos, lower, upper, lower
    return -1, np.nan, np.nan, np.nan


def _fvg_leg_stats(
    exec_bars: pd.DataFrame, direction: int, start: int, end: int, mss_pos: int
) -> tuple[int, float, int]:
    high = exec_bars["high"].to_numpy(dtype=float)
    low = exec_bars["low"].to_numpy(dtype=float)
    left = max(2, int(start))
    right = min(int(end), len(exec_bars) - 1)
    count = 0
    largest_width = np.nan
    mss_bar_flag = 0
    for pos in range(left, right + 1):
        width = np.nan
        if direction > 0 and low[pos] > high[pos - 2]:
            width = float(low[pos] - high[pos - 2])
        elif direction < 0 and high[pos] < low[pos - 2]:
            width = float(low[pos - 2] - high[pos])
        if np.isfinite(width) and width > 0:
            count += 1
            largest_width = width if not np.isfinite(largest_width) else max(largest_width, width)
            if pos == int(mss_pos):
                mss_bar_flag = 1
    return count, largest_width, mss_bar_flag


def _range_extreme(values: np.ndarray, start: int, end: int, mode: str) -> float:
    if end < start:
        return np.nan
    segment = values[max(0, start) : min(len(values), end + 1)]
    if not len(segment):
        return np.nan
    return float(np.nanmin(segment) if mode == "min" else np.nanmax(segment))


def build_mss_fvg_events(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    *,
    execution_minutes: int,
    reference_mode: str,
    config: MSS2Config | None = None,
    project_timezone: str | int | float | None = "+8",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build first-sweep -> displacement -> close-MSS -> FVG pullback events.

    ``reference_mode='recent'`` uses the latest causally known order-1 opposite
    pivot that existed before the sweep.  ``reference_mode='structural'`` uses
    the latest pivot already confirmed to order >=2.  Neither mode uses future
    eventual pivot order.
    """
    cfg = (config or MSS2Config()).validate()
    if reference_mode not in {"recent", "structural"}:
        raise ValueError("reference_mode must be recent or structural")
    minutes = int(execution_minutes)
    if minutes not in {1, 2}:
        raise ValueError("execution_minutes must be 1 or 2")
    bars1 = normalize_1m_bars(primary_1m)
    exec_bars = aggregate_bars(bars1, minutes)
    pivots = build_execution_pivots(exec_bars, minutes, cfg)
    high_refs = _PivotReferenceIndex(pivots, "high", cfg.execution_confirmation_orders)
    low_refs = _PivotReferenceIndex(pivots, "low", cfg.execution_confirmation_orders)
    close_index = SegmentThresholdIndex(exec_bars["close"].to_numpy(dtype=float))
    low_index = SegmentThresholdIndex(exec_bars["low"].to_numpy(dtype=float))
    high_index = SegmentThresholdIndex(exec_bars["high"].to_numpy(dtype=float))
    base_low_index = SegmentThresholdIndex(bars1["low"].to_numpy(dtype=float))
    base_high_index = SegmentThresholdIndex(bars1["high"].to_numpy(dtype=float))
    base_low = bars1["low"].to_numpy(dtype=float)
    base_high = bars1["high"].to_numpy(dtype=float)
    high = exec_bars["high"].to_numpy(dtype=float)
    low = exec_bars["low"].to_numpy(dtype=float)
    open_ = exec_bars["open"].to_numpy(dtype=float)
    close = exec_bars["close"].to_numpy(dtype=float)
    tr = _true_range(exec_bars)
    atr_pre = tr.shift(1).rolling(cfg.atr_window, min_periods=max(5, cfg.atr_window // 2)).mean().to_numpy(dtype=float)
    abs_close_change = np.abs(np.diff(close, prepend=np.nan))
    path_cum = np.nancumsum(np.where(np.isfinite(abs_close_change), abs_close_change, 0.0))
    max_mss_bars = max(1, int(np.ceil(cfg.max_mss_minutes / minutes)))
    max_entry_wait_1m = max(1, int(cfg.max_entry_wait_minutes))
    break_eps = cfg.mss_break_epsilon_bps / 10_000.0
    exec_index = exec_bars.index
    rows: list[dict[str, object]] = []

    swept = classified_lifecycle.loc[pd.to_numeric(classified_lifecycle["sweep_pos_1m"], errors="coerce").fillna(-1).ge(0)]
    reporter = ProgressReporter(
        f"[mss-{minutes}m-{reference_mode}]", total=len(swept), every=max(1, len(swept) // 100), enabled=show_progress
    )
    for loop_i, source in enumerate(swept.itertuples(index=False), start=1):
        reporter.update(loop_i)
        base = source._asdict()
        sweep_time = pd.Timestamp(base["sweep_bar_time_1m"])
        sweep_exec_pos = int(exec_index.searchsorted(sweep_time, side="right")) - 1
        if sweep_exec_pos < 0 or sweep_exec_pos >= len(exec_bars):
            continue
        sweep_exec_end = pd.Timestamp(exec_bars["bar_end_time"].iloc[sweep_exec_pos])
        arm_pos = sweep_exec_pos + 1
        if arm_pos >= len(exec_bars):
            continue
        direction = int(base["trade_direction"])
        min_order = 1 if reference_mode == "recent" else 2
        ref_index = high_refs if direction > 0 else low_refs
        # Strict causality: the MSS reference swing must already be known before
        # the execution bar that contains the sweep starts.  We deliberately do
        # not allow the sweep bar itself to become the right-confirmation bar.
        reference_known_cutoff = pd.Timestamp(exec_index[sweep_exec_pos])
        ref_price, ref_pivot_pos, ref_confirmed_order, ref_available_time = ref_index.latest_before(
            sweep_pos=sweep_exec_pos,
            known_time=reference_known_cutoff,
            min_order=min_order,
        )
        if not np.isfinite(ref_price):
            continue
        end_pos = min(len(exec_bars) - 1, arm_pos + max_mss_bars - 1)
        if direction > 0:
            mss_pos = close_index.first_geq(arm_pos, end_pos, ref_price * (1.0 + break_eps))
        else:
            mss_pos = close_index.first_leq(arm_pos, end_pos, ref_price * (1.0 - break_eps))
        if mss_pos < 0:
            continue
        if direction > 0:
            sweep_extreme = _range_extreme(low, sweep_exec_pos, mss_pos, "min")
            directional_move = float(close[mss_pos] - sweep_extreme)
            break_distance = float(close[mss_pos] - ref_price)
        else:
            sweep_extreme = _range_extreme(high, sweep_exec_pos, mss_pos, "max")
            directional_move = float(sweep_extreme - close[mss_pos])
            break_distance = float(ref_price - close[mss_pos])
        pre_atr = float(atr_pre[sweep_exec_pos]) if sweep_exec_pos < len(atr_pre) else np.nan
        path_start = max(1, sweep_exec_pos)
        path_len = float(path_cum[mss_pos] - path_cum[path_start - 1]) if mss_pos >= path_start else 0.0
        efficiency = directional_move / path_len if path_len > EPS else np.nan
        body = abs(float(close[mss_pos] - open_[mss_pos]))
        candle_range = max(float(high[mss_pos] - low[mss_pos]), EPS)
        fvg_pos, fvg_lower, fvg_upper, fvg_proximal = _first_fvg_in_range(
            exec_bars, direction, sweep_exec_pos, mss_pos
        )
        fvg_width = float(fvg_upper - fvg_lower) if fvg_pos >= 0 else np.nan
        fvg_count, largest_fvg_width, mss_bar_fvg_flag = _fvg_leg_stats(
            exec_bars, direction, sweep_exec_pos, mss_pos, mss_pos
        )
        # The signal timeframe is 1m or 2m, but the resting order is executed
        # against the original 1m naked K path.  This avoids optimistic intrabar
        # ordering assumptions inside a 2m candle.
        entry_fill_pos_1m = -1
        entry_fill_pos_exec = -1
        if fvg_pos >= 0 and np.isfinite(fvg_proximal):
            order_live_time = pd.Timestamp(exec_bars["bar_end_time"].iloc[mss_pos])
            entry_start_1m = int(bars1.index.searchsorted(order_live_time, side="left"))
            entry_end_1m = min(len(bars1) - 1, entry_start_1m + max_entry_wait_1m - 1)
            if entry_start_1m <= entry_end_1m:
                if direction > 0:
                    candidate = base_low_index.first_leq(entry_start_1m, entry_end_1m, fvg_proximal)
                else:
                    candidate = base_high_index.first_geq(entry_start_1m, entry_end_1m, fvg_proximal)
                if candidate >= 0 and base_low[candidate] <= fvg_proximal <= base_high[candidate]:
                    entry_fill_pos_1m = int(candidate)
                    entry_fill_pos_exec = int(exec_index.searchsorted(bars1.index[candidate], side="right")) - 1
        record = dict(base)
        record.update(
            {
                "execution_minutes": minutes,
                "reference_mode": reference_mode,
                "sweep_exec_pos": sweep_exec_pos,
                "sweep_exec_bar_time": exec_index[sweep_exec_pos],
                "sweep_exec_available_time": sweep_exec_end,
                "mss_reference_price": ref_price,
                "mss_reference_pivot_pos": ref_pivot_pos,
                "mss_reference_confirmed_order": ref_confirmed_order,
                "mss_reference_available_time": ref_available_time,
                "mss_reference_known_cutoff": reference_known_cutoff,
                "mss_pos": int(mss_pos),
                "mss_bar_time": exec_index[mss_pos],
                "mss_available_time": pd.Timestamp(exec_bars["bar_end_time"].iloc[mss_pos]),
                "bars_to_mss": int(mss_pos - sweep_exec_pos),
                "minutes_to_mss": int((mss_pos - sweep_exec_pos) * minutes),
                "sweep_extreme": sweep_extreme,
                "pre_sweep_atr": pre_atr,
                "displacement_atr": directional_move / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "break_distance_atr": break_distance / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "path_efficiency": efficiency,
                "mss_body_atr": body / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "mss_body_ratio": body / candle_range,
                "fvg_pos": int(fvg_pos),
                "fvg_available_time": pd.Timestamp(exec_bars["bar_end_time"].iloc[fvg_pos]) if fvg_pos >= 0 else pd.NaT,
                "fvg_lower": fvg_lower,
                "fvg_upper": fvg_upper,
                "fvg_proximal": fvg_proximal,
                "fvg_width_atr": fvg_width / pre_atr if np.isfinite(fvg_width) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "fvg_count_in_leg": int(fvg_count),
                "largest_fvg_width_atr": largest_fvg_width / pre_atr if np.isfinite(largest_fvg_width) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "mss_bar_fvg_flag": int(mss_bar_fvg_flag),
                "first_fvg_minutes_before_mss": int((mss_pos - fvg_pos) * minutes) if fvg_pos >= 0 else np.nan,
                "has_displacement_fvg": int(fvg_pos >= 0),
                "entry_fill_pos_1m": entry_fill_pos_1m,
                "entry_fill_pos_exec": entry_fill_pos_exec,
                "entry_time": bars1.index[entry_fill_pos_1m] if entry_fill_pos_1m >= 0 else pd.NaT,
                "entry_available_time": bars1.index[entry_fill_pos_1m] + pd.Timedelta(minutes=1) if entry_fill_pos_1m >= 0 else pd.NaT,
                "entry_price": fvg_proximal if entry_fill_pos_1m >= 0 else np.nan,
            }
        )
        rows.append(record)
    reporter.close()
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events.insert(0, "event_id", [f"MSS2_{minutes}M_{reference_mode.upper()}_{i+1:08d}" for i in range(len(events))])
    events = attach_session_context(events, "mss_available_time", project_timezone=project_timezone)
    return events.sort_values(["mss_available_time", "level_id"], kind="stable").reset_index(drop=True)


def attach_sweep_baseline_outcomes(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    *,
    config: MSS2Config | None = None,
    project_timezone: str | int | float | None = "+8",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Measure sweep-only forward paths from the next 1m open.

    This provides the control baseline needed to test whether MSS/displacement/FVG
    adds information beyond the liquidity sweep itself.  All forward columns are
    labels and must never be reused as event-time features.
    """
    cfg = (config or MSS2Config()).validate()
    bars = normalize_1m_bars(primary_1m)
    if classified_lifecycle.empty:
        return pd.DataFrame()
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    swept = classified_lifecycle.loc[
        pd.to_numeric(classified_lifecycle["sweep_pos_1m"], errors="coerce").fillna(-1).ge(0)
    ].copy()
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter(
        "[sweep-baseline]", total=len(swept), every=max(1, len(swept) // 100), enabled=show_progress
    )
    for loop_i, source in enumerate(swept.itertuples(index=False), start=1):
        reporter.update(loop_i)
        row = source._asdict()
        sweep_pos = int(row["sweep_pos_1m"])
        entry_pos = sweep_pos + 1
        if entry_pos >= len(bars):
            continue
        direction = int(row["trade_direction"])
        entry = float(open_[entry_pos])
        row["sweep_baseline_entry_pos_1m"] = entry_pos
        row["sweep_baseline_entry_time"] = bars.index[entry_pos]
        row["sweep_baseline_entry_price"] = entry
        for horizon in cfg.outcome_horizons_minutes:
            end = min(len(bars) - 1, entry_pos + int(horizon))
            row[f"sweep_close_return_{int(horizon)}m"] = direction * (float(close[end]) / entry - 1.0)
            if direction > 0:
                mfe = max(0.0, float(np.nanmax(high[entry_pos : end + 1]) / entry - 1.0))
                mae = max(0.0, float(1.0 - np.nanmin(low[entry_pos : end + 1]) / entry))
            else:
                mfe = max(0.0, float(1.0 - np.nanmin(low[entry_pos : end + 1]) / entry))
                mae = max(0.0, float(np.nanmax(high[entry_pos : end + 1]) / entry - 1.0))
            row[f"sweep_mfe_{int(horizon)}m"] = mfe
            row[f"sweep_mae_{int(horizon)}m"] = mae
        rows.append(row)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "sweep_event_id", [f"SWEEP_{i+1:08d}" for i in range(len(out))])
    return attach_session_context(
        out, "sweep_available_time_1m", project_timezone=project_timezone
    ).sort_values(["sweep_available_time_1m", "level_id"], kind="stable").reset_index(drop=True)


class _ActiveNearestLiquidityIndex:
    """Sweep-line active liquidity index that can return the nearest active price."""

    def __init__(self, levels: pd.DataFrame, base_index: pd.DatetimeIndex, side: str, timeframe: str | None = None):
        part = levels.loc[levels["pivot_side"].eq(side)].copy()
        if timeframe is not None:
            part = part.loc[part["source_timeframe"].eq(timeframe)].copy()
        self.part = part
        self.base_index = base_index
        self.prices = np.sort(part["level_price"].dropna().astype(float).unique()) if not part.empty else np.array([], dtype=float)
        self.rank = {float(p): i for i, p in enumerate(self.prices)}
        self.additions: dict[int, list[float]] = {}
        self.removals: dict[int, list[float]] = {}
        for row in part.itertuples(index=False):
            self.additions.setdefault(int(row.active_pos_1m), []).append(float(row.level_price))
            if int(row.sweep_pos_1m) >= 0:
                # For target selection, a level touched anywhere in the same 1m
                # entry bar is treated as already consumed; intrabar ordering is
                # unknowable from OHLC and must not be assumed favorable.
                self.removals.setdefault(int(row.sweep_pos_1m), []).append(float(row.level_price))
        self.update_positions = sorted(set(self.additions) | set(self.removals))
        self.pointer = 0
        self.current_pos = -1
        self.tree = FenwickTree(len(self.prices))

    def advance(self, pos: int) -> None:
        if pos < self.current_pos:
            raise ValueError("active liquidity index requires nondecreasing event positions")
        while self.pointer < len(self.update_positions) and self.update_positions[self.pointer] <= pos:
            update_pos = self.update_positions[self.pointer]
            for price in self.removals.get(update_pos, []):
                self.tree.add(self.rank[price], -1)
            for price in self.additions.get(update_pos, []):
                self.tree.add(self.rank[price], 1)
            self.pointer += 1
        self.current_pos = int(pos)

    def _count_rank(self, left: int, right: int) -> int:
        return self.tree.range_sum(left, right)

    def nearest_above(self, price: float) -> float:
        if not len(self.prices):
            return np.nan
        left = int(np.searchsorted(self.prices, float(price), side="right"))
        if left >= len(self.prices) or self._count_rank(left, len(self.prices)) <= 0:
            return np.nan
        lo, hi = left, len(self.prices) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._count_rank(left, mid + 1) > 0:
                hi = mid
            else:
                lo = mid + 1
        return float(self.prices[lo])

    def nearest_below(self, price: float) -> float:
        if not len(self.prices):
            return np.nan
        right = int(np.searchsorted(self.prices, float(price), side="left"))
        if right <= 0 or self._count_rank(0, right) <= 0:
            return np.nan
        lo, hi = 0, right - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count_rank(mid, right) > 0:
                lo = mid
            else:
                hi = mid - 1
        return float(self.prices[lo])


def attach_execution_outcomes(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    events: pd.DataFrame,
    *,
    execution_minutes: int,
    config: MSS2Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Attach FVG-limit fill outcomes, structural stop, opposing-liquidity target, and MFE/MAE.

    Same-bar stop/target collisions are resolved stop-first (pessimistic).  The
    entry order is never allowed on the MSS/FVG close bar; ``entry_fill_pos`` is
    strictly after ``mss_pos``.
    """
    cfg = (config or MSS2Config()).validate()
    if events.empty:
        return events.copy()
    minutes = int(execution_minutes)
    bars1 = normalize_1m_bars(primary_1m)
    # Signal logic is built on 1m/2m, but once a resting order is live all
    # fills/stops/targets are evaluated on original 1m OHLC to remove 2m
    # intrabar ordering ambiguity.
    exec_bars = aggregate_bars(bars1, minutes)
    high = bars1["high"].to_numpy(dtype=float)
    low = bars1["low"].to_numpy(dtype=float)
    close = bars1["close"].to_numpy(dtype=float)
    result = events.copy().sort_values(["entry_fill_pos_1m", "event_id"], kind="stable").reset_index(drop=True)

    # Opposing-liquidity targets are queried at actual entry time using the
    # 1m lifecycle, never using levels formed/confirmed later.
    buy_all = _ActiveNearestLiquidityIndex(classified_lifecycle, bars1.index, "high", None)
    sell_all = _ActiveNearestLiquidityIndex(classified_lifecycle, bars1.index, "low", None)
    buy_15 = _ActiveNearestLiquidityIndex(classified_lifecycle, bars1.index, "high", "15m")
    sell_15 = _ActiveNearestLiquidityIndex(classified_lifecycle, bars1.index, "low", "15m")

    max_hold_bars = max(1, int(cfg.max_outcome_minutes))
    stop_buffer = cfg.stop_buffer_bps / 10_000.0
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter(
        f"[outcomes-{minutes}m]", total=len(result), every=max(1, len(result) // 100), enabled=show_progress
    )
    for loop_i, source in enumerate(result.itertuples(index=False), start=1):
        reporter.update(loop_i)
        row = source._asdict()
        entry_pos = int(row.get("entry_fill_pos_1m", -1))
        if entry_pos < 0 or entry_pos >= len(bars1):
            row["filled_flag"] = 0
            rows.append(row)
            continue
        row["filled_flag"] = 1
        direction = int(row["trade_direction"])
        entry = float(row["entry_price"])
        sweep_extreme = float(row["sweep_extreme"])
        stop = sweep_extreme * (1.0 - stop_buffer) if direction > 0 else sweep_extreme * (1.0 + stop_buffer)
        risk = (entry - stop) if direction > 0 else (stop - entry)
        row["stop_price"] = stop
        row["risk_price"] = risk
        row["risk_bps"] = risk / entry * 10_000.0 if entry > EPS else np.nan
        base_entry_pos = int(entry_pos)
        if direction > 0:
            buy_all.advance(base_entry_pos)
            buy_15.advance(base_entry_pos)
            target_any = buy_all.nearest_above(entry)
            target_15 = buy_15.nearest_above(entry)
        else:
            sell_all.advance(base_entry_pos)
            sell_15.advance(base_entry_pos)
            target_any = sell_all.nearest_below(entry)
            target_15 = sell_15.nearest_below(entry)
        row["opposing_liquidity_target_any"] = target_any
        row["opposing_liquidity_target_15m"] = target_15
        end = min(len(bars1) - 1, entry_pos + max_hold_bars)
        if risk <= EPS:
            row["valid_risk_flag"] = 0
            rows.append(row)
            continue
        row["valid_risk_flag"] = 1
        adverse_high = high[entry_pos : end + 1]
        adverse_low = low[entry_pos : end + 1]
        favorable_high = high[entry_pos + 1 : end + 1] if entry_pos < end else np.array([entry])
        favorable_low = low[entry_pos + 1 : end + 1] if entry_pos < end else np.array([entry])
        if direction > 0:
            mfe = max(0.0, float(np.nanmax(favorable_high) - entry)) / risk
            mae = max(0.0, float(entry - np.nanmin(adverse_low))) / risk
        else:
            mfe = max(0.0, float(entry - np.nanmin(favorable_low))) / risk
            mae = max(0.0, float(np.nanmax(adverse_high) - entry)) / risk
        row["mfe_r_180m"] = mfe
        row["mae_r_180m"] = mae

        for horizon_min in cfg.outcome_horizons_minutes:
            horizon_bars = max(1, int(horizon_min))
            pos = min(len(bars1) - 1, entry_pos + horizon_bars)
            gross = direction * (float(close[pos]) / entry - 1.0)
            row[f"close_return_{int(horizon_min)}m"] = gross

        for r_target in cfg.fixed_r_targets:
            target = entry + direction * float(r_target) * risk
            outcome = "timeout"
            exit_pos = end
            exit_price = float(close[end])
            for pos in range(entry_pos, end + 1):
                stop_hit = low[pos] <= stop if direction > 0 else high[pos] >= stop
                target_hit = high[pos] >= target if direction > 0 else low[pos] <= target
                if stop_hit:
                    outcome, exit_pos, exit_price = "stop", pos, stop
                    break
                if pos > entry_pos and target_hit:
                    outcome, exit_pos, exit_price = "target", pos, target
                    break
            token = str(float(r_target)).replace(".", "p")
            row[f"r{token}_outcome"] = outcome
            row[f"r{token}_exit_pos"] = int(exit_pos)
            row[f"r{token}_gross_return"] = direction * (exit_price / entry - 1.0)

        for target_name, target in (("liq15", target_15), ("liqany", target_any)):
            if not np.isfinite(target):
                row[f"{target_name}_outcome"] = "no_target"
                row[f"{target_name}_gross_return"] = np.nan
                continue
            if direction > 0 and target <= entry or direction < 0 and target >= entry:
                row[f"{target_name}_outcome"] = "invalid_target"
                row[f"{target_name}_gross_return"] = np.nan
                continue
            outcome = "timeout"
            exit_price = float(close[end])
            for pos in range(entry_pos, end + 1):
                stop_hit = low[pos] <= stop if direction > 0 else high[pos] >= stop
                target_hit = high[pos] >= target if direction > 0 else low[pos] <= target
                if stop_hit:
                    outcome, exit_price = "stop", stop
                    break
                if pos > entry_pos and target_hit:
                    outcome, exit_price = "target", float(target)
                    break
            row[f"{target_name}_outcome"] = outcome
            row[f"{target_name}_gross_return"] = direction * (exit_price / entry - 1.0)
        rows.append(row)
    reporter.close()
    out = pd.DataFrame(rows)
    return out.sort_values(["mss_available_time", "event_id"], kind="stable").reset_index(drop=True)


def split_features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically separate causal event features from forward outcome labels."""
    if frame.empty:
        return frame.copy(), frame.copy()
    future_prefixes = (
        "future_eventual_order_",
        "entry_",
        "filled_flag",
        "stop_price",
        "risk_price",
        "risk_bps",
        "opposing_liquidity_target_",
        "valid_risk_flag",
        "mfe_r_",
        "mae_r_",
        "close_return_",
        "r1p0_",
        "r2p0_",
        "r3p0_",
        "liq15_",
        "liqany_",
    )
    ids = [name for name in ("event_id", "level_id", "execution_minutes", "reference_mode") if name in frame.columns]
    label_columns = ids + [name for name in frame.columns if name.startswith(future_prefixes)]
    label_columns = list(dict.fromkeys(label_columns))
    labels = frame.loc[:, label_columns].copy()
    feature_columns = [name for name in frame.columns if name not in set(label_columns) or name in ids]
    features = frame.loc[:, feature_columns].copy()
    forbidden = [name for name in features.columns if name.startswith(future_prefixes)]
    if forbidden:
        raise RuntimeError(f"future outcome columns leaked into feature table: {forbidden}")
    return features, labels


def causal_audit(
    levels: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, int]:
    result: dict[str, int] = {
        "levels": int(len(levels)),
        "events": int(len(events)),
        "level_available_before_pivot_bar_end": 0,
        "mss_available_before_sweep_exec_available": 0,
        "entry_not_after_mss": 0,
        "mss_reference_not_pre_sweep": 0,
        "mss_reference_available_after_sweep_bar_start": 0,
    }
    if not levels.empty:
        result["level_available_before_pivot_bar_end"] = int(
            (pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).sum()
        )
    if not events.empty:
        result["mss_available_before_sweep_exec_available"] = int(
            (pd.to_datetime(events["mss_available_time"]) <= pd.to_datetime(events["sweep_exec_available_time"])).sum()
        )
        filled = events.loc[pd.to_numeric(events.get("entry_fill_pos_1m", -1), errors="coerce").fillna(-1).ge(0)]
        if not filled.empty:
            result["entry_not_after_mss"] = int(
                (pd.to_datetime(filled["entry_time"]) < pd.to_datetime(filled["mss_available_time"])).sum()
            )
        result["mss_reference_not_pre_sweep"] = int(
            (pd.to_numeric(events["mss_reference_pivot_pos"]) >= pd.to_numeric(events["sweep_exec_pos"])).sum()
        )
        if "mss_reference_available_time" in events.columns and "sweep_exec_bar_time" in events.columns:
            result["mss_reference_available_after_sweep_bar_start"] = int(
                (pd.to_datetime(events["mss_reference_available_time"]) > pd.to_datetime(events["sweep_exec_bar_time"])).sum()
            )
    return result
