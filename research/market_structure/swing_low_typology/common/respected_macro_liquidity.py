#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal respected macro-liquidity features.

This module deliberately rejects the idea that every recent low is liquidity.
A level is eligible only after at least two *causally confirmed* higher-timeframe
pivot lows occur near the same price.  By default at least one respect must come
from 1H or 4H, so two 15m pivots alone are not promoted to macro liquidity.  The second confirmation makes the level
available; nothing is backfilled to earlier 1m bars.

The level is then tracked until its first sweep or a predeclared expiry.  Sweep
is an event transition, not a persistent ``price is below an old low`` state.
A same-bar or short-window close reclaim is distinguished from acceptance below.
Observed aggressive selling is treated only as a proxy for stop/forced flow.

All returned features are available at the current closed 1m bar.  Future bars
are never used to create a feature for an earlier candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12
CONTEXT_GROUP = "Q1_respected_macro_context"
EVENT_GROUP = "Q2_first_sweep_reclaim"
ORDERFLOW_GROUP = "Q3_sweep_orderflow"


@dataclass(frozen=True)
class RespectedLiquidityBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    diagnostics: pd.DataFrame
    group_membership: pd.DataFrame
    levels: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_divide(numerator: np.ndarray | pd.Series, denominator: np.ndarray | pd.Series) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=float)
    return np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > EPS))


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    return positive.median() if not positive.empty else pd.Timedelta(minutes=1)


def _pivot_events(
    bars: pd.DataFrame,
    *,
    minutes: int,
    left_bars: int,
    right_bars: int,
    timeframe_weight: float,
) -> pd.DataFrame:
    """Return causally available pivot-low events for one timeframe."""

    if minutes < 5 or left_bars < 1 or right_bars < 1:
        raise ValueError("macro pivot configuration must be positive")
    low_1m = _numeric(bars, "low")
    close_1m = _numeric(bars, "close")
    rule = f"{int(minutes)}min"
    htf = pd.DataFrame(
        {
            "low": low_1m.resample(rule, label="left", closed="left").min(),
            "close": close_1m.resample(rule, label="left", closed="left").last(),
        }
    ).dropna()
    minimum = left_bars + right_bars + 1
    if len(htf) < minimum:
        return pd.DataFrame()

    values = htf["low"].to_numpy(dtype=float)
    pivot = np.ones(len(values), dtype=bool)
    pivot[:left_bars] = False
    pivot[len(values) - right_bars :] = False
    for lag in range(1, left_bars + 1):
        pivot &= values < np.roll(values, lag)
    for lead in range(1, right_bars + 1):
        pivot &= values <= np.roll(values, -lead)
    positions = np.flatnonzero(pivot)
    if not len(positions):
        return pd.DataFrame()

    delta = pd.Timedelta(minutes=int(minutes))
    formed = htf.index[positions]
    # A left-labelled bar closes one delta after its start.  The pivot is known
    # only after all right-side bars have also closed.
    available = formed + (right_bars + 1) * delta
    reaction_end = np.minimum(positions + right_bars, len(htf) - 1)
    reaction_close = np.array(
        [htf["close"].iloc[pos + 1 : end + 1].max() for pos, end in zip(positions, reaction_end)],
        dtype=float,
    )
    levels = values[positions]
    reaction_bp = np.maximum(_safe_divide(reaction_close - levels, levels) * 10_000.0, 0.0)
    return (
        pd.DataFrame(
            {
                "pivot_timeframe_min": int(minutes),
                "pivot_weight": float(timeframe_weight),
                "pivot_level": levels,
                "pivot_formed_time": formed,
                "pivot_available_time": available,
                "pivot_reaction_bp": reaction_bp,
            }
        )
        .sort_values(
            ["pivot_available_time", "pivot_formed_time", "pivot_timeframe_min", "pivot_level"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_respected_level_table(
    bars: pd.DataFrame,
    *,
    pivot_minutes: Sequence[int] = (15, 60, 240),
    pivot_weights: Sequence[float] = (1.0, 2.0, 4.0),
    left_bars: int = 2,
    right_bars: int = 2,
    cluster_tolerance_bp: float = 25.0,
    minimum_respects: int = 2,
    minimum_macro_timeframe_min: int = 60,
    minimum_respect_separation_minutes: int = 60,
    formation_max_days: int = 45,
    expiry_days_by_timeframe: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build non-overlapping respected levels from repeated HTF pivots.

    Provisional clusters use only pivot events already available at each step.
    A cluster is emitted once it reaches ``minimum_respects``; later pivots do
    not retroactively strengthen the emitted level.  This keeps the level table
    simple and strictly causal for the first increment study.
    """

    if len(tuple(pivot_minutes)) != len(tuple(pivot_weights)):
        raise ValueError("pivot_minutes and pivot_weights must have equal length")
    if minimum_respects < 2:
        raise ValueError("minimum_respects must be >= 2")
    if minimum_macro_timeframe_min < min(int(value) for value in pivot_minutes):
        raise ValueError("minimum_macro_timeframe_min cannot be below the smallest pivot timeframe")
    if minimum_respect_separation_minutes < 1:
        raise ValueError("minimum_respect_separation_minutes must be positive")
    events = []
    for minutes, weight in zip(pivot_minutes, pivot_weights):
        part = _pivot_events(
            bars,
            minutes=int(minutes),
            left_bars=int(left_bars),
            right_bars=int(right_bars),
            timeframe_weight=float(weight),
        )
        if not part.empty:
            events.append(part)
    if not events:
        return pd.DataFrame(), pd.DataFrame()
    # Multiple timeframe pivots can become available at the same timestamp.
    # Pandas' default quicksort is not stable, so sorting only by availability
    # can assign a different processing order when the same history is rebuilt
    # from a truncated prefix.  That changes sequential level ids and, when
    # simultaneous pivots share a provisional cluster, can also change which
    # causal respects are included in the emitted level.  Use a fully
    # deterministic chronological tie-break order.
    pivots = (
        pd.concat(events, ignore_index=True)
        .sort_values(
            [
                "pivot_available_time",
                "pivot_formed_time",
                "pivot_timeframe_min",
                "pivot_level",
                "pivot_weight",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    tolerance = float(cluster_tolerance_bp) / 10_000.0
    max_age = pd.Timedelta(days=int(formation_max_days))
    provisional: list[dict[str, object]] = []
    emitted: list[dict[str, object]] = []
    next_id = 1

    for row in pivots.itertuples(index=False):
        now = pd.Timestamp(row.pivot_available_time)
        provisional = [item for item in provisional if now - pd.Timestamp(item["first_available_time"]) <= max_age]
        level = float(row.pivot_level)
        best_index: int | None = None
        best_distance = np.inf
        for index, item in enumerate(provisional):
            center = float(item["weighted_level_sum"]) / max(float(item["weight_sum"]), EPS)
            distance = abs(level / center - 1.0)
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance
        event = {
            "level": level,
            "weight": float(row.pivot_weight),
            "timeframe": int(row.pivot_timeframe_min),
            "available_time": now,
            "formed_time": pd.Timestamp(row.pivot_formed_time),
            "reaction_bp": float(row.pivot_reaction_bp),
        }
        if best_index is None:
            provisional.append(
                {
                    "events": [event],
                    "weighted_level_sum": level * float(row.pivot_weight),
                    "weight_sum": float(row.pivot_weight),
                    "first_available_time": now,
                }
            )
            continue
        cluster = provisional[best_index]
        cluster_events = list(cluster["events"]) + [event]  # type: ignore[arg-type]
        cluster["events"] = cluster_events
        cluster["weighted_level_sum"] = float(cluster["weighted_level_sum"]) + level * float(row.pivot_weight)
        cluster["weight_sum"] = float(cluster["weight_sum"]) + float(row.pivot_weight)
        separation = pd.Timedelta(minutes=int(minimum_respect_separation_minutes))
        formed_sorted = sorted(pd.Timestamp(item["formed_time"]) for item in cluster_events)
        respect_times: list[pd.Timestamp] = []
        for formed_time in formed_sorted:
            if not respect_times or formed_time - respect_times[-1] >= separation:
                respect_times.append(formed_time)
        if len(respect_times) < minimum_respects:
            continue
        if max(int(item["timeframe"]) for item in cluster_events) < int(minimum_macro_timeframe_min):
            continue

        weighted_level = float(cluster["weighted_level_sum"]) / max(float(cluster["weight_sum"]), EPS)
        timeframes = [int(item["timeframe"]) for item in cluster_events]
        reactions = [float(item["reaction_bp"]) for item in cluster_events]
        available_times = [pd.Timestamp(item["available_time"]) for item in cluster_events]
        max_tf = max(timeframes)
        expiry_map = expiry_days_by_timeframe or {15: 14, 60: 45, 240: 120}
        expiry_days = int(expiry_map.get(max_tf, max(expiry_map.values())))
        strength = (
            float(cluster["weight_sum"])
            + min(float(np.median(reactions)) / 100.0, 5.0)
            + 0.5 * len(set(timeframes))
        )
        emitted.append(
            {
                "level_id": next_id,
                "level_price": weighted_level,
                "available_time": max(available_times),
                "expiry_time": max(available_times) + pd.Timedelta(days=expiry_days),
                "respect_count": len(respect_times),
                "timeframe_count": len(set(timeframes)),
                "max_timeframe_min": max_tf,
                "strength": strength,
                "median_reaction_bp": float(np.median(reactions)),
                "respect_span_bars_1m": int((max(respect_times) - min(respect_times)) / pd.Timedelta(minutes=1)),
                "source_timeframes": "|".join(str(value) for value in sorted(set(timeframes))),
            }
        )
        next_id += 1
        provisional.pop(best_index)

    levels = pd.DataFrame(emitted)
    diagnostics = pd.DataFrame(
        [
            {
                "metric": "confirmed_pivot_events",
                "value": int(len(pivots)),
            },
            {
                "metric": "respected_level_count",
                "value": int(len(levels)),
            },
            {
                "metric": "unqualified_provisional_clusters",
                "value": int(len(provisional)),
            },
        ]
    )
    return levels, diagnostics


def _feature_row(name: str, group: str, description: str, causal_rule: str) -> dict[str, object]:
    return {
        "feature": name,
        "feature_group": group,
        "source": "respected_macro_liquidity",
        "description": description,
        "causal_rule": causal_rule,
    }


def build_respected_macro_liquidity_features(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    pivot_minutes: Sequence[int] = (15, 60, 240),
    pivot_weights: Sequence[float] = (1.0, 2.0, 4.0),
    left_bars: int = 2,
    right_bars: int = 2,
    cluster_tolerance_bp: float = 25.0,
    minimum_respects: int = 2,
    minimum_macro_timeframe_min: int = 60,
    minimum_respect_separation_minutes: int = 60,
    formation_max_days: int = 45,
    approach_tolerance_bp: float = 35.0,
    reclaim_window_bars: int = 3,
    accept_below_bars: int = 3,
    accept_depth_bp: float = 75.0,
    show_progress: bool = False,
) -> RespectedLiquidityBuildResult:
    """Attach respected macro level context and first-sweep process features."""

    required_bars = {
        "open", "high", "low", "close", "notional", "trades_count", "delta_notional",
        "buy_notional", "sell_notional", "large_buy_notional", "large_sell_notional", "large_delta_notional",
    }
    missing = sorted(required_bars.difference(bars.columns))
    if missing:
        raise RuntimeError(f"respected liquidity builder missing trade-bar fields: {missing}")
    required_candidates = {"event_id", "extreme_pos", "feature_available_time"}
    missing_candidates = sorted(required_candidates.difference(candidates.columns))
    if missing_candidates:
        raise RuntimeError(f"respected liquidity builder missing candidate fields: {missing_candidates}")
    index = pd.DatetimeIndex(bars.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise RuntimeError("bars index must be unique and increasing")
    positions = pd.to_numeric(candidates["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if positions.min(initial=0) < 0 or positions.max(initial=0) >= len(bars):
        raise RuntimeError("candidate positions outside bars")

    levels, level_diagnostics = build_respected_level_table(
        bars,
        pivot_minutes=pivot_minutes,
        pivot_weights=pivot_weights,
        left_bars=left_bars,
        right_bars=right_bars,
        cluster_tolerance_bp=cluster_tolerance_bp,
        minimum_respects=minimum_respects,
        minimum_macro_timeframe_min=minimum_macro_timeframe_min,
        minimum_respect_separation_minutes=minimum_respect_separation_minutes,
        formation_max_days=formation_max_days,
    )

    n_candidates = len(candidates)
    candidate_times = pd.to_datetime(candidates["feature_available_time"]).to_numpy(dtype="datetime64[ns]")
    bar_delta = _infer_bar_delta(index)
    # Pandas may preserve a microsecond-resolution DatetimeIndex when data is
    # loaded from SQLite (notably with newer Pandas versions).  ``Timestamp.value``
    # is always nanoseconds, while ``DatetimeIndex.view("i8")`` uses the index
    # native unit.  Comparing those raw integers silently sends every level to
    # the end of the bar array.  Normalize explicitly to ns before searchsorted.
    bar_available_ns = (
        pd.DatetimeIndex(index + bar_delta)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )
    candidate_positions = positions
    if len(candidate_positions) > 1 and np.any(np.diff(candidate_positions) < 0):
        raise RuntimeError("candidates must be sorted by extreme_pos")

    close = _numeric(bars, "close").to_numpy(dtype=float)
    low = _numeric(bars, "low").to_numpy(dtype=float)
    notional = _numeric(bars, "notional")
    trades = _numeric(bars, "trades_count")
    delta = _numeric(bars, "delta_notional")
    buy = _numeric(bars, "buy_notional")
    sell = _numeric(bars, "sell_notional")
    large_buy = _numeric(bars, "large_buy_notional")
    large_sell = _numeric(bars, "large_sell_notional")
    large_delta = _numeric(bars, "large_delta_notional")

    prior_notional = notional.shift(1).rolling(60, min_periods=20).mean()
    prior_trades = trades.shift(1).rolling(60, min_periods=20).mean()
    notional_intensity = _safe_divide(notional, prior_notional)
    trades_intensity = _safe_divide(trades, prior_trades)
    neg_delta_ratio = np.maximum(-_safe_divide(delta, notional), 0.0)
    aggressive_sell_ratio = _safe_divide(sell, buy + sell)
    large_sell_ratio = _safe_divide(large_sell, large_buy + large_sell)
    large_neg_delta_ratio = np.maximum(-_safe_divide(large_delta, notional), 0.0)

    # Candidate arrays.  NaN means no qualified macro level was active.
    nearest_abs_distance = np.full(n_candidates, np.inf, dtype=float)
    nearest_distance = np.full(n_candidates, np.nan, dtype=float)
    nearest_strength = np.full(n_candidates, np.nan, dtype=float)
    nearest_age = np.full(n_candidates, np.nan, dtype=float)
    nearest_respects = np.full(n_candidates, np.nan, dtype=float)
    nearest_tf = np.full(n_candidates, np.nan, dtype=float)
    nearest_reaction = np.full(n_candidates, np.nan, dtype=float)
    nearest_span = np.full(n_candidates, np.nan, dtype=float)
    level_count_25 = np.zeros(n_candidates, dtype=np.int16)
    level_count_50 = np.zeros(n_candidates, dtype=np.int16)
    level_count_100 = np.zeros(n_candidates, dtype=np.int16)
    approach = np.zeros(n_candidates, dtype=bool)

    event_strength = np.full(n_candidates, -np.inf, dtype=float)
    first_sweep = np.zeros(n_candidates, dtype=bool)
    pending_sweep = np.zeros(n_candidates, dtype=bool)
    reclaim = np.zeros(n_candidates, dtype=bool)
    accept_below = np.zeros(n_candidates, dtype=bool)
    bars_since_sweep = np.full(n_candidates, np.nan, dtype=float)
    sweep_depth = np.full(n_candidates, np.nan, dtype=float)
    reclaim_strength = np.full(n_candidates, np.nan, dtype=float)
    sweep_level_strength = np.full(n_candidates, np.nan, dtype=float)
    sweep_notional_intensity = np.full(n_candidates, np.nan, dtype=float)
    sweep_trades_intensity = np.full(n_candidates, np.nan, dtype=float)
    sweep_neg_delta = np.full(n_candidates, np.nan, dtype=float)
    sweep_aggressive_sell = np.full(n_candidates, np.nan, dtype=float)
    sweep_large_sell = np.full(n_candidates, np.nan, dtype=float)
    sweep_large_neg_delta = np.full(n_candidates, np.nan, dtype=float)
    post_sell_decay = np.full(n_candidates, np.nan, dtype=float)
    post_delta_recovery = np.full(n_candidates, np.nan, dtype=float)
    post_price_recovery = np.full(n_candidates, np.nan, dtype=float)

    diagnostics_rows: list[dict[str, object]] = []
    availability_violations = 0
    levels_overlapping_candidate_time = 0
    levels_with_valid_bar_window = 0
    levels_below_at_activation = 0
    levels_with_candidate_overlap = 0
    candidate_time_min = pd.Timestamp(candidate_times.min()) if n_candidates else pd.NaT
    candidate_time_max = pd.Timestamp(candidate_times.max()) if n_candidates else pd.NaT
    reporter = (
        ProgressReporter("[liquidity] respected macro levels", total=max(len(levels), 1), every=max(1, len(levels) // 100))
        if ProgressReporter is not None and show_progress
        else None
    )

    for level_index, level_row in enumerate(levels.itertuples(index=False), start=1):
        level_price = float(level_row.level_price)
        active_time = pd.Timestamp(level_row.available_time)
        expiry_time = pd.Timestamp(level_row.expiry_time)
        if n_candidates and active_time <= candidate_time_max and expiry_time >= candidate_time_min:
            levels_overlapping_candidate_time += 1
        active_pos = int(np.searchsorted(bar_available_ns, active_time.value, side="left"))
        expiry_pos = int(np.searchsorted(bar_available_ns, expiry_time.value, side="right") - 1)
        if active_pos >= len(index) or expiry_pos < active_pos:
            diagnostics_rows.append(
                {
                    "level_id": int(level_row.level_id),
                    "status": "outside_bar_range",
                    "available_time": active_time,
                    "expiry_time": expiry_time,
                    "level_price": level_price,
                }
            )
            if reporter is not None:
                reporter.update(level_index)
            continue
        levels_with_valid_bar_window += 1
        expiry_pos = min(expiry_pos, len(index) - 1)
        # A newly qualified support level must still be below the market.  If
        # price is already below at availability, it is not a clean future
        # liquidity target and is skipped.
        if close[active_pos] <= level_price:
            levels_below_at_activation += 1
            diagnostics_rows.append(
                {
                    "level_id": int(level_row.level_id),
                    "status": "below_at_activation",
                    "available_time": active_time,
                    "expiry_time": expiry_time,
                    "level_price": level_price,
                }
            )
            if reporter is not None:
                reporter.update(level_index)
            continue

        segment_low = low[active_pos : expiry_pos + 1]
        swept_rel = np.flatnonzero(segment_low < level_price)
        sweep_pos = active_pos + int(swept_rel[0]) if len(swept_rel) else None
        resolution_pos = expiry_pos
        reclaim_pos: int | None = None
        accept_pos: int | None = None
        status = "expired_unswept"
        if sweep_pos is not None:
            event_end = min(expiry_pos, sweep_pos + int(reclaim_window_bars))
            consecutive_below = 0
            for pos in range(sweep_pos, event_end + 1):
                if close[pos] >= level_price:
                    reclaim_pos = pos
                    status = "reclaimed"
                    break
                consecutive_below += 1
                depth_bp = max((level_price - low[pos]) / level_price * 10_000.0, 0.0)
                if consecutive_below >= int(accept_below_bars) or depth_bp >= float(accept_depth_bp):
                    accept_pos = pos
                    status = "accepted_below"
                    break
            if reclaim_pos is None and accept_pos is None:
                accept_pos = event_end
                status = "unreclaimed_timeout"
            resolution_pos = reclaim_pos if reclaim_pos is not None else int(accept_pos)

        active_end = resolution_pos
        left = int(np.searchsorted(candidate_positions, active_pos, side="left"))
        right = int(np.searchsorted(candidate_positions, active_end, side="right"))
        if right > left:
            levels_with_candidate_overlap += 1
            subset_index = np.arange(left, right)
            subset_positions = candidate_positions[left:right]
            availability_violations += int(np.sum(candidate_times[left:right] < np.datetime64(active_time)))
            distance = (close[subset_positions] / level_price - 1.0) * 10_000.0
            absolute = np.abs(distance)
            level_count_25[subset_index] += (absolute <= 25.0).astype(np.int16)
            level_count_50[subset_index] += (absolute <= 50.0).astype(np.int16)
            level_count_100[subset_index] += (absolute <= 100.0).astype(np.int16)
            approach[subset_index] |= low[subset_positions] <= level_price * (1.0 + float(approach_tolerance_bp) / 10_000.0)
            choose = absolute < nearest_abs_distance[subset_index]
            chosen = subset_index[choose]
            nearest_abs_distance[chosen] = absolute[choose]
            nearest_distance[chosen] = distance[choose]
            nearest_strength[chosen] = float(level_row.strength)
            nearest_age[chosen] = subset_positions[choose] - active_pos
            nearest_respects[chosen] = float(level_row.respect_count)
            nearest_tf[chosen] = float(level_row.max_timeframe_min)
            nearest_reaction[chosen] = float(level_row.median_reaction_bp)
            nearest_span[chosen] = float(level_row.respect_span_bars_1m)

        if sweep_pos is not None:
            event_left = int(np.searchsorted(candidate_positions, sweep_pos, side="left"))
            event_right = int(np.searchsorted(candidate_positions, resolution_pos, side="right"))
            if event_right > event_left:
                event_indices = np.arange(event_left, event_right)
                event_positions = candidate_positions[event_left:event_right]
                choose_event = float(level_row.strength) > event_strength[event_indices]
                chosen = event_indices[choose_event]
                chosen_positions = event_positions[choose_event]
                event_strength[chosen] = float(level_row.strength)
                since = chosen_positions - sweep_pos
                first_sweep[chosen] = since == 0
                pending_sweep[chosen] = (since > 0) & (chosen_positions < resolution_pos)
                reclaim[chosen] = reclaim_pos is not None and chosen_positions == reclaim_pos
                accept_below[chosen] = accept_pos is not None and chosen_positions == accept_pos
                bars_since_sweep[chosen] = since
                depth = np.maximum((level_price - low[sweep_pos]) / level_price * 10_000.0, 0.0)
                sweep_depth[chosen] = depth
                reclaim_strength[chosen] = np.maximum((close[chosen_positions] - level_price) / level_price * 10_000.0, 0.0)
                sweep_level_strength[chosen] = float(level_row.strength)
                sweep_notional_intensity[chosen] = notional_intensity[sweep_pos]
                sweep_trades_intensity[chosen] = trades_intensity[sweep_pos]
                sweep_neg_delta[chosen] = neg_delta_ratio[sweep_pos]
                sweep_aggressive_sell[chosen] = aggressive_sell_ratio[sweep_pos]
                sweep_large_sell[chosen] = large_sell_ratio[sweep_pos]
                sweep_large_neg_delta[chosen] = large_neg_delta_ratio[sweep_pos]
                post_sell_decay[chosen] = neg_delta_ratio[sweep_pos] - neg_delta_ratio[chosen_positions]
                cumulative_delta = pd.Series(delta.iloc[sweep_pos : resolution_pos + 1].to_numpy(dtype=float)).cumsum().to_numpy()
                cumulative_notional = pd.Series(notional.iloc[sweep_pos : resolution_pos + 1].to_numpy(dtype=float)).cumsum().to_numpy()
                recovery_ratio = _safe_divide(cumulative_delta, cumulative_notional)
                post_delta_recovery[chosen] = recovery_ratio[since.astype(int)]
                post_price_recovery[chosen] = (close[chosen_positions] / level_price - 1.0) * 10_000.0

        diagnostics_rows.append(
            {
                "level_id": int(level_row.level_id),
                "status": status,
                "available_time": active_time,
                "level_price": level_price,
                "strength": float(level_row.strength),
                "max_timeframe_min": int(level_row.max_timeframe_min),
                "respect_count": int(level_row.respect_count),
                "sweep_time": index[sweep_pos] + bar_delta if sweep_pos is not None else pd.NaT,
                "resolution_time": index[resolution_pos] + bar_delta if sweep_pos is not None else pd.NaT,
            }
        )
        if reporter is not None:
            reporter.update(level_index)
    if reporter is not None:
        reporter.close()

    output = {
        "rml_has_active_level": np.isfinite(nearest_distance).astype(float),
        "rml_nearest_distance_close_bp": nearest_distance,
        "rml_nearest_abs_distance_bp": np.where(np.isfinite(nearest_abs_distance), nearest_abs_distance, np.nan),
        "rml_nearest_strength": nearest_strength,
        "rml_nearest_age_bars": nearest_age,
        "rml_nearest_respect_count": nearest_respects,
        "rml_nearest_max_timeframe_min": nearest_tf,
        "rml_nearest_median_reaction_bp": nearest_reaction,
        "rml_nearest_respect_span_bars": nearest_span,
        "rml_active_level_count_25bp": level_count_25.astype(float),
        "rml_active_level_count_50bp": level_count_50.astype(float),
        "rml_active_level_count_100bp": level_count_100.astype(float),
        "rml_approach_active_level": approach.astype(float),
        "rml_first_sweep": first_sweep.astype(float),
        "rml_pending_sweep": pending_sweep.astype(float),
        "rml_reclaim": reclaim.astype(float),
        "rml_accept_below": accept_below.astype(float),
        "rml_bars_since_sweep": bars_since_sweep,
        "rml_sweep_depth_bp": sweep_depth,
        "rml_reclaim_strength_bp": reclaim_strength,
        "rml_swept_level_strength": sweep_level_strength,
        "rml_sweep_notional_intensity": sweep_notional_intensity,
        "rml_sweep_trades_intensity": sweep_trades_intensity,
        "rml_sweep_negative_delta_ratio": sweep_neg_delta,
        "rml_sweep_aggressive_sell_ratio": sweep_aggressive_sell,
        "rml_sweep_large_sell_ratio": sweep_large_sell,
        "rml_sweep_large_negative_delta_ratio": sweep_large_neg_delta,
        "rml_post_sweep_sell_decay": post_sell_decay,
        "rml_post_sweep_delta_recovery": post_delta_recovery,
        "rml_post_sweep_price_recovery_bp": post_price_recovery,
    }

    context_names = {
        "rml_has_active_level", "rml_nearest_distance_close_bp", "rml_nearest_abs_distance_bp",
        "rml_nearest_strength", "rml_nearest_age_bars", "rml_nearest_respect_count",
        "rml_nearest_max_timeframe_min", "rml_nearest_median_reaction_bp",
        "rml_nearest_respect_span_bars", "rml_active_level_count_25bp",
        "rml_active_level_count_50bp", "rml_active_level_count_100bp", "rml_approach_active_level",
    }
    event_names = {
        "rml_first_sweep", "rml_pending_sweep", "rml_reclaim", "rml_accept_below",
        "rml_bars_since_sweep", "rml_sweep_depth_bp", "rml_reclaim_strength_bp",
        "rml_swept_level_strength",
    }
    dictionary = []
    membership = []
    for name in output:
        if name in context_names:
            group = CONTEXT_GROUP
            description = "causally active repeated higher-timeframe support context"
        elif name in event_names:
            group = EVENT_GROUP
            description = "first sweep, pending, reclaim or acceptance state of a respected macro level"
        else:
            group = ORDERFLOW_GROUP
            description = "observable trade-bar order flow around the first sweep event"
        dictionary.append(
            _feature_row(
                name,
                group,
                description,
                "level requires repeated confirmed HTF pivots; feature uses current/prior closed bars only",
            )
        )
        membership.append({"feature": name, "feature_group": group})

    feature_frame = candidates.reset_index(drop=True).copy()
    feature_frame = pd.concat([feature_frame, pd.DataFrame({name: np.asarray(value, dtype=np.float32) for name, value in output.items()})], axis=1)
    diagnostics = pd.concat(
        [
            level_diagnostics.assign(scope="level_construction"),
            pd.DataFrame(diagnostics_rows).assign(scope="level_lifecycle"),
            pd.DataFrame(
                [
                    {
                        "scope": "aggregate",
                        "metric": "candidate_active_level_coverage",
                        "value": float(np.isfinite(nearest_distance).mean()) if n_candidates else np.nan,
                        "availability_violations": int(availability_violations),
                        "levels_overlapping_candidate_time": int(levels_overlapping_candidate_time),
                        "levels_with_valid_bar_window": int(levels_with_valid_bar_window),
                        "levels_below_at_activation": int(levels_below_at_activation),
                        "levels_with_candidate_overlap": int(levels_with_candidate_overlap),
                        "first_sweep_candidates": int(first_sweep.sum()),
                        "reclaim_candidates": int(reclaim.sum()),
                        "accept_below_candidates": int(accept_below.sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return RespectedLiquidityBuildResult(
        frame=feature_frame,
        dictionary=pd.DataFrame(dictionary),
        diagnostics=diagnostics,
        group_membership=pd.DataFrame(membership),
        levels=levels,
    )
