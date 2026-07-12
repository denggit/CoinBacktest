#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal mechanism features for hierarchical swing-low typology research.

Every feature in this module is computed from the labelled extreme bar or older
trade bars.  Future bars are never inspected.  The routines operate on bounded
event windows and pre-converted NumPy arrays so the full trade-bar frame is not
copied for each event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover - standalone fallback
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12

MECHANISM_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "confirmation_time",
        "confirmation_available_time",
        "completion_bars",
        "realized_confirmation_move_pct",
        "parent_cluster_id",
        "parent_distance_to_centroid",
        "parent_split",
        "source_subcluster_id",
        "source_subcluster_distance",
        "split",
        "year",
    }
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "notional",
    "trades_count",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)


@dataclass(frozen=True)
class SupportTest:
    order: int
    local_pos: int
    timestamp: pd.Timestamp
    low_price: float
    low_distance_bp: float
    interval_bars: float
    drawdown_depth_bp: float
    rebound_bp: float
    negative_delta_ratio: float
    sell_price_impact: float
    notional_intensity: float
    large_sell_ratio: float


def _safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= EPS:
        return float(default)
    return float(num / den)


def _finite(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean(values: Iterable[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: Iterable[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if arr.size else float("nan")


def _std(values: Iterable[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")


def _linear_slope(values: Iterable[float] | np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= EPS:
        return 0.0
    return float(np.dot(x, y - y.mean()) / den)


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _phase_slices(length: int, bins: int) -> list[slice]:
    edges = np.linspace(0, length, bins + 1, dtype=int)
    out: list[slice] = []
    for idx in range(bins):
        start, end = int(edges[idx]), int(edges[idx + 1])
        if end <= start:
            end = min(length, start + 1)
        out.append(slice(start, end))
    return out


def _numeric_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = [column for column in REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise RuntimeError(f"Mechanism features require rich trade bars; missing={missing}")
    return {
        column: pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype=float, copy=False)
        for column in REQUIRED_COLUMNS
    }


def _local_minima(values: np.ndarray, radius: int = 2) -> list[int]:
    arr = np.asarray(values, dtype=float)
    out: list[int] = []
    for idx in range(radius, max(radius, len(arr) - radius)):
        value = arr[idx]
        if not np.isfinite(value):
            continue
        window = arr[idx - radius : idx + radius + 1]
        if np.isfinite(window).any() and value <= float(np.nanmin(window)):
            out.append(idx)
    return out


def _merge_nearby_tests(candidates: Sequence[int], low: np.ndarray, min_gap: int) -> list[int]:
    if not candidates:
        return []
    merged: list[int] = []
    for idx in sorted({int(x) for x in candidates}):
        if not merged or idx - merged[-1] >= min_gap:
            merged.append(idx)
            continue
        previous = merged[-1]
        if np.isfinite(low[idx]) and (not np.isfinite(low[previous]) or low[idx] < low[previous]):
            merged[-1] = idx
    return merged


def _support_tests(
    *,
    timestamps: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    notional: np.ndarray,
    delta: np.ndarray,
    large_sell: np.ndarray,
    large_gross: np.ndarray,
    support_tolerance_bp: float,
    min_test_gap: int,
    rebound_horizon: int,
    minimum_separation_rebound_bp: float,
) -> list[SupportTest]:
    final_idx = len(low) - 1
    support = float(low[final_idx])
    if not np.isfinite(support) or support <= 0:
        return []
    threshold = support * (1.0 + float(support_tolerance_bp) / 10_000.0)
    candidates = [idx for idx in _local_minima(low, radius=2) if low[idx] <= threshold]
    candidates.append(final_idx)
    merged_positions = _merge_nearby_tests(candidates, low, max(1, int(min_test_gap)))
    test_positions: list[int] = []
    for idx in merged_positions:
        if not test_positions:
            test_positions.append(idx)
            continue
        previous = test_positions[-1]
        segment_high = high[previous : idx + 1]
        rebound_bp = (
            (_safe_ratio(float(np.nanmax(segment_high)), float(low[previous])) - 1.0) * 10_000.0
            if segment_high.size and np.isfinite(segment_high).any()
            else 0.0
        )
        if rebound_bp >= float(minimum_separation_rebound_bp):
            test_positions.append(idx)
        elif np.isfinite(low[idx]) and low[idx] <= low[previous]:
            test_positions[-1] = idx
    if not test_positions or test_positions[-1] != final_idx:
        if test_positions:
            previous = test_positions[-1]
            segment_high = high[previous : final_idx + 1]
            rebound_bp = (
                (_safe_ratio(float(np.nanmax(segment_high)), float(low[previous])) - 1.0) * 10_000.0
                if segment_high.size and np.isfinite(segment_high).any()
                else 0.0
            )
            if rebound_bp < float(minimum_separation_rebound_bp):
                test_positions[-1] = final_idx
            else:
                test_positions.append(final_idx)
        else:
            test_positions.append(final_idx)

    baseline_notional = _median(notional)
    tests: list[SupportTest] = []
    previous_test = -1
    for order, idx in enumerate(test_positions, start=1):
        pre_start = max(previous_test + 1, idx - 12, 0)
        preceding_high = float(np.nanmax(high[pre_start : idx + 1])) if np.isfinite(high[pre_start : idx + 1]).any() else np.nan
        depth_bp = (_safe_ratio(preceding_high, low[idx]) - 1.0) * 10_000.0

        if idx < final_idx:
            next_test = test_positions[order] if order < len(test_positions) else final_idx
            rebound_end = min(final_idx, idx + max(1, int(rebound_horizon)), max(idx + 1, next_test - 1))
            future_high = high[idx + 1 : rebound_end + 1]
            rebound_bp = (
                (_safe_ratio(float(np.nanmax(future_high)), low[idx]) - 1.0) * 10_000.0
                if future_high.size and np.isfinite(future_high).any()
                else np.nan
            )
        else:
            rebound_bp = np.nan

        zone_start = max(0, idx - 1)
        zone_end = min(final_idx + 1, idx + 2)
        zone_notional = float(np.nansum(notional[zone_start:zone_end]))
        zone_delta = float(np.nansum(delta[zone_start:zone_end]))
        negative_delta_ratio = max(0.0, -_safe_ratio(zone_delta, zone_notional, default=0.0))
        sell_price_impact = _safe_ratio(max(0.0, depth_bp / 10_000.0), negative_delta_ratio)
        zone_notional_median = _median(notional[zone_start:zone_end])
        notional_intensity = _safe_ratio(zone_notional_median, baseline_notional)
        zone_large_sell = float(np.nansum(large_sell[zone_start:zone_end]))
        zone_large_gross = float(np.nansum(large_gross[zone_start:zone_end]))
        large_sell_ratio = _safe_ratio(zone_large_sell, zone_large_gross)
        tests.append(
            SupportTest(
                order=order,
                local_pos=idx,
                timestamp=pd.Timestamp(timestamps[idx]),
                low_price=float(low[idx]),
                low_distance_bp=(_safe_ratio(low[idx], support) - 1.0) * 10_000.0,
                interval_bars=float(idx - previous_test) if previous_test >= 0 else np.nan,
                drawdown_depth_bp=float(depth_bp),
                rebound_bp=float(rebound_bp),
                negative_delta_ratio=float(negative_delta_ratio),
                sell_price_impact=float(sell_price_impact),
                notional_intensity=float(notional_intensity),
                large_sell_ratio=float(large_sell_ratio),
            )
        )
        previous_test = idx
    return tests


def _test_features(tests: Sequence[SupportTest], high: np.ndarray, close: np.ndarray, low: np.ndarray) -> dict[str, float]:
    prior = list(tests[:-1])
    lows = np.asarray([test.low_distance_bp for test in tests], dtype=float)
    intervals = np.asarray([test.interval_bars for test in tests[1:]], dtype=float)
    depths = np.asarray([test.drawdown_depth_bp for test in prior], dtype=float)
    rebounds = np.asarray([test.rebound_bp for test in prior], dtype=float)
    negative_delta = np.asarray([test.negative_delta_ratio for test in tests], dtype=float)
    impacts = np.asarray([test.sell_price_impact for test in tests], dtype=float)
    activity = np.asarray([test.notional_intensity for test in tests], dtype=float)
    large_sell = np.asarray([test.large_sell_ratio for test in tests], dtype=float)

    if activity.size >= 3:
        thirds = np.array_split(activity, 3)
        u_shape = _mean(np.r_[thirds[0], thirds[-1]]) - _mean(thirds[1])
    else:
        u_shape = np.nan

    prior_support = _median([test.low_price for test in prior])
    final_test = tests[-1] if tests else None
    final_break_bp = (
        max(0.0, (1.0 - _safe_ratio(final_test.low_price, prior_support)) * 10_000.0)
        if final_test is not None and np.isfinite(prior_support)
        else np.nan
    )
    final_close_reclaim_bp = (
        (_safe_ratio(float(close[-1]), prior_support) - 1.0) * 10_000.0
        if np.isfinite(prior_support)
        else np.nan
    )
    final_wick_reclaim_share = _safe_ratio(
        float(close[-1] - low[-1]),
        float(high[-1] - low[-1]),
    )

    return {
        "support_test_count": float(len(tests)),
        "support_prior_test_count": float(len(prior)),
        "support_test_interval_median": _median(intervals),
        "support_test_interval_cv": _safe_ratio(_std(intervals), _mean(intervals)),
        "support_low_dispersion_bp": _std(lows),
        "support_low_level_slope_bp": _linear_slope(lows),
        "test_depth_weakening_slope": -_linear_slope(depths),
        "test_rebound_strengthening_slope": _linear_slope(rebounds),
        "test_sell_pressure_decay_slope": -_linear_slope(negative_delta),
        "test_sell_impact_decay_slope": -_linear_slope(impacts),
        "test_activity_slope": _linear_slope(activity),
        "test_activity_u_shape": float(u_shape),
        "test_large_sell_decay_slope": -_linear_slope(large_sell),
        "test_large_sell_concentration": float(np.nanmax(large_sell)) if np.isfinite(large_sell).any() else np.nan,
        "final_support_break_bp": float(final_break_bp),
        "final_close_reclaim_bp": float(final_close_reclaim_bp),
        "final_wick_reclaim_share": float(final_wick_reclaim_share),
    }


def _path_features(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    notional: np.ndarray,
    delta: np.ndarray,
    large_delta: np.ndarray,
    large_gross: np.ndarray,
    phase_bins: int,
) -> dict[str, float]:
    slices = _phase_slices(len(close), phase_bins)
    phase_close = np.asarray([_median(close[sl]) for sl in slices], dtype=float)
    phase_low = np.asarray([float(np.nanmin(low[sl])) if np.isfinite(low[sl]).any() else np.nan for sl in slices])
    phase_notional = np.asarray([_median(notional[sl]) for sl in slices], dtype=float)
    phase_delta_ratio = np.asarray(
        [_safe_ratio(float(np.nansum(delta[sl])), float(np.nansum(notional[sl]))) for sl in slices], dtype=float
    )
    phase_large_delta_ratio = np.asarray(
        [_safe_ratio(float(np.nansum(large_delta[sl])), float(np.nansum(large_gross[sl]))) for sl in slices], dtype=float
    )
    phase_returns = np.divide(
        phase_close[1:], phase_close[:-1], out=np.full(max(0, len(phase_close) - 1), np.nan), where=np.abs(phase_close[:-1]) > EPS
    ) - 1.0
    negative_phase_returns = -phase_returns[np.isfinite(phase_returns) & (phase_returns < 0)]
    uniformity = _safe_ratio(1.0, 1.0 + _safe_ratio(_std(negative_phase_returns), _mean(negative_phase_returns), 0.0), 0.0)
    thirds = np.array_split(np.arange(len(phase_returns)), 3) if len(phase_returns) else [np.array([], dtype=int)] * 3
    first_return = _mean(phase_returns[thirds[0]])
    last_return = _mean(phase_returns[thirds[-1]])
    price_acceleration = first_return - last_return

    cumulative_price = np.log(np.maximum(phase_close, EPS)) - np.log(max(float(phase_close[0]), EPS))
    cumulative_cvd = np.nancumsum(np.where(np.isfinite(phase_delta_ratio), phase_delta_ratio, 0.0))
    valid_sync = np.isfinite(cumulative_price) & np.isfinite(cumulative_cvd)
    price_cvd_sync = (
        float(np.corrcoef(cumulative_price[valid_sync], cumulative_cvd[valid_sync])[0, 1])
        if int(valid_sync.sum()) >= 4 and _std(cumulative_price[valid_sync]) > EPS and _std(cumulative_cvd[valid_sync]) > EPS
        else np.nan
    )
    price_return = _safe_ratio(float(close[-1]), float(close[0])) - 1.0
    cvd_ratio = _safe_ratio(float(np.nansum(delta)), float(np.nansum(notional)))
    divergence = price_return - cvd_ratio

    negative_large = np.isfinite(phase_large_delta_ratio) & (phase_large_delta_ratio < 0)
    large_sell_persistence = _safe_ratio(float(_longest_true_run(negative_large)), float(len(negative_large)))
    thirds_phase = np.array_split(np.arange(len(phase_delta_ratio)), 3)
    first_sell = max(0.0, -_mean(phase_delta_ratio[thirds_phase[0]]))
    last_sell = max(0.0, -_mean(phase_delta_ratio[thirds_phase[-1]]))
    sell_pressure_decay = first_sell - last_sell

    phase_log_activity = np.log1p(np.maximum(phase_notional, 0.0))
    first_activity = _mean(phase_log_activity[thirds_phase[0]])
    middle_activity = _mean(phase_log_activity[thirds_phase[1]])
    last_activity = _mean(phase_log_activity[thirds_phase[-1]])
    activity_compression = first_activity - last_activity
    activity_u_shape = _mean([first_activity, last_activity]) - middle_activity

    ranges = np.divide(high - low, np.maximum(close, EPS))
    phase_range = np.asarray([_median(ranges[sl]) for sl in slices], dtype=float)
    range_compression = _mean(phase_range[thirds_phase[0]]) - _mean(phase_range[thirds_phase[-1]])

    bar_returns = np.divide(close, open_, out=np.full_like(close, np.nan), where=np.abs(open_) > EPS) - 1.0
    delta_ratio_bar = np.divide(delta, notional, out=np.full_like(delta, np.nan), where=np.abs(notional) > EPS)
    thirds_bar = np.array_split(np.arange(len(bar_returns)), 3)
    impact_parts: list[float] = []
    for indices in thirds_bar:
        negative_delta = float(np.nansum(np.minimum(delta_ratio_bar[indices], 0.0)))
        negative_return = float(np.nansum(np.minimum(bar_returns[indices], 0.0)))
        impact_parts.append(_safe_ratio(-negative_return, -negative_delta))
    sell_impact_decay = impact_parts[0] - impact_parts[-1] if len(impact_parts) >= 2 else np.nan

    prior_running_low = np.r_[low[0], np.minimum.accumulate(low[:-1])]
    no_new_low = low >= prior_running_low * 0.9998
    negative_delta_no_new_low = _mean(((delta_ratio_bar < 0) & no_new_low).astype(float))

    return {
        "price_decline_uniformity": float(uniformity),
        "price_decline_acceleration": float(price_acceleration),
        "price_phase_direction_changes": float(np.sum(np.sign(phase_returns[1:]) != np.sign(phase_returns[:-1]))) if len(phase_returns) > 1 else np.nan,
        "price_cvd_path_sync": float(price_cvd_sync),
        "price_cvd_path_divergence": float(divergence),
        "large_sell_phase_persistence": float(large_sell_persistence),
        "sell_pressure_decay": float(sell_pressure_decay),
        "sell_impact_decay": float(sell_impact_decay),
        "negative_delta_no_new_low_share": float(negative_delta_no_new_low),
        "activity_compression": float(activity_compression),
        "activity_u_shape": float(activity_u_shape),
        "range_compression": float(range_compression),
        "phase_low_slope": float(_linear_slope(np.divide(phase_low, low[-1]) - 1.0)),
    }


def _metadata_from_event(event: object) -> dict[str, object]:
    def value(name: str, default: object = np.nan) -> object:
        return getattr(event, name, default)

    return {
        "event_id": value("event_id"),
        "extreme_time": pd.Timestamp(value("extreme_time")),
        "feature_available_time": pd.Timestamp(value("feature_available_time")),
        "extreme_pos": int(value("extreme_pos")),
        "extreme_price": float(value("extreme_price")),
        "confirmation_time": pd.Timestamp(value("confirmation_time")),
        "confirmation_available_time": pd.Timestamp(value("confirmation_available_time")),
        "completion_bars": int(value("completion_bars")),
        "realized_confirmation_move_pct": float(value("realized_confirmation_move_pct")),
        "parent_cluster_id": str(value("parent_cluster_id", "C3")),
        "parent_distance_to_centroid": float(value("parent_distance_to_centroid", np.nan)),
        "parent_split": str(value("parent_split", value("split", ""))),
        "source_subcluster_id": str(value("subcluster_id", value("source_subcluster_id", ""))),
        "source_subcluster_distance": float(value("distance_to_train_centroid", value("source_subcluster_distance", np.nan))),
        "split": str(value("split", "")),
        "year": int(pd.Timestamp(value("extreme_time")).year),
    }


def extract_event_mechanism_features(
    *,
    timestamps: pd.DatetimeIndex,
    arrays: dict[str, np.ndarray],
    event: object,
    lookback: int,
    phase_bins: int,
    support_tolerance_bp: float,
    min_test_gap: int,
    rebound_horizon: int,
    minimum_separation_rebound_bp: float = 15.0,
) -> tuple[dict[str, object], list[SupportTest]]:
    """Extract one event using a bounded window ending at the extreme bar."""

    pos = int(getattr(event, "extreme_pos"))
    start = pos - int(lookback) + 1
    if start < 0:
        raise ValueError(f"event {getattr(event, 'event_id', '<unknown>')} lacks {lookback} history bars")
    sl = slice(start, pos + 1)
    local_timestamps = timestamps[sl]
    open_ = arrays["open"][sl]
    high = arrays["high"][sl]
    low = arrays["low"][sl]
    close = arrays["close"][sl]
    notional = arrays["notional"][sl]
    delta = arrays["delta_notional"][sl]
    large_buy = arrays["large_buy_notional"][sl]
    large_sell = arrays["large_sell_notional"][sl]
    large_delta = arrays["large_delta_notional"][sl]
    large_gross = large_buy + large_sell

    tests = _support_tests(
        timestamps=local_timestamps,
        open_=open_,
        high=high,
        low=low,
        close=close,
        notional=notional,
        delta=delta,
        large_sell=large_sell,
        large_gross=large_gross,
        support_tolerance_bp=support_tolerance_bp,
        min_test_gap=min_test_gap,
        rebound_horizon=rebound_horizon,
        minimum_separation_rebound_bp=minimum_separation_rebound_bp,
    )
    row = _metadata_from_event(event)
    row.update(
        _path_features(
            open_=open_,
            high=high,
            low=low,
            close=close,
            notional=notional,
            delta=delta,
            large_delta=large_delta,
            large_gross=large_gross,
            phase_bins=phase_bins,
        )
    )
    row.update(_test_features(tests, high, close, low))

    # Causal composite mechanism observables.  These are still features, not labels.
    row["absorption_observable"] = _mean(
        [
            row.get("negative_delta_no_new_low_share", np.nan),
            max(0.0, float(row.get("sell_pressure_decay", np.nan))),
            max(0.0, float(row.get("sell_impact_decay", np.nan))),
            max(0.0, float(row.get("test_sell_impact_decay_slope", np.nan))),
        ]
    )
    row["compression_observable"] = _mean(
        [
            max(0.0, float(row.get("activity_compression", np.nan))),
            max(0.0, float(row.get("range_compression", np.nan))),
            _safe_ratio(1.0, 1.0 + max(0.0, float(row.get("support_low_dispersion_bp", np.nan))), 0.0),
        ]
    )
    break_bp = float(row.get("final_support_break_bp", np.nan))
    reclaim_bp = float(row.get("final_close_reclaim_bp", np.nan))
    row["spring_observable"] = (
        max(0.0, break_bp) * max(0.0, reclaim_bp) / 10_000.0
        if np.isfinite(break_bp) and np.isfinite(reclaim_bp)
        else np.nan
    )
    row["repeated_support_observable"] = _safe_ratio(
        max(0.0, float(row.get("support_prior_test_count", np.nan))),
        1.0 + max(0.0, float(row.get("support_low_dispersion_bp", np.nan))) / max(1.0, support_tolerance_bp),
    )
    row["slow_accumulation_observable"] = _mean(
        [
            max(0.0, float(row.get("test_rebound_strengthening_slope", np.nan))),
            max(0.0, float(row.get("test_sell_pressure_decay_slope", np.nan))),
            max(0.0, float(row.get("test_large_sell_decay_slope", np.nan))),
            max(0.0, float(row.get("test_activity_u_shape", np.nan))),
        ]
    )
    return row, tests


def build_mechanism_features(
    bars: pd.DataFrame,
    stage2_assignments: pd.DataFrame,
    *,
    lookback: int = 240,
    phase_bins: int = 12,
    support_tolerance_bp: float = 25.0,
    min_test_gap: int = 4,
    rebound_horizon: int = 30,
    minimum_separation_rebound_bp: float = 15.0,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build causal event features and detailed historical support-test paths."""

    if lookback < 60:
        raise ValueError("lookback must be >= 60")
    if phase_bins < 6:
        raise ValueError("phase_bins must be >= 6")
    arrays = _numeric_arrays(bars)
    timestamps = pd.DatetimeIndex(bars.index)
    events = stage2_assignments.copy()
    events["extreme_pos"] = pd.to_numeric(events["extreme_pos"], errors="raise").astype(int)
    events = events[events["extreme_pos"] >= lookback - 1].sort_values("extreme_time").reset_index(drop=True)
    reporter = (
        ProgressReporter("[features] mechanism paths", total=len(events), every=max(1, int(progress_every)))
        if ProgressReporter is not None
        else None
    )

    rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for idx, event in enumerate(events.itertuples(index=False)):
        row, tests = extract_event_mechanism_features(
            timestamps=timestamps,
            arrays=arrays,
            event=event,
            lookback=lookback,
            phase_bins=phase_bins,
            support_tolerance_bp=support_tolerance_bp,
            min_test_gap=min_test_gap,
            rebound_horizon=rebound_horizon,
            minimum_separation_rebound_bp=minimum_separation_rebound_bp,
        )
        rows.append(row)
        for test in tests:
            test_rows.append(
                {
                    "event_id": row["event_id"],
                    "extreme_time": row["extreme_time"],
                    "split": row["split"],
                    "source_subcluster_id": row["source_subcluster_id"],
                    "test_order": test.order,
                    "test_time": test.timestamp,
                    "bars_to_extreme": int((lookback - 1) - test.local_pos),
                    "low_price": test.low_price,
                    "low_distance_bp": test.low_distance_bp,
                    "interval_bars": test.interval_bars,
                    "drawdown_depth_bp": test.drawdown_depth_bp,
                    "rebound_bp": test.rebound_bp,
                    "negative_delta_ratio": test.negative_delta_ratio,
                    "sell_price_impact": test.sell_price_impact,
                    "notional_intensity": test.notional_intensity,
                    "large_sell_ratio": test.large_sell_ratio,
                }
            )
        if reporter is not None and idx + 1 < len(events):
            reporter.update(idx + 1)
    if reporter is not None:
        reporter.close()

    features = pd.DataFrame(rows).sort_values("extreme_time").reset_index(drop=True)
    tests = pd.DataFrame(test_rows)
    dictionary = build_mechanism_feature_dictionary(
        [column for column in features.columns if column not in MECHANISM_METADATA_COLUMNS]
    )
    return features, dictionary, tests


def build_mechanism_feature_dictionary(feature_columns: Sequence[str]) -> pd.DataFrame:
    family_map = {
        "price_decline_uniformity": "trend_path",
        "price_decline_acceleration": "trend_path",
        "price_phase_direction_changes": "trend_path",
        "price_cvd_path_sync": "price_flow_path",
        "price_cvd_path_divergence": "price_flow_path",
        "large_sell_phase_persistence": "large_flow_path",
        "sell_pressure_decay": "orderflow_decay",
        "sell_impact_decay": "price_flow_path",
        "negative_delta_no_new_low_share": "price_flow_path",
        "activity_compression": "activity_path",
        "activity_u_shape": "activity_path",
        "range_compression": "price_path",
        "phase_low_slope": "price_path",
        "support_test_count": "support_tests",
        "support_prior_test_count": "support_tests",
        "support_test_interval_median": "support_tests",
        "support_test_interval_cv": "support_tests",
        "support_low_dispersion_bp": "support_tests",
        "support_low_level_slope_bp": "support_tests",
        "test_depth_weakening_slope": "support_test_sequence",
        "test_rebound_strengthening_slope": "support_test_sequence",
        "test_sell_pressure_decay_slope": "support_test_sequence",
        "test_sell_impact_decay_slope": "support_test_sequence",
        "test_activity_slope": "support_test_sequence",
        "test_activity_u_shape": "support_test_sequence",
        "test_large_sell_decay_slope": "support_test_sequence",
        "test_large_sell_concentration": "support_test_sequence",
        "final_support_break_bp": "spring_path",
        "final_close_reclaim_bp": "spring_path",
        "final_wick_reclaim_share": "spring_path",
        "absorption_observable": "mechanism_observable",
        "compression_observable": "mechanism_observable",
        "spring_observable": "mechanism_observable",
        "repeated_support_observable": "mechanism_observable",
        "slow_accumulation_observable": "mechanism_observable",
    }
    labels = {
        "price_decline_uniformity": "下跌路径均匀度",
        "price_decline_acceleration": "末段下跌加速度",
        "price_cvd_path_sync": "价格与CVD路径同步",
        "price_cvd_path_divergence": "价格与CVD路径背离",
        "large_sell_phase_persistence": "大单卖压阶段持续性",
        "sell_pressure_decay": "主动卖压衰减",
        "sell_impact_decay": "负Delta价格冲击衰减",
        "negative_delta_no_new_low_share": "负Delta但不创新低占比",
        "activity_compression": "成交活动压缩",
        "activity_u_shape": "成交活动先缩后放",
        "range_compression": "价格振幅压缩",
        "support_test_count": "支撑测试次数",
        "support_test_interval_median": "支撑测试间隔中位数",
        "support_low_dispersion_bp": "支撑测试低点离散度",
        "test_depth_weakening_slope": "测试深度减弱趋势",
        "test_rebound_strengthening_slope": "测试后反弹增强趋势",
        "test_sell_pressure_decay_slope": "测试卖压衰减趋势",
        "test_sell_impact_decay_slope": "测试卖压冲击衰减趋势",
        "test_activity_u_shape": "测试成交先缩后放",
        "test_large_sell_decay_slope": "测试大单卖压衰减趋势",
        "final_support_break_bp": "最终假跌破幅度",
        "final_close_reclaim_bp": "最终收盘收复支撑幅度",
        "absorption_observable": "吸收机制观测分数",
        "compression_observable": "压缩机制观测分数",
        "spring_observable": "spring/假跌破观测分数",
        "repeated_support_observable": "反复测试支撑观测分数",
        "slow_accumulation_observable": "缓慢吸筹观测分数",
    }
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "family": family_map.get(feature, "mechanism_path"),
                "label": labels.get(feature, feature.replace("_", " ")),
                "causal_cutoff": "extreme bar close or older",
            }
            for feature in feature_columns
        ]
    )


def build_path_profiles(
    bars: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    type_column: str,
    lookback: int = 240,
    phase_bins: int = 24,
    max_samples_per_type_split: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate causal price/CVD/activity/large-flow paths for each type."""

    arrays = _numeric_arrays(bars)
    close = arrays["close"]
    low = arrays["low"]
    notional = arrays["notional"]
    trades_count = arrays["trades_count"]
    delta = arrays["delta_notional"]
    large_delta = arrays["large_delta_notional"]
    large_gross = arrays["large_buy_notional"] + arrays["large_sell_notional"]
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, object]] = []

    for (type_id, split), group in assignments.groupby([type_column, "split"], sort=True):
        valid = group[pd.to_numeric(group["extreme_pos"], errors="coerce") >= lookback - 1]
        if len(valid) > max_samples_per_type_split:
            chosen = rng.choice(valid.index.to_numpy(), size=max_samples_per_type_split, replace=False)
            valid = valid.loc[chosen]
        metric_paths: dict[str, list[np.ndarray]] = {
            "price": [],
            "cvd": [],
            "notional_intensity": [],
            "trade_count_intensity": [],
            "large_flow": [],
            "low_distance": [],
        }
        for event in valid.itertuples(index=False):
            pos = int(event.extreme_pos)
            sl = slice(pos - lookback + 1, pos + 1)
            event_notional = notional[sl]
            event_trades = trades_count[sl]
            event_large_gross = large_gross[sl]
            total_notional = float(np.nansum(event_notional))
            total_large = float(np.nansum(event_large_gross))
            extreme = float(low[pos])
            metric_paths["price"].append(close[sl] / extreme - 1.0)
            metric_paths["low_distance"].append(low[sl] / extreme - 1.0)
            metric_paths["cvd"].append(np.nancumsum(np.where(np.isfinite(delta[sl]), delta[sl], 0.0)) / max(total_notional, EPS))
            metric_paths["large_flow"].append(
                np.nancumsum(np.where(np.isfinite(large_delta[sl]), large_delta[sl], 0.0)) / max(total_large, EPS)
            )
            notional_base = _median(event_notional)
            trades_base = _median(event_trades)
            metric_paths["notional_intensity"].append(
                event_notional / notional_base
                if np.isfinite(notional_base) and notional_base > EPS
                else np.full(lookback, np.nan)
            )
            metric_paths["trade_count_intensity"].append(
                event_trades / trades_base
                if np.isfinite(trades_base) and trades_base > EPS
                else np.full(lookback, np.nan)
            )

        for metric, paths in metric_paths.items():
            if not paths:
                continue
            matrix = np.asarray(paths, dtype=float)
            for phase, phase_slice in enumerate(_phase_slices(lookback, phase_bins), start=1):
                values = np.nanmedian(matrix[:, phase_slice], axis=1)
                rows.append(
                    {
                        "type_id": type_id,
                        "split": split,
                        "metric": metric,
                        "phase": phase,
                        "sample_count": int(len(values)),
                        "median": float(np.nanmedian(values)),
                        "q25": float(np.nanquantile(values, 0.25)),
                        "q75": float(np.nanquantile(values, 0.75)),
                    }
                )
    return pd.DataFrame(rows)


def build_future_perturbation_audit(
    bars: pd.DataFrame,
    stage2_assignments: pd.DataFrame,
    *,
    lookback: int = 240,
    phase_bins: int = 12,
    support_tolerance_bp: float = 25.0,
    min_test_gap: int = 4,
    rebound_horizon: int = 30,
    minimum_separation_rebound_bp: float = 15.0,
    future_bars: int = 60,
    sample_size: int = 24,
    random_state: int = 42,
) -> pd.DataFrame:
    """Actually perturb post-extreme raw bars and verify feature invariance.

    Only a bounded local window is copied for each sampled event.  This avoids
    duplicating the full multi-year trade-bar frame while still proving that
    future OHLC, activity, Delta, and large-flow changes cannot alter any 03
    mechanism feature.
    """

    from types import SimpleNamespace

    valid = stage2_assignments.copy()
    valid["extreme_pos"] = pd.to_numeric(valid["extreme_pos"], errors="coerce")
    valid = valid[
        (valid["extreme_pos"] >= lookback - 1)
        & (valid["extreme_pos"] + int(future_bars) < len(bars))
    ]
    if valid.empty:
        return pd.DataFrame(
            [{"check": "raw_future_perturbation", "passed": False, "detail": "no auditable events"}]
        )
    rng = np.random.default_rng(int(random_state))
    if len(valid) > int(sample_size):
        chosen = rng.choice(valid.index.to_numpy(), size=int(sample_size), replace=False)
        valid = valid.loc[chosen]

    rows: list[dict[str, object]] = []
    for source in valid.itertuples(index=False):
        pos = int(source.extreme_pos)
        start = pos - lookback + 1
        end = min(len(bars), pos + int(future_bars) + 1)
        local = bars.iloc[start:end].copy()
        local_event_data = source._asdict()
        local_event_data["extreme_pos"] = lookback - 1
        local_event = SimpleNamespace(**local_event_data)

        original_arrays = _numeric_arrays(local)
        original, _ = extract_event_mechanism_features(
            timestamps=pd.DatetimeIndex(local.index),
            arrays=original_arrays,
            event=local_event,
            lookback=lookback,
            phase_bins=phase_bins,
            support_tolerance_bp=support_tolerance_bp,
            min_test_gap=min_test_gap,
            rebound_horizon=rebound_horizon,
            minimum_separation_rebound_bp=minimum_separation_rebound_bp,
        )

        perturbed = local.copy()
        future_start = lookback
        changed_cells = 0
        for column in REQUIRED_COLUMNS:
            values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            tail = values[future_start:]
            if tail.size == 0:
                continue
            if column in {"open", "high", "low", "close"}:
                tail = np.maximum(EPS, tail * rng.uniform(0.50, 1.50, size=tail.size))
            elif column in {"delta_notional", "large_delta_notional"}:
                scale = np.nanmedian(np.abs(tail))
                scale = float(scale) if np.isfinite(scale) and scale > EPS else 1.0
                tail = rng.normal(0.0, 25.0 * scale, size=tail.size)
            else:
                scale = rng.uniform(0.05, 8.0, size=tail.size)
                tail = np.maximum(0.0, np.nan_to_num(tail, nan=0.0) * scale + rng.uniform(0.0, 1.0, size=tail.size))
            changed_cells += int(tail.size)
            values[future_start:] = tail
            perturbed[column] = values

        perturbed_arrays = _numeric_arrays(perturbed)
        changed, _ = extract_event_mechanism_features(
            timestamps=pd.DatetimeIndex(perturbed.index),
            arrays=perturbed_arrays,
            event=local_event,
            lookback=lookback,
            phase_bins=phase_bins,
            support_tolerance_bp=support_tolerance_bp,
            min_test_gap=min_test_gap,
            rebound_horizon=rebound_horizon,
            minimum_separation_rebound_bp=minimum_separation_rebound_bp,
        )
        feature_names = sorted(
            key for key in original if key not in MECHANISM_METADATA_COLUMNS
        )
        differences: list[float] = []
        mismatches: list[str] = []
        for feature in feature_names:
            left = float(original.get(feature, np.nan))
            right = float(changed.get(feature, np.nan))
            if np.isnan(left) and np.isnan(right):
                continue
            diff = abs(left - right) if np.isfinite(left) and np.isfinite(right) else np.inf
            differences.append(diff)
            if not np.isfinite(diff) or diff > 1e-12:
                mismatches.append(feature)
        rows.append(
            {
                "event_id": source.event_id,
                "extreme_time": source.extreme_time,
                "changed_future_cells": changed_cells,
                "compared_feature_count": len(feature_names),
                "maximum_absolute_difference": max(differences, default=0.0),
                "mismatch_count": len(mismatches),
                "mismatch_features": ",".join(mismatches[:20]),
                "passed": len(mismatches) == 0 and changed_cells > 0,
            }
        )
    return pd.DataFrame(rows).sort_values("extreme_time").reset_index(drop=True)
