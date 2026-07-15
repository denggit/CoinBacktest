#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal respected-macro first-sweep event construction for research 12.

A lifecycle event is created only after a respected higher-timeframe level has
already become available.  Two deployable decision paths are emitted:

* ``sweep``   -- the first closed 1m bar whose low trades below the level;
* ``reclaim`` -- the closed 1m bar that first closes back above the level
  within the fixed reclaim window.

Both paths use the decision bar only after it closes.  Downstream labels must
therefore enter at the next 1m open.  No future lifecycle state is attached to
the sweep decision unless it is already known on the same closed sweep bar.
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

from research.market_structure.swing_low_typology.common.respected_macro_liquidity import (
    build_respected_level_table,
)

EPS = 1e-12
LEVEL_GROUP = "E1_level_sweep_geometry"
ORDERFLOW_GROUP = "E2_sweep_orderflow_absorption"
RECLAIM_GROUP = "E3_reclaim_process"


@dataclass(frozen=True)
class FirstSweepEventBuildResult:
    decisions: pd.DataFrame
    levels: pd.DataFrame
    lifecycle: pd.DataFrame
    dictionary: pd.DataFrame
    group_membership: pd.DataFrame
    diagnostics: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index, name=column)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_divide(numerator: np.ndarray | pd.Series | float, denominator: np.ndarray | pd.Series | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=float), where=np.abs(den) > EPS)


def _bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=1)
    diffs = pd.Series(index[1:] - index[:-1])
    value = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    return pd.Timedelta(value)


def _feature_row(name: str, group: str, description: str) -> dict[str, object]:
    return {
        "feature": name,
        "feature_group": group,
        "source": "respected_macro_first_sweep",
        "description": description,
        "causal_rule": "respected level pre-exists decision bar; current/prior closed 1m bars only",
    }


def build_first_sweep_event_decisions(
    bars: pd.DataFrame,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    pivot_minutes: Sequence[int] = (15, 60, 240),
    pivot_weights: Sequence[float] = (1.0, 2.0, 4.0),
    left_bars: int = 2,
    right_bars: int = 2,
    cluster_tolerance_bp: float = 25.0,
    minimum_respects: int = 2,
    minimum_macro_timeframe_min: int = 60,
    minimum_respect_separation_minutes: int = 60,
    formation_max_days: int = 45,
    reclaim_window_bars: int = 3,
    accept_below_bars: int = 3,
    accept_depth_bp: float = 75.0,
    show_progress: bool = False,
) -> FirstSweepEventBuildResult:
    """Create one sweep decision and, when observed, one reclaim decision per level."""

    required = {
        "open", "high", "low", "close", "notional", "trades_count", "delta_notional",
        "buy_notional", "sell_notional", "large_buy_notional", "large_sell_notional",
        "large_delta_notional",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise RuntimeError(f"first-sweep builder missing trade-bar fields: {missing}")
    if reclaim_window_bars < 0:
        raise ValueError("reclaim_window_bars must be non-negative")

    data = bars.sort_index()
    index = pd.DatetimeIndex(data.index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError("bars index must be unique and increasing")
    delta_time = _bar_delta(index)
    index_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    available_ns = (index + delta_time).to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)

    levels, level_diag = build_respected_level_table(
        data,
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

    open_ = _numeric(data, "open").to_numpy(dtype=float)
    high = _numeric(data, "high").to_numpy(dtype=float)
    low = _numeric(data, "low").to_numpy(dtype=float)
    close = _numeric(data, "close").to_numpy(dtype=float)
    notional_s = _numeric(data, "notional")
    trades_s = _numeric(data, "trades_count")
    delta_s = _numeric(data, "delta_notional")
    buy_s = _numeric(data, "buy_notional")
    sell_s = _numeric(data, "sell_notional")
    large_buy_s = _numeric(data, "large_buy_notional")
    large_sell_s = _numeric(data, "large_sell_notional")
    large_delta_s = _numeric(data, "large_delta_notional")

    prior_notional = notional_s.shift(1).rolling(60, min_periods=20).mean()
    prior_trades = trades_s.shift(1).rolling(60, min_periods=20).mean()
    notional_intensity = _safe_divide(notional_s, prior_notional)
    trades_intensity = _safe_divide(trades_s, prior_trades)
    negative_delta_ratio = np.maximum(-_safe_divide(delta_s, notional_s), 0.0)
    aggressive_sell_ratio = _safe_divide(sell_s, buy_s + sell_s)
    large_sell_ratio = _safe_divide(large_sell_s, large_buy_s + large_sell_s)
    large_negative_delta_ratio = np.maximum(-_safe_divide(large_delta_s, notional_s), 0.0)

    decisions: list[dict[str, object]] = []
    lifecycle_rows: list[dict[str, object]] = []
    skipped_below_activation = 0
    outside_bar_range = 0
    availability_violations = 0
    reporter = (
        ProgressReporter("[events] respected first sweeps", total=max(len(levels), 1), every=max(1, len(levels) // 100))
        if ProgressReporter is not None and show_progress
        else None
    )

    research_start = pd.Timestamp(research_start)
    research_end_exclusive = pd.Timestamp(research_end_exclusive)

    for ordinal, level in enumerate(levels.itertuples(index=False), start=1):
        level_price = float(level.level_price)
        active_time = pd.Timestamp(level.available_time)
        expiry_time = pd.Timestamp(level.expiry_time)
        # The level must exist before the sweep bar starts.  A 1m bar whose
        # close makes the HTF pivot available cannot retrospectively sweep that
        # newly known level with its already-observed intrabar low.
        active_pos = int(np.searchsorted(index_ns, active_time.value, side="left"))
        expiry_pos = int(np.searchsorted(available_ns, expiry_time.value, side="right") - 1)
        if active_pos >= len(index) or expiry_pos < active_pos:
            outside_bar_range += 1
            lifecycle_rows.append({
                "level_id": int(level.level_id), "status": "outside_bar_range", "available_time": active_time,
                "expiry_time": expiry_time, "level_price": level_price,
            })
            if reporter is not None:
                reporter.update(ordinal)
            continue
        expiry_pos = min(expiry_pos, len(index) - 1)
        if active_time > index[active_pos]:
            availability_violations += 1
        if close[active_pos] <= level_price:
            skipped_below_activation += 1
            lifecycle_rows.append({
                "level_id": int(level.level_id), "status": "below_at_activation", "available_time": active_time,
                "expiry_time": expiry_time, "level_price": level_price,
            })
            if reporter is not None:
                reporter.update(ordinal)
            continue

        swept = np.flatnonzero(low[active_pos : expiry_pos + 1] < level_price)
        if not len(swept):
            lifecycle_rows.append({
                "level_id": int(level.level_id), "status": "expired_unswept", "available_time": active_time,
                "expiry_time": expiry_time, "level_price": level_price,
            })
            if reporter is not None:
                reporter.update(ordinal)
            continue

        sweep_pos = active_pos + int(swept[0])
        reclaim_pos: int | None = None
        accept_pos: int | None = None
        event_end = min(expiry_pos, sweep_pos + int(reclaim_window_bars))
        consecutive_below = 0
        for pos in range(sweep_pos, event_end + 1):
            if close[pos] >= level_price:
                reclaim_pos = pos
                break
            consecutive_below += 1
            penetration_bp = max((level_price - low[pos]) / level_price * 10_000.0, 0.0)
            if consecutive_below >= int(accept_below_bars) or penetration_bp >= float(accept_depth_bp):
                accept_pos = pos
                break
        if reclaim_pos is not None:
            status = "reclaimed"
            resolution_pos = reclaim_pos
        elif accept_pos is not None:
            status = "accepted_below"
            resolution_pos = accept_pos
        else:
            status = "unreclaimed_timeout"
            resolution_pos = event_end

        sweep_available = index[sweep_pos] + delta_time
        reclaim_available = index[reclaim_pos] + delta_time if reclaim_pos is not None else pd.NaT
        lifecycle_id = f"LFS_{int(level.level_id):07d}"
        prior_pos = max(active_pos, sweep_pos - 1)
        prior_close = close[prior_pos]
        sweep_range = max(high[sweep_pos] - low[sweep_pos], EPS)
        lower_wick = max(min(open_[sweep_pos], close[sweep_pos]) - low[sweep_pos], 0.0)
        body = abs(close[sweep_pos] - open_[sweep_pos])
        sweep_depth_bp = max((level_price - low[sweep_pos]) / level_price * 10_000.0, 0.0)
        sweep_close_vs_level_bp = (close[sweep_pos] / level_price - 1.0) * 10_000.0
        downside_close_move = max((prior_close - close[sweep_pos]) / max(prior_close, EPS), 0.0)
        sweep_negative_delta = float(negative_delta_ratio[sweep_pos])
        price_response_efficiency = downside_close_move / max(sweep_negative_delta, 1e-6)
        absorption_proxy = sweep_negative_delta / max(downside_close_move * 100.0 + 1e-4, 1e-4)

        common = {
            "lifecycle_id": lifecycle_id,
            "level_id": int(level.level_id),
            "level_price": level_price,
            "level_available_time": active_time,
            "level_expiry_time": expiry_time,
            "level_strength": float(level.strength),
            "level_respect_count": int(level.respect_count),
            "level_timeframe_count": int(level.timeframe_count),
            "level_max_timeframe_min": int(level.max_timeframe_min),
            "level_median_reaction_bp": float(level.median_reaction_bp),
            "level_respect_span_bars": int(level.respect_span_bars_1m),
            "level_source_timeframes": str(level.source_timeframes),
            "sweep_pos": int(sweep_pos),
            "sweep_time": index[sweep_pos],
            "sweep_available_time": sweep_available,
            "reclaim_pos": int(reclaim_pos) if reclaim_pos is not None else -1,
            "reclaim_time": index[reclaim_pos] if reclaim_pos is not None else pd.NaT,
            "reclaim_available_time": reclaim_available,
            "resolution_pos": int(resolution_pos),
            "lifecycle_status": status,
            "reclaim_lag_bars": int(reclaim_pos - sweep_pos) if reclaim_pos is not None else -1,
            "same_bar_reclaim": float(reclaim_pos == sweep_pos) if reclaim_pos is not None else 0.0,
            "fse_level_strength": float(level.strength),
            "fse_level_respect_count": float(level.respect_count),
            "fse_level_timeframe_count": float(level.timeframe_count),
            "fse_level_max_timeframe_min": float(level.max_timeframe_min),
            "fse_level_median_reaction_bp": float(level.median_reaction_bp),
            "fse_level_respect_span_bars": float(level.respect_span_bars_1m),
            "fse_level_age_bars_at_sweep": float(sweep_pos - active_pos),
            "fse_pre_sweep_close_distance_bp": float((prior_close / level_price - 1.0) * 10_000.0),
            "fse_sweep_depth_bp": float(sweep_depth_bp),
            "fse_sweep_close_vs_level_bp": float(sweep_close_vs_level_bp),
            "fse_sweep_lower_wick_fraction": float(lower_wick / sweep_range),
            "fse_sweep_body_fraction": float(body / sweep_range),
            "fse_sweep_range_pct": float(sweep_range / max(close[sweep_pos], EPS)),
            "fse_sweep_notional_intensity": float(notional_intensity[sweep_pos]),
            "fse_sweep_trades_intensity": float(trades_intensity[sweep_pos]),
            "fse_sweep_negative_delta_ratio": sweep_negative_delta,
            "fse_sweep_aggressive_sell_ratio": float(aggressive_sell_ratio[sweep_pos]),
            "fse_sweep_large_sell_ratio": float(large_sell_ratio[sweep_pos]),
            "fse_sweep_large_negative_delta_ratio": float(large_negative_delta_ratio[sweep_pos]),
            "fse_sweep_price_response_efficiency": float(price_response_efficiency),
            "fse_sweep_absorption_proxy": float(absorption_proxy),
            "fse_same_bar_reclaim": float(reclaim_pos == sweep_pos) if reclaim_pos is not None else 0.0,
        }

        sweep_time_in_range = research_start <= sweep_available < research_end_exclusive
        if sweep_time_in_range:
            row = dict(common)
            row.update({
                "event_id": f"{lifecycle_id}_SWEEP",
                "decision_path": "sweep",
                "extreme_pos": int(sweep_pos),
                "extreme_time": index[sweep_pos],
                "feature_available_time": sweep_available,
                "causal_region_id": lifecycle_id,
                # Reclaim-process fields available on the sweep bar only when
                # the same closed bar already reclaimed.
                "fse_reclaim_lag_bars": 0.0 if reclaim_pos == sweep_pos else np.nan,
                "fse_reclaim_strength_bp": max(sweep_close_vs_level_bp, 0.0) if reclaim_pos == sweep_pos else np.nan,
                "fse_max_penetration_to_decision_bp": sweep_depth_bp,
                "fse_process_cumulative_delta_ratio": float(delta_s.iloc[sweep_pos] / max(notional_s.iloc[sweep_pos], EPS)),
                "fse_process_aggressive_sell_ratio": float(aggressive_sell_ratio[sweep_pos]),
                "fse_post_sweep_sell_decay": 0.0,
                "fse_post_sweep_delta_recovery": 0.0,
                "fse_post_sweep_price_recovery_bp": sweep_close_vs_level_bp,
                "fse_process_notional_vs_sweep": 1.0,
            })
            decisions.append(row)

        if reclaim_pos is not None and research_start <= reclaim_available < research_end_exclusive:
            process_slice = slice(sweep_pos, reclaim_pos + 1)
            process_notional = float(notional_s.iloc[process_slice].sum())
            process_delta = float(delta_s.iloc[process_slice].sum())
            process_sell = float(sell_s.iloc[process_slice].sum())
            process_buy = float(buy_s.iloc[process_slice].sum())
            max_penetration = max((level_price - float(np.nanmin(low[process_slice]))) / level_price * 10_000.0, 0.0)
            current_negative_delta = float(negative_delta_ratio[reclaim_pos])
            row = dict(common)
            row.update({
                "event_id": f"{lifecycle_id}_RECLAIM",
                "decision_path": "reclaim",
                "extreme_pos": int(reclaim_pos),
                "extreme_time": index[reclaim_pos],
                "feature_available_time": reclaim_available,
                "causal_region_id": lifecycle_id,
                "fse_reclaim_lag_bars": float(reclaim_pos - sweep_pos),
                "fse_reclaim_strength_bp": float(max((close[reclaim_pos] / level_price - 1.0) * 10_000.0, 0.0)),
                "fse_max_penetration_to_decision_bp": float(max_penetration),
                "fse_process_cumulative_delta_ratio": float(process_delta / max(process_notional, EPS)),
                "fse_process_aggressive_sell_ratio": float(process_sell / max(process_buy + process_sell, EPS)),
                "fse_post_sweep_sell_decay": float(sweep_negative_delta - current_negative_delta),
                "fse_post_sweep_delta_recovery": float((process_delta / max(process_notional, EPS)) - (delta_s.iloc[sweep_pos] / max(notional_s.iloc[sweep_pos], EPS))),
                "fse_post_sweep_price_recovery_bp": float((close[reclaim_pos] / level_price - 1.0) * 10_000.0),
                "fse_process_notional_vs_sweep": float(process_notional / max(float(notional_s.iloc[sweep_pos]), EPS)),
            })
            decisions.append(row)

        lifecycle_rows.append({
            "level_id": int(level.level_id),
            "lifecycle_id": lifecycle_id,
            "status": status,
            "available_time": active_time,
            "expiry_time": expiry_time,
            "level_price": level_price,
            "level_strength": float(level.strength),
            "max_timeframe_min": int(level.max_timeframe_min),
            "sweep_time": sweep_available,
            "reclaim_time": reclaim_available,
            "reclaim_lag_bars": int(reclaim_pos - sweep_pos) if reclaim_pos is not None else -1,
            "sweep_in_research_period": bool(sweep_time_in_range),
        })
        if reporter is not None:
            reporter.update(ordinal)
    if reporter is not None:
        reporter.close()

    decision_frame = pd.DataFrame(decisions)
    if not decision_frame.empty:
        decision_frame = decision_frame.sort_values(["extreme_pos", "decision_path", "level_id"]).reset_index(drop=True)
        decision_frame["event_id"] = decision_frame["event_id"].astype(str)
        if decision_frame["event_id"].duplicated().any():
            raise RuntimeError("duplicate first-sweep decision event_id")
        if (pd.to_datetime(decision_frame["level_available_time"]) > pd.to_datetime(decision_frame["feature_available_time"])).any():
            raise RuntimeError("first-sweep event used a level before it was available")

    level_features = (
        "fse_level_strength", "fse_level_respect_count", "fse_level_timeframe_count",
        "fse_level_max_timeframe_min", "fse_level_median_reaction_bp",
        "fse_level_respect_span_bars", "fse_level_age_bars_at_sweep",
        "fse_pre_sweep_close_distance_bp", "fse_sweep_depth_bp",
        "fse_sweep_close_vs_level_bp", "fse_sweep_lower_wick_fraction",
        "fse_sweep_body_fraction", "fse_sweep_range_pct", "fse_same_bar_reclaim",
    )
    orderflow_features = (
        "fse_sweep_notional_intensity", "fse_sweep_trades_intensity",
        "fse_sweep_negative_delta_ratio", "fse_sweep_aggressive_sell_ratio",
        "fse_sweep_large_sell_ratio", "fse_sweep_large_negative_delta_ratio",
        "fse_sweep_price_response_efficiency", "fse_sweep_absorption_proxy",
    )
    reclaim_features = (
        "fse_reclaim_lag_bars", "fse_reclaim_strength_bp",
        "fse_max_penetration_to_decision_bp", "fse_process_cumulative_delta_ratio",
        "fse_process_aggressive_sell_ratio", "fse_post_sweep_sell_decay",
        "fse_post_sweep_delta_recovery", "fse_post_sweep_price_recovery_bp",
        "fse_process_notional_vs_sweep",
    )
    dictionary = pd.DataFrame(
        [*[_feature_row(name, LEVEL_GROUP, "respected level and first-sweep geometry") for name in level_features],
         *[_feature_row(name, ORDERFLOW_GROUP, "observable sweep-bar sell pressure and price-response proxy") for name in orderflow_features],
         *[_feature_row(name, RECLAIM_GROUP, "causally observed sweep-to-reclaim process at the reclaim decision") for name in reclaim_features]]
    )
    membership = pd.DataFrame(
        [{"feature": name, "feature_group": group}
         for group, names in ((LEVEL_GROUP, level_features), (ORDERFLOW_GROUP, orderflow_features), (RECLAIM_GROUP, reclaim_features))
         for name in names]
    )
    lifecycle = pd.DataFrame(lifecycle_rows)
    aggregate = pd.DataFrame([
        {"metric": "confirmed_level_count", "value": int(len(levels))},
        {"metric": "lifecycle_rows", "value": int(len(lifecycle))},
        {"metric": "swept_levels", "value": int(lifecycle["status"].isin(["reclaimed", "accepted_below", "unreclaimed_timeout"]).sum()) if not lifecycle.empty else 0},
        {"metric": "reclaimed_levels", "value": int(lifecycle["status"].eq("reclaimed").sum()) if not lifecycle.empty else 0},
        {"metric": "sweep_decisions", "value": int((decision_frame.get("decision_path", pd.Series(dtype=str)) == "sweep").sum())},
        {"metric": "reclaim_decisions", "value": int((decision_frame.get("decision_path", pd.Series(dtype=str)) == "reclaim").sum())},
        {"metric": "skipped_below_activation", "value": int(skipped_below_activation)},
        {"metric": "outside_bar_range", "value": int(outside_bar_range)},
        {"metric": "availability_violations", "value": int(availability_violations)},
    ])
    diagnostics = pd.concat([level_diag.assign(scope="level_construction"), aggregate.assign(scope="aggregate")], ignore_index=True, sort=False)
    return FirstSweepEventBuildResult(
        decisions=decision_frame,
        levels=levels,
        lifecycle=lifecycle,
        dictionary=dictionary,
        group_membership=membership,
        diagnostics=diagnostics,
    )
