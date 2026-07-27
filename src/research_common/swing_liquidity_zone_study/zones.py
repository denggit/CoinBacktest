#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal same-bar Swing Low zone aggregation and online impulse de-duplication."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import ZoneStudyConfig

EPS = 1e-12


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _float_array(values: Iterable[object]) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    try:
        return np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").to_numpy(dtype=float)


def _safe_median(values: Iterable[object]) -> float:
    data = _float_array(values)
    return float(np.nanmedian(data)) if np.isfinite(data).any() else np.nan


def _safe_mean(values: Iterable[object]) -> float:
    data = _float_array(values)
    return float(np.nanmean(data)) if np.isfinite(data).any() else np.nan


def _safe_max(values: Iterable[object]) -> float:
    data = _float_array(values)
    return float(np.nanmax(data)) if np.isfinite(data).any() else np.nan


def _safe_min(values: Iterable[object]) -> float:
    data = _float_array(values)
    return float(np.nanmin(data)) if np.isfinite(data).any() else np.nan


def _cluster_same_bar(group: pd.DataFrame, tolerance_bp: float) -> list[pd.DataFrame]:
    ordered = group.sort_values(["level_price", "source_timeframe_min", "level_id"], kind="mergesort")
    clusters: list[list[int]] = []
    current: list[int] = []
    anchor = np.nan
    for idx, row in ordered.iterrows():
        price = float(row["level_price"])
        if not current:
            current = [idx]
            anchor = price
            continue
        gap_bp = abs(price / anchor - 1.0) * 10_000.0 if anchor > EPS else np.inf
        if gap_bp <= float(tolerance_bp):
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
            anchor = price
    if current:
        clusters.append(current)
    return [ordered.loc[indexes].copy() for indexes in clusters]


def _timeframe_signature(frame: pd.DataFrame) -> str:
    values = (
        frame[["source_timeframe", "source_timeframe_min"]]
        .drop_duplicates()
        .sort_values("source_timeframe_min")
        ["source_timeframe"]
        .astype(str)
        .tolist()
    )
    return "|".join(values)


def _zone_record(cluster: pd.DataFrame, bars: pd.DataFrame, zone_ordinal: int) -> dict[str, object]:
    event_pos = int(cluster["sweep_pos"].iloc[0])
    if event_pos < 0 or event_pos >= len(bars):
        raise ValueError(f"zone event_pos out of range: {event_pos}")
    event_time = pd.Timestamp(bars.index[event_pos])
    available_time = event_time + pd.Timedelta(minutes=1)
    level_prices = pd.to_numeric(cluster["level_price"], errors="coerce").to_numpy(dtype=float)
    weights = np.sqrt(pd.to_numeric(cluster["source_timeframe_min"], errors="coerce").to_numpy(dtype=float))
    weights *= np.maximum(pd.to_numeric(cluster["confirmed_order_at_sweep"], errors="coerce").fillna(1).to_numpy(dtype=float), 1.0)
    valid_weight = np.isfinite(level_prices) & np.isfinite(weights) & (weights > 0)
    weighted_center = (
        float(np.average(level_prices[valid_weight], weights=weights[valid_weight]))
        if valid_weight.any()
        else float(np.nanmedian(level_prices))
    )
    floor_price = float(np.nanmin(level_prices))
    ceiling_price = float(np.nanmax(level_prices))
    event_low = float(bars["low"].iloc[event_pos])
    event_high = float(bars["high"].iloc[event_pos])
    event_open = float(bars["open"].iloc[event_pos])
    event_close = float(bars["close"].iloc[event_pos])
    bar_range = max(event_high - event_low, EPS)
    latest_available = pd.to_datetime(cluster["initial_available_time"], errors="coerce").max()
    earliest_pivot = pd.to_datetime(cluster["pivot_time"], errors="coerce").min()
    latest_pivot = pd.to_datetime(cluster["pivot_time"], errors="coerce").max()
    ages = pd.to_numeric(cluster["age_minutes_at_sweep"], errors="coerce")
    touches = pd.to_numeric(cluster["touch_episode_count_before_sweep"], errors="coerce").fillna(0)
    approaches = pd.to_numeric(cluster["approach_episode_count_before_sweep"], errors="coerce").fillna(0)
    tf_minutes = pd.to_numeric(cluster["source_timeframe_min"], errors="coerce")
    order = pd.to_numeric(cluster["confirmed_order_at_sweep"], errors="coerce").fillna(1)
    level_ids = pd.to_numeric(cluster["level_id"], errors="raise").astype(np.int64).tolist()
    width_bp = (ceiling_price / max(floor_price, EPS) - 1.0) * 10_000.0
    return {
        "zone_event_id": f"SZ_{zone_ordinal:08d}",
        "event_kind": "swing_zone_sweep",
        "event_pos": event_pos,
        "event_bar_time": event_time,
        "event_available_time": available_time,
        "zone_latest_level_available_time": latest_available,
        "zone_member_level_ids": "|".join(str(v) for v in level_ids),
        "zone_member_count": int(len(cluster)),
        "zone_timeframe_count": int(cluster["source_timeframe"].astype(str).nunique()),
        "zone_timeframes": _timeframe_signature(cluster),
        "zone_primary_timeframe": str(cluster.loc[tf_minutes.idxmax(), "source_timeframe"]),
        "zone_max_timeframe_min": int(tf_minutes.max()),
        "zone_has_15m": bool((tf_minutes == 15).any()),
        "zone_has_30m": bool((tf_minutes == 30).any()),
        "zone_has_1H": bool((tf_minutes == 60).any()),
        "zone_has_4H": bool((tf_minutes == 240).any()),
        "zone_has_1D": bool((tf_minutes == 1440).any()),
        "zone_floor_price": floor_price,
        "zone_ceiling_price": ceiling_price,
        "zone_center_price": weighted_center,
        "zone_width_bp": width_bp,
        "zone_earliest_pivot_time": earliest_pivot,
        "zone_latest_pivot_time": latest_pivot,
        "zone_formation_span_minutes": float((latest_pivot - earliest_pivot).total_seconds() / 60.0),
        "zone_age_min_minutes": float(ages.min()) if ages.notna().any() else np.nan,
        "zone_age_median_minutes": float(ages.median()) if ages.notna().any() else np.nan,
        "zone_age_max_minutes": float(ages.max()) if ages.notna().any() else np.nan,
        "zone_fresh_member_share": float((touches <= 0).mean()),
        "zone_all_members_fresh": bool((touches <= 0).all()),
        "zone_prior_touch_min": int(touches.min()),
        "zone_prior_touch_median": float(touches.median()),
        "zone_prior_touch_max": int(touches.max()),
        "zone_prior_approach_median": float(approaches.median()),
        "zone_confirmed_order_min": int(order.min()),
        "zone_confirmed_order_median": float(order.median()),
        "zone_confirmed_order_max": int(order.max()),
        "zone_pivot_range_bp_median": _safe_median(cluster.get("pivot_range_bp", [])),
        "zone_pivot_range_bp_max": _safe_max(cluster.get("pivot_range_bp", [])),
        "zone_pivot_close_location_median": _safe_median(cluster.get("pivot_close_location", [])),
        "zone_pivot_lower_wick_fraction_median": _safe_median(cluster.get("pivot_lower_wick_fraction", [])),
        "zone_left_high_range_20_bp_median": _safe_median(cluster.get("left_high_range_20_bp", [])),
        "zone_left_high_range_20_bp_max": _safe_max(cluster.get("left_high_range_20_bp", [])),
        "zone_confirmation_reaction_close_bp_median": _safe_median(cluster.get("confirmation_reaction_close_bp", [])),
        "zone_confirmation_reaction_close_bp_max": _safe_max(cluster.get("confirmation_reaction_close_bp", [])),
        "zone_pivot_notional_vs_past20_median": _safe_median(cluster.get("pivot_notional_vs_past20", [])),
        "zone_pivot_trades_vs_past20_median": _safe_median(cluster.get("pivot_trades_count_vs_past20", [])),
        "zone_pivot_delta_ratio_median": _safe_median(cluster.get("pivot_delta_ratio", [])),
        "sweep_low": event_low,
        "sweep_bar_open": event_open,
        "sweep_bar_high": event_high,
        "sweep_bar_close": event_close,
        "sweep_depth_below_floor_bp": max((floor_price - event_low) / max(floor_price, EPS) * 10_000.0, 0.0),
        "sweep_depth_below_center_bp": max((weighted_center - event_low) / max(weighted_center, EPS) * 10_000.0, 0.0),
        "sweep_close_vs_floor_bp": (event_close / max(floor_price, EPS) - 1.0) * 10_000.0,
        "sweep_close_vs_ceiling_bp": (event_close / max(ceiling_price, EPS) - 1.0) * 10_000.0,
        "sweep_bar_range_bp": bar_range / max(event_open, EPS) * 10_000.0,
        "sweep_bar_close_location": (event_close - event_low) / bar_range,
        "sweep_bar_lower_wick_fraction": (min(event_open, event_close) - event_low) / bar_range,
        "same_bar_raw_level_sweeps": int(len(cluster)),
    }


def _attach_online_impulses(events: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    cfg = config.validate()
    out = events.sort_values(["event_pos", "zone_ceiling_price"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    impulse_ids: list[str] = []
    first_flags: list[bool] = []
    impulse_number = 0
    last_pos = -10**18
    running_floor = np.nan
    running_ceiling = np.nan
    for row in out.itertuples(index=False):
        pos = int(row.event_pos)
        center = float(row.zone_center_price)
        joins = False
        if impulse_number > 0 and pos - last_pos <= int(cfg.impulse_gap_bars):
            tolerance = float(cfg.impulse_price_tolerance_bp) / 10_000.0
            lower = float(running_floor) * (1.0 - tolerance)
            upper = float(running_ceiling) * (1.0 + tolerance)
            joins = lower <= center <= upper
        if not joins:
            impulse_number += 1
            running_floor = float(row.zone_floor_price)
            running_ceiling = float(row.zone_ceiling_price)
            first_flags.append(True)
        else:
            running_floor = min(float(running_floor), float(row.zone_floor_price))
            running_ceiling = max(float(running_ceiling), float(row.zone_ceiling_price))
            first_flags.append(False)
        impulse_ids.append(f"SI_{impulse_number:08d}")
        last_pos = pos
    out["online_impulse_id"] = impulse_ids
    out["is_impulse_first_event"] = np.asarray(first_flags, dtype=bool)
    out["impulse_observation_number"] = out.groupby("online_impulse_id", sort=False).cumcount() + 1
    out["impulse_zone_event_count_so_far"] = out["impulse_observation_number"].astype(np.int32)
    return out


def zone_merge_sensitivity_summary(
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: ZoneStudyConfig,
    *,
    tolerances_bp: Iterable[float] | None = None,
    research_start: pd.Timestamp | None = None,
    research_end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Count zone/online-impulse events across tolerances without building wide records."""

    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    swept = lifecycle.loc[pd.to_numeric(lifecycle.get("sweep_pos"), errors="coerce").fillna(-1).ge(0), ["sweep_pos", "level_price"]].copy()
    if swept.empty:
        return pd.DataFrame()
    swept["sweep_pos"] = pd.to_numeric(swept["sweep_pos"], errors="raise").astype(np.int64)
    start = pd.Timestamp(research_start) if research_start is not None else pd.Timestamp.min
    end = pd.Timestamp(research_end_exclusive) if research_end_exclusive is not None else pd.Timestamp.max
    raw_mask = (bars.index[swept["sweep_pos"].to_numpy()] + pd.Timedelta(minutes=1) >= start) & (bars.index[swept["sweep_pos"].to_numpy()] + pd.Timedelta(minutes=1) < end)
    swept = swept.loc[raw_mask].copy()
    raw_count = len(swept)
    rows: list[dict[str, object]] = []
    for tolerance in (tuple(tolerances_bp) if tolerances_bp is not None else cfg.zone_merge_sensitivity_bp):
        minimal: list[dict[str, object]] = []
        ordinal = 0
        member_counts: list[int] = []
        for pos, group in swept.groupby("sweep_pos", sort=True):
            prices = np.sort(pd.to_numeric(group["level_price"], errors="coerce").dropna().to_numpy(dtype=float))
            if not len(prices):
                continue
            cluster_start = 0
            anchor = float(prices[0])
            for i in range(1, len(prices) + 1):
                split = i == len(prices) or abs(float(prices[i]) / max(anchor, EPS) - 1.0) * 10_000.0 > float(tolerance)
                if not split:
                    continue
                part = prices[cluster_start:i]
                ordinal += 1
                member_counts.append(len(part))
                minimal.append({
                    "zone_event_id": f"SENS_{ordinal}", "event_pos": int(pos),
                    "zone_floor_price": float(part.min()), "zone_ceiling_price": float(part.max()),
                    "zone_center_price": float(np.median(part)),
                })
                if i < len(prices):
                    cluster_start = i
                    anchor = float(prices[i])
        frame = _attach_online_impulses(pd.DataFrame(minimal), cfg) if minimal else pd.DataFrame()
        first_count = int(frame["is_impulse_first_event"].sum()) if len(frame) else 0
        counts = pd.Series(member_counts, dtype=float)
        rows.append({
            "zone_merge_tolerance_bp": float(tolerance),
            "raw_level_sweeps": int(raw_count),
            "same_bar_zone_events": int(len(frame)),
            "online_impulse_first_events": first_count,
            "raw_to_zone_reduction": 1.0 - len(frame) / max(raw_count, 1),
            "raw_to_impulse_reduction": 1.0 - first_count / max(raw_count, 1),
            "median_members_per_zone": float(counts.median()) if len(counts) else np.nan,
            "p99_members_per_zone": float(counts.quantile(0.99)) if len(counts) else np.nan,
        })
    return pd.DataFrame(rows)


def build_sweep_zone_events(
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: ZoneStudyConfig,
    *,
    tolerance_bp: float | None = None,
) -> pd.DataFrame:
    """Aggregate levels swept on the same closed 1m bar into causal price zones."""

    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    if lifecycle.empty:
        return pd.DataFrame()
    swept = lifecycle.loc[pd.to_numeric(lifecycle["sweep_pos"], errors="coerce").fillna(-1).ge(0)].copy()
    if swept.empty:
        return pd.DataFrame()
    swept["sweep_pos"] = pd.to_numeric(swept["sweep_pos"], errors="raise").astype(np.int64)
    tolerance = float(cfg.zone_merge_tolerance_bp if tolerance_bp is None else tolerance_bp)
    rows: list[dict[str, object]] = []
    ordinal = 0
    for _, same_bar in swept.groupby("sweep_pos", sort=True):
        for cluster in _cluster_same_bar(same_bar, tolerance):
            ordinal += 1
            rows.append(_zone_record(cluster, bars, ordinal))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _attach_online_impulses(out, cfg)
    if out["zone_event_id"].duplicated().any():
        raise RuntimeError("duplicate zone_event_id")
    invalid = pd.to_datetime(out["zone_latest_level_available_time"]) > pd.to_datetime(out["event_available_time"])
    if bool(invalid.any()):
        raise RuntimeError(f"zone contains level unavailable at event time: {int(invalid.sum())}")
    return out
