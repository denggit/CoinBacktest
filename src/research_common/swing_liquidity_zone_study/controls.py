#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic matched non-zone downside-impulse controls for R03."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import ZoneStudyConfig
from .features import attach_causal_market_features, build_causal_market_feature_frame


def _fixed_period(ts: pd.Series) -> pd.Series:
    value = pd.to_datetime(ts, errors="coerce")
    return pd.Series(
        np.select(
            [value < pd.Timestamp("2025-01-01"), value < pd.Timestamp("2025-10-01")],
            ["EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="BOOKS_2025Q4_2026H1",
        ),
        index=ts.index,
        dtype="object",
    )


def _bucket(frame: pd.DataFrame) -> pd.Series:
    atr = pd.cut(
        pd.to_numeric(frame["pre_atr_60m_vs_past7d"], errors="coerce"),
        [-np.inf, 0.75, 1.0, 1.25, 1.5, np.inf],
        labels=False,
    )
    pre = pd.cut(
        pd.to_numeric(frame["pre_return_60m"], errors="coerce") * 10_000.0,
        [-np.inf, -200, -100, -50, 0, 50, np.inf],
        labels=False,
    )
    down = pd.cut(
        pd.to_numeric(frame["bar_downside_to_pre_atr_60m"], errors="coerce"),
        [-np.inf, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=False,
    )
    month = pd.to_datetime(frame["event_available_time"]).dt.to_period("M").astype(str)
    period = _fixed_period(pd.to_datetime(frame["event_available_time"]))
    return period.astype(str) + "|" + month + "|" + atr.astype("Int64").astype(str) + "|" + pre.astype("Int64").astype(str) + "|" + down.astype("Int64").astype(str)


def build_matched_controls(
    zone_events: pd.DataFrame,
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: ZoneStudyConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = config.validate()
    if zone_events.empty or cfg.control_max_per_zone <= 0:
        return pd.DataFrame()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    positions = np.arange(len(bars), dtype=np.int64)
    if feature_frame is None:
        feature_frame = build_causal_market_feature_frame(bars, cfg)
    elif not feature_frame.index.equals(bars.index):
        raise ValueError("feature_frame index must match primary bars")
    available_time = index + pd.Timedelta(minutes=1)
    in_window = (available_time >= pd.Timestamp(research_start)) & (available_time < pd.Timestamp(research_end_exclusive))
    downside = pd.to_numeric(feature_frame["bar_downside_to_pre_atr_60m"], errors="coerce").to_numpy(dtype=float)
    eligible = in_window & np.isfinite(downside) & (downside >= float(cfg.control_min_downside_atr))
    raw_sweep_positions = set(pd.to_numeric(lifecycle.get("sweep_pos", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    excluded = np.zeros(len(bars), dtype=bool)
    for pos in raw_sweep_positions:
        left = max(0, int(pos) - int(cfg.control_exclusion_bars))
        right = min(len(excluded), int(pos) + int(cfg.control_exclusion_bars) + 1)
        excluded[left:right] = True
    candidate_positions = positions[eligible & ~excluded]
    if not len(candidate_positions):
        return pd.DataFrame()
    candidate_low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)[candidate_positions]
    base = pd.DataFrame(
        {
            "zone_event_id": [f"CTRL_CAND_{int(i):09d}" for i in candidate_positions],
            "event_kind": "non_zone_downside_control",
            "event_pos": candidate_positions,
            "event_bar_time": index[candidate_positions],
            "event_available_time": available_time[candidate_positions],
            "zone_floor_price": candidate_low,
            "zone_ceiling_price": candidate_low,
            "zone_center_price": candidate_low,
            "sweep_low": candidate_low,
        }
    )
    candidate = attach_causal_market_features(base, bars, cfg, feature_frame=feature_frame)
    zone = zone_events.loc[zone_events["is_impulse_first_event"].astype(bool)].copy().reset_index(drop=True)
    zone["_match_bucket"] = _bucket(zone)
    candidate["_match_bucket"] = _bucket(candidate)
    pools: dict[str, deque[int]] = defaultdict(deque)
    for bucket, group in candidate.groupby("_match_bucket", sort=False):
        ordered = group.sort_values("event_pos", kind="mergesort")
        pools[str(bucket)].extend(ordered.index.astype(int).tolist())
    chosen_rows: list[pd.Series] = []
    used: set[int] = set()
    for _, row in zone.sort_values("event_pos", kind="mergesort").iterrows():
        bucket = str(row["_match_bucket"])
        pool = pools.get(bucket)
        if not pool:
            continue
        picked = None
        while pool:
            idx = int(pool.popleft())
            if idx not in used:
                picked = idx
                break
        if picked is None:
            continue
        used.add(picked)
        control = candidate.loc[picked].copy()
        control["matched_zone_event_id"] = str(row["zone_event_id"])
        control["zone_event_id"] = f"CTRL_{str(row['zone_event_id'])}"
        chosen_rows.append(control)
    if not chosen_rows:
        return pd.DataFrame()
    out = pd.DataFrame(chosen_rows).reset_index(drop=True)
    out["is_impulse_first_event"] = True
    out["online_impulse_id"] = out["zone_event_id"]
    out["impulse_observation_number"] = 1
    out["zone_member_count"] = 0
    out["zone_timeframe_count"] = 0
    out["zone_primary_timeframe"] = "CONTROL"
    out["zone_max_timeframe_min"] = 0
    out["zone_width_bp"] = 0.0
    out["zone_latest_level_available_time"] = pd.NaT
    return out.drop(columns=["_match_bucket"], errors="ignore")
