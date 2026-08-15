#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Future-only spatial labels for R02 candidate price zones."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LatentLiquidityPoolForecastConfig


def _future_extreme(values: pd.Series, window: int, how: str) -> pd.Series:
    shifted = values.shift(-1)
    rev = shifted.iloc[::-1]
    if how == "min":
        rolled = rev.rolling(window, min_periods=window).min()
    else:
        rolled = rev.rolling(window, min_periods=window).max()
    return rolled.iloc[::-1]


def attach_touch_labels(zones: pd.DataFrame, minute_bars: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if zones.empty:
        return zones.copy()
    bars = minute_bars.sort_index(kind="mergesort")
    helper = pd.DataFrame(index=bars.index)
    for horizon in config.touch_horizons_minutes:
        helper[f"future_low_{horizon}m"] = _future_extreme(pd.to_numeric(bars["low"], errors="coerce"), int(horizon), "min")
        helper[f"future_high_{horizon}m"] = _future_extreme(pd.to_numeric(bars["high"], errors="coerce"), int(horizon), "max")
    helper["decision_time"] = helper.index + pd.Timedelta(minutes=1)
    merged = pd.merge(zones, helper.reset_index(drop=True), on="decision_time", how="left", validate="many_to_one")
    down = merged["zone_side"].eq("DOWN").to_numpy()
    near_price = merged.get("zone_near_price", merged["zone_price"]).to_numpy(dtype=float)
    near_distance = merged.get("zone_near_distance_bp", merged["zone_distance_bp"]).to_numpy(dtype=float)
    for horizon in config.touch_horizons_minutes:
        low = pd.to_numeric(merged[f"future_low_{horizon}m"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(merged[f"future_high_{horizon}m"], errors="coerce").to_numpy(dtype=float)
        complete = np.isfinite(low) & np.isfinite(high)
        merged[f"touch_label_complete_{horizon}m"] = complete
        down_touch = np.where(near_distance <= 1e-9, low < near_price, low <= near_price)
        up_touch = np.where(near_distance <= 1e-9, high > near_price, high >= near_price)
        touched = np.where(down, down_touch, up_touch)
        # Incomplete future windows must never be silently treated as negative labels.
        merged[f"touch_{horizon}m"] = pd.array(np.where(complete, touched, False), dtype="boolean")
        merged.drop(columns=[f"future_low_{horizon}m", f"future_high_{horizon}m"], inplace=True)
    merged["primary_touch_label_complete"] = merged[f"touch_label_complete_{config.primary_horizon_minutes}m"].astype(bool)
    return merged


def attach_release_labels(zones: pd.DataFrame, episodes: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if zones.empty:
        return zones.copy()
    out = zones.copy()
    n = len(out)
    out["release_within_horizon"] = False
    out["favorable_release"] = False
    out["continuation_release"] = False
    out["time_to_release_minutes"] = np.nan
    out["release_density_proxy"] = np.nan
    out["release_episode_size"] = np.nan
    out["release_score"] = np.nan
    out["release_path_cluster"] = -1
    out["sweep_depth_bp"] = np.nan
    out["reversal_after_extreme_bp"] = np.nan
    out["time_to_extreme_seconds"] = np.nan
    out["release_outcome_type"] = "NONE"
    offsets = np.asarray(config.zone_offsets_bp, dtype=float)
    half = float(config.zone_half_width_bp)
    horizon = pd.Timedelta(minutes=int(config.primary_horizon_minutes))
    eps = episodes.copy()
    eps["event_time"] = pd.to_datetime(eps["event_time"], errors="coerce")
    by_side = {side: group.sort_values("event_time", kind="mergesort").reset_index(drop=True) for side, group in eps.groupby("event_side", sort=False)}
    times_by_side = {side: group["event_time"].to_numpy(dtype="datetime64[ns]") for side, group in by_side.items()}

    grouped = out.groupby("decision_time", sort=False).groups
    for decision_time, idxs in grouped.items():
        t = pd.Timestamp(decision_time)
        current = float(out.loc[idxs[0], "current_price"])
        for side in ("DOWN", "UP"):
            group = by_side.get(side)
            if group is None or group.empty:
                continue
            times = times_by_side[side]
            left = int(np.searchsorted(times, np.datetime64(t, "ns"), side="right"))
            right = int(np.searchsorted(times, np.datetime64(t + horizon, "ns"), side="left"))
            if right <= left:
                continue
            future = group.iloc[left:right]
            ref = future["event_reference_price"].to_numpy(dtype=float)
            distance = np.where(side == "DOWN", (current - ref) / current * 1e4, (ref - current) / current * 1e4)
            valid = np.isfinite(distance) & (distance > 0) & (distance <= offsets[-1] + half)
            if not valid.any():
                continue
            future = future.iloc[np.flatnonzero(valid)].reset_index(drop=True)
            distance = distance[valid]
            nearest_idx = np.abs(distance[:, None] - offsets[None, :]).argmin(axis=1)
            nearest_gap = np.abs(distance - offsets[nearest_idx])
            valid_zone = nearest_gap <= half
            if not valid_zone.any():
                continue
            future = future.iloc[np.flatnonzero(valid_zone)].reset_index(drop=True)
            nearest_idx = nearest_idx[valid_zone]
            side_rows = list(idxs)[0:len(offsets)] if side == "DOWN" else list(idxs)[len(offsets):]
            # Multiple future episodes may map to one zone; use the first causal future release.
            for zone_pos in np.unique(nearest_idx):
                candidates = future.iloc[np.flatnonzero(nearest_idx == zone_pos)]
                first = candidates.sort_values("event_time", kind="mergesort").iloc[0]
                row = side_rows[int(zone_pos)]
                out.at[row, "release_within_horizon"] = True
                out.at[row, "favorable_release"] = bool(first["favorable_reversal"])
                out.at[row, "continuation_release"] = str(first["outcome_type"]) == "ACCEPT_CONTINUATION"
                out.at[row, "time_to_release_minutes"] = (pd.Timestamp(first["event_time"]) - t).total_seconds() / 60.0
                out.at[row, "release_density_proxy"] = float(first["release_density_proxy"])
                out.at[row, "release_episode_size"] = float(first["release_episode_size"])
                out.at[row, "release_score"] = float(first["release_score"]) if pd.notna(first["release_score"]) else np.nan
                out.at[row, "release_path_cluster"] = int(first["path_cluster"])
                out.at[row, "sweep_depth_bp"] = float(first["future_extension_bp"])
                out.at[row, "reversal_after_extreme_bp"] = float(first["future_reversal_after_extreme_bp"])
                out.at[row, "time_to_extreme_seconds"] = float(first["future_time_to_extreme_seconds"])
                out.at[row, "release_outcome_type"] = str(first["outcome_type"])
    out["release_on_touch"] = out["release_within_horizon"] & out[f"touch_{config.primary_horizon_minutes}m"].astype(bool)
    out["favorable_on_release"] = out["favorable_release"] & out["release_within_horizon"]
    return out


def deterministic_control_sample(frame: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    """Keep a weighted model sample plus complete lattices for location audit.

    The model sample uses inverse-probability row controls.  Independently, a
    deterministic fraction of decision-time x side groups retains *every* zone
    so top-zone diagnostics never choose from a partially sampled price grid.
    Audit-only rows have zero model weight and are excluded from model fitting.
    """
    if frame.empty:
        return frame.copy()
    primary = f"touch_{config.primary_horizon_minutes}m"
    release = frame["release_within_horizon"].astype(bool).to_numpy()
    touched = frame[primary].astype(bool).to_numpy()
    ids = frame["zone_id"].astype(str)
    hashed = pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype=np.uint64)
    touched_threshold = int(config.touched_control_keep_fraction * 1_000_000)
    untouched_threshold = int(config.untouched_control_keep_fraction * 1_000_000)
    sampled_touched = touched & ~release & ((hashed % 1_000_000) < touched_threshold)
    sampled_untouched = ~touched & ~release & ((hashed % 1_000_000) < untouched_threshold)
    model_keep = release | sampled_touched | sampled_untouched

    group_key = frame["decision_time"].astype(str) + "|" + frame["zone_side"].astype(str)
    group_hash = pd.util.hash_pandas_object(group_key, index=False).to_numpy(dtype=np.uint64)
    audit_threshold = int(config.full_lattice_audit_group_fraction * 1_000_000)
    audit_group = (group_hash % 1_000_000) < audit_threshold

    keep = model_keep | audit_group
    out = frame.loc[keep].copy().reset_index(drop=True)
    kept_model = model_keep[keep]
    kept_audit = audit_group[keep]
    out["model_sample_keep"] = kept_model
    out["full_lattice_audit_group"] = kept_audit
    out["sample_weight"] = 0.0
    t = out[primary].astype(bool)
    r = out["release_within_horizon"].astype(bool)
    m = out["model_sample_keep"].astype(bool)
    out.loc[m & r, "sample_weight"] = 1.0
    out.loc[m & t & ~r, "sample_weight"] = 1.0 / config.touched_control_keep_fraction
    out.loc[m & ~t & ~r, "sample_weight"] = 1.0 / config.untouched_control_keep_fraction
    return out
