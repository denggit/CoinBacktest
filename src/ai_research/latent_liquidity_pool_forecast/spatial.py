#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal temporal-spatial candidate features for latent liquidity pools."""
from __future__ import annotations

from dataclasses import replace
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_path_atlas.config import DEFAULT_CONFIG as ATLAS_CONFIG
from src.ai_research.latent_liquidity_path_atlas.features import _path_feature_series
from src.ai_research.latent_liquidity_path_atlas.macro import build_macro_path_context, normalize_minute_bars
from src.ai_research.latent_liquidity_path_atlas.time_axis import as_datetime_ns

from .config import LatentLiquidityPoolForecastConfig


def period_for_time(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.to_datetime(values)
    return np.select(
        [t < pd.Timestamp("2025-01-01"), t < pd.Timestamp("2025-10-01")],
        ["TRAIN_2023_2024", "VALIDATION_2025Q1_Q3"],
        default="HOLDOUT_2025Q4_2026H1",
    )


def build_snapshot_context(
    minute_bars: pd.DataFrame,
    second_bars: pd.DataFrame,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp,
    config: LatentLiquidityPoolForecastConfig,
) -> pd.DataFrame:
    """Build decision snapshots using completed data only."""
    minute = normalize_minute_bars(minute_bars)
    if minute.empty:
        return pd.DataFrame()
    atlas_cfg = replace(
        ATLAS_CONFIG,
        macro_windows_minutes=config.macro_windows_minutes,
        macro_context_minutes=config.macro_context_minutes,
        path_windows_seconds=config.micro_windows_seconds,
        pre_context_seconds=config.micro_context_seconds,
    )
    macro = build_macro_path_context(minute, atlas_cfg)
    if macro.empty:
        return pd.DataFrame()
    decision = pd.to_datetime(macro["macro_available_time"], errors="coerce")
    mask = (
        decision.ge(core_start)
        & decision.le(core_end)
        & decision.dt.minute.mod(config.snapshot_interval_minutes).eq(0)
        & decision.dt.second.eq(0)
    )
    out = macro.loc[mask].copy()
    out = out.rename(columns={"macro_available_time": "decision_time", "macro_pre_event_close": "current_price"})
    out["decision_time"] = as_datetime_ns(out["decision_time"])
    out["feature_available_time"] = out["decision_time"]
    out["period"] = period_for_time(out["decision_time"])

    if not second_bars.empty:
        sec = second_bars.copy()
        sec.index = as_datetime_ns(sec.index, errors="coerce")
        sec = sec.loc[~sec.index.isna()].sort_index(kind="mergesort")
        additions = _path_feature_series(sec, atlas_cfg)
        sample_index = pd.DatetimeIndex(out["decision_time"]) - pd.Timedelta(seconds=1)
        micro_keep = (
            "path_ret_", "path_range_bp_", "path_efficiency_", "path_realized_vol_",
            "path_notional_intensity_", "path_delta_share_", "path_trades_intensity_",
            "path_max_trade_ratio_", "path_turnover_per_range_intensity_",
            "path_pressure_without_progress_", "path_travel_bp_",
        )
        for name, series in additions.items():
            if not name.startswith(micro_keep):
                continue
            out[f"micro_{name}"] = series.reindex(sample_index).to_numpy()
    macro_keep_prefixes = (
        "macro_ret_", "macro_range_bp_", "macro_efficiency_", "macro_realized_vol_",
        "macro_drawdown_from_high_", "macro_rally_from_low_", "macro_notional_intensity_",
        "macro_delta_share_", "macro_trades_intensity_", "macro_travel_bp_",
        "macro_overlap_ratio_", "macro_pressure_without_progress_",
        "macro_impact_bp_per_million_", "macro_price_residency_proxy_",
    )
    meta = {"decision_time", "feature_available_time", "period", "current_price", "macro_bar_start_time"}
    # Raw notional is kept only long enough to build a causal position-buildup proxy.
    raw_notional = {f"macro_notional_{w}m" for w in config.macro_windows_minutes}
    keep = [name for name in out.columns if name in meta or name in raw_notional or name.startswith(macro_keep_prefixes) or name.startswith("micro_")]
    return out.loc[:, keep].reset_index(drop=True)


def expand_zone_lattice(snapshots: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    offsets = np.asarray(config.zone_offsets_bp, dtype=np.float32)
    n = len(snapshots)
    repeated = snapshots.loc[snapshots.index.repeat(len(offsets) * 2)].reset_index(drop=True)
    repeated["zone_side"] = np.tile(np.repeat(np.array(["DOWN", "UP"], dtype=object), len(offsets)), n)
    repeated["zone_distance_bp"] = np.tile(np.tile(offsets, 2), n)
    current = repeated["current_price"].to_numpy(dtype=float)
    dist = repeated["zone_distance_bp"].to_numpy(dtype=float) / 1e4
    down = repeated["zone_side"].eq("DOWN").to_numpy()
    repeated["zone_price"] = np.where(down, current * (1.0 - dist), current * (1.0 + dist))
    near_bp = np.maximum(repeated["zone_distance_bp"].to_numpy(dtype=float) - float(config.zone_half_width_bp), 0.0)
    far_bp = repeated["zone_distance_bp"].to_numpy(dtype=float) + float(config.zone_half_width_bp)
    repeated["zone_near_distance_bp"] = near_bp.astype(np.float32)
    repeated["zone_far_distance_bp"] = far_bp.astype(np.float32)
    repeated["zone_near_price"] = np.where(down, current * (1.0 - near_bp / 1e4), current * (1.0 + near_bp / 1e4))
    repeated["zone_far_price"] = np.where(down, current * (1.0 - far_bp / 1e4), current * (1.0 + far_bp / 1e4))
    repeated["side_is_down"] = down.astype(np.int8)
    repeated["zone_id"] = (
        repeated["decision_time"].astype(str) + "_" + repeated["zone_side"].astype(str) + "_" + repeated["zone_distance_bp"].astype(str)
    )
    return add_zone_path_features(repeated, config)


def _rolling_level(current: np.ndarray, metric: np.ndarray, kind: str) -> np.ndarray:
    # drawdown=(close-high)/high -> high=close/(1+drawdown)
    # rally=(close-low)/low -> low=close/(1+rally)
    if kind == "HIGH":
        denom = 1.0 + metric
    else:
        denom = 1.0 + metric
    with np.errstate(divide="ignore", invalid="ignore"):
        return current / denom


def add_zone_path_features(frame: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    current = out["current_price"].to_numpy(dtype=float)
    zone = out["zone_price"].to_numpy(dtype=float)
    is_down = out["zone_side"].eq("DOWN").to_numpy()
    nesting = np.zeros(len(out), dtype=np.int16)
    untouched = np.zeros(len(out), dtype=np.int16)
    for w in config.macro_windows_minutes:
        dd_name, rally_name = f"macro_drawdown_from_high_{w}m", f"macro_rally_from_low_{w}m"
        if dd_name not in out or rally_name not in out:
            continue
        dd = pd.to_numeric(out[dd_name], errors="coerce").to_numpy(dtype=float)
        rally = pd.to_numeric(out[rally_name], errors="coerce").to_numpy(dtype=float)
        high = _rolling_level(current, dd, "HIGH")
        low = _rolling_level(current, rally, "LOW")
        boundary = np.where(is_down, low, high)
        signed_gap = np.where(is_down, (zone - low) / current * 1e4, (high - zone) / current * 1e4)
        out[f"zone_boundary_gap_{w}m_bp"] = signed_gap.astype(np.float32)
        near = np.abs((zone - boundary) / current * 1e4) <= 10.0
        outside = np.where(is_down, zone < low, zone > high)
        nesting += near.astype(np.int16)
        untouched += outside.astype(np.int16)
        out[f"zone_near_boundary_{w}m"] = near.astype(np.int8)
        out[f"zone_untouched_{w}m"] = outside.astype(np.int8)
        notional_name = f"macro_notional_{w}m"
        if notional_name in out:
            notional = pd.to_numeric(out[notional_name], errors="coerce").clip(lower=0)
            out[f"zone_buildup_log_notional_{w}m"] = np.where(outside, np.log1p(notional / 1_000_000.0), 0.0).astype(np.float32)
        ret_name = f"macro_ret_{w}m"
        if ret_name in out:
            ret = pd.to_numeric(out[ret_name], errors="coerce").fillna(0.0).to_numpy(dtype=float) * 1e4
            out[f"zone_directional_path_{w}m_bp"] = np.where(is_down, -ret, ret).astype(np.float32)
    out["zone_boundary_nesting_count"] = nesting
    out["zone_untouched_window_count"] = untouched
    return out


def attach_swing_spatial_features(
    zones: pd.DataFrame,
    lifecycle: pd.DataFrame,
    config: LatentLiquidityPoolForecastConfig,
) -> pd.DataFrame:
    """Attach all-active 15m+ unswept Swing proximity as a supplemental family.

    A chronological sweep-line inventory avoids scanning the whole lifecycle at
    every snapshot.  Old levels remain active indefinitely until their first
    causal sweep; no "recent Swing only" shortcut is used.  Candidate zones for
    each snapshot/side are evaluated as one broadcast matrix for speed.
    """
    if zones.empty:
        return zones.copy()
    out = zones.copy()
    bands = tuple(float(x) for x in config.swing_band_bp)
    defaults = {
        "swing_nearest_distance_bp": np.nan,
        "swing_nearest_age_minutes": np.nan,
        "swing_oldest_age_within_25bp_minutes": np.nan,
        "swing_timeframe_diversity_25bp": 0,
    }
    for band in bands:
        defaults[f"swing_count_{int(band)}bp"] = 0
        defaults[f"swing_timeframe_diversity_{int(band)}bp"] = 0
    for key, value in defaults.items():
        out[key] = value
    if lifecycle.empty:
        return out

    levels = lifecycle.copy()
    levels["initial_available_time"] = pd.to_datetime(levels["initial_available_time"], errors="coerce")
    levels["sweep_available_time"] = pd.to_datetime(levels["sweep_available_time"], errors="coerce")
    levels = levels.loc[
        pd.to_numeric(levels["source_timeframe_min"], errors="coerce").ge(15)
        & levels["initial_available_time"].notna()
    ].copy()
    if levels.empty:
        return out
    levels = levels.sort_values("initial_available_time", kind="mergesort").reset_index(drop=True)
    add_times = levels["initial_available_time"].to_numpy(dtype="datetime64[ns]")
    removals = levels.loc[levels["sweep_available_time"].notna(), ["level_id", "sweep_available_time"]].sort_values("sweep_available_time", kind="mergesort")
    rem_times = removals["sweep_available_time"].to_numpy(dtype="datetime64[ns]")
    rem_ids = removals["level_id"].to_numpy()
    active: dict[object, tuple[str, float, pd.Timestamp, str]] = {}
    add_pos = 0
    rem_pos = 0
    grouped = out.groupby("decision_time", sort=True).groups
    for decision_time, indices in grouped.items():
        t = pd.Timestamp(decision_time)
        t64 = np.datetime64(t, "ns")
        while add_pos < len(levels) and add_times[add_pos] <= t64:
            row = levels.iloc[add_pos]
            active[row["level_id"]] = (
                str(row["level_side"]), float(row["level_price"]),
                pd.Timestamp(row["initial_available_time"]), str(row["source_timeframe"]),
            )
            add_pos += 1
        while rem_pos < len(rem_times) and rem_times[rem_pos] <= t64:
            active.pop(rem_ids[rem_pos], None)
            rem_pos += 1
        if not active:
            continue
        for zone_side, level_side in (("DOWN", "LOW"), ("UP", "HIGH")):
            side_rows = [idx for idx in indices if out.at[idx, "zone_side"] == zone_side]
            records = [r for r in active.values() if r[0] == level_side]
            if not side_rows or not records:
                continue
            zone_prices = out.loc[side_rows, "zone_price"].to_numpy(dtype=float)
            level_prices = np.asarray([r[1] for r in records], dtype=float)
            ages = np.asarray([max((t - r[2]).total_seconds() / 60.0, 0.0) for r in records], dtype=float)
            timeframes = np.asarray([r[3] for r in records], dtype=object)
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.abs(level_prices[:, None] / zone_prices[None, :] - 1.0) * 1e4
            dist[~np.isfinite(dist)] = np.inf
            nearest = np.argmin(dist, axis=0)
            nearest_dist = dist[nearest, np.arange(len(side_rows))]
            out.loc[side_rows, "swing_nearest_distance_bp"] = nearest_dist
            out.loc[side_rows, "swing_nearest_age_minutes"] = ages[nearest]
            for band in bands:
                mask = dist <= band
                key = int(band)
                out.loc[side_rows, f"swing_count_{key}bp"] = mask.sum(axis=0).astype(np.int32)
                diversity = []
                for col in range(mask.shape[1]):
                    diversity.append(len(set(timeframes[mask[:, col]].tolist())))
                out.loc[side_rows, f"swing_timeframe_diversity_{key}bp"] = np.asarray(diversity, dtype=np.int16)
            near25 = dist <= 25.0
            oldest = np.full(len(side_rows), np.nan, dtype=float)
            diversity25 = np.zeros(len(side_rows), dtype=np.int16)
            for col in range(near25.shape[1]):
                mask = near25[:, col]
                if mask.any():
                    oldest[col] = float(np.max(ages[mask]))
                    diversity25[col] = len(set(timeframes[mask].tolist()))
            out.loc[side_rows, "swing_oldest_age_within_25bp_minutes"] = oldest
            out.loc[side_rows, "swing_timeframe_diversity_25bp"] = diversity25
    return out
