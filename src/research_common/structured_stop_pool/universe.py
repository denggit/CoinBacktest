#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 reuse/rebuild and R09 unique-zone/control universe construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas import AtlasConfig, build_level_lifecycle, build_swing_low_universe
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_zone_study import (
    ZoneStudyConfig,
    attach_causal_market_features,
    build_causal_market_feature_frame,
    build_matched_controls,
    build_sweep_zone_events,
)

from .config import StructuredStopPoolConfig
from .structure import FAMILY_COLUMNS, attach_zone_hypotheses, build_level_structure_features


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for name in frame.columns:
        if name.endswith("_time") or "available_time" in name or name in {"pivot_time", "event_bar_time"}:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
    return frame


def load_or_build_r02(
    report_dir: Path,
    primary: pd.DataFrame,
    config: StructuredStopPoolConfig,
    *,
    rebuild_if_missing: bool,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    level_path = report_dir / "03_swing_level_table.csv"
    lifecycle_path = report_dir / "04_level_lifecycle_table.csv"
    if level_path.exists() and lifecycle_path.exists():
        levels = _read_csv(level_path)
        lifecycle = _read_csv(lifecycle_path)
        return levels, lifecycle, "R02_REPORT_CACHE"
    if not rebuild_if_missing:
        raise FileNotFoundError(
            f"R02 report missing: {level_path} / {lifecycle_path}. "
            "Run R02 first or pass --rebuild-r02-if-missing."
        )
    cfg = config.validate()
    atlas = AtlasConfig(
        timeframes=cfg.timeframes,
        confirmation_orders=(1, 2, 3, 5),
        approach_distance_bp=200.0,
        touch_distance_bp=5.0,
        sweep_epsilon_bp=0.01,
        acceptance_depth_bp=50.0,
        acceptance_consecutive_closes=3,
        resolution_horizon_bars=180,
        forward_horizons=(5, 15, 30, 60, 180),
        confluence_tolerances_bp=(5.0, 10.0, 25.0, 50.0),
    ).validate()
    levels = build_swing_low_universe(primary, atlas)
    lifecycle = build_level_lifecycle(primary, levels, atlas, show_progress=show_progress)
    return levels, lifecycle, "R02_REBUILT_IN_R09"


def audit_r02_bar_alignment(lifecycle: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    bars = normalize_primary_bars(primary)
    swept = lifecycle.loc[pd.to_numeric(lifecycle.get("sweep_pos"), errors="coerce").ge(0)].copy()
    if swept.empty:
        return pd.DataFrame([{"check": "r02_sweep_alignment", "violations": 0, "status": "PASS"}])
    pos = pd.to_numeric(swept["sweep_pos"], errors="raise").astype(np.int64).to_numpy()
    outside = (pos < 0) | (pos >= len(bars))
    expected = pd.Series(pd.NaT, index=swept.index, dtype="datetime64[ns]")
    valid = ~outside
    expected.loc[valid] = (bars.index[pos[valid]] + pd.Timedelta(minutes=1)).to_numpy()
    actual = pd.to_datetime(swept["sweep_available_time"], errors="coerce")
    mismatch = int((expected.notna() & actual.notna() & expected.ne(actual)).sum()) + int(outside.sum())
    return pd.DataFrame(
        [
            {"check": "r02_sweep_pos_within_primary", "violations": int(outside.sum()), "status": "PASS" if not outside.any() else "FAIL"},
            {"check": "r02_sweep_available_time_matches_primary", "violations": mismatch, "status": "PASS" if mismatch == 0 else "FAIL"},
        ]
    )


def build_r09_universe(
    levels: pd.DataFrame,
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredStopPoolConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    max_events: int = 0,
    include_controls: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    level_features, structure_thresholds = build_level_structure_features(levels, bars, cfg)
    zone_cfg = ZoneStudyConfig(
        zone_merge_tolerance_bp=float(cfg.zone_merge_tolerance_bp),
        zone_merge_sensitivity_bp=(float(cfg.zone_merge_tolerance_bp),),
        impulse_gap_bars=int(cfg.impulse_gap_bars),
        impulse_price_tolerance_bp=float(cfg.impulse_price_tolerance_bp),
        path_horizons=tuple(int(v) for v in cfg.path_horizons),
        tp_returns=tuple(float(v) for v in cfg.tp_returns),
        structural_break_epsilon_bp=float(cfg.structural_break_epsilon_bp),
        control_exclusion_bars=int(cfg.control_exclusion_bars),
        control_min_downside_atr=float(cfg.control_min_downside_atr),
        control_max_per_zone=1 if include_controls else 0,
    ).validate()
    all_zones = build_sweep_zone_events(lifecycle, bars, zone_cfg)
    ts = pd.to_datetime(all_zones["event_available_time"], errors="coerce")
    all_zones = all_zones.loc[(ts >= research_start) & (ts < research_end_exclusive)].reset_index(drop=True)
    market_features = build_causal_market_feature_frame(bars, zone_cfg)
    all_zones = attach_causal_market_features(all_zones, bars, zone_cfg, feature_frame=market_features)
    zones = all_zones.loc[all_zones["is_impulse_first_event"].astype(bool)].copy().reset_index(drop=True)
    zones = attach_zone_hypotheses(zones, level_features)
    if int(max_events) > 0 and len(zones) > int(max_events):
        zones = zones.sort_values(["event_pos", "zone_event_id"], kind="mergesort").head(int(max_events)).reset_index(drop=True)
    controls = pd.DataFrame()
    if include_controls:
        controls = build_matched_controls(
            zones,
            lifecycle,
            bars,
            zone_cfg,
            research_start=research_start,
            research_end_exclusive=research_end_exclusive,
            feature_frame=market_features,
        )
        if not controls.empty:
            for family in FAMILY_COLUMNS:
                controls[family] = False
                controls[f"{family}_member_count"] = 0
            controls["zone_structured_family_count"] = 0
            controls["zone_has_any_structured_family"] = False
            controls["independent_multitimeframe_confluence"] = False
            controls["zone_member_structure_available_time_max"] = pd.NaT
            controls["zone_member_primary_family_timeframes"] = ""
    return level_features, structure_thresholds, zones, controls, all_zones
