#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal R10 hypothesis universe built from R02/R09 structural features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas import AtlasConfig, attach_active_confluence
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.structured_stop_pool import (
    FAMILY_COLUMNS as R09_FAMILY_COLUMNS,
    StructuredStopPoolConfig,
    build_level_structure_features,
)

from .config import StructuredPullbackConfig

BASELINE_ID = "B0"
FAMILY_IDS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
ALL_FAMILY_IDS = (BASELINE_ID, *FAMILY_IDS)

FAMILY_NAMES = {
    "B0": "All confirmed Higher-Low pullbacks",
    "P1": "Large-decline first Higher-Low pullback",
    "P2": "BOS first Higher-Low pullback",
    "P3": "Layered-base Higher-Low pullback",
    "P4": "Strong-displacement Higher-Low pullback",
    "P5": "Base-breakout Higher-Low pullback",
    "P6": "Multi-timeframe-confluence Higher-Low pullback",
    "P7": "Trend-continuation Higher-Low pullback",
    "P8": "Failed-breakdown recovery Higher-Low pullback",
}

FAMILY_RATIONALES = {
    "B0": "Broad causal benchmark: buy a later retest of any confirmed Higher Low and invalidate below the prior structural low.",
    "P1": "After a meaningful decline and rebound, the first Higher Low may be the first place reversal longs cluster entries.",
    "P2": "A causal break of the previous bearish Swing High can make the subsequent Higher Low a more credible trend-change pullback.",
    "P3": "Two near-equal older lows plus a Higher Low create a layered base; the structural stop belongs below the lower base floor.",
    "P4": "A Higher Low that itself launched strong right-confirmation displacement may be more memorable when later retested.",
    "P5": "A base breakout followed by a Higher Low pullback may combine base participants and breakout participants.",
    "P6": "A Higher Low overlapping active levels from another timeframe may attract multiple trading horizons before the retest occurs.",
    "P7": "A mature Higher-High/Higher-Low sequence tests whether trend-continuation pullbacks outperform first-reversal pullbacks.",
    "P8": "A failed breakdown followed by a Higher Low may create a stronger reversal narrative and a clearer structural invalidation.",
}


def hypothesis_definitions() -> pd.DataFrame:
    """Return the predeclared R10 hypotheses and structural-stop semantics."""

    rows = [
        {
            "family_id": family_id,
            "name": FAMILY_NAMES[family_id],
            "behavioral_rationale": FAMILY_RATIONALES[family_id],
            "entry_rule": "After causal Higher-Low confirmation, rest a buy limit at the confirmed Higher-Low price until the next same-timeframe Swing Low becomes available.",
            "stop_rule": (
                "5bp below the prior structural Swing Low; P3/P5 use 5bp below the lower of the previous two base lows."
            ),
            "target_rules": "Prior upswing high H0 plus fixed 1R/2R/3R targets; same-bar ambiguity is resolved conservatively.",
        }
        for family_id in ALL_FAMILY_IDS
    ]
    return pd.DataFrame(rows)


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for name in frame.columns:
        if name.endswith("_time") or "available_time" in name or name in {"pivot_time", "pivot_bar_end_time"}:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
    for name in frame.columns:
        if name.startswith(("hyp_", "is_", "previous_", "prior_lows_", "base_breakout_", "bos_", "higher_high_", "failed_breakdown_", "any_")):
            values = frame[name]
            if values.dtype == object:
                lowered = values.astype(str).str.strip().str.lower()
                if lowered.isin({"true", "false", "nan", "none", ""}).all():
                    frame[name] = lowered.eq("true")
    return frame


def load_or_build_r09_level_features(
    report_dir: Path,
    levels: pd.DataFrame,
    primary: pd.DataFrame,
    *,
    rebuild_if_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Reuse the complete R09 level table or causally rebuild the same features."""

    path = report_dir / "18_level_structure_feature_table.csv.gz"
    if path.exists():
        frame = _read_csv(path)
        if frame["level_id"].duplicated().any():
            raise RuntimeError("duplicate level_id in cached R09 level feature table")
        return frame, pd.DataFrame(), "R09_REPORT_CACHE"
    if not rebuild_if_missing:
        raise FileNotFoundError(
            f"R09 level feature cache missing: {path}. Run R09 first or pass --rebuild-r09-if-missing."
        )
    r09_cfg = StructuredStopPoolConfig().validate()
    features, thresholds = build_level_structure_features(levels, primary, r09_cfg)
    return features, thresholds, "R09_REBUILT_IN_R10"


def _atlas_config(cfg: StructuredPullbackConfig) -> AtlasConfig:
    return AtlasConfig(
        timeframes=cfg.timeframes,
        confirmation_orders=(1, 2, 3, 5),
        approach_distance_bp=200.0,
        touch_distance_bp=5.0,
        sweep_epsilon_bp=0.01,
        acceptance_depth_bp=50.0,
        acceptance_consecutive_closes=3,
        resolution_horizon_bars=180,
        forward_horizons=(5, 15, 30, 60, 180),
        confluence_tolerances_bp=(float(cfg.confluence_tolerance_bp),),
    ).validate()


def _confluence_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _datetime_ns(values: pd.Series | pd.Index) -> np.ndarray:
    """Return nanosecond epochs independent of pandas datetime storage unit."""

    parsed = pd.to_datetime(values, errors="coerce")
    return np.asarray(parsed, dtype="datetime64[ns]").astype(np.int64, copy=False)


def _attach_formation_confluence(
    features: pd.DataFrame,
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredPullbackConfig,
) -> pd.DataFrame:
    """Count active multi-timeframe levels when the pullback order becomes legal."""

    if features.empty:
        return features.copy()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    index_ns = _datetime_ns(index)
    out = features.copy()
    signal_time = pd.to_datetime(out["structure_available_time"], errors="coerce")
    signal_ns = _datetime_ns(signal_time)
    event_pos = np.searchsorted(index_ns, signal_ns, side="left").astype(np.int64)
    events = pd.DataFrame(
        {
            "event_id": [f"R10_FORM_{int(value):08d}" for value in out["level_id"]],
            "event_pos": event_pos,
            "level_id": pd.to_numeric(out["level_id"], errors="raise").astype(np.int64),
            "level_price": pd.to_numeric(out["level_price"], errors="coerce"),
        }
    )
    attached = attach_active_confluence(events, lifecycle, _atlas_config(config))
    token = _confluence_token(config.confluence_tolerance_bp)
    columns = [
        "level_id",
        f"active_level_count_{token}bp",
        f"active_timeframe_count_{token}bp",
    ]
    out = out.merge(attached.loc[:, columns], on="level_id", how="left", validate="one_to_one")
    out["formation_multitimeframe_confluence"] = (
        pd.to_numeric(out[f"active_timeframe_count_{token}bp"], errors="coerce").fillna(0).ge(2)
    )
    return out


def _period(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    return pd.Series(
        np.select(
            [ts < pd.Timestamp("2025-01-01"), ts < pd.Timestamp("2025-10-01")],
            ["EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="LATE_2025Q4_2026H1",
        ),
        index=values.index,
        dtype="object",
    )


def _next_same_timeframe_signal(features: pd.DataFrame) -> pd.Series:
    ordered = features.sort_values(
        ["source_timeframe_min", "structure_available_time", "pivot_time", "level_id"],
        kind="mergesort",
    ).copy()
    ordered["next_same_timeframe_structure_available_time"] = ordered.groupby(
        "source_timeframe_min", sort=False
    )["structure_available_time"].shift(-1)
    return ordered.set_index("level_id")["next_same_timeframe_structure_available_time"]


def build_pullback_candidate_universe(
    level_features: pd.DataFrame,
    lifecycle: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredPullbackConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    max_candidates: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build unique level candidates and an expanded family table.

    Membership uses only information available at ``structure_available_time``.
    The next same-timeframe Swing Low is an execution-time cancellation event,
    never a feature used to admit the candidate.
    """

    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    required = {
        "level_id",
        "source_timeframe",
        "source_timeframe_min",
        "pivot_time",
        "level_price",
        "structure_available_time",
        "previous_swing_low_price",
        "previous2_swing_low_price",
        "current_leg_high_price",
        "is_higher_low",
        *[name for name in R09_FAMILY_COLUMNS if name != R09_FAMILY_COLUMNS[5]],
    }
    missing = sorted(required.difference(level_features.columns))
    if missing:
        raise ValueError(f"R09 level features missing columns: {missing}")

    features = level_features.copy()
    features["structure_available_time"] = pd.to_datetime(features["structure_available_time"], errors="coerce")
    features["pivot_time"] = pd.to_datetime(features["pivot_time"], errors="coerce")
    features = features.loc[
        features["is_higher_low"].astype(bool)
        & features["structure_available_time"].ge(pd.Timestamp(research_start))
        & features["structure_available_time"].lt(pd.Timestamp(research_end_exclusive))
    ].copy()
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Restrict to configured timeframes and attach causal confluence at order activation.
    allowed_minutes = {int(value) for _, value in cfg.timeframes}
    features = features.loc[
        pd.to_numeric(features["source_timeframe_min"], errors="coerce").isin(allowed_minutes)
    ].copy()
    features = _attach_formation_confluence(features, lifecycle, bars, cfg)

    next_signal = _next_same_timeframe_signal(level_features)
    features["next_same_timeframe_structure_available_time"] = features["level_id"].map(next_signal)
    features["period"] = _period(features["structure_available_time"])

    numeric_columns = (
        "level_price",
        "previous_swing_low_price",
        "previous2_swing_low_price",
        "current_leg_high_price",
        "source_timeframe_min",
        "predecessor_decline_atr",
        "rebound_before_current_atr",
        "higher_low_gap_atr",
        "pullback_fraction_of_rebound",
        "confirmation_reaction_high_bp",
    )
    for name in numeric_columns:
        if name in features.columns:
            features[name] = pd.to_numeric(features[name], errors="coerce")

    features["default_structural_anchor_price"] = features["previous_swing_low_price"]
    features["layered_base_anchor_price"] = np.fmin(
        features["previous_swing_low_price"].to_numpy(dtype=float),
        features["previous2_swing_low_price"].to_numpy(dtype=float),
    )
    features["entry_limit_price"] = features["level_price"]
    features["structural_target_h0_price"] = features["current_leg_high_price"]
    features["family_B0"] = True
    features["family_P1"] = features[R09_FAMILY_COLUMNS[0]].astype(bool)
    features["family_P2"] = features[R09_FAMILY_COLUMNS[1]].astype(bool)
    features["family_P3"] = features[R09_FAMILY_COLUMNS[2]].astype(bool)
    features["family_P4"] = features[R09_FAMILY_COLUMNS[3]].astype(bool)
    features["family_P5"] = features[R09_FAMILY_COLUMNS[4]].astype(bool)
    features["family_P6"] = features["formation_multitimeframe_confluence"].astype(bool)
    features["family_P7"] = features[R09_FAMILY_COLUMNS[6]].astype(bool)
    features["family_P8"] = features[R09_FAMILY_COLUMNS[7]].astype(bool)

    # Broad geometric validity; family-specific anchor validity is checked after expansion.
    features["base_geometry_valid"] = (
        features["entry_limit_price"].gt(0)
        & features["default_structural_anchor_price"].gt(0)
        & features["structural_target_h0_price"].gt(features["entry_limit_price"])
    )
    features = features.sort_values(
        ["structure_available_time", "source_timeframe_min", "pivot_time", "level_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if int(max_candidates) > 0 and len(features) > int(max_candidates):
        features = features.head(int(max_candidates)).copy()

    # Whitelist causal feature columns. Future order labels from R02 are intentionally excluded.
    causal_columns = [
        "level_id",
        "source_timeframe",
        "source_timeframe_min",
        "pivot_time",
        "level_price",
        "structure_available_time",
        "next_same_timeframe_structure_available_time",
        "period",
        "entry_limit_price",
        "structural_target_h0_price",
        "default_structural_anchor_price",
        "layered_base_anchor_price",
        "predecessor_decline_atr",
        "rebound_before_current_atr",
        "higher_low_gap_atr",
        "pullback_fraction_of_rebound",
        "confirmation_reaction_high_bp",
        "left_high_range_20_bp",
        "prior_two_low_gap_atr",
        "consecutive_higher_low_count",
        "bos_before_current_low",
        "higher_high_before_current_low",
        "failed_breakdown_previous_low",
        "formation_multitimeframe_confluence",
        "base_geometry_valid",
        *[f"family_{family_id}" for family_id in ALL_FAMILY_IDS],
    ]
    causal_columns += [
        name
        for name in features.columns
        if name.startswith(("active_level_count_", "active_timeframe_count_"))
    ]
    causal_columns = [name for name in dict.fromkeys(causal_columns) if name in features.columns]
    unique = features.loc[:, causal_columns].copy()

    rows: list[pd.DataFrame] = []
    for family_id in ALL_FAMILY_IDS:
        member = unique.loc[unique[f"family_{family_id}"].astype(bool)].copy()
        if member.empty:
            continue
        member["family_id"] = family_id
        member["family_name"] = FAMILY_NAMES[family_id]
        if family_id in {"P3", "P5"}:
            member["structural_anchor_price"] = member["layered_base_anchor_price"]
            member["anchor_rule"] = "LOWER_OF_PREVIOUS_TWO_SWING_LOWS"
        else:
            member["structural_anchor_price"] = member["default_structural_anchor_price"]
            member["anchor_rule"] = "PREVIOUS_SWING_LOW"
        member["stop_price"] = member["structural_anchor_price"] * (
            1.0 - float(cfg.stop_buffer_bp) / 10_000.0
        )
        member["risk_distance_return"] = (
            member["entry_limit_price"] - member["stop_price"]
        ) / member["entry_limit_price"]
        member["h0_reward_return"] = (
            member["structural_target_h0_price"] - member["entry_limit_price"]
        ) / member["entry_limit_price"]
        member["h0_reward_risk_ratio"] = (
            member["h0_reward_return"] / member["risk_distance_return"]
        )
        member["family_geometry_valid"] = (
            member["base_geometry_valid"].astype(bool)
            & member["structural_anchor_price"].gt(0)
            & member["stop_price"].lt(member["entry_limit_price"])
            & member["risk_distance_return"].gt(0)
            & member["h0_reward_return"].gt(0)
        )
        member["candidate_family_id"] = [
            f"R10_{family_id}_{int(level_id):08d}" for level_id in member["level_id"]
        ]
        rows.append(member)
    family = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not family.empty:
        family = family.sort_values(
            ["structure_available_time", "source_timeframe_min", "level_id", "family_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        if family["candidate_family_id"].duplicated().any():
            raise RuntimeError("duplicate R10 candidate_family_id")
    return unique, family
