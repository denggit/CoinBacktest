#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load only the report columns needed by R08."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "checkpoint_id", "zone_event_id", "period", "event_available_time",
    "checkpoint_available_time", "elapsed_bars",
    "checkpoint_high", "checkpoint_low", "checkpoint_close",
    "zone_floor_reclaimed",
    "no_new_low_3bars", "no_new_low_5bars", "no_new_low_10bars",
    "micro_high_break_3bars", "micro_high_break_5bars", "micro_high_break_10bars",
)
LABEL_COLUMNS = (
    "checkpoint_id", "entry_reference_time", "entry_reference_price",
    "future_mfe_15m",
    "future_mfe_60m", "future_mae_60m", "future_no_lower_low_60m",
    "future_reversal_dominant_60m", "future_continuation_dominant_60m",
    "future_mfe_180m", "future_mae_180m", "future_close_return_180m",
    "future_large_mfe_0p5_180m", "future_large_mfe_1_180m", "future_large_mfe_2_180m",
)
R07_COLUMNS = (
    "checkpoint_id", "zone_event_id", "period", "new_low_attempt_index",
    "fp_downside_bp_per_sell_million", "fp_low3_large_sell_share", "fp_causal_valid",
)

BOOL_COLUMNS = {
    "new_low_attempt_flag", "zone_floor_reclaimed", "zone_ceiling_reclaimed",
    "no_new_low_3bars", "no_new_low_5bars", "no_new_low_10bars",
    "micro_high_break_3bars", "micro_high_break_5bars", "micro_high_break_10bars",
    "future_label_complete_15m", "future_no_lower_low_15m", "future_label_complete_30m",
    "future_no_lower_low_30m", "future_label_complete_60m", "future_no_lower_low_60m",
    "future_reversal_dominant_60m", "future_continuation_dominant_60m",
    "future_label_complete_180m", "future_no_lower_low_180m", "future_reversal_dominant_180m",
    "future_continuation_dominant_180m", "future_large_mfe_0p5_180m", "future_large_mfe_1_180m",
    "future_large_mfe_2_180m", "zone_has_1H", "zone_has_4H", "zone_has_1D",
    "zone_all_members_fresh", "fp_causal_valid",
}
TIME_COLUMNS = {"event_available_time", "checkpoint_time", "checkpoint_available_time", "entry_reference_time"}


def _read(path: Path, usecols: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    out = pd.read_csv(path, usecols=list(usecols), low_memory=False)
    for c in TIME_COLUMNS.intersection(out.columns):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in BOOL_COLUMNS.intersection(out.columns):
        if not pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
    return out


def load_r04(r04_dir: Path, start: pd.Timestamp, end: pd.Timestamp, max_events: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _read(r04_dir / "13_checkpoint_feature_table.csv.gz", FEATURE_COLUMNS)
    features = features.loc[(features["event_available_time"] >= start) & (features["event_available_time"] <= end)].copy()
    if max_events > 0:
        keep = features[["zone_event_id", "event_available_time"]].drop_duplicates().sort_values(["event_available_time", "zone_event_id"], kind="mergesort").head(max_events)["zone_event_id"]
        features = features.loc[features["zone_event_id"].isin(set(keep))].copy()
    wanted = set(features["checkpoint_id"])
    # Labels are read with a compact schema; filtering before the join prevents
    # the smoke path from carrying the full 709k-row label table further.
    labels = _read(r04_dir / "14_checkpoint_label_table.csv.gz", LABEL_COLUMNS)
    labels = labels.loc[labels["checkpoint_id"].isin(wanted)].copy()
    merged = features.merge(labels, on="checkpoint_id", how="inner", validate="one_to_one", sort=False)
    merged = merged.sort_values(["zone_event_id", "elapsed_bars", "checkpoint_available_time", "checkpoint_id"], kind="mergesort").reset_index(drop=True)
    events = merged.groupby("zone_event_id", sort=False).head(1).copy()
    if events.empty:
        raise RuntimeError("R08 found no R04 events in requested window")
    return merged, events


def load_r07_opportunity(r07_dir: Path, event_ids: set[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    path = r07_dir / "19_all_attempt_feature_table.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=list(R07_COLUMNS)), {}
    data = _read(path, R07_COLUMNS)
    data = data.loc[data["zone_event_id"].isin(event_ids)].copy()
    data = data.loc[data["fp_causal_valid"].fillna(False)]
    data["new_low_attempt_index"] = pd.to_numeric(data["new_low_attempt_index"], errors="coerce")
    data = data.sort_values(["zone_event_id", "new_low_attempt_index", "checkpoint_id"], kind="mergesort").groupby("zone_event_id", sort=False).head(1)
    thresholds: dict[str, float] = {}
    lift = r07_dir / "09_footprint_frozen_quantile_lift.csv"
    if lift.exists():
        table = pd.read_csv(lift)
        for feature, direction in (("fp_downside_bp_per_sell_million", "HIGH"), ("fp_low3_large_sell_share", "LOW")):
            rows = table.loc[(table["reference_period"] == "EARLY_2023_2024") & (table["feature"] == feature) & (table["outcome"] == "future_large_mfe_1_180m") & (table["favorable_direction"] == direction)]
            if not rows.empty:
                thresholds[feature] = float(rows.iloc[0]["frozen_threshold"])
    if "fp_downside_bp_per_sell_million" not in thresholds:
        early = data.loc[data["period"] == "EARLY_2023_2024", "fp_downside_bp_per_sell_million"]
        thresholds["fp_downside_bp_per_sell_million"] = float(pd.to_numeric(early, errors="coerce").quantile(0.75))
    if "fp_low3_large_sell_share" not in thresholds:
        early = data.loc[data["period"] == "EARLY_2023_2024", "fp_low3_large_sell_share"]
        thresholds["fp_low3_large_sell_share"] = float(pd.to_numeric(early, errors="coerce").quantile(0.25))
    impact = pd.to_numeric(data["fp_downside_bp_per_sell_million"], errors="coerce")
    large = pd.to_numeric(data["fp_low3_large_sell_share"], errors="coerce")
    data["fp_high_impact_flag"] = impact >= thresholds["fp_downside_bp_per_sell_million"]
    data["fp_low_large_sell_flag"] = large <= thresholds["fp_low3_large_sell_share"]
    data["fp_opportunity_score"] = data[["fp_high_impact_flag", "fp_low_large_sell_flag"]].sum(axis=1).astype("int8")
    return data, thresholds
