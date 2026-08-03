#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load broad and matched R07 event universes without future leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


R04_FEATURE_FILE = "13_checkpoint_feature_table.csv.gz"
R04_LABEL_FILE = "14_checkpoint_label_table.csv.gz"
R04_STATIC_FILE = "11_static_zone_event_features.csv"
R06_FEATURE_FILE = "19_attempt_feature_table.csv.gz"
R06_LABEL_FILE = "20_attempt_label_table.csv.gz"

R04_FEATURE_COLUMNS = (
    "checkpoint_id",
    "zone_event_id",
    "event_kind",
    "period",
    "event_available_time",
    "checkpoint_time",
    "checkpoint_available_time",
    "elapsed_bars",
    "zone_floor_price",
    "zone_ceiling_price",
    "zone_center_price",
    "sweep_low",
    "checkpoint_open",
    "checkpoint_high",
    "checkpoint_low",
    "checkpoint_close",
    "running_low_since_sweep",
    "new_low_attempt_flag",
    "new_low_attempt_index",
    "bars_since_new_low_attempt",
    "new_low_extension_bp",
    "new_low_extension_to_pre_atr_240m",
    "attempt_delta_notional",
    "attempt_sell_notional",
    "attempt_extension_vs_previous",
    "close_vs_zone_floor_bp",
    "close_vs_running_low_bp",
    "running_low_vs_zone_floor_bp",
    "delta_ratio_1m",
    "sell_share_1m",
    "large_delta_ratio_1m",
    "price_change_1m_bp",
    "downside_bp_per_sell_million_1m",
    "downside_bp_per_abs_negative_delta_million_1m",
    "delta_ratio_5m",
    "sell_share_5m",
    "price_change_5m_bp",
    "entry_reference_time",
    "entry_reference_price",
)

R04_LABEL_COLUMNS = (
    "checkpoint_id",
    "zone_event_id",
    "period",
    "future_label_complete_15m",
    "future_mfe_15m",
    "future_mae_15m",
    "future_close_return_15m",
    "future_no_lower_low_15m",
    "future_label_complete_30m",
    "future_mfe_30m",
    "future_mae_30m",
    "future_close_return_30m",
    "future_no_lower_low_30m",
    "future_label_complete_60m",
    "future_mfe_60m",
    "future_mae_60m",
    "future_close_return_60m",
    "future_no_lower_low_60m",
    "future_reversal_dominant_60m",
    "future_continuation_dominant_60m",
    "future_label_complete_180m",
    "future_mfe_180m",
    "future_mae_180m",
    "future_close_return_180m",
    "future_no_lower_low_180m",
    "future_reversal_dominant_180m",
    "future_continuation_dominant_180m",
    "future_large_mfe_0p5_180m",
    "future_large_mfe_1_180m",
    "future_large_mfe_2_180m",
)

STATIC_COLUMNS = (
    "zone_event_id",
    "zone_member_count",
    "zone_timeframe_count",
    "zone_primary_timeframe",
    "zone_max_timeframe_min",
    "zone_has_1H",
    "zone_has_4H",
    "zone_has_1D",
    "zone_width_bp",
    "zone_age_median_minutes",
    "zone_age_max_minutes",
    "zone_fresh_member_share",
    "zone_all_members_fresh",
    "zone_prior_touch_median",
    "zone_prior_touch_max",
    "zone_confirmed_order_max",
    "zone_left_high_range_20_bp_max",
    "zone_confirmation_reaction_close_bp_max",
    "sweep_depth_below_floor_bp",
    "pre_atr_240m_bp",
    "pre_return_60m",
    "pre_down_efficiency_60m",
    "current_delta_ratio",
    "sweep_depth_to_pre_atr_240m",
)


def _existing_columns(path: Path, requested: Iterable[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    return [name for name in requested if name in header]


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def _coerce_times(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in out.columns:
        if name.endswith("_time") or name.endswith("_available_time"):
            out[name] = pd.to_datetime(out[name], errors="coerce")
    return out


def _read_filtered_csv(
    path: Path,
    *,
    usecols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_column: str,
    chunksize: int = 150_000,
    require_new_low: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk[time_column] = pd.to_datetime(chunk[time_column], errors="coerce")
        mask = chunk[time_column].between(start, end, inclusive="both")
        if require_new_low and "new_low_attempt_flag" in chunk.columns:
            mask &= _bool_series(chunk["new_low_attempt_flag"])
        selected = chunk.loc[mask].copy()
        if len(selected):
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def load_r04_attempt_universe(
    report_dir: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_events: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every causal new-low attempt plus physically separate future labels."""

    root = Path(report_dir)
    feature_path = root / R04_FEATURE_FILE
    label_path = root / R04_LABEL_FILE
    static_path = root / R04_STATIC_FILE
    for path in (feature_path, label_path, static_path):
        if not path.exists():
            raise FileNotFoundError(f"R07 source missing: {path}")

    feature_cols = _existing_columns(feature_path, R04_FEATURE_COLUMNS)
    features = _read_filtered_csv(
        feature_path,
        usecols=feature_cols,
        start=start,
        end=end,
        time_column="checkpoint_available_time",
        require_new_low=True,
    )
    features = _coerce_times(features)
    if features.empty:
        raise RuntimeError("R04 contains no new-low attempts in the requested interval")
    features["attempt_id"] = features["checkpoint_id"].astype(str)
    features["attempt_universe"] = "ALL_NEW_LOW_ATTEMPTS"
    features["selection_uses_future"] = False

    if max_events > 0 and len(features) > max_events:
        features = features.sort_values(
            ["period", "checkpoint_available_time", "checkpoint_id"], kind="mergesort"
        ).head(max_events).copy()
    ids = set(features["checkpoint_id"].astype(str))

    label_cols = _existing_columns(label_path, R04_LABEL_COLUMNS)
    labels: list[pd.DataFrame] = []
    for chunk in pd.read_csv(label_path, usecols=label_cols, chunksize=150_000, low_memory=False):
        keep = chunk["checkpoint_id"].astype(str).isin(ids)
        if keep.any():
            labels.append(chunk.loc[keep].copy())
    label_frame = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(columns=label_cols)
    if len(label_frame) != len(features):
        missing = ids.difference(set(label_frame.get("checkpoint_id", pd.Series(dtype=str)).astype(str)))
        raise RuntimeError(
            f"R04 label join incomplete features={len(features):,} labels={len(label_frame):,} "
            f"missing_sample={sorted(missing)[:5]}"
        )
    label_frame["attempt_id"] = label_frame["checkpoint_id"].astype(str)

    static_cols = _existing_columns(static_path, STATIC_COLUMNS)
    static = pd.read_csv(static_path, usecols=static_cols, low_memory=False)
    static = static.drop_duplicates("zone_event_id", keep="last")
    features = features.merge(static, on="zone_event_id", how="left", validate="many_to_one")
    forbidden = [name for name in features.columns if name.startswith("future_")]
    if forbidden:
        raise RuntimeError(f"future label leakage into R07 feature universe: {forbidden}")
    return features.reset_index(drop=True), label_frame.reset_index(drop=True)


def load_r06_matched_universe(
    report_dir: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_pairs: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load R06 oracle/prior/control cohorts for matched mechanism diagnostics."""

    root = Path(report_dir)
    feature_path = root / R06_FEATURE_FILE
    label_path = root / R06_LABEL_FILE
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"R06 matched source missing: {feature_path if not feature_path.exists() else label_path}"
        )
    header = pd.read_csv(feature_path, nrows=0).columns.tolist()
    features = _read_filtered_csv(
        feature_path,
        usecols=header,
        start=start,
        end=end,
        time_column="checkpoint_available_time",
        require_new_low=False,
    )
    features = _coerce_times(features)
    if max_pairs > 0 and "pair_id" in features.columns:
        pair_ids = (
            features.loc[features["cohort"] == "ORACLE_TURN", ["period", "pair_id"]]
            .drop_duplicates()
            .sort_values(["period", "pair_id"], kind="mergesort")
            .head(max_pairs)["pair_id"]
            .astype(str)
        )
        selected = set(pair_ids)
        features = features.loc[features["pair_id"].astype(str).isin(selected)].copy()
    ids = set(features["window_id"].astype(str))
    labels: list[pd.DataFrame] = []
    label_header = pd.read_csv(label_path, nrows=0).columns.tolist()
    for chunk in pd.read_csv(label_path, usecols=label_header, chunksize=100_000, low_memory=False):
        keep = chunk["window_id"].astype(str).isin(ids)
        if keep.any():
            labels.append(chunk.loc[keep].copy())
    label_frame = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(columns=label_header)
    forbidden = [name for name in features.columns if name.startswith("future_")]
    if forbidden:
        raise RuntimeError(f"future label leakage into matched R07 features: {forbidden}")
    return features.reset_index(drop=True), label_frame.reset_index(drop=True)
