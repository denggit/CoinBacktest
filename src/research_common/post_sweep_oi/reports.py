#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptive reports for R05 Binance OI post-sweep research."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import PostSweepOIConfig
from .align import first_per_event


def data_quality_report(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: object,
    coverage_days: pd.DataFrame,
) -> pd.DataFrame:
    present = features.get("oi_context_present", pd.Series(False, index=features.index)).eq(True)
    ages = pd.to_numeric(features.get("oi_age_seconds"), errors="coerce")
    rows = [
        {"metric": "checkpoints", "value": len(features)},
        {"metric": "events", "value": int(features["zone_event_id"].nunique()) if len(features) else 0},
        {"metric": "oi_aligned_checkpoints", "value": int(present.sum())},
        {"metric": "oi_alignment_rate", "value": float(present.mean()) if len(features) else 0.0},
        {"metric": "oi_median_age_seconds", "value": float(ages.median()) if ages.notna().any() else np.nan},
        {"metric": "oi_max_age_seconds", "value": float(ages.max()) if ages.notna().any() else np.nan},
        {"metric": "oi_causal_violations", "value": int((features.get("oi_causal_flag", False) == False).where(present, False).sum()) if len(features) else 0},
        {"metric": "oi_db_rows", "value": getattr(coverage, "rows", np.nan)},
        {"metric": "oi_db_start", "value": getattr(coverage, "start", None)},
        {"metric": "oi_db_end", "value": getattr(coverage, "end", None)},
        {"metric": "oi_complete_days", "value": getattr(coverage, "complete_days", np.nan)},
        {"metric": "oi_partial_days", "value": getattr(coverage, "partial_days", np.nan)},
        {"metric": "oi_missing_days", "value": getattr(coverage, "missing_days", np.nan)},
        {"metric": "oi_error_days", "value": getattr(coverage, "error_days", np.nan)},
        {"metric": "coverage_rows", "value": len(coverage_days)},
        {"metric": "taker_ratio_non_null_rate", "value": float(features.get("taker_volume_imbalance", pd.Series(dtype=float)).notna().mean()) if len(features) else 0.0},
        {"metric": "top_trader_ratio_non_null_rate", "value": float(features.get("top_trader_position_long_share", pd.Series(dtype=float)).notna().mean()) if len(features) else 0.0},
        {"metric": "feature_label_row_match", "value": len(features) == len(labels)},
    ]
    return pd.DataFrame(rows)


def coverage_by_period(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    out = []
    for period, group in features.groupby("period", dropna=False):
        present = group["oi_context_present"].eq(True)
        out.append({
            "period": period,
            "checkpoints": len(group),
            "events": group["zone_event_id"].nunique(),
            "aligned": int(present.sum()),
            "alignment_rate": float(present.mean()),
            "median_age_seconds": pd.to_numeric(group["oi_age_seconds"], errors="coerce").median(),
            "taker_ratio_non_null_rate": group.get("taker_volume_imbalance", pd.Series(index=group.index, dtype=float)).notna().mean(),
        })
    return pd.DataFrame(out)


def position_flow_state_summary(full: pd.DataFrame, horizons: Iterable[int] = (15, 60, 180)) -> pd.DataFrame:
    base = first_per_event(full)
    return _group_outcomes(base, ["period", "position_flow_state_5m"], horizons)


def fixed_oi_bin_summary(full: pd.DataFrame, config: PostSweepOIConfig) -> pd.DataFrame:
    """Use rounded, predeclared OI change bins; no fitted cut points."""

    base = first_per_event(full)
    rows: list[pd.DataFrame] = []
    edges = [-np.inf, -0.01, -0.005, -0.001, 0.001, 0.005, 0.01, np.inf]
    labels = ["<=-1%", "-1..-0.5%", "-0.5..-0.1%", "-0.1..0.1%", "0.1..0.5%", "0.5..1%", ">1%"]
    for window in ("5m", "15m", "30m", "1h", "4h"):
        col = f"oi_base_change_{window}"
        if col not in base.columns:
            continue
        data = base.copy()
        data["oi_bin"] = pd.cut(pd.to_numeric(data[col], errors="coerce"), bins=edges, labels=labels, include_lowest=True, right=False)
        summary = _group_outcomes(data, ["period", "oi_bin"], (15, 60, 180))
        summary.insert(0, "feature", col)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def new_low_attempt_oi_summary(full: pd.DataFrame) -> pd.DataFrame:
    attempts = full.loc[full["new_low_attempt_flag"].eq(True)].copy()
    if attempts.empty:
        return pd.DataFrame()
    # One first occurrence of each state per event prevents duration weighting.
    attempts = attempts.sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    attempts = attempts.drop_duplicates(["zone_event_id", "delta_oi_state_5m"], keep="first")
    return _group_outcomes(attempts, ["period", "delta_oi_state_5m"], (15, 60, 180))


def attempt_mechanism_summary(full: pd.DataFrame) -> pd.DataFrame:
    attempts = full.loc[
        full["new_low_attempt_flag"].eq(True)
        & full["prior_attempt_checkpoint_id"].notna()
    ].copy()
    if attempts.empty:
        return pd.DataFrame()
    attempts = attempts.sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    attempts = attempts.drop_duplicates(["zone_event_id", "attempt_mechanism_state"], keep="first")
    summary = _group_outcomes(attempts, ["period", "attempt_mechanism_state"], (15, 60, 180))
    extra = attempts.groupby(["period", "attempt_mechanism_state"], dropna=False).agg(
        median_oi_change_since_prior=("oi_base_change_since_prior_attempt", "median"),
        median_sell_impact_ratio=("sell_impact_ratio_vs_prior_attempt", "median"),
        median_delta_impact_ratio=("delta_impact_ratio_vs_prior_attempt", "median"),
        median_new_low_extension_bp=("new_low_extension_bp", "median"),
    ).reset_index()
    return summary.merge(extra, on=["period", "attempt_mechanism_state"], how="left")


def oracle_pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    fields = [
        "oi_base_change_5m", "oi_base_change_15m", "oi_base_change_1h",
        "oi_base", "delta_ratio_1m", "sell_share_1m", "price_change_1m_bp",
        "downside_bp_per_sell_million_1m",
        "downside_bp_per_abs_negative_delta_million_1m",
        "close_vs_running_low_bp", "new_low_extension_bp",
        "future_oi_base_change_15m", "future_oi_base_change_30m",
        "future_oi_base_change_60m", "future_mfe_60m", "future_mae_60m",
    ]
    rows = []
    for period, group in pairs.groupby("period", dropna=False):
        row: dict[str, object] = {"period": period, "pairs": len(group)}
        for name in fields:
            oracle_col, prior_col = f"oracle_{name}", f"prior_{name}"
            if oracle_col not in group.columns or prior_col not in group.columns:
                continue
            oracle = pd.to_numeric(group[oracle_col], errors="coerce")
            prior = pd.to_numeric(group[prior_col], errors="coerce")
            diff = oracle - prior
            row[f"oracle_median_{name}"] = _median_or_nan(oracle)
            row[f"prior_median_{name}"] = _median_or_nan(prior)
            row[f"median_diff_{name}"] = _median_or_nan(diff)
            row[f"oracle_gt_prior_rate_{name}"] = (oracle > prior).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rebound_oi_path_summary(full: pd.DataFrame) -> pd.DataFrame:
    """Classify future price/OI paths; future OI is a label, not a feature."""

    anchor = first_per_event(full)
    rows: list[pd.DataFrame] = []
    for horizon in (15, 30, 60, 180):
        price_col = f"future_close_return_{horizon}m"
        oi_col = f"future_oi_base_change_{horizon}m"
        if price_col not in anchor.columns or oi_col not in anchor.columns:
            continue
        data = anchor.copy()
        price = pd.to_numeric(data[price_col], errors="coerce")
        oi = pd.to_numeric(data[oi_col], errors="coerce")
        data["future_price_oi_state"] = np.select(
            [(price > 0) & (oi < 0), (price > 0) & (oi > 0), (price < 0) & (oi > 0), (price < 0) & (oi < 0)],
            ["PRICE_UP_OI_DOWN", "PRICE_UP_OI_UP", "PRICE_DOWN_OI_UP", "PRICE_DOWN_OI_DOWN"],
            default="MISSING_OR_FLAT",
        )
        group = data.groupby(["period", "future_price_oi_state"], dropna=False).agg(
            events=("zone_event_id", "nunique"),
            median_price_return=(price_col, "median"),
            median_oi_change=(oi_col, "median"),
            median_mfe_180m=("future_mfe_180m", "median"),
            median_mae_180m=("future_mae_180m", "median"),
        ).reset_index()
        group.insert(0, "horizon_minutes", horizon)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def large_mfe_oi_profile(full: pd.DataFrame, config: PostSweepOIConfig) -> pd.DataFrame:
    """Event-level profiles at the first post-sweep checkpoint."""

    anchor = first_per_event(full)
    feature_cols = [
        "oi_base_change_5m", "oi_base_change_15m", "oi_base_change_30m",
        "oi_base_change_1h", "oi_base_change_4h", "taker_volume_imbalance",
        "top_trader_position_long_share", "global_account_long_share",
        "delta_ratio_5m", "price_change_5m_bp", "sweep_depth_to_pre_atr_240m",
        "zone_prior_touch_median", "zone_left_high_range_20_bp_max",
        "zone_confirmation_reaction_close_bp_max",
    ]
    rows = []
    for threshold in config.large_mfe_returns:
        flag = pd.to_numeric(anchor["future_mfe_180m"], errors="coerce") >= threshold
        for period, group in anchor.groupby("period", dropna=False):
            group_flag = flag.loc[group.index]
            for name in feature_cols:
                if name not in group.columns:
                    continue
                values = pd.to_numeric(group[name], errors="coerce")
                pos, neg = values[group_flag], values[~group_flag]
                rows.append({
                    "threshold": threshold,
                    "period": period,
                    "feature": name,
                    "large_events": int(group.loc[group_flag, "zone_event_id"].nunique()),
                    "other_events": int(group.loc[~group_flag, "zone_event_id"].nunique()),
                    "large_median": pos.median(),
                    "other_median": neg.median(),
                    "median_difference": pos.median() - neg.median(),
                    "large_non_null": int(pos.notna().sum()),
                    "other_non_null": int(neg.notna().sum()),
                })
    return pd.DataFrame(rows)


def taker_ratio_summary(full: pd.DataFrame) -> pd.DataFrame:
    base = first_per_event(full)
    if "taker_volume_imbalance" not in base.columns:
        return pd.DataFrame()
    edges = [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf]
    labels = ["<=-0.3", "-0.3..-0.1", "-0.1..0.1", "0.1..0.3", ">0.3"]
    base["taker_imbalance_bin"] = pd.cut(
        pd.to_numeric(base["taker_volume_imbalance"], errors="coerce"),
        bins=edges, labels=labels, include_lowest=True, right=False,
    )
    return _group_outcomes(base, ["period", "taker_imbalance_bin"], (15, 60, 180))


def causal_audit(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    present = features["oi_context_present"].eq(True)
    oi_time = pd.to_datetime(features["oi_available_time"], errors="coerce")
    checkpoint = pd.to_datetime(features["checkpoint_available_time"], errors="coerce")
    tests = [
        ("oi_available_not_after_checkpoint", int((present & (oi_time > checkpoint)).sum())),
        ("oi_context_has_non_negative_age", int((present & (pd.to_numeric(features["oi_age_seconds"], errors="coerce") < 0)).sum())),
        ("checkpoint_id_unique_features", int(features["checkpoint_id"].duplicated().sum())),
        ("checkpoint_id_unique_labels", int(labels["checkpoint_id"].duplicated().sum())),
        ("feature_label_row_count_match", abs(len(features) - len(labels))),
        ("future_columns_in_features", sum(name.startswith("future_") for name in features.columns)),
    ]
    return pd.DataFrame([{"test": name, "violations": int(value)} for name, value in tests])


def compact_event_sample(full: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    anchor = first_per_event(full)
    sort_cols = ["future_mfe_180m", "future_close_return_180m"]
    return anchor.sort_values(sort_cols, ascending=False, kind="mergesort").head(int(sample_rows)).reset_index(drop=True)



def _median_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan

def _group_outcomes(frame: pd.DataFrame, groups: list[str], horizons: Iterable[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    named = {
        "rows": ("checkpoint_id", "size"),
        "events": ("zone_event_id", "nunique"),
        "median_elapsed_bars": ("elapsed_bars", "median"),
        "median_oi_change_5m": ("oi_base_change_5m", "median"),
        "median_oi_change_15m": ("oi_base_change_15m", "median"),
    }
    for horizon in horizons:
        for metric in ("future_mfe", "future_mae", "future_close_return"):
            col = f"{metric}_{horizon}m"
            if col in frame.columns:
                named[f"median_{col}"] = (col, "median")
        no_low = f"future_no_lower_low_{horizon}m"
        if no_low in frame.columns:
            frame = frame.copy()
            frame[f"_{no_low}_num"] = frame[no_low].astype("boolean").astype("Float64")
            named[f"rate_{no_low}"] = (f"_{no_low}_num", "mean")
    return frame.groupby(groups, observed=False, dropna=False).agg(**named).reset_index()
