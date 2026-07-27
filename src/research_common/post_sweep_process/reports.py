#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptive reports for R04 post-sweep process research."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import PostSweepConfig

EPS = 1e-12


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _mean_bool(series: pd.Series) -> float:
    values = pd.Series(series, copy=False).astype("boolean")
    return float(values.mean()) if values.notna().any() else np.nan


def _group_metrics(frame: pd.DataFrame, horizon: int) -> dict[str, float | int]:
    complete_col = f"future_label_complete_{horizon}m"
    if complete_col in frame.columns:
        frame = frame.loc[frame[complete_col].fillna(False).astype(bool)]
    if frame.empty:
        return {
            "checkpoints": 0,
            "events": 0,
            "median_future_mfe": np.nan,
            "p75_future_mfe": np.nan,
            "median_future_mae": np.nan,
            "p25_future_mae": np.nan,
            "median_future_close_return": np.nan,
            "positive_future_close_rate": np.nan,
            "no_lower_low_rate": np.nan,
            "reversal_dominant_rate": np.nan,
            "continuation_dominant_rate": np.nan,
        }
    mfe = pd.to_numeric(frame[f"future_mfe_{horizon}m"], errors="coerce")
    mae = pd.to_numeric(frame[f"future_mae_{horizon}m"], errors="coerce")
    close_return = pd.to_numeric(frame[f"future_close_return_{horizon}m"], errors="coerce")
    return {
        "checkpoints": int(len(frame)),
        "events": int(frame["zone_event_id"].nunique()),
        "median_future_mfe": float(mfe.median()),
        "p75_future_mfe": float(mfe.quantile(0.75)),
        "median_future_mae": float(mae.median()),
        "p25_future_mae": float(mae.quantile(0.25)),
        "median_future_close_return": float(close_return.median()),
        "positive_future_close_rate": float((close_return > 0).mean()),
        "no_lower_low_rate": _mean_bool(frame[f"future_no_lower_low_{horizon}m"]),
        "reversal_dominant_rate": _mean_bool(frame[f"future_reversal_dominant_{horizon}m"]),
        "continuation_dominant_rate": _mean_bool(frame[f"future_continuation_dominant_{horizon}m"]),
    }


def causal_audit(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    forbidden = [
        name for name in features.columns
        if name.startswith("future_") or name.startswith("entry_reference_") or "oracle" in name.lower()
    ]
    checks.append({"check": "no_future_fields_in_features", "violations": len(forbidden), "detail": ",".join(forbidden[:20])})
    if not features.empty:
        event_available = pd.to_datetime(features["event_available_time"], errors="coerce")
        checkpoint_available = pd.to_datetime(features["checkpoint_available_time"], errors="coerce")
        checkpoint_time = pd.to_datetime(features["checkpoint_time"], errors="coerce")
        checks.append({
            "check": "event_available_not_after_checkpoint",
            "violations": int((event_available > checkpoint_available).fillna(True).sum()),
            "detail": "event_available_time <= checkpoint_available_time",
        })
        checks.append({
            "check": "closed_checkpoint_available_time",
            "violations": int((checkpoint_available < checkpoint_time + pd.Timedelta(minutes=1)).fillna(True).sum()),
            "detail": "checkpoint available only after its 1m bar closes",
        })
    if not labels.empty and "entry_reference_time" in labels.columns:
        entry = pd.to_datetime(labels["entry_reference_time"], errors="coerce")
        checkpoint_available = pd.to_datetime(labels["checkpoint_available_time"], errors="coerce")
        valid = entry.notna()
        checks.append({
            "check": "entry_not_before_checkpoint_available",
            "violations": int((entry[valid] < checkpoint_available[valid]).sum()),
            "detail": "hypothetical label path begins at next 1m open",
        })
    return pd.DataFrame(checks)


def new_low_attempt_summary(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    attempts = checkpoints.loc[checkpoints["new_low_attempt_flag"].astype(bool)].copy()
    if attempts.empty:
        return pd.DataFrame()
    attempts["attempt_bucket"] = np.where(
        pd.to_numeric(attempts["new_low_attempt_index"], errors="coerce") >= 5,
        "5+",
        pd.to_numeric(attempts["new_low_attempt_index"], errors="coerce").astype("Int64").astype(str),
    )
    rows: list[dict[str, object]] = []
    horizon = max(cfg.future_horizons)
    for (period, bucket), group in attempts.groupby(["period", "attempt_bucket"], sort=False):
        row: dict[str, object] = {"period": period, "attempt_bucket": bucket}
        row.update(_group_metrics(group, horizon))
        row.update({
            "median_extension_bp": _median(group["new_low_extension_bp"]),
            "median_extension_to_atr240": _median(group["new_low_extension_to_pre_atr_240m"]),
            "median_attempt_delta": _median(group["attempt_delta_notional"]),
            "median_attempt_sell": _median(group["attempt_sell_notional"]),
            "median_extension_vs_previous": _median(group["attempt_extension_vs_previous"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_path_summary(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    rows: list[dict[str, object]] = []
    for (period, elapsed), group in checkpoints.groupby(["period", "elapsed_bars"], sort=False):
        for horizon in cfg.future_horizons:
            row: dict[str, object] = {"period": period, "elapsed_bars": int(elapsed), "horizon_minutes": int(horizon)}
            row.update(_group_metrics(group, int(horizon)))
            rows.append(row)
    return pd.DataFrame(rows)


def _confirmation_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    false = pd.Series(False, index=frame.index)
    delta3 = pd.to_numeric(frame.get("delta_ratio_3m", np.nan), errors="coerce")
    extension_ratio = pd.to_numeric(frame.get("attempt_extension_vs_previous", np.nan), errors="coerce")
    masks = {
        "ALL_CHECKPOINTS": pd.Series(True, index=frame.index),
        "NEW_LOW_ATTEMPT": frame.get("new_low_attempt_flag", false).fillna(False).astype(bool),
        "NO_NEW_LOW_3": frame.get("no_new_low_3bars", false).fillna(False).astype(bool),
        "NO_NEW_LOW_5": frame.get("no_new_low_5bars", false).fillna(False).astype(bool),
        "CVD_NEW_LOW_WITHOUT_PRICE_NEW_LOW": frame.get("cvd_new_low_without_price_new_low", false).fillna(False).astype(bool),
        "NEGATIVE_DELTA_WITHOUT_PRICE_NEW_LOW": frame.get("negative_delta_without_price_new_low", false).fillna(False).astype(bool),
        "MICRO_HIGH_BREAK_3": frame.get("micro_high_break_3bars", false).fillna(False).astype(bool),
        "MICRO_HIGH_BREAK_5": frame.get("micro_high_break_5bars", false).fillna(False).astype(bool),
        "ZONE_FLOOR_RECLAIMED": frame.get("zone_floor_reclaimed", false).fillna(False).astype(bool),
        "ZONE_CEILING_RECLAIMED": frame.get("zone_ceiling_reclaimed", false).fillna(False).astype(bool),
        "POSITIVE_3M_DELTA": delta3 > 0,
        "SHRINKING_NEW_LOW_EXTENSION": frame.get("new_low_attempt_flag", false).fillna(False).astype(bool) & (extension_ratio < 1.0),
    }
    masks["NO_NEW_LOW_5_AND_MICRO_BREAK_3"] = masks["NO_NEW_LOW_5"] & masks["MICRO_HIGH_BREAK_3"]
    masks["CVD_DIVERGENCE_AND_MICRO_BREAK_3"] = masks["CVD_NEW_LOW_WITHOUT_PRICE_NEW_LOW"] & masks["MICRO_HIGH_BREAK_3"]
    masks["POSITIVE_DELTA_AND_MICRO_BREAK_3"] = masks["POSITIVE_3M_DELTA"] & masks["MICRO_HIGH_BREAK_3"]
    masks["NEG_DELTA_STALL_AND_MICRO_BREAK_3"] = masks["NEGATIVE_DELTA_WITHOUT_PRICE_NEW_LOW"] & masks["MICRO_HIGH_BREAK_3"]
    return masks


def confirmation_state_summary(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    rows: list[dict[str, object]] = []
    masks = _confirmation_masks(checkpoints)
    for state, mask in masks.items():
        subset = checkpoints.loc[mask]
        if subset.empty:
            continue
        for period, group in subset.groupby("period", sort=False):
            for horizon in cfg.future_horizons:
                row: dict[str, object] = {
                    "confirmation_state": state,
                    "period": period,
                    "horizon_minutes": int(horizon),
                    "median_elapsed_bars": _median(group["elapsed_bars"]),
                }
                row.update(_group_metrics(group, int(horizon)))
                rows.append(row)
    return pd.DataFrame(rows)


def large_mfe_summary(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    horizon = max(cfg.future_horizons)
    fixed_elapsed = {1, 3, 5, 10, 15, 30, 60, 120, 180}
    data = checkpoints.loc[checkpoints["elapsed_bars"].isin(fixed_elapsed)].copy()
    complete_col = f"future_label_complete_{horizon}m"
    if complete_col in data.columns:
        data = data.loc[data[complete_col].fillna(False).astype(bool)].copy()
    rows: list[dict[str, object]] = []
    mfe = pd.to_numeric(data[f"future_mfe_{horizon}m"], errors="coerce")
    for threshold in cfg.large_mfe_returns:
        for (period, elapsed), group in data.assign(_large=mfe >= threshold).groupby(["period", "elapsed_bars"], sort=False):
            large = group["_large"].fillna(False).astype(bool)
            rows.append({
                "period": period,
                "elapsed_bars": int(elapsed),
                "horizon_minutes": horizon,
                "large_mfe_threshold": float(threshold),
                "checkpoints": int(len(group)),
                "events": int(group["zone_event_id"].nunique()),
                "large_mfe_checkpoints": int(large.sum()),
                "large_mfe_rate": float(large.mean()),
                "median_mfe": _median(group[f"future_mfe_{horizon}m"]),
                "median_mae": _median(group[f"future_mae_{horizon}m"]),
            })
    return pd.DataFrame(rows)


def _curated_features(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "elapsed_bars",
        "new_low_attempt_index",
        "bars_since_new_low_attempt",
        "new_low_extension_bp",
        "new_low_extension_to_pre_atr_240m",
        "attempt_extension_vs_previous",
        "close_vs_zone_floor_bp",
        "close_vs_running_low_bp",
        "running_low_vs_zone_floor_bp",
        "cum_delta_ratio_since_sweep",
        "delta_ratio_1m",
        "delta_ratio_3m",
        "delta_ratio_5m",
        "delta_ratio_15m",
        "sell_share_3m",
        "sell_share_5m",
        "large_delta_ratio_5m",
        "price_change_3m_bp",
        "price_change_5m_bp",
        "downside_bp_per_sell_million_3m",
        "downside_bp_per_sell_million_5m",
        "downside_bp_per_abs_negative_delta_million_5m",
        "price_to_delta_slope_5m",
        # Joined static event features when supplied.
        "zone_max_timeframe_min",
        "zone_member_count",
        "zone_timeframe_count",
        "zone_age_median_minutes",
        "zone_prior_touch_median",
        "zone_confirmed_order_max",
        "zone_left_high_range_20_bp_max",
        "zone_confirmation_reaction_close_bp_max",
        "sweep_depth_to_pre_atr_240m",
        "pre_return_60m",
        "pre_down_efficiency_60m",
    ]
    return [name for name in candidates if name in frame.columns]


def large_mfe_feature_profile(
    checkpoints: pd.DataFrame,
    config: PostSweepConfig,
    *,
    event_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    horizon = max(cfg.future_horizons)
    fixed_elapsed = {1, 3, 5, 10, 15, 30, 60}
    data = checkpoints.loc[checkpoints["elapsed_bars"].isin(fixed_elapsed)].copy()
    complete_col = f"future_label_complete_{horizon}m"
    if complete_col in data.columns:
        data = data.loc[data[complete_col].fillna(False).astype(bool)].copy()
    if event_features is not None and not event_features.empty:
        static = event_features.drop_duplicates("zone_event_id").copy()
        keep = ["zone_event_id"] + [name for name in _curated_features(static) if name != "zone_event_id"]
        data = data.merge(static.loc[:, list(dict.fromkeys(keep))], on="zone_event_id", how="left", suffixes=("", "_event"), validate="many_to_one")
    features = _curated_features(data)
    rows: list[dict[str, object]] = []
    future_mfe = pd.to_numeric(data[f"future_mfe_{horizon}m"], errors="coerce")
    for threshold in cfg.large_mfe_returns:
        flagged = future_mfe >= float(threshold)
        for (period, elapsed), group_idx in data.groupby(["period", "elapsed_bars"], sort=False).groups.items():
            idx = pd.Index(group_idx)
            large = data.loc[idx[flagged.loc[idx].fillna(False).to_numpy()]]
            base = data.loc[idx[~flagged.loc[idx].fillna(False).to_numpy()]]
            if len(large) < 20 or len(base) < 20:
                continue
            for feature in features:
                large_values = pd.to_numeric(large[feature], errors="coerce").dropna()
                base_values = pd.to_numeric(base[feature], errors="coerce").dropna()
                if len(large_values) < 20 or len(base_values) < 20:
                    continue
                pooled = np.sqrt((large_values.var(ddof=1) + base_values.var(ddof=1)) / 2.0)
                rows.append({
                    "period": period,
                    "elapsed_bars": int(elapsed),
                    "large_mfe_threshold": float(threshold),
                    "horizon_minutes": horizon,
                    "feature": feature,
                    "large_count": int(len(large_values)),
                    "baseline_count": int(len(base_values)),
                    "large_median": float(large_values.median()),
                    "baseline_median": float(base_values.median()),
                    "median_difference": float(large_values.median() - base_values.median()),
                    "large_mean": float(large_values.mean()),
                    "baseline_mean": float(base_values.mean()),
                    "standardized_mean_difference": float((large_values.mean() - base_values.mean()) / pooled) if np.isfinite(pooled) and pooled > EPS else np.nan,
                })
    return pd.DataFrame(rows)


def oracle_turning_point_table(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    """Future-labelled earliest durable-reversal checkpoint per event.

    This table is descriptive only. Selecting the row uses future information and
    must never be treated as a live signal.
    """

    if checkpoints.empty:
        return pd.DataFrame()
    cfg = config.validate()
    horizon = 60 if 60 in cfg.future_horizons else max(cfg.future_horizons)
    complete = checkpoints.get(f"future_label_complete_{horizon}m", pd.Series(True, index=checkpoints.index)).fillna(False).astype(bool)
    mask = (
        complete
        & checkpoints[f"future_no_lower_low_{horizon}m"].fillna(False).astype(bool)
        & (pd.to_numeric(checkpoints[f"future_mfe_{horizon}m"], errors="coerce") >= float(cfg.reversal_mfe_return))
        & checkpoints[f"future_reversal_dominant_{horizon}m"].fillna(False).astype(bool)
    )
    candidates = checkpoints.loc[mask].sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    if candidates.empty:
        return pd.DataFrame()
    out = candidates.groupby("zone_event_id", sort=False, as_index=False).first()
    out["oracle_selection_uses_future"] = True
    out["oracle_definition"] = f"earliest no-lower-low {horizon}m + MFE>={cfg.reversal_mfe_return:g} + reversal-dominant"
    return out



def orderflow_fixed_bin_summary(checkpoints: pd.DataFrame, config: PostSweepConfig) -> pd.DataFrame:
    """Summarize broad predeclared order-flow bins without fitting cut points."""

    if checkpoints.empty:
        return pd.DataFrame(columns=[
            "dimension", "bin_value", "period", "horizon_minutes", "checkpoints", "events",
            "median_future_mfe", "median_future_mae", "median_future_close_return",
            "no_lower_low_rate", "reversal_dominant_rate", "continuation_dominant_rate",
        ])
    cfg = config.validate()
    specs: dict[str, tuple[list[float], list[str]]] = {
        "delta_ratio_5m": (
            [-np.inf, -0.30, -0.10, 0.0, 0.10, 0.30, np.inf],
            ["<=-0.30", "-0.30~-0.10", "-0.10~0", "0~0.10", "0.10~0.30", ">0.30"],
        ),
        "cum_delta_ratio_since_sweep": (
            [-np.inf, -0.30, -0.10, 0.0, 0.10, 0.30, np.inf],
            ["<=-0.30", "-0.30~-0.10", "-0.10~0", "0~0.10", "0.10~0.30", ">0.30"],
        ),
        "sell_share_5m": (
            [-np.inf, 0.45, 0.55, 0.65, 0.75, np.inf],
            ["<=0.45", "0.45~0.55", "0.55~0.65", "0.65~0.75", ">0.75"],
        ),
        "attempt_extension_vs_previous": (
            [-np.inf, 0.50, 0.80, 1.20, 2.00, np.inf],
            ["<=0.50", "0.50~0.80", "0.80~1.20", "1.20~2.00", ">2.00"],
        ),
        "bars_since_new_low_attempt": (
            [-np.inf, 1.5, 3.5, 5.5, 10.5, 15.5, np.inf],
            ["0-1", "2-3", "4-5", "6-10", "11-15", "16+"],
        ),
    }
    rows: list[dict[str, object]] = []
    for dimension, (edges, labels) in specs.items():
        if dimension not in checkpoints.columns:
            continue
        value = pd.to_numeric(checkpoints[dimension], errors="coerce")
        bins = pd.cut(value, edges, labels=labels, include_lowest=True, right=True)
        source = checkpoints.assign(_bin=bins).dropna(subset=["_bin"])
        for (period, bin_value), group in source.groupby(["period", "_bin"], observed=True, sort=False):
            for horizon in cfg.future_horizons:
                row: dict[str, object] = {
                    "dimension": dimension,
                    "bin_value": str(bin_value),
                    "period": period,
                    "horizon_minutes": int(horizon),
                }
                row.update(_group_metrics(group, int(horizon)))
                rows.append(row)
    return pd.DataFrame(rows)

def period_stability_summary(confirmation_summary: pd.DataFrame) -> pd.DataFrame:
    if confirmation_summary.empty:
        return pd.DataFrame()
    source = confirmation_summary.loc[confirmation_summary["horizon_minutes"].isin([15, 60, 180])].copy()
    rows: list[dict[str, object]] = []
    for (state, horizon), group in source.groupby(["confirmation_state", "horizon_minutes"], sort=False):
        if group["period"].nunique() < 2:
            continue
        returns = pd.to_numeric(group["median_future_close_return"], errors="coerce")
        reversal = pd.to_numeric(group["reversal_dominant_rate"], errors="coerce")
        rows.append({
            "confirmation_state": state,
            "horizon_minutes": int(horizon),
            "periods": int(group["period"].nunique()),
            "minimum_events_per_period": int(pd.to_numeric(group["events"], errors="coerce").min()),
            "all_period_median_return_positive": bool((returns > 0).all()),
            "median_return_min": float(returns.min()),
            "median_return_max": float(returns.max()),
            "reversal_rate_min": float(reversal.min()),
            "reversal_rate_max": float(reversal.max()),
        })
    return pd.DataFrame(rows)
