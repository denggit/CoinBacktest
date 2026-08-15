#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Residual-distance, distance-cell and causal audits for R02.3.1b."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import TargetConsistencyConfig


def _rho(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2 or np.nanstd(a[valid]) <= 1e-12 or np.nanstd(b[valid]) <= 1e-12:
        return np.nan
    value = spearmanr(a[valid], b[valid]).statistic
    return float(value) if np.isfinite(value) else np.nan


def target_consistency_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = frame["r02_3_1b_source_eligible"].astype(bool)
    for (period, side), group in frame.loc[eligible].groupby(["period", "zone_side"], sort=True):
        rows.append({
            "period": period,
            "zone_side": side,
            "rows": int(len(group)),
            "release_rate": float(group["release_observed_180s"].mean()),
            "distance_vs_raw_log_density_spearman": _rho(group["zone_distance_bp"], group["raw_log_release_density"]),
            "distance_vs_legacy_residual_spearman": _rho(group["zone_distance_bp"], group["legacy_excess_residual"]),
            "distance_vs_formula_only_residual_spearman": _rho(group["zone_distance_bp"], group["formula_only_excess_residual"]),
            "distance_vs_target_consistent_residual_spearman": _rho(group["zone_distance_bp"], group["target_consistent_excess_residual"]),
            "mean_legacy_residual": float(pd.to_numeric(group["legacy_excess_residual"], errors="coerce").mean()),
            "mean_formula_only_residual": float(pd.to_numeric(group["formula_only_excess_residual"], errors="coerce").mean()),
            "mean_target_consistent_residual": float(pd.to_numeric(group["target_consistent_excess_residual"], errors="coerce").mean()),
            "mean_transform_gap": float(pd.to_numeric(group["transform_gap_formula_only_minus_legacy"], errors="coerce").mean()),
            "mean_objective_gap": float(pd.to_numeric(group["objective_gap_mean_minus_huber"], errors="coerce").mean()),
            "mean_total_expected_log_correction": float(pd.to_numeric(group["total_expected_log_correction"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def distance_cell_audit(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame["r02_3_1b_source_eligible"].astype(bool)
    work = frame.loc[eligible].copy()
    work["zone_distance_bp"] = pd.to_numeric(work["zone_distance_bp"], errors="coerce")
    rows: list[dict[str, object]] = []
    for (period, side, distance), group in work.groupby(["period", "zone_side", "zone_distance_bp"], sort=True):
        rows.append({
            "period": period,
            "zone_side": side,
            "zone_distance_bp": float(distance),
            "rows": int(len(group)),
            "release_rate": float(group["release_observed_180s"].mean()),
            "actual_mean_log_density": float(pd.to_numeric(group["raw_log_release_density"], errors="coerce").mean()),
            "legacy_expected_log_mean": float(pd.to_numeric(group["legacy_expected_log_proxy"], errors="coerce").mean()),
            "formula_only_expected_log_mean": float(pd.to_numeric(group["formula_only_expected_log_density"], errors="coerce").mean()),
            "target_consistent_expected_log_mean": float(pd.to_numeric(group["mean_aligned_expected_log_density"], errors="coerce").mean()),
            "legacy_residual_mean": float(pd.to_numeric(group["legacy_excess_residual"], errors="coerce").mean()),
            "formula_only_residual_mean": float(pd.to_numeric(group["formula_only_excess_residual"], errors="coerce").mean()),
            "target_consistent_residual_mean": float(pd.to_numeric(group["target_consistent_excess_residual"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def yearly_stability(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["r02_3_1b_source_eligible"].astype(bool)].copy()
    work["year"] = pd.to_datetime(work["decision_time"], errors="coerce").dt.year
    rows: list[dict[str, object]] = []
    for (year, side), group in work.groupby(["year", "zone_side"], sort=True):
        rows.append({
            "year": int(year),
            "zone_side": side,
            "rows": int(len(group)),
            "release_rate": float(group["release_observed_180s"].mean()),
            "mean_actual_log_density": float(pd.to_numeric(group["raw_log_release_density"], errors="coerce").mean()),
            "mean_expected_log_density": float(pd.to_numeric(group["mean_aligned_expected_log_density"], errors="coerce").mean()),
            "mean_target_consistent_residual": float(pd.to_numeric(group["target_consistent_excess_residual"], errors="coerce").mean()),
            "distance_vs_raw_log_density_spearman": _rho(group["zone_distance_bp"], group["raw_log_release_density"]),
            "distance_vs_target_consistent_residual_spearman": _rho(group["zone_distance_bp"], group["target_consistent_excess_residual"]),
        })
    return pd.DataFrame(rows)


def transform_gap_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = frame.loc[frame["r02_3_1b_source_eligible"].astype(bool)].copy()
    for (period, side), group in work.groupby(["period", "zone_side"], sort=True):
        legacy = pd.to_numeric(group["legacy_expected_log_proxy"], errors="coerce")
        formula = pd.to_numeric(group["formula_only_expected_log_density"], errors="coerce")
        mean = pd.to_numeric(group["mean_aligned_expected_log_density"], errors="coerce")
        rows.append({
            "period": period,
            "zone_side": side,
            "rows": int(len(group)),
            "mean_legacy_expected_log_proxy": float(legacy.mean()),
            "mean_formula_only_expected_log": float(formula.mean()),
            "mean_mean_aligned_expected_log": float(mean.mean()),
            "mean_abs_formula_vs_legacy_gap": float((formula - legacy).abs().mean()),
            "mean_abs_mean_vs_formula_gap": float((mean - formula).abs().mean()),
            "mean_abs_total_correction": float((mean - legacy).abs().mean()),
            "distance_vs_transform_gap_spearman": _rho(group["zone_distance_bp"], group["transform_gap_formula_only_minus_legacy"]),
            "distance_vs_objective_gap_spearman": _rho(group["zone_distance_bp"], group["objective_gap_mean_minus_huber"]),
        })
    return pd.DataFrame(rows)


def causal_audit(
    frame: pd.DataFrame,
    source_gate: pd.DataFrame,
    feature_audit: pd.DataFrame,
    fold_audit: pd.DataFrame,
    config: TargetConsistencyConfig,
) -> pd.DataFrame:
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    touch = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    touch_available = pd.to_datetime(frame["first_touch_available_time"], errors="coerce")
    eligible = frame["r02_3_1b_source_eligible"].astype(bool)
    group_sizes = frame.groupby(["decision_time", "zone_side"], sort=False).size()
    source_fail = int(source_gate["status"].astype(str).eq("FAIL").sum()) if not source_gate.empty else 0
    feature_fail = int(feature_audit["status"].astype(str).eq("FAIL").sum()) if not feature_audit.empty else 1
    fold_fail = int((~fold_audit["causal_fit_before_prediction"].astype(bool)).sum()) if not fold_audit.empty else 1
    train_bad = int((
        frame["period"].astype(str).eq(config.train_period)
        & eligible
        & ~frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS")
    ).sum())
    future_bad = int((
        frame["period"].astype(str).isin([config.calibration_period, config.holdout_period])
        & eligible
        & ~frame["nuisance_prediction_source"].astype(str).eq("TRAIN_FULL_FROZEN")
    ).sum())
    mismatch_used = int((eligible & ~frame["r02_touch_consistent"].astype(bool)).sum())
    split_overlap = int((eligible & ~frame["split_purge_eligible"].astype(bool)).sum())
    identity = (
        pd.to_numeric(frame.loc[eligible, "mean_aligned_expected_log_density"], errors="coerce")
        - pd.to_numeric(frame.loc[eligible, "nuisance_p_release"], errors="coerce")
        * pd.to_numeric(frame.loc[eligible, "nuisance_mean_log_density_if_release"], errors="coerce")
    ).abs()
    identity_max = float(identity.max()) if len(identity) else np.nan
    nuisance_cols = feature_audit["feature"].astype(str).tolist() if not feature_audit.empty else []
    swing_nuisance = [c for c in nuisance_cols if c.startswith("swing_")]
    path_nuisance = [c for c in nuisance_cols if c.startswith(("zone_boundary_", "zone_buildup_", "zone_untouched_"))]
    objectives = set(fold_audit.get("positive_log_primary_objective", pd.Series(dtype=str)).astype(str))

    rows = [
        {"check": "r01_1_r02_source_gate", "value": source_fail, "status": "PASS" if source_fail == 0 else "FAIL"},
        {"check": "complete_lattice_25_zones_per_side", "value": int(group_sizes.ne(config.expected_zone_count).sum()), "status": "PASS" if len(group_sizes) and not group_sizes.ne(config.expected_zone_count).any() else "FAIL"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "first_touch_available_strictly_after_decision", "value": int((eligible & touch_available.le(decision)).sum()), "status": "PASS" if not (eligible & touch_available.le(decision)).any() else "FAIL"},
        {"check": "first_touch_inside_exclusive_12h_horizon", "value": int((eligible & touch.ge(decision + pd.Timedelta(hours=12))).sum()), "status": "PASS" if not (eligible & touch.ge(decision + pd.Timedelta(hours=12))).any() else "FAIL"},
        {"check": "upstream_r02_touch_mismatch_rows_used", "value": mismatch_used, "status": "PASS" if mismatch_used == 0 else "FAIL"},
        {"check": "period_boundary_overlap_rows_used", "value": split_overlap, "status": "PASS" if split_overlap == 0 else "FAIL"},
        {"check": "nuisance_features_group_level_or_distance_only", "value": feature_fail, "status": "PASS" if feature_fail == 0 else "FAIL"},
        {"check": "nuisance_excludes_swing", "value": len(swing_nuisance), "status": "PASS" if not swing_nuisance else "FAIL"},
        {"check": "nuisance_excludes_zone_specific_path", "value": len(path_nuisance), "status": "PASS" if not path_nuisance else "FAIL"},
        {"check": "train_nuisance_predictions_expanding_oos_only", "value": train_bad, "status": "PASS" if train_bad == 0 else "FAIL"},
        {"check": "future_nuisance_predictions_full_train_frozen_only", "value": future_bad, "status": "PASS" if future_bad == 0 else "FAIL"},
        {"check": "nuisance_expanding_fit_strictly_before_prediction", "value": fold_fail, "status": "PASS" if fold_fail == 0 else "FAIL"},
        {"check": "positive_log_primary_objective_is_l2_mean", "value": ",".join(sorted(objectives)), "status": "PASS" if objectives == {"regression_l2"} else "FAIL"},
        {"check": "expected_log_hurdle_identity_max_abs_error", "value": identity_max, "status": "PASS" if np.isfinite(identity_max) and identity_max <= 1e-12 else "FAIL"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ]
    return pd.DataFrame(rows)
