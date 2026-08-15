#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1 report tables, residualization gates and cumulative review pack."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, HurdleResidualizationConfig
from .modeling import ModelBundle


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def source_quality(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame["first_touch_observed"].astype(bool)
    complete = frame["first_touch_label_complete"].astype(bool)
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    touch = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    available = pd.to_datetime(frame["first_touch_available_time"], errors="coerce")
    return pd.DataFrame([
        {"metric": "rows", "value": int(len(frame))},
        {"metric": "exact_first_touch_rows", "value": int(observed.sum())},
        {"metric": "complete_first_touch_rows", "value": int(complete.sum())},
        {"metric": "touch_bar_timestamp_equals_decision", "value": int((observed & touch.eq(decision)).sum())},
        {"metric": "touch_available_not_after_decision", "value": int((observed & available.le(decision)).sum())},
        {"metric": "upstream_r02_touch_mismatch_quarantined", "value": int((complete & ~frame["r02_touch_consistent"].astype(bool)).sum())},
        {"metric": "split_boundary_purged_rows", "value": int((~frame["split_purge_eligible"].astype(bool)).sum())},
        {"metric": "r02_3_1_source_eligible_rows", "value": int(frame["r02_3_1_source_eligible"].astype(bool).sum())},
        {"metric": "train_expanding_oos_rows", "value": int(frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS").sum())},
        {"metric": "future_train_frozen_rows", "value": int(frame["nuisance_prediction_source"].astype(str).eq("TRAIN_FULL_FROZEN").sum())},
    ])


def residual_stability(frame: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sf = frame.loc[frame["r02_3_1_source_eligible"].astype(bool)].copy()
    for (period, side), group in sf.groupby(["period", "zone_side"], sort=True):
        d = pd.to_numeric(group["zone_distance_bp"], errors="coerce").to_numpy(dtype=float)
        raw = pd.to_numeric(group["raw_release_density"], errors="coerce").to_numpy(dtype=float)
        ex = pd.to_numeric(group["excess_liquidity_residual"], errors="coerce").to_numpy(dtype=float)
        raw_rev = pd.to_numeric(group["raw_reversal_quality"], errors="coerce").to_numpy(dtype=float)
        rev = pd.to_numeric(group["reversal_quality_residual"], errors="coerce").to_numpy(dtype=float)

        def rho(x: np.ndarray, y: np.ndarray) -> float:
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < 2 or np.nanstd(x[valid]) <= 1e-12 or np.nanstd(y[valid]) <= 1e-12:
                return np.nan
            value = spearmanr(x[valid], y[valid]).statistic
            return float(value) if np.isfinite(value) else np.nan

        raw_rho = rho(d, raw)
        ex_rho = rho(d, ex)
        release = group.loc[group["release_observed_180s"].astype(bool)].copy()
        dr = pd.to_numeric(release["zone_distance_bp"], errors="coerce").to_numpy(dtype=float)
        rr = pd.to_numeric(release["raw_reversal_quality"], errors="coerce").to_numpy(dtype=float)
        rres = pd.to_numeric(release["reversal_quality_residual"], errors="coerce").to_numpy(dtype=float)
        raw_rev_rho = rho(dr, rr)
        rev_rho = rho(dr, rres)
        rows.append({
            "period": period,
            "zone_side": side,
            "rows": int(len(group)),
            "release_rows": int(len(release)),
            "mean_actual_density": float(pd.to_numeric(group["raw_release_density"], errors="coerce").mean()),
            "mean_nuisance_expected_density": float(pd.to_numeric(group["nuisance_expected_density"], errors="coerce").mean()),
            "mean_actual_to_expected_ratio": float(pd.to_numeric(group["density_vs_nuisance_expected_ratio"], errors="coerce").mean()),
            "mean_excess_liquidity_residual": float(pd.to_numeric(group["excess_liquidity_residual"], errors="coerce").mean()),
            "distance_vs_raw_density_spearman": raw_rho,
            "distance_vs_excess_residual_spearman": ex_rho,
            "absolute_distance_bias_reduction": abs(raw_rho) - abs(ex_rho) if np.isfinite(raw_rho) and np.isfinite(ex_rho) else np.nan,
            "distance_vs_raw_reversal_spearman": raw_rev_rho,
            "distance_vs_reversal_residual_spearman": rev_rho,
            "absolute_reversal_distance_bias_reduction": abs(raw_rev_rho) - abs(rev_rho) if np.isfinite(raw_rev_rho) and np.isfinite(rev_rho) else np.nan,
        })
    return pd.DataFrame(rows)


def swing_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side, task), sf in metrics.groupby(["period", "zone_side", "task"], sort=True):
        path = sf.loc[sf["model"].eq("PATH_NO_SWING")]
        full = sf.loc[sf["model"].eq("FULL_WITH_SWING")]
        if path.empty or full.empty:
            continue
        p, f = path.iloc[0], full.iloc[0]
        rows.append({
            "period": period, "zone_side": side, "task": task,
            "path_mean_group_spearman": p.get("mean_group_spearman", np.nan),
            "full_mean_group_spearman": f.get("mean_group_spearman", np.nan),
            "swing_spearman_uplift": float(f.get("mean_group_spearman", np.nan) - p.get("mean_group_spearman", np.nan)),
            "path_ndcg3": p.get("mean_ndcg3", np.nan), "full_ndcg3": f.get("mean_ndcg3", np.nan),
            "swing_ndcg3_uplift": float(f.get("mean_ndcg3", np.nan) - p.get("mean_ndcg3", np.nan)),
        })
    return pd.DataFrame(rows)


def causal_audit(
    frame: pd.DataFrame,
    models: ModelBundle,
    source_gate: pd.DataFrame,
    feature_audit: pd.DataFrame,
    fold_audit: pd.DataFrame,
    config: HurdleResidualizationConfig,
) -> pd.DataFrame:
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    touch = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    touch_available = pd.to_datetime(frame["first_touch_available_time"], errors="coerce")
    eligible = frame["r02_3_1_source_eligible"].astype(bool)
    group_sizes = frame.groupby(["decision_time", "zone_side"], sort=False).size()
    source_fail = int(source_gate["status"].astype(str).eq("FAIL").sum()) if not source_gate.empty else 0
    path_cols = sorted({c for b in models.by_side.values() for c in b.path_columns})
    leaked_prefixes = (
        "touch_", "release_", "favorable_", "continuation_", "time_to_", "sweep_", "reversal_",
        "p_touch", "p_release", "p_favorable", "pred_", "pool_score", "high_strength", "ft_", "first_touch",
        "ranking_", "expected_", "excess_", "density_vs_", "nuisance_", "raw_", "r02_3_1_",
    )
    leaked_quality = {"r02_touch_consistent", "r02_3_source_eligible", "split_purge_eligible"}
    leaked = [c for c in path_cols if c.startswith(leaked_prefixes) or c in leaked_quality]
    swing = [c for c in path_cols if c.startswith("swing_")]
    nuisance_activity = [c for c in path_cols if c.startswith((
        "macro_notional_intensity_", "macro_trades_intensity_", "macro_realized_vol_", "macro_range_bp_",
        "micro_path_notional_intensity_", "micro_path_trades_intensity_", "micro_path_realized_vol_", "micro_path_range_bp_",
    ))]
    feature_fail = int(feature_audit["status"].astype(str).eq("FAIL").sum()) if not feature_audit.empty else 1
    fold_fail = int((~fold_audit["causal_fit_before_prediction"].astype(bool)).sum()) if not fold_audit.empty else 1
    train_bad_source = int((
        frame["period"].astype(str).eq(config.train_period)
        & eligible
        & ~frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS")
    ).sum())
    future_bad_source = int((
        frame["period"].astype(str).isin([config.calibration_period, config.holdout_period])
        & eligible
        & ~frame["nuisance_prediction_source"].astype(str).eq("TRAIN_FULL_FROZEN")
    ).sum())
    mismatch_used = int((eligible & ~frame["r02_touch_consistent"].astype(bool)).sum())
    split_overlap_used = int((eligible & ~frame["split_purge_eligible"].astype(bool)).sum())
    exp = pd.to_numeric(frame.loc[eligible, "nuisance_expected_density"], errors="coerce")
    positive_expected = float(exp.gt(0).mean()) if len(exp) else 0.0
    rows = [
        {"check": "r01_1_r02_source_gate", "value": source_fail, "status": "PASS" if source_fail == 0 else "FAIL"},
        {"check": "complete_lattice_25_zones_per_side", "value": int(group_sizes.ne(config.expected_zone_count).sum()), "status": "PASS" if len(group_sizes) and not group_sizes.ne(config.expected_zone_count).any() else "FAIL"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "first_touch_available_strictly_after_decision", "value": int((eligible & touch_available.le(decision)).sum()), "status": "PASS" if not (eligible & touch_available.le(decision)).any() else "FAIL"},
        {"check": "first_touch_inside_exclusive_12h_horizon", "value": int((eligible & touch.ge(decision + pd.Timedelta(hours=12))).sum()), "status": "PASS" if not (eligible & touch.ge(decision + pd.Timedelta(hours=12))).any() else "FAIL"},
        {"check": "upstream_r02_touch_mismatch_rows_used", "value": mismatch_used, "status": "PASS" if mismatch_used == 0 else "FAIL"},
        {"check": "period_boundary_overlap_rows_used", "value": split_overlap_used, "status": "PASS" if split_overlap_used == 0 else "FAIL"},
        {"check": "nuisance_features_group_level_or_distance_only", "value": feature_fail, "status": "PASS" if feature_fail == 0 else "FAIL"},
        {"check": "train_nuisance_predictions_expanding_oos_only", "value": train_bad_source, "status": "PASS" if train_bad_source == 0 else "FAIL"},
        {"check": "future_nuisance_predictions_full_train_frozen_only", "value": future_bad_source, "status": "PASS" if future_bad_source == 0 else "FAIL"},
        {"check": "nuisance_expanding_fit_strictly_before_prediction", "value": fold_fail, "status": "PASS" if fold_fail == 0 else "FAIL"},
        {"check": "nuisance_expected_density_not_degenerate_zero", "value": positive_expected, "status": "PASS" if positive_expected > 0.95 else "FAIL"},
        {"check": "primary_ranker_has_no_future_or_target_columns", "value": len(leaked), "status": "PASS" if not leaked else "FAIL"},
        {"check": "primary_ranker_excludes_swing", "value": len(swing), "status": "PASS" if not swing else "FAIL"},
        {"check": "primary_ranker_excludes_raw_distance", "value": int("zone_distance_bp" in path_cols), "status": "PASS" if "zone_distance_bp" not in path_cols else "FAIL"},
        {"check": "primary_ranker_excludes_nuisance_activity_features", "value": len(nuisance_activity), "status": "PASS" if not nuisance_activity else "FAIL"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ]
    return pd.DataFrame(rows)


def _cell(frame: pd.DataFrame, period: str, side: str) -> pd.Series | None:
    x = frame.loc[frame["period"].astype(str).eq(period) & frame["zone_side"].astype(str).eq(side)]
    return None if x.empty else x.iloc[0]


def _metric(metrics: pd.DataFrame, period: str, side: str, task: str, model: str) -> float:
    x = metrics.loc[
        metrics["period"].astype(str).eq(period)
        & metrics["zone_side"].astype(str).eq(side)
        & metrics["task"].astype(str).eq(task)
        & metrics["model"].astype(str).eq(model),
        "mean_group_spearman",
    ]
    return float(x.iloc[0]) if len(x) else np.nan


def _reg(regression: pd.DataFrame, period: str, side: str, task: str) -> float:
    x = regression.loc[
        regression["period"].astype(str).eq(period)
        & regression["zone_side"].astype(str).eq(side)
        & regression["task"].astype(str).eq(task), "spearman"
    ]
    return float(x.iloc[0]) if len(x) else np.nan


def _top(top: pd.DataFrame, period: str, side: str, task: str, model: str, col: str) -> float:
    x = top.loc[
        top["period"].astype(str).eq(period)
        & top["zone_side"].astype(str).eq(side)
        & top["task"].astype(str).eq(task)
        & top["model"].astype(str).eq(model), col
    ]
    return float(x.iloc[0]) if len(x) else np.nan


def _decision(
    metrics: pd.DataFrame,
    regression: pd.DataFrame,
    top: pd.DataFrame,
    stability: pd.DataFrame,
    causal: pd.DataFrame,
    config: HurdleResidualizationConfig,
) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].astype(str).eq("FAIL").any():
        return "BLOCKED_R02_3_1_QUALITY_OR_CAUSAL_FAILURE", ["A source, nuisance or causal gate failed. No signal interpretation or promotion is allowed."]

    reasons: list[str] = []
    residualization_ok = True
    for side in ("DOWN", "UP"):
        for period in (config.calibration_period, config.holdout_period):
            row = _cell(stability, period, side)
            if row is None:
                residualization_ok = False
                reasons.append(f"{side} {period}: residual-stability cell missing.")
                continue
            raw_ex_corr = abs(float(row.get("distance_vs_raw_density_spearman", np.nan)))
            ex_corr = abs(float(row.get("distance_vs_excess_residual_spearman", np.nan)))
            raw_rev_corr = abs(float(row.get("distance_vs_raw_reversal_spearman", np.nan)))
            rev_corr = abs(float(row.get("distance_vs_reversal_residual_spearman", np.nan)))
            reasons.append(
                f"{side} {period}: |distance raw/excess residual|={raw_ex_corr:.3f}/{ex_corr:.3f}; "
                f"|distance raw/reversal residual|={raw_rev_corr:.3f}/{rev_corr:.3f}."
            )
            if not np.isfinite(ex_corr) or ex_corr > config.residualization_max_abs_distance_spearman:
                residualization_ok = False
            if np.isfinite(raw_ex_corr) and raw_ex_corr >= 0.05 and ex_corr >= raw_ex_corr:
                residualization_ok = False
            if not np.isfinite(rev_corr) or rev_corr > config.reversal_residualization_max_abs_distance_spearman:
                residualization_ok = False
            if np.isfinite(raw_rev_corr) and raw_rev_corr >= 0.05 and rev_corr >= raw_rev_corr:
                residualization_ok = False
    if not residualization_ok:
        return "BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED", reasons + [
            "The hurdle nuisance model did not remove distance/activity contamination sufficiently out of sample. Do not add Range/Footprint/OI until this quality gate is fixed."
        ]

    promote: list[str] = []
    weak_stable = False
    for side in ("DOWN", "UP"):
        ex_val = _metric(metrics, config.calibration_period, side, "EXCESS_RESIDUAL", "PATH_NO_SWING")
        ex_hold = _metric(metrics, config.holdout_period, side, "EXCESS_RESIDUAL", "PATH_NO_SWING")
        rv_val = _metric(metrics, config.calibration_period, side, "REVERSAL_RESIDUAL", "PATH_NO_SWING")
        rv_hold = _metric(metrics, config.holdout_period, side, "REVERSAL_RESIDUAL", "PATH_NO_SWING")
        base_values = np.asarray([
            _metric(metrics, config.holdout_period, side, "EXCESS_RESIDUAL", "NUISANCE_EXPECTED"),
            _metric(metrics, config.holdout_period, side, "EXCESS_RESIDUAL", "DISTANCE_NEAR"),
            _metric(metrics, config.holdout_period, side, "EXCESS_RESIDUAL", "DISTANCE_FAR"),
        ], dtype=float)
        base_hold = float(np.nanmax(base_values)) if np.isfinite(base_values).any() else np.nan
        ratio = _top(top, config.holdout_period, side, "EXCESS_RESIDUAL", "PATH_NO_SWING", "top1_median_actual_to_nuisance_expected_density_ratio")
        oracle = _top(top, config.holdout_period, side, "EXCESS_RESIDUAL", "PATH_NO_SWING", "oracle_strongest_zone_in_top3_rate")
        touched = _top(top, config.holdout_period, side, "EXCESS_RESIDUAL", "PATH_NO_SWING", "top1_touched")
        sweep = _reg(regression, config.holdout_period, side, "SWEEP_DEPTH")
        reasons.append(
            f"{side}: excess val/hold={ex_val:.3f}/{ex_hold:.3f}, hold best nuisance/distance baseline={base_hold:.3f}; "
            f"reversal val/hold={rv_val:.3f}/{rv_hold:.3f}; Top-1 actual/expected={ratio:.3f}, oracle-top3={oracle:.3f}, "
            f"sweep-depth={sweep:.3f}, touched={int(touched) if np.isfinite(touched) else 0}."
        )
        stable_ex = np.isfinite(ex_val) and np.isfinite(ex_hold) and ex_val >= config.continue_min_stable_spearman and ex_hold >= config.continue_min_stable_spearman
        stable_rv = np.isfinite(rv_val) and np.isfinite(rv_hold) and rv_val >= config.continue_min_stable_spearman and rv_hold >= config.continue_min_stable_spearman
        weak_stable = weak_stable or stable_ex or stable_rv
        passed = bool(
            np.isfinite(ex_val) and ex_val >= config.promotion_min_excess_spearman
            and np.isfinite(ex_hold) and ex_hold >= config.promotion_min_excess_spearman
            and ex_hold > base_hold
            and np.isfinite(rv_val) and rv_val >= config.promotion_min_reversal_spearman
            and np.isfinite(rv_hold) and rv_hold >= config.promotion_min_reversal_spearman
            and np.isfinite(ratio) and ratio >= config.promotion_min_top1_actual_expected_ratio
            and np.isfinite(oracle) and oracle >= config.promotion_min_oracle_top3_rate
            and np.isfinite(sweep) and sweep >= config.promotion_min_sweep_depth_spearman
            and np.isfinite(touched) and touched >= config.minimum_top1_touched
        )
        if passed:
            promote.append(side)
    if promote:
        return (
            f"PROMOTE_{'_AND_'.join(promote)}_TO_R02_4_CAUSAL_PASSIVE_LIMIT_SWEEP_GEOMETRY_STUDY",
            reasons + ["Promotion remains research-only; no live approval and no retrospectively known stop/extreme may be used."],
        )
    if weak_stable:
        return "CONTINUE_R02_3_1_WITH_RANGE_FOOTPRINT_OI_INCREMENT", reasons + [
            "Residualized No-Swing signal is weak but cross-period stable. Only now may independent Range/Footprint/OI feature-family increments be tested on these frozen residual targets."
        ]
    return "STOP_R02_3_1_RESIDUAL_SPATIAL_EDGE_WEAK", reasons + [
        "After removing distance/activity nuisance, neither residual pool strength nor residual reversal quality is stably rankable. Do not rescue with Swing or parameter grids."
    ]


def write_reports(
    *,
    config: HurdleResidualizationConfig,
    source_gate: pd.DataFrame,
    frame: pd.DataFrame,
    nuisance_feature_audit: pd.DataFrame,
    nuisance_fold_audit: pd.DataFrame,
    nuisance_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    regression: pd.DataFrame,
    top: pd.DataFrame,
    importance: pd.DataFrame,
    causal: pd.DataFrame,
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    source_q = source_quality(frame)
    stability = residual_stability(frame, config)
    ablation = swing_ablation(metrics)
    family = importance.groupby(["zone_side", "task", "model", "feature_family"], sort=True)["importance_share"].sum().reset_index()
    eligible = frame["r02_3_1_source_eligible"].astype(bool)
    summary = pd.DataFrame([{
        "rows": int(len(frame)),
        "groups": int(frame.groupby(["decision_time", "zone_side"], sort=False).ngroups),
        "source_eligible_rows": int(eligible.sum()),
        "train_expanding_oos_eligible_rows": int((eligible & frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS")).sum()),
        "excess_rankable_groups": int(frame.loc[frame["excess_residual_group_eligible"].astype(bool), "ranking_group"].nunique()),
        "reversal_rankable_groups": int(frame.loc[frame["reversal_residual_group_eligible"].astype(bool), "ranking_group"].nunique()),
        "primary_target": "log1p(actual first-touch release density) - log1p(hurdle nuisance expected density)",
        "hurdle_expected_density": "P(release>0 | distance+group activity) * smearing-adjusted E[density | release>0,distance+group activity]",
        "reversal_target": "raw reversal quality - nuisance expected reversal quality",
        "nuisance_train_prediction": "expanding past-only OOS",
        "future_nuisance_prediction": "full 2023-2024 TRAIN frozen",
        "swing_in_primary_model": False,
        "raw_distance_in_primary_model": False,
        "nuisance_activity_in_primary_model": False,
    }])
    sample_cols = [c for c in (
        "zone_id", "decision_time", "period", "zone_side", "zone_distance_bp", "first_touch_time", "first_touch_available_time",
        "r02_touch_consistent", "r02_3_1_source_eligible", "nuisance_prediction_source", "nuisance_p_release",
        "nuisance_pred_log_density_if_release", "nuisance_pred_density_if_release", "nuisance_expected_density", "raw_release_density",
        "density_vs_nuisance_expected_ratio", "excess_liquidity_residual", "raw_reversal_quality",
        "nuisance_expected_reversal_quality", "reversal_quality_residual", "sweep_depth_target_bp", "reversal_room_target_bp",
        "score_excess_path_no_swing", "score_reversal_path_no_swing", "score_joint_residual_path_no_swing",
        "pred_sweep_depth_bp", "pred_reversal_room_bp",
    ) if c in frame.columns]
    sample = frame.sort_values(["period", "zone_side", "score_joint_residual_path_no_swing"], ascending=[True, True, False], kind="mergesort").loc[:, sample_cols].head(50_000)
    tables = {
        "01_source_gate.csv": source_gate,
        "02_source_quality_and_quarantine.csv": source_q,
        "03_nuisance_feature_audit.csv": nuisance_feature_audit,
        "04_nuisance_expanding_fold_audit.csv": nuisance_fold_audit,
        "05_nuisance_model_metrics.csv": nuisance_metrics,
        "06_dataset_summary.csv": summary,
        "07_residualization_stability.csv": stability,
        "08_ranking_metrics.csv": metrics,
        "09_sweep_geometry_metrics.csv": regression,
        "10_top_zone_summary.csv": top,
        "11_swing_ablation.csv": ablation,
        "12_feature_importance.csv": importance,
        "13_feature_family_importance.csv": family,
        "14_causal_audit.csv": causal,
        "15_top_zone_sample.csv": sample,
    }
    for name, table in tables.items():
        _write(table, root / name)
    decision, reasons = _decision(metrics, regression, top, stability, causal, config)
    lines = [
        f"# {MODEL_NAME} {STAGE_ID} decision", "", "## Primary decision", "", f"`{decision}`", "", "## Evidence", "",
        *[f"- {r}" for r in reasons], "", "## Frozen interpretation", "",
        "- R02.3 median/IQR normalization is retired because zero-inflated first-touch release density made TRAIN medians and many IQRs exactly zero.",
        "- R02.3.1 uses a two-part hurdle nuisance model: release probability plus positive-density magnitude. Distance, calendar/session and broad group-level activity/volatility may enter nuisance only.",
        "- TRAIN residual labels use expanding past-only out-of-sample nuisance predictions. Validation/Holdout use one nuisance family frozen on all 2023-2024 TRAIN.",
        "- Primary residual rankers exclude raw distance, nuisance activity features and Swing. Swing remains an identical-target ablation only.",
        "- Reversal Quality is also residualized against the same mechanical nuisance family so a simple far-distance reversal effect cannot masquerade as path edge.",
        "- Sweep Depth and Reversal Room remain separate geometry regressions and do not define the residual target.",
        "- R01.3 post-confirmation market entry remains stopped. This stage does not place passive orders and does not claim sealed validation or live approval.",
        "- 2025Q4-2026H1 remains a development holdout because prior stages have inspected it.",
    ]
    (root / "16_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "sealed_validation_claim": False,
        "live_approved": False,
        "config": config.to_dict(),
        "reports": list(tables) + ["16_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R02.3.1 as a zero-inflated hurdle nuisance-residualization study. First verify that expected density is non-degenerate and that distance correlation of Excess Residual and Reversal Residual is materially reduced in Validation/Holdout. Then evaluate PATH_NO_SWING cross-sectional ranking versus nuisance/distance baselines, Swing ablation, and retained sweep geometry. TRAIN nuisance predictions must be expanding past-only OOS; future periods must use full-TRAIN frozen nuisance models. Do not reinterpret this as a Swing strategy, raw-density model, limit-order backtest, sealed validation, or live approval.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(
            report_dir=root,
            experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02_3_1",
            edge_id="RESEARCH_ONLY_HURDLE_NUISANCE_RESIDUAL_LIQUIDITY_RANKING",
            title=f"{MODEL_NAME} {STAGE_ID}",
            decision_focus="zero-inflated nuisance residualization, residual liquidity/reversal ranking, Swing ablation, sweep geometry and causal audit",
        ))
    return root, decision
