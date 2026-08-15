#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer and hard target-consistency gate for R02.3.1b."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .audit import distance_cell_audit, target_consistency_stability, transform_gap_audit, yearly_stability
from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, TargetConsistencyConfig


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
        {"metric": "r02_3_1b_source_eligible_rows", "value": int(frame["r02_3_1b_source_eligible"].astype(bool).sum())},
        {"metric": "train_expanding_oos_rows", "value": int(frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS").sum())},
        {"metric": "future_train_frozen_rows", "value": int(frame["nuisance_prediction_source"].astype(str).eq("TRAIN_FULL_FROZEN").sum())},
    ])


def _cell(stability: pd.DataFrame, period: str, side: str) -> pd.Series | None:
    x = stability.loc[
        stability["period"].astype(str).eq(period)
        & stability["zone_side"].astype(str).eq(side)
    ]
    return None if x.empty else x.iloc[0]


def _metric(metrics: pd.DataFrame, period: str, side: str, task: str, col: str) -> float:
    x = metrics.loc[
        metrics["period"].astype(str).eq(period)
        & metrics["zone_side"].astype(str).eq(side)
        & metrics["task"].astype(str).eq(task),
        col,
    ]
    return float(x.iloc[0]) if len(x) else np.nan


def _decision(
    stability: pd.DataFrame,
    nuisance_metrics: pd.DataFrame,
    causal: pd.DataFrame,
    config: TargetConsistencyConfig,
) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].astype(str).eq("FAIL").any():
        return "BLOCKED_R02_3_1B_QUALITY_OR_CAUSAL_FAILURE", [
            "A source, target-identity or causal gate failed. Do not interpret residual quality."
        ]

    reasons: list[str] = []
    residual_ok = True
    for side in ("DOWN", "UP"):
        for period in (config.calibration_period, config.holdout_period):
            row = _cell(stability, period, side)
            if row is None:
                residual_ok = False
                reasons.append(f"{side} {period}: target-consistency stability cell missing.")
                continue
            raw = abs(float(row["distance_vs_raw_log_density_spearman"]))
            legacy = abs(float(row["distance_vs_legacy_residual_spearman"]))
            formula = abs(float(row["distance_vs_formula_only_residual_spearman"]))
            corrected = abs(float(row["distance_vs_target_consistent_residual_spearman"]))
            reasons.append(
                f"{side} {period}: |distance raw/legacy/formula-only/mean-aligned residual|="
                f"{raw:.3f}/{legacy:.3f}/{formula:.3f}/{corrected:.3f}."
            )
            if not np.isfinite(corrected) or corrected > config.residualization_max_abs_distance_spearman:
                residual_ok = False
            if np.isfinite(raw) and raw >= 0.05 and corrected >= raw:
                residual_ok = False

    if not residual_ok:
        return "BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED", reasons + [
            "Even after same-scale hurdle expectation and an L2 conditional-log mean, Validation/Holdout still retain too much mechanical distance dependence. Do not add Range/Footprint/OI and do not train a new PATH ranker."
        ]

    drift_flags: list[str] = []
    for side in ("DOWN", "UP"):
        for period in (config.calibration_period, config.holdout_period):
            auc = _metric(nuisance_metrics, period, side, "RELEASE_HURDLE", "roc_auc")
            ratio = _metric(nuisance_metrics, period, side, "MEAN_ALIGNED_EXPECTED_LOG_PRIMARY", "actual_to_predicted_ratio")
            if np.isfinite(auc) and auc < config.diagnostic_release_auc_floor:
                drift_flags.append(f"{side} {period} release AUC={auc:.3f}")
            if np.isfinite(ratio) and not (
                config.diagnostic_expected_log_mean_ratio_low <= ratio <= config.diagnostic_expected_log_mean_ratio_high
            ):
                drift_flags.append(f"{side} {period} actual/predicted expected-log ratio={ratio:.3f}")
    if drift_flags:
        return "PASS_R02_3_1B_TARGET_SCALE_BUT_NUISANCE_REGIME_DRIFT_REMAINS", reasons + [
            "Target construction now passes the distance gate, but frozen 2023-2024 nuisance calibration/discrimination still shows future-period drift: " + "; ".join(drift_flags) + ".",
            "Next stage should diagnose nuisance-regime conditioning only. Do not train the PATH ranker yet.",
        ]
    return "PASS_R02_3_1B_TARGET_CONSISTENCY_READY_FOR_FROZEN_PATH_RETEST", reasons + [
        "The same-scale target passes the frozen distance gate without a material nuisance-drift diagnostic. The next stage may retest PATH_NO_SWING on this frozen target before any new data family is added."
    ]


def write_reports(
    *,
    config: TargetConsistencyConfig,
    source_gate: pd.DataFrame,
    frame: pd.DataFrame,
    nuisance_feature_audit: pd.DataFrame,
    nuisance_fold_audit: pd.DataFrame,
    nuisance_metrics: pd.DataFrame,
    causal: pd.DataFrame,
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    quality = source_quality(frame)
    stability = target_consistency_stability(frame)
    cells = distance_cell_audit(frame)
    yearly = yearly_stability(frame)
    gaps = transform_gap_audit(frame)
    eligible = frame["r02_3_1b_source_eligible"].astype(bool)

    summary = pd.DataFrame([{
        "rows": int(len(frame)),
        "groups": int(frame.groupby(["decision_time", "zone_side"], sort=False).ngroups),
        "source_eligible_rows": int(eligible.sum()),
        "train_expanding_oos_eligible_rows": int((eligible & frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS")).sum()),
        "target_variable": "Z=log1p(first-touch release density)",
        "legacy_proxy": "log1p(P(release)*smearing_adjusted_positive_density)",
        "formula_only_expectation": "P(release)*Huber_location(log1p(density)|release)",
        "primary_expectation": "P(release)*L2_mean(log1p(density)|release)",
        "primary_residual": "Z - primary_expectation",
        "path_ranker_trained": False,
        "new_data_family_added": False,
        "swing_used": False,
    }])

    sample_cols = [c for c in (
        "zone_id", "decision_time", "period", "zone_side", "zone_distance_bp",
        "first_touch_time", "first_touch_available_time", "r02_touch_consistent",
        "r02_3_1b_source_eligible", "nuisance_prediction_source", "release_observed_180s",
        "raw_release_density", "raw_log_release_density", "nuisance_p_release",
        "nuisance_huber_log_density_if_release", "nuisance_mean_log_density_if_release",
        "legacy_expected_log_proxy", "formula_only_expected_log_density", "mean_aligned_expected_log_density",
        "legacy_excess_residual", "formula_only_excess_residual", "target_consistent_excess_residual",
        "transform_gap_formula_only_minus_legacy", "objective_gap_mean_minus_huber", "total_expected_log_correction",
    ) if c in frame.columns]
    sample = frame.loc[eligible, sample_cols].sort_values(
        ["period", "zone_side", "decision_time", "zone_distance_bp"], kind="mergesort"
    ).head(50_000)

    tables = {
        "01_source_gate.csv": source_gate,
        "02_source_quality_and_quarantine.csv": quality,
        "03_nuisance_feature_audit.csv": nuisance_feature_audit,
        "04_nuisance_expanding_fold_audit.csv": nuisance_fold_audit,
        "05_nuisance_model_metrics.csv": nuisance_metrics,
        "06_dataset_summary.csv": summary,
        "07_target_consistency_stability.csv": stability,
        "08_distance_cell_residual_audit.csv": cells,
        "09_yearly_stability.csv": yearly,
        "10_transform_gap_audit.csv": gaps,
        "11_causal_audit.csv": causal,
        "13_target_consistency_sample.csv": sample,
    }
    for name, table in tables.items():
        _write(table, root / name)

    decision, reasons = _decision(stability, nuisance_metrics, causal, config)
    lines = [
        f"# {MODEL_NAME} {STAGE_ID} decision",
        "",
        "## Primary decision",
        "",
        f"`{decision}`",
        "",
        "## Evidence",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Frozen interpretation",
        "",
        "- R02.3.1 real run is frozen as BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED; this stage does not rewrite that historical result.",
        "- R02.3.1b is an audit-only correction. It does not train a PATH ranker, add Range/Footprint/OI/Books, backtest entries, or approve live trading.",
        "- The primary target is Z=log1p(release density). Its hurdle expectation is computed on the same scale as P(release)*E[Z|release,X].",
        "- Positive-log conditional mean uses a fixed L2 objective. A parallel Huber model is diagnostic only, so transform mismatch and objective mismatch can be separated.",
        "- TRAIN predictions remain expanding past-only OOS with purge. Validation/Holdout use one full-TRAIN-frozen nuisance family.",
        "- Swing and zone-specific path structure are prohibited from nuisance estimation.",
        "- 2025Q4-2026H1 remains development holdout, not sealed validation.",
    ]
    (root / "12_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "sealed_validation_claim": False,
        "live_approved": False,
        "path_ranker_trained": False,
        "new_data_family_added": False,
        "config": config.to_dict(),
        "reports": list(tables) + ["12_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R02.3.1b strictly as a target-construction and nuisance-residualization audit. Compare legacy, formula-only and mean-aligned residual distance dependence in Validation/Holdout; inspect exact distance-cell residual means, yearly stability, release-hurdle drift, and causal audit. No PATH ranker, new data family, execution backtest, sealed validation or live approval is part of this stage.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(
            report_dir=root,
            experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02_3_1B",
            edge_id="RESEARCH_ONLY_TARGET_CONSISTENCY_AUDIT",
            title=f"{MODEL_NAME} {STAGE_ID}",
            decision_focus="same-scale hurdle target consistency, distance-cell residual audit, nuisance drift and causal integrity",
        ))
    return root, decision
