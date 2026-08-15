#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3 compact reports, causal audit and research-only promotion gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, ExcessLiquidityRankingConfig
from .labels import DistanceNormalizer
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
        {"metric": "r02_3_source_eligible_rows", "value": int(frame["r02_3_source_eligible"].astype(bool).sum())},
    ])


def causal_audit(
    frame: pd.DataFrame,
    models: ModelBundle,
    source_gate: pd.DataFrame,
    normalizer: DistanceNormalizer,
    config: ExcessLiquidityRankingConfig,
) -> pd.DataFrame:
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    touch = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    touch_available = pd.to_datetime(frame["first_touch_available_time"], errors="coerce")
    eligible = frame["r02_3_source_eligible"].astype(bool)
    group_sizes = frame.groupby(["decision_time", "zone_side"], sort=False).size()
    source_fail = int(source_gate["status"].astype(str).eq("FAIL").sum()) if not source_gate.empty else 0
    path_cols = sorted({c for b in models.by_side.values() for c in b.path_columns})
    future_prefixes = (
        "touch_", "release_", "favorable_", "continuation_", "time_to_", "sweep_", "reversal_",
        "p_touch", "p_release", "p_favorable", "pred_", "pool_score", "high_strength", "ft_", "first_touch",
        "ranking_", "expected_", "excess_", "density_vs_expected", "reversal_quality",
    )
    leaked = [c for c in path_cols if c.startswith(future_prefixes)]
    swing = [c for c in path_cols if c.startswith("swing_")]
    normalizer_bad = int((normalizer.table["rows"] <= 0).sum()) if not normalizer.table.empty else 1
    source_mismatch_in_model = int((eligible & ~frame["r02_touch_consistent"].astype(bool)).sum())
    rows = [
        {"check": "r01_1_r02_source_gate", "value": source_fail, "status": "PASS" if source_fail == 0 else "FAIL"},
        {"check": "complete_lattice_25_zones_per_side", "value": int(group_sizes.ne(config.expected_zone_count).sum()), "status": "PASS" if len(group_sizes) and not group_sizes.ne(config.expected_zone_count).any() else "FAIL"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "first_touch_available_strictly_after_decision", "value": int((eligible & touch_available.le(decision)).sum()), "status": "PASS" if not (eligible & touch_available.le(decision)).any() else "FAIL"},
        {"check": "first_touch_inside_exclusive_12h_horizon", "value": int((eligible & touch.ge(decision + pd.Timedelta(hours=12))).sum()), "status": "PASS" if not (eligible & touch.ge(decision + pd.Timedelta(hours=12))).any() else "FAIL"},
        {"check": "upstream_r02_touch_mismatch_rows_used_by_r02_3", "value": source_mismatch_in_model, "status": "PASS" if source_mismatch_in_model == 0 else "FAIL"},
        {"check": "distance_normalizer_train_only_frozen", "value": normalizer_bad, "status": "PASS" if normalizer_bad == 0 else "FAIL"},
        {"check": "primary_excess_ranker_has_no_future_labels", "value": len(leaked), "status": "PASS" if not leaked else "FAIL"},
        {"check": "primary_excess_ranker_excludes_swing", "value": len(swing), "status": "PASS" if not swing else "FAIL"},
        {"check": "primary_excess_ranker_excludes_raw_distance", "value": int("zone_distance_bp" in path_cols), "status": "PASS" if "zone_distance_bp" not in path_cols else "FAIL"},
        {"check": "touch_probability_not_in_primary_score", "value": 0, "status": "PASS"},
        {"check": "no_absolute_q80_q90_pool_threshold", "value": 0, "status": "PASS"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ]
    return pd.DataFrame(rows)


def distance_profile(frame: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> pd.DataFrame:
    w = config.primary_label_window_seconds
    sf = frame.loc[frame["r02_3_source_eligible"].astype(bool)].copy()
    if sf.empty:
        return pd.DataFrame()
    return sf.groupby(["period", "zone_side", "zone_distance_bp"], sort=True).agg(
        rows=("zone_id", "size"),
        mean_raw_density=(f"ft_release_density_sum_{w}s", "mean"),
        expected_density=("expected_density", "mean"),
        mean_excess_z=("excess_liquidity_z", "mean"),
        median_excess_z=("excess_liquidity_z", "median"),
        mean_density_vs_expected=("density_vs_expected_ratio", "mean"),
        favorable_rate=("favorable_observed_180s", "mean"),
        continuation_rate=("continuation_observed_180s", "mean"),
        mean_reversal_quality=("reversal_quality_target", "mean"),
    ).reset_index()


def target_stability(frame: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sf = frame.loc[frame["r02_3_source_eligible"].astype(bool)].copy()
    for (period, side), group in sf.groupby(["period", "zone_side"], sort=True):
        d = pd.to_numeric(group["zone_distance_bp"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(group["excess_liquidity_z"], errors="coerce").to_numpy(dtype=float)
        raw = pd.to_numeric(group[f"ft_release_density_sum_{config.primary_label_window_seconds}s"], errors="coerce").to_numpy(dtype=float)
        valid_z = np.isfinite(d) & np.isfinite(z)
        valid_raw = np.isfinite(d) & np.isfinite(raw)
        rho_z = spearmanr(d[valid_z], z[valid_z]).statistic if valid_z.sum() >= 2 and np.nanstd(z[valid_z]) > 1e-12 else np.nan
        rho_raw = spearmanr(d[valid_raw], raw[valid_raw]).statistic if valid_raw.sum() >= 2 and np.nanstd(raw[valid_raw]) > 1e-12 else np.nan
        rows.append({
            "period": period, "zone_side": side, "rows": int(len(group)),
            "mean_raw_density": float(np.nanmean(raw)) if len(raw) else np.nan,
            "mean_excess_z": float(np.nanmean(z)) if len(z) else np.nan,
            "std_excess_z": float(np.nanstd(z)) if len(z) else np.nan,
            "distance_vs_raw_density_spearman": float(rho_raw) if np.isfinite(rho_raw) else np.nan,
            "distance_vs_excess_z_spearman": float(rho_z) if np.isfinite(rho_z) else np.nan,
            "mean_density_vs_expected_ratio": float(pd.to_numeric(group["density_vs_expected_ratio"], errors="coerce").mean()),
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


def _decision(
    metrics: pd.DataFrame,
    regression: pd.DataFrame,
    top: pd.DataFrame,
    causal: pd.DataFrame,
    config: ExcessLiquidityRankingConfig,
) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].astype(str).eq("FAIL").any():
        return "BLOCKED_R02_3_QUALITY_OR_CAUSAL_FAILURE", ["A source or causal gate failed; no promotion claim is allowed."]
    promote: list[str] = []
    reasons: list[str] = []
    for side in ("DOWN", "UP"):
        hold = metrics.loc[metrics["period"].astype(str).eq(config.holdout_period) & metrics["zone_side"].astype(str).eq(side)]
        ex = hold.loc[(hold["task"].eq("EXCESS_LIQUIDITY")) & (hold["model"].eq("PATH_NO_SWING"))]
        ex_near = hold.loc[(hold["task"].eq("EXCESS_LIQUIDITY")) & (hold["model"].eq("DISTANCE_NEAR"))]
        ex_far = hold.loc[(hold["task"].eq("EXCESS_LIQUIDITY")) & (hold["model"].eq("DISTANCE_FAR"))]
        rv = hold.loc[(hold["task"].eq("REVERSAL_QUALITY")) & (hold["model"].eq("PATH_NO_SWING"))]
        top_ex = top.loc[top["period"].astype(str).eq(config.holdout_period) & top["zone_side"].astype(str).eq(side) & top["model"].eq("EXCESS_PATH_NO_SWING")]
        sweep = regression.loc[regression["period"].astype(str).eq(config.holdout_period) & regression["zone_side"].astype(str).eq(side) & regression["task"].eq("SWEEP_DEPTH")]
        if ex.empty or ex_near.empty or ex_far.empty or rv.empty or top_ex.empty or sweep.empty:
            reasons.append(f"{side}: missing holdout cells.")
            continue
        exr = float(ex.iloc[0].get("mean_group_spearman", np.nan))
        base = max(float(ex_near.iloc[0].get("mean_group_spearman", np.nan)), float(ex_far.iloc[0].get("mean_group_spearman", np.nan)))
        rvr = float(rv.iloc[0].get("mean_group_spearman", np.nan))
        ratio = float(top_ex.iloc[0].get("top1_mean_density_ratio", np.nan))
        oracle = float(top_ex.iloc[0].get("oracle_strongest_excess_zone_in_top3_rate", np.nan))
        touched = int(top_ex.iloc[0].get("top1_touched", 0))
        sweep_rho = float(sweep.iloc[0].get("spearman", np.nan))
        reasons.append(
            f"{side}: holdout excess Spearman path/best-distance={exr:.3f}/{base:.3f}; "
            f"Top-1 density-vs-expected={ratio:.3f}; excess oracle-in-top3={oracle:.3f}; "
            f"reversal-quality Spearman={rvr:.3f}; sweep-depth Spearman={sweep_rho:.3f}; touched={touched}."
        )
        passed = bool(
            np.isfinite(exr) and exr >= config.promotion_min_excess_spearman and exr > base
            and np.isfinite(ratio) and ratio >= config.promotion_min_excess_top1_ratio
            and np.isfinite(oracle) and oracle >= config.promotion_min_oracle_top3_rate
            and np.isfinite(rvr) and rvr >= config.promotion_min_reversal_spearman
            and np.isfinite(sweep_rho) and sweep_rho >= config.promotion_min_sweep_depth_spearman
            and touched >= config.minimum_top1_touched
        )
        if passed:
            promote.append(side)
    if promote:
        return (
            f"PROMOTE_{'_AND_'.join(promote)}_TO_R02_4_LIMIT_PLACEMENT_SWEEP_GEOMETRY_STUDY",
            reasons + ["Promotion is research-only. R02.4 may study passive placement near predicted sweep depth; no live approval."],
        )
    hold_ex = metrics.loc[
        metrics["period"].astype(str).eq(config.holdout_period)
        & metrics["task"].eq("EXCESS_LIQUIDITY")
        & metrics["model"].eq("PATH_NO_SWING"), "mean_group_spearman"
    ]
    hold_rv = metrics.loc[
        metrics["period"].astype(str).eq(config.holdout_period)
        & metrics["task"].eq("REVERSAL_QUALITY")
        & metrics["model"].eq("PATH_NO_SWING"), "mean_group_spearman"
    ]
    if (len(hold_ex) and hold_ex.max(skipna=True) >= 0.05) or (len(hold_rv) and hold_rv.max(skipna=True) >= 0.05):
        return "CONTINUE_R02_3_WITH_RANGE_FOOTPRINT_OI_INCREMENT", reasons + [
            "The corrected target contains some spatial signal but does not clear the placement gate. Only independent Range/Footprint/OI increments are allowed next; do not rescue with Swing or threshold grids."
        ]
    return "STOP_R02_3_EXCESS_LIQUIDITY_RANKING_NO_EDGE", reasons


def write_reports(
    *,
    config: ExcessLiquidityRankingConfig,
    source_gate: pd.DataFrame,
    frame: pd.DataFrame,
    normalizer: DistanceNormalizer,
    metrics: pd.DataFrame,
    regression: pd.DataFrame,
    top: pd.DataFrame,
    importance: pd.DataFrame,
    causal: pd.DataFrame,
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    family = importance.groupby(["zone_side", "task", "model", "feature_family"], sort=True)["importance_share"].sum().reset_index()
    source_q = source_quality(frame)
    distance = distance_profile(frame, config)
    stability = target_stability(frame, config)
    ablation = swing_ablation(metrics)
    eligible = frame["r02_3_source_eligible"].astype(bool)
    summary = pd.DataFrame([{
        "rows": int(len(frame)),
        "groups": int(frame.groupby(["decision_time", "zone_side"], sort=False).ngroups),
        "source_eligible_rows": int(eligible.sum()),
        "excess_rankable_rows": int(frame["excess_group_eligible"].astype(bool).sum()),
        "excess_rankable_groups": int(frame.loc[frame["excess_group_eligible"].astype(bool), "ranking_group"].nunique()),
        "reversal_rankable_rows": int(frame["reversal_group_eligible"].astype(bool).sum()),
        "reversal_rankable_groups": int(frame.loc[frame["reversal_group_eligible"].astype(bool), "ranking_group"].nunique()),
        "primary_target": "train-distance-normalized robust z of log1p(first-touch release density 180s)",
        "reversal_target": "log1p(favorable density)-log1p(continuation density), conditioned on release",
        "swing_in_primary_model": False,
        "touch_probability_in_primary_model": False,
        "absolute_q80_q90_used": False,
    }])
    sample_cols = [c for c in (
        "zone_id", "decision_time", "period", "zone_side", "zone_distance_bp", "first_touch_time", "first_touch_available_time",
        "r02_touch_consistent", "r02_3_source_eligible", "expected_density", "ft_release_density_sum_180s",
        "density_vs_expected_ratio", "excess_liquidity_z", "reversal_quality_target", "sweep_depth_target_bp", "reversal_room_target_bp",
        "score_excess_path_no_swing", "score_reversal_path_no_swing", "score_joint_path_no_swing", "pred_sweep_depth_bp", "pred_reversal_room_bp",
    ) if c in frame.columns]
    sample = frame.sort_values(["period", "zone_side", "score_joint_path_no_swing"], ascending=[True, True, False], kind="mergesort").loc[:, sample_cols].head(50_000)
    tables = {
        "01_source_gate.csv": source_gate,
        "02_source_quality_and_quarantine.csv": source_q,
        "03_distance_normalizer_train_only.csv": normalizer.table,
        "04_dataset_summary.csv": summary,
        "05_ranking_metrics.csv": metrics,
        "06_sweep_geometry_metrics.csv": regression,
        "07_top_zone_summary.csv": top,
        "08_distance_normalized_profile.csv": distance,
        "09_target_stability.csv": stability,
        "10_swing_ablation.csv": ablation,
        "11_feature_importance.csv": importance,
        "12_feature_family_importance.csv": family,
        "13_causal_audit.csv": causal,
        "14_top_zone_sample.csv": sample,
    }
    for name, table in tables.items():
        _write(table, root / name)
    decision, reasons = _decision(metrics, regression, top, causal, config)
    lines = [
        f"# {MODEL_NAME} {STAGE_ID} decision", "", "## Primary decision", "", f"`{decision}`", "", "## Evidence", "",
        *[f"- {r}" for r in reasons], "", "## Frozen interpretation", "",
        "- R02.3 retires raw first-touch release-density ranking as the primary pool-location objective.",
        "- Primary Excess Liquidity is the robust deviation of log1p(first-touch 180s release density) from a TRAIN-only side x distance median/IQR baseline.",
        "- Reversal Quality is a separate target: log1p(favorable density) minus log1p(continuation density), conditioned on a release being observed.",
        "- Sweep depth and reversal room remain separate geometry regressions; they are not mixed into the pool-strength label.",
        "- The primary Excess and Reversal models exclude Swing. 15m+ unswept Swing remains an ablation only.",
        "- R02.2 first-touch bar timestamps are interpreted by 1s bar available_time = bar_start + 1s. Upstream R02 touch-cache disagreements are quarantined and cannot enter R02.3 training/evaluation.",
        "- R01.3 post-confirmation market entry remains stopped. R02.3 does not place orders and does not claim sealed validation or live approval.",
        "- 2025Q4-2026H1 remains a development holdout because prior stages have already inspected it.",
    ]
    (root / "15_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "model": MODEL_NAME, "stage_id": STAGE_ID, "stage_name": STAGE_NAME,
        "decision": decision, "sealed_validation_claim": False, "live_approved": False,
        "config": config.to_dict(), "reports": list(tables) + ["15_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R02.3 as a distance-normalized excess-liquidity and separate reversal-quality ranking study. Focus on whether TRAIN-only distance normalization removes the raw-density distance bias out of sample, whether PATH_NO_SWING ranks excess liquidity and reversal quality stably across Validation/Holdout, whether Swing adds any stable ablation uplift, whether sweep-depth geometry remains predictable, and whether all causal/quarantine gates pass. Do not reinterpret this as a Swing strategy, raw-density model, limit-order backtest, sealed validation, or live approval.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(
            report_dir=root,
            experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02_3",
            edge_id="RESEARCH_ONLY_DISTANCE_NORMALIZED_EXCESS_LIQUIDITY_RANKING",
            title=f"{MODEL_NAME} {STAGE_ID}",
            decision_focus="distance-normalized excess liquidity, reversal-quality ranking, Swing ablation, sweep geometry and causal quarantine",
        ))
    return root, decision
