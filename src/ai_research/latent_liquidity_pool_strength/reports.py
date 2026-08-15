#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact diagnostics and gate for R02.1 conditional pool strength."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, LatentLiquidityPoolStrengthConfig


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def calibration_thresholds(audit: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    rows = []
    cal = audit.loc[audit["period"].astype(str).eq(config.calibration_period)]
    for side, sf in cal.groupby("zone_side", sort=True):
        values = pd.to_numeric(sf["pool_strength_score"], errors="coerce")
        values = values[np.isfinite(values)]
        rows.append({
            "zone_side": side,
            "period": config.calibration_period,
            "score_quantile": 0.90,
            "pool_strength_score_threshold": float(values.quantile(0.90)) if len(values) else np.nan,
            "audit_rows": int(len(values)),
        })
    return pd.DataFrame(rows)


def score_deciles(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf0 in audit.groupby(["period", "zone_side"], sort=True):
        sf = sf0.copy()
        score = pd.to_numeric(sf["pool_strength_score"], errors="coerce")
        valid = np.isfinite(score)
        sf = sf.loc[valid].copy(); score = score[valid]
        if len(sf) < 20:
            continue
        try:
            sf["score_decile"] = pd.qcut(score.rank(method="first"), 10, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for decile, df in sf.groupby("score_decile", sort=True):
            touched = df.loc[df["touch_720m"].astype(bool)]
            released = touched.loc[touched["release_episode_count"].gt(0)]
            rows.append({
                "period": period, "zone_side": side, "score_decile": int(decile), "rows": len(df),
                "touch_rate": float(df["touch_720m"].mean()), "touched_rows": len(touched),
                "high_strength_rate_if_touched": float(touched["high_strength_label"].mean()) if len(touched) else np.nan,
                "mean_release_density_if_touched": float(touched["release_density_sum"].mean()) if len(touched) else np.nan,
                "mean_release_episode_count_if_touched": float(touched["release_episode_count"].mean()) if len(touched) else np.nan,
                "favorable_any_if_release": float(released["favorable_episode_count"].gt(0).mean()) if len(released) else np.nan,
                "continuation_any_if_release": float(released["continuation_episode_count"].gt(0).mean()) if len(released) else np.nan,
                "mean_score": float(df["pool_strength_score"].mean()),
                "mean_zone_distance_bp": float(df["zone_distance_bp"].mean()),
            })
    return pd.DataFrame(rows)


def top1_zone_summary(audit: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in audit.groupby(["period", "zone_side"], sort=True):
        if sf.empty:
            continue
        touched_all = sf.loc[sf["touch_720m"].astype(bool)]
        baseline_hs = float(touched_all["high_strength_label"].mean()) if len(touched_all) else np.nan
        baseline_density = float(touched_all["release_density_sum"].mean()) if len(touched_all) else np.nan
        order = sf.sort_values(["decision_time", "pool_strength_score", "zone_distance_bp"], ascending=[True, False, True], kind="mergesort")
        chosen = order.groupby("decision_time", sort=False).head(1)
        touched = chosen.loc[chosen["touch_720m"].astype(bool)]
        released = touched.loc[touched["release_episode_count"].gt(0)]
        hs = float(touched["high_strength_label"].mean()) if len(touched) else np.nan
        density = float(touched["release_density_sum"].mean()) if len(touched) else np.nan
        rows.append({
            "period": period, "zone_side": side, "audit_groups": int(sf["decision_time"].nunique()),
            "top1_rows": len(chosen), "top1_touch_rate": float(chosen["touch_720m"].mean()),
            "top1_touched_rows": len(touched), "top1_high_strength_rate_if_touched": hs,
            "baseline_high_strength_rate_if_touched": baseline_hs,
            "high_strength_lift": hs / baseline_hs if np.isfinite(hs) and np.isfinite(baseline_hs) and baseline_hs > 0 else np.nan,
            "top1_mean_density_if_touched": density, "baseline_mean_density_if_touched": baseline_density,
            "density_lift": density / baseline_density if np.isfinite(density) and np.isfinite(baseline_density) and baseline_density > 0 else np.nan,
            "top1_release_rate_if_touched": float(touched["release_episode_count"].gt(0).mean()) if len(touched) else np.nan,
            "top1_favorable_any_if_release": float(released["favorable_episode_count"].gt(0).mean()) if len(released) else np.nan,
            "top1_continuation_any_if_release": float(released["continuation_episode_count"].gt(0).mean()) if len(released) else np.nan,
            "top1_mean_sweep_depth_bp": float(released["sweep_depth_weighted_bp"].mean()) if len(released) else np.nan,
            "top1_mean_reversal_room_bp": float(released["reversal_room_weighted_bp"].mean()) if len(released) else np.nan,
            "top1_mean_distance_bp": float(chosen["zone_distance_bp"].mean()),
            "top1_mean_score": float(chosen["pool_strength_score"].mean()),
        })
    return pd.DataFrame(rows)


def q90_zone_summary(audit: pd.DataFrame, thresholds: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    rows = []
    threshold_map = dict(zip(thresholds.get("zone_side", []), thresholds.get("pool_strength_score_threshold", [])))
    for (period, side), sf in audit.groupby(["period", "zone_side"], sort=True):
        threshold = float(threshold_map.get(side, np.nan))
        selected = sf.loc[pd.to_numeric(sf["pool_strength_score"], errors="coerce").ge(threshold)] if np.isfinite(threshold) else sf.iloc[0:0]
        touched = selected.loc[selected["touch_720m"].astype(bool)]
        released = touched.loc[touched["release_episode_count"].gt(0)]
        rows.append({
            "period": period, "zone_side": side, "threshold": threshold, "selected_rows": len(selected),
            "selected_touch_rate": float(selected["touch_720m"].mean()) if len(selected) else np.nan,
            "touched_rows": len(touched), "high_strength_rate_if_touched": float(touched["high_strength_label"].mean()) if len(touched) else np.nan,
            "mean_density_if_touched": float(touched["release_density_sum"].mean()) if len(touched) else np.nan,
            "favorable_any_if_release": float(released["favorable_episode_count"].gt(0).mean()) if len(released) else np.nan,
            "continuation_any_if_release": float(released["continuation_episode_count"].gt(0).mean()) if len(released) else np.nan,
            "mean_distance_bp": float(selected["zone_distance_bp"].mean()) if len(selected) else np.nan,
        })
    return pd.DataFrame(rows)


def causal_audit(frame: pd.DataFrame, audit: pd.DataFrame, feature_columns: tuple[str, ...], source_gate: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    future_names = {
        "high_strength_label", "release_episode_count", "release_density_sum", "release_density_max",
        "release_episode_size_sum", "release_score_sum", "favorable_episode_count", "continuation_episode_count",
        "favorable_density_sum", "continuation_density_sum", "sweep_depth_weighted_bp", "reversal_room_weighted_bp",
        "first_release_minutes", "release_density_log", "release_count_log", "release_size_log", "release_peak_log",
    }
    leaked = sorted(set(feature_columns) & future_names)
    sub15 = [name for name in feature_columns if name.startswith("swing_") and any(token in name.lower() for token in ("1s", "5s", "1m", "3m", "5m"))]
    complete_groups = audit.groupby(["decision_time", "zone_side"], sort=False).size() if not audit.empty else pd.Series(dtype=int)
    expected = 25
    release_without_touch = frame["release_within_horizon"].astype(bool) & ~frame["touch_720m"].astype(bool)
    source_fail = int(source_gate["status"].astype(str).eq("FAIL").sum()) if not source_gate.empty else 0
    rows = [
        {"check": "r01_1_source_gate", "value": source_fail, "status": "PASS" if source_fail == 0 else "FAIL"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "release_implies_touch_exclusive_horizon", "value": int(release_without_touch.sum()), "status": "PASS" if not release_without_touch.any() else "FAIL"},
        {"check": "future_strength_labels_excluded_from_features", "value": len(leaked), "status": "PASS" if not leaked else "FAIL"},
        {"check": "primary_pool_score_excludes_touch_probability", "value": int("p_touch" in feature_columns), "status": "PASS" if "p_touch" not in feature_columns else "FAIL"},
        {"check": "primary_pool_score_excludes_swing", "value": int(any(name.startswith("swing_") for name in feature_columns)), "status": "PASS" if not any(name.startswith("swing_") for name in feature_columns) else "FAIL"},
        {"check": "sub15m_swing_forbidden", "value": len(sub15), "status": "PASS" if not sub15 else "FAIL"},
        {"check": "full_lattice_audit_groups_complete", "value": int(complete_groups.ne(expected).sum()) if len(complete_groups) else -1, "status": "PASS" if len(complete_groups) and not complete_groups.ne(expected).any() else "FAIL"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ]
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, top1: pd.DataFrame, causal: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].astype(str).eq("FAIL").any():
        return "BLOCKED_R02_1_QUALITY_OR_CAUSAL_FAILURE", ["A source or causal gate failed."]
    promote: list[str] = []
    reasons: list[str] = []
    for side in ("DOWN", "UP"):
        hold = metrics.loc[metrics["period"].eq(config.holdout_period) & metrics["zone_side"].eq(side)]
        def row(task: str):
            return hold.loc[hold["task"].eq(task)]
        sp, sf, sb = row("HIGH_STRENGTH_PATH_NO_SWING"), row("HIGH_STRENGTH_FULL_WITH_SWING"), row("HIGH_STRENGTH_DISTANCE_BASELINE")
        dp, db = row("DENSITY_PATH_NO_SWING"), row("DENSITY_DISTANCE_BASELINE")
        fav, depth = row("FAVORABLE_PATH_NO_SWING"), row("SWEEP_DEPTH_PATH_NO_SWING")
        if any(x.empty for x in (sp, sf, sb, dp, db, fav, depth)):
            reasons.append(f"{side}: missing holdout model cells.")
            continue
        path_auc = float(sp.iloc[0].get("roc_auc", np.nan)); full_auc = float(sf.iloc[0].get("roc_auc", np.nan)); base_auc = float(sb.iloc[0].get("roc_auc", np.nan))
        density_rho = float(dp.iloc[0].get("spearman", np.nan)); density_base_rho = float(db.iloc[0].get("spearman", np.nan))
        fav_auc = float(fav.iloc[0].get("roc_auc", np.nan)); depth_rho = float(depth.iloc[0].get("spearman", np.nan))
        top = top1.loc[top1["period"].eq(config.holdout_period) & top1["zone_side"].eq(side)]
        top_ok = False
        if not top.empty:
            tr = top.iloc[0]
            top_ok = bool(
                int(tr.get("top1_touched_rows", 0)) >= config.minimum_top1_touched
                and float(tr.get("high_strength_lift", np.nan)) >= 1.5
                and float(tr.get("density_lift", np.nan)) >= 1.35
            )
        predictive = bool(
            np.isfinite(path_auc) and path_auc >= 0.60
            and path_auc - base_auc >= 0.03
            and np.isfinite(density_rho) and density_rho >= 0.20
            and density_rho - density_base_rho >= 0.05
            and np.isfinite(fav_auc) and fav_auc >= 0.62
            and np.isfinite(depth_rho) and depth_rho >= 0.20
        )
        reasons.append(
            f"{side}: holdout strength AUC path/full/base={path_auc:.3f}/{full_auc:.3f}/{base_auc:.3f}; "
            f"Swing uplift={full_auc-path_auc:.3f}; density Spearman path/base={density_rho:.3f}/{density_base_rho:.3f}; "
            f"favorable AUC={fav_auc:.3f}; sweep-depth Spearman={depth_rho:.3f}; top1 quality={'PASS' if top_ok else 'FAIL'}."
        )
        if predictive and top_ok:
            promote.append(side)
    if promote:
        return f"PROMOTE_{'_AND_'.join(promote)}_TO_R02_2_LIMIT_PLACEMENT_DEPTH_STUDY", reasons + ["Promotion is development-only: next stage studies causal limit placement near predicted pool depth; it is not live approval."]
    hold_strength = metrics.loc[metrics["period"].eq(config.holdout_period) & metrics["task"].eq("HIGH_STRENGTH_PATH_NO_SWING"), "roc_auc"]
    hold_density = metrics.loc[metrics["period"].eq(config.holdout_period) & metrics["task"].eq("DENSITY_PATH_NO_SWING"), "spearman"]
    if (hold_strength.max(skipna=True) >= 0.55 if len(hold_strength) else False) or (hold_density.max(skipna=True) >= 0.15 if len(hold_density) else False):
        return "CONTINUE_R02_1_WITH_RANGE_FOOTPRINT_OI_INCREMENT", reasons + ["Conditional pool-strength signal exists but did not clear the baseline gate. Add independent microstructure/OI evidence before any placement study."]
    return "STOP_R02_1_POOL_STRENGTH_NO_PREDICTIVE_EDGE", reasons


def write_reports(*, config: LatentLiquidityPoolStrengthConfig, source_gate: pd.DataFrame, frame: pd.DataFrame, audit: pd.DataFrame, thresholds_train: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame, causal: pd.DataFrame, skip_review_pack: bool) -> tuple[Path, str]:
    root = config.report_path; root.mkdir(parents=True, exist_ok=True)
    deciles = score_deciles(audit)
    score_thresholds = calibration_thresholds(audit, config)
    top1 = top1_zone_summary(audit, config)
    q90 = q90_zone_summary(audit, score_thresholds, config)
    family = importance.groupby(["task", "feature_family"], sort=True)["importance_share"].sum().reset_index()
    distance = audit.loc[audit["touch_720m"].astype(bool)].groupby(["period", "zone_side", "zone_distance_bp"], sort=True).agg(
        touched_rows=("zone_id", "size"), high_strength_rate=("high_strength_label", "mean"),
        mean_release_density=("release_density_sum", "mean"), mean_release_count=("release_episode_count", "mean"),
        favorable_any_rate=("favorable_episode_count", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())),
        continuation_any_rate=("continuation_episode_count", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())),
    ).reset_index()
    source_summary = pd.DataFrame([{
        "rows": len(frame), "audit_rows": len(audit), "snapshots": int(frame["decision_time"].nunique()),
        "touched_rows": int(frame["touch_720m"].sum()), "release_rows": int(frame["release_episode_count"].gt(0).sum()),
        "primary_score": "p_strength_path_no_swing", "touch_probability_in_primary_score": False, "swing_in_primary_score": False,
    }])
    tables = {
        "01_source_gate.csv": source_gate,
        "02_dataset_summary.csv": source_summary,
        "03_train_strength_thresholds.csv": thresholds_train,
        "04_model_metrics.csv": metrics,
        "05_strength_score_deciles.csv": deciles,
        "06_feature_importance.csv": importance,
        "07_feature_family_importance.csv": family,
        "08_calibration_score_thresholds.csv": score_thresholds,
        "09_top1_zone_summary.csv": top1,
        "10_q90_zone_summary.csv": q90,
        "11_distance_strength_profile.csv": distance,
        "12_causal_audit.csv": causal,
        "13_top_zone_sample.csv": audit.sort_values("pool_strength_score", ascending=False).loc[:, [c for c in (
            "zone_id", "decision_time", "period", "zone_side", "current_price", "zone_price", "zone_distance_bp", "touch_720m",
            "release_episode_count", "release_density_sum", "release_density_max", "high_strength_label", "favorable_episode_count",
            "continuation_episode_count", "sweep_depth_weighted_bp", "reversal_room_weighted_bp", "p_strength_path", "p_strength_full",
            "p_strength_baseline", "pred_density_path", "pred_density_full", "pred_density_baseline", "p_favorable_path",
            "p_continuation_path", "pred_sweep_depth_path_bp", "pool_strength_score",
        ) if c in audit.columns]].head(50_000),
    }
    for name, df in tables.items():
        _write(df, root / name)
    decision, reasons = decide(metrics, top1, causal, config)
    lines = [f"# {MODEL_NAME} {STAGE_ID} decision", "", "## Primary decision", "", f"`{decision}`", "", "## Evidence", ""]
    lines += [f"- {r}" for r in reasons]
    lines += ["", "## Scope", "", "- R02.1 deconfounds pool strength from arrival probability: the primary pool-strength score never multiplies or consumes Touch probability.", "- The primary model excludes Swing entirely; 15m+ all-unswept Swing is retained only in a full-model ablation.", "- Strength labels aggregate all realized R01.1 release Episodes in the zone during the exclusive 12h horizon; untouched zones are not treated as zero-liquidity evidence.", "- This stage does not place orders and does not revive R01.3 post-confirmation market entry.", "- The 2025Q4-2026H1 period is development holdout, not sealed validation or live approval."]
    (root / "14_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "model": MODEL_NAME, "stage_id": STAGE_ID, "stage_name": STAGE_NAME, "decision": decision,
        "sealed_validation_claim": False, "live_approved": False, "config": config.to_dict(),
        "reports": list(tables) + ["14_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R02.1 as a conditional-on-touch latent liquidity-pool strength model. Primary score is path-only/no-Swing and excludes Touch probability. Focus on holdout high-strength AUC vs strict distance baseline, density Spearman, Swing ablation, top-1 price-zone lift, favorable-vs-continuation separation, sweep-depth prediction, and causal gates. Do not reinterpret it as a Swing strategy or as live approval.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(
            report_dir=root, experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02_1",
            edge_id="RESEARCH_ONLY_CONDITIONAL_POOL_STRENGTH",
            title=f"{MODEL_NAME} {STAGE_ID}",
            decision_focus="arrival-independent pool strength, release density, favorable/continuation and sweep depth",
        ))
    return root, decision
