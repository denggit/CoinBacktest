#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact R02 reports and promotion gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, LatentLiquidityPoolForecastConfig


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def score_deciles(pred: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in pred.groupby(["period", "zone_side"], sort=True):
        work = group.copy()
        ranks = work["pool_score"].rank(method="first", pct=True)
        work["score_decile"] = np.ceil(ranks * 10).clip(1, 10).astype(int)
        for decile, d in work.groupby("score_decile", sort=True):
            rows.append({
                "period": keys[0], "zone_side": keys[1], "score_decile": int(decile),
                "rows": len(d), "touch_rate": float(d[f"touch_{config.primary_horizon_minutes}m"].mean()) if f"touch_{config.primary_horizon_minutes}m" in d else np.nan,
                "release_rate": float(d["release_within_horizon"].mean()),
                "favorable_release_rate": float(d["favorable_release"].mean()),
                "mean_sweep_depth_bp": float(d.loc[d["release_within_horizon"], "sweep_depth_bp"].mean()),
                "mean_reversal_room_bp": float(d.loc[d["release_within_horizon"], "reversal_after_extreme_bp"].mean()),
                "mean_zone_distance_bp": float(d["zone_distance_bp"].mean()),
            })
    return pd.DataFrame(rows)


def calibration_thresholds(pred: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    rows = []
    for side, group in pred.loc[pred["period"].eq(config.calibration_period)].groupby("zone_side", sort=True):
        values = pd.to_numeric(group["pool_score"], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({"zone_side": side, "quantile": config.selection_quantile, "threshold": float(values.quantile(config.selection_quantile)), "calibration_rows": len(values), "threshold_source_period": config.calibration_period, "holdout_used_for_threshold": False})
    return pd.DataFrame(rows)


def top_zone_summary(pred: pd.DataFrame, thresholds: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if thresholds.empty or pred.empty:
        return pd.DataFrame()
    mapping = thresholds.set_index("zone_side")["threshold"].to_dict()
    population = pred.copy()
    population["audit_day"] = pd.to_datetime(population["decision_time"], errors="coerce").dt.floor("D")
    population_stats: dict[tuple[str, str], tuple[int, int]] = {}
    for keys, g in population.groupby(["period", "zone_side"], sort=True):
        population_stats[(str(keys[0]), str(keys[1]))] = (g.groupby("decision_time", sort=False).ngroups, int(g["audit_day"].nunique()))
    threshold = population["zone_side"].map(mapping).astype(float)
    score = pd.to_numeric(population["pool_score"], errors="coerce")
    work = population.loc[score.ge(threshold) & threshold.notna()].copy()
    if not work.empty:
        work = work.sort_values(["decision_time", "zone_side", "pool_score"], ascending=[True, True, False], kind="mergesort")
        work = work.drop_duplicates(["decision_time", "zone_side"], keep="first")
    rows = []
    for keys, (audit_groups, audit_days) in population_stats.items():
        period, side = keys
        g = work.loc[work["period"].astype(str).eq(period) & work["zone_side"].astype(str).eq(side)] if not work.empty else work
        release_count = int(g["release_within_horizon"].astype(bool).sum()) if not g.empty else 0
        rows.append({
            "period": period, "zone_side": side, "audit_groups": int(audit_groups), "audit_days": int(audit_days),
            "snapshots_selected": len(g),
            "selected_rate_per_group": float(len(g) / max(audit_groups, 1)),
            "selected_per_audit_day": float(len(g) / max(audit_days, 1)),
            "touch_rate": float(g[f"touch_{config.primary_horizon_minutes}m"].mean()) if not g.empty else np.nan,
            "release_count": release_count,
            "release_rate": float(g["release_within_horizon"].mean()) if not g.empty else np.nan,
            "favorable_release_rate": float(g["favorable_release"].mean()) if not g.empty else np.nan,
            "favorable_given_release": float(g.loc[g["release_within_horizon"], "favorable_release"].mean()) if release_count else np.nan,
            "continuation_release_rate": float(g["continuation_release"].mean()) if not g.empty else np.nan,
            "mean_zone_distance_bp": float(g["zone_distance_bp"].mean()) if not g.empty else np.nan,
            "mean_sweep_depth_bp": float(g.loc[g["release_within_horizon"], "sweep_depth_bp"].mean()) if release_count else np.nan,
            "mean_reversal_room_bp": float(g.loc[g["release_within_horizon"], "reversal_after_extreme_bp"].mean()) if release_count else np.nan,
            "mean_pool_score": float(g["pool_score"].mean()) if not g.empty else np.nan,
        })
    return pd.DataFrame(rows)


def causal_audit(frame: pd.DataFrame, feature_columns: tuple[str, ...], source_gate: pd.DataFrame, config: LatentLiquidityPoolForecastConfig, *, audit_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures = int(source_gate.get("status", pd.Series(dtype=str)).astype(str).eq("FAIL").sum())
    rows.append({"check": "r01_1_source_gate", "value": failures, "status": "PASS" if failures == 0 else "FAIL"})
    if frame.empty:
        return pd.DataFrame(rows + [{"check": "spatial_rows", "value": 0, "status": "FAIL"}])
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    future_cols = [c for c in feature_columns if c.startswith(("future_", "touch_", "release_", "favorable_", "continuation_", "sweep_depth", "reversal_after"))]
    sub15_swing = [c for c in frame.columns if c.startswith("swing_") and any(token in c.lower() for token in ("1s", "5s", "1m", "3m", "5m"))]
    audit = audit_frame if audit_frame is not None else frame.iloc[0:0]
    expected_zones = len(config.zone_offsets_bp)
    if not audit.empty:
        group_sizes = audit.groupby(["decision_time", "zone_side"], sort=False).size()
        incomplete_lattice_groups = int(group_sizes.ne(expected_zones).sum())
    else:
        incomplete_lattice_groups = -1
    rows.extend([
        {"check": "spatial_rows", "value": len(frame), "status": "PASS"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "primary_future_label_complete", "value": int((~frame.get("primary_touch_label_complete", pd.Series(False, index=frame.index)).astype(bool)).sum()), "status": "PASS" if "primary_touch_label_complete" in frame and frame["primary_touch_label_complete"].astype(bool).all() else "FAIL"},
        {"check": "release_implies_primary_touch", "value": int((frame["release_within_horizon"].astype(bool) & ~frame[f"touch_{config.primary_horizon_minutes}m"].astype(bool)).sum()), "status": "PASS" if not (frame["release_within_horizon"].astype(bool) & ~frame[f"touch_{config.primary_horizon_minutes}m"].astype(bool)).any() else "FAIL"},
        {"check": "full_lattice_audit_groups_complete", "value": incomplete_lattice_groups, "status": "PASS" if incomplete_lattice_groups == 0 else ("WARN" if audit_frame is None else "FAIL")},
        {"check": "future_labels_excluded_from_features", "value": len(future_cols), "status": "PASS" if not future_cols else "FAIL"},
        {"check": "sub15m_swing_forbidden", "value": len(sub15_swing), "status": "PASS" if not sub15_swing else "FAIL"},
        {"check": "swing_is_supplement_not_admission", "value": int(any(name.startswith("swing_") for name in feature_columns)), "status": "PASS"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ])
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, top: pd.DataFrame, causal: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].eq("FAIL").any():
        return "BLOCKED_R02_QUALITY_OR_CAUSAL_FAILURE", ["A source or causal gate failed."]
    reasons: list[str] = []
    promote = []
    for side in ("DOWN", "UP"):
        hold = metrics.loc[metrics["period"].eq(config.holdout_period) & metrics["zone_side"].eq(side)]
        full = hold.loc[hold["task"].eq("RELEASE_FULL")]
        path = hold.loc[hold["task"].eq("RELEASE_PATH_NO_SWING")]
        base = hold.loc[hold["task"].eq("RELEASE_DISTANCE_BASELINE")]
        fav = hold.loc[hold["task"].eq("FAVORABLE_FULL")]
        depth = hold.loc[hold["task"].eq("SWEEP_DEPTH")]
        if any(x.empty for x in (full, path, base, fav, depth)):
            reasons.append(f"{side}: missing complete holdout model cells.")
            continue
        full_auc = float(full.iloc[0].get("roc_auc", np.nan)); path_auc = float(path.iloc[0].get("roc_auc", np.nan)); base_auc = float(base.iloc[0].get("roc_auc", np.nan)); fav_auc = float(fav.iloc[0].get("roc_auc", np.nan)); rho = float(depth.iloc[0].get("spearman", np.nan))
        swing_increment = full_auc - path_auc
        path_increment = path_auc - base_auc
        top_cell = top.loc[top["period"].eq(config.holdout_period) & top["zone_side"].eq(side)]
        top_ok = (
            not top_cell.empty
            and int(top_cell.iloc[0].get("release_count", 0)) >= config.minimum_top_zone_releases
            and float(top_cell.iloc[0]["release_rate"]) > 0
            and float(top_cell.iloc[0]["favorable_given_release"]) >= 0.35
        )
        predictive = bool(np.isfinite(full_auc) and full_auc >= 0.60 and path_increment >= 0.025 and np.isfinite(fav_auc) and fav_auc >= 0.58 and np.isfinite(rho) and rho >= 0.20)
        reasons.append(f"{side}: holdout release AUC full/path/base={full_auc:.3f}/{path_auc:.3f}/{base_auc:.3f}, path uplift={path_increment:.3f}, Swing uplift={swing_increment:.3f}, favorable AUC={fav_auc:.3f}, sweep-depth Spearman={rho:.3f}, top-zone quality={'PASS' if top_ok else 'FAIL'}.")
        if predictive and top_ok:
            promote.append(side)
    if promote:
        reasons.append("R02 is post-R01 development evidence only. Promotion means a separate causal limit-placement study, not live approval.")
        return f"PROMOTE_{'_AND_'.join(promote)}_TO_R02_1_LIMIT_PLACEMENT_STUDY", reasons
    max_auc = metrics.loc[metrics["period"].eq(config.holdout_period) & metrics["task"].eq("RELEASE_FULL"), "roc_auc"].max()
    if pd.notna(max_auc) and float(max_auc) >= 0.55:
        return "CONTINUE_R02_WITH_RANGE_FOOTPRINT_OI_INCREMENT", reasons + ["Spatial location signal exists but did not clear the baseline R02 gate; add independent microstructure data before abandoning the pool-location thesis."]
    return "STOP_R02_POOL_LOCATION_BASELINE_NO_PREDICTIVE_EDGE", reasons


def write_reports(*, config: LatentLiquidityPoolForecastConfig, source_gate: pd.DataFrame, frame: pd.DataFrame, audit_frame: pd.DataFrame, spatial_rows: int, spatial_snapshots: int, metrics: pd.DataFrame, deciles: pd.DataFrame, importance: pd.DataFrame, thresholds: pd.DataFrame, top: pd.DataFrame, causal: pd.DataFrame, source_rows_scanned: int, feature_columns: tuple[str, ...], skip_review_pack: bool) -> tuple[Path, str]:
    root = config.report_path; root.mkdir(parents=True, exist_ok=True)
    source_summary = pd.DataFrame([{"source_rows_scanned": source_rows_scanned, "spatial_rows": int(spatial_rows), "modeling_rows": len(frame), "full_lattice_audit_rows": len(audit_frame), "snapshots": int(spatial_snapshots), "release_rows_model_sample": int(frame["release_within_horizon"].sum()) if not frame.empty else 0, "favorable_release_rows_model_sample": int(frame["favorable_release"].sum()) if not frame.empty else 0}])
    label_summary = audit_frame.groupby(["period", "zone_side", "zone_distance_bp"], sort=True).agg(rows=("zone_id", "size"), touch_rate=(f"touch_{config.primary_horizon_minutes}m", "mean"), release_rate=("release_within_horizon", "mean"), favorable_release_rate=("favorable_release", "mean"), continuation_rate=("continuation_release", "mean"), mean_sweep_depth_bp=("sweep_depth_bp", "mean"), mean_reversal_room_bp=("reversal_after_extreme_bp", "mean")).reset_index()
    swing_importance = importance.groupby(["task", "feature_family"], sort=True)["importance_share"].sum().reset_index()
    tables = {
        "01_source_gate.csv": source_gate,
        "02_dataset_summary.csv": source_summary,
        "03_zone_label_summary.csv": label_summary,
        "04_model_metrics.csv": metrics,
        "05_pool_score_deciles.csv": deciles,
        "06_feature_importance.csv": importance,
        "07_feature_family_importance.csv": swing_importance,
        "08_calibration_thresholds.csv": thresholds,
        "09_top_zone_summary.csv": top,
        "10_causal_audit.csv": causal,
        "11_top_zone_sample.csv": audit_frame.loc[:, [c for c in (
            "zone_id", "decision_time", "period", "zone_side", "current_price", "zone_price", "zone_distance_bp",
            f"touch_{config.primary_horizon_minutes}m", "release_within_horizon", "favorable_release",
            "continuation_release", "time_to_release_minutes", "sweep_depth_bp", "reversal_after_extreme_bp",
            "p_touch", "p_release_baseline", "p_release_path", "p_release_full", "p_favorable_path",
            "p_favorable_full", "pred_sweep_depth_bp", "pred_reversal_room_bp", "pred_room_after_sweep_bp", "pool_score"
        ) if c in audit_frame.columns]].sort_values("pool_score", ascending=False).head(50_000),
    }
    for name, df in tables.items(): _write(df, root / name)
    decision, reasons = decide(metrics, top, causal, config)
    text = [f"# {MODEL_NAME} {STAGE_ID} decision", "", "## Primary decision", "", f"`{decision}`", "", "## Evidence", ""] + [f"- {x}" for x in reasons] + ["", "## Scope", "", "- R02 predicts price-zone liquidity before the release event; release-time burst information is label-only.", "- Swing is limited to all active 15m+ unswept levels and is tested only as incremental supplemental information.", "- R02 does not place orders and does not revive the stopped R01.3 post-confirmation entry branch.", "- This stage was designed after prior holdout review, so it is development evidence, not sealed validation or live approval."]
    (root / "12_decision.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    manifest = {"model": MODEL_NAME, "stage_id": STAGE_ID, "stage_name": STAGE_NAME, "decision": decision, "source_rows_scanned": source_rows_scanned, "spatial_rows": int(spatial_rows), "modeling_rows": len(frame), "full_lattice_audit_rows": len(audit_frame), "feature_columns": list(feature_columns), "sealed_validation_claim": False, "live_approved": False, "config": config.to_dict(), "reports": list(tables) + ["12_decision.md"]}
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text("Review R02 as a pre-event time-price liquidity-pool forecast. Focus on holdout release-on-touch AUC versus distance-only baseline, the no-Swing path model versus full model, favorable-release ranking, sweep-depth forecast, top-zone concentration, and causal availability. Swing is supplemental only; do not reinterpret this as a Swing sweep strategy.\n", encoding="utf-8")
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(report_dir=root, experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02", edge_id="RESEARCH_ONLY_LATENT_LIQUIDITY_POOL_LOCATION", title=f"{MODEL_NAME} {STAGE_ID}", decision_focus="pre-event pool location, release probability, favorable reversal and sweep depth"))
    return root, decision
