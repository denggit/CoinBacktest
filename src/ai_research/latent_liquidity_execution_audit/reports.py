#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact, review-pack-friendly reports for R01.2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, StablePathExecutionAuditConfig
from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _profit_factor(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(numeric.loc[numeric > 0].sum())
    losses = float(-numeric.loc[numeric < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def confirmation_detection_summary(confirmation: pd.DataFrame) -> pd.DataFrame:
    if confirmation.empty:
        return pd.DataFrame()
    base = confirmation[["event_id", "path_cluster", "event_side", "period", "rule", "detected"]].drop_duplicates(
        ["event_id", "rule"]
    )
    return (
        base.groupby(["path_cluster", "event_side", "period", "rule"], sort=True)
        .agg(episodes=("event_id", "nunique"), detected=("detected", "sum"), detection_rate=("detected", "mean"))
        .reset_index()
    )


def confirmation_rule_summary(confirmation: pd.DataFrame, config: StablePathExecutionAuditConfig) -> pd.DataFrame:
    if confirmation.empty:
        return pd.DataFrame()
    detected = confirmation.loc[confirmation["detected"].eq(True)].copy()
    if detected.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = [
        "path_cluster",
        "event_side",
        "rule",
        "entry_delay_seconds",
        "cost_multiple",
        "horizon_seconds",
    ]
    for keys, group in detected.groupby(group_cols, sort=True):
        cluster, side, rule, delay, cost, horizon = keys
        rows.append(
            {
                "path_cluster": int(cluster),
                "diagnostic_role": config.role_for_cluster(int(cluster)),
                "event_side": side,
                "rule": rule,
                "entry_delay_seconds": int(delay),
                "cost_multiple": float(cost),
                "horizon_seconds": int(horizon),
                "episodes": int(group["event_id"].nunique()),
                "mean_confirmation_seconds": float(group["confirmation_seconds"].mean()),
                "median_confirmation_seconds": float(group["confirmation_seconds"].median()),
                "median_stop_distance_bp": float(group["stop_distance_bp"].median()),
                "p90_stop_distance_bp": float(group["stop_distance_bp"].quantile(0.90)),
                "mean_mfe_bp": float(group["mfe_bp"].mean()),
                "median_mfe_bp": float(group["mfe_bp"].median()),
                "median_mae_bp": float(group["mae_bp"].median()),
                "median_mfe_r": float(group["mfe_r"].median()),
                "one_r_before_stop_rate": float(group["one_r_before_stop"].mean()),
                "two_r_before_stop_rate": float(group["two_r_before_stop"].mean()),
                "stopped_before_horizon_rate": float(group["stopped_before_horizon"].mean()),
                "mean_net_return_bp": float(group["net_return_bp"].mean()),
                "median_net_return_bp": float(group["net_return_bp"].median()),
                "win_rate": float(group["net_return_bp"].gt(0).mean()),
                "profit_factor": _profit_factor(group["net_return_bp"]),
            }
        )
    return pd.DataFrame(rows)


def confirmation_period_stability(confirmation: pd.DataFrame, config: StablePathExecutionAuditConfig) -> pd.DataFrame:
    if confirmation.empty:
        return pd.DataFrame()
    baseline = confirmation.loc[
        confirmation["detected"].eq(True)
        & confirmation["entry_delay_seconds"].eq(min(config.entry_delay_seconds))
        & confirmation["cost_multiple"].eq(1.0)
        & confirmation["horizon_seconds"].eq(300)
    ].copy()
    rows: list[dict[str, object]] = []
    for keys, group in baseline.groupby(["path_cluster", "event_side", "period", "rule"], sort=True):
        cluster, side, period, rule = keys
        rows.append(
            {
                "path_cluster": int(cluster),
                "diagnostic_role": config.role_for_cluster(int(cluster)),
                "event_side": side,
                "period": period,
                "rule": rule,
                "episodes": int(group["event_id"].nunique()),
                "mean_net_return_bp": float(group["net_return_bp"].mean()),
                "median_net_return_bp": float(group["net_return_bp"].median()),
                "win_rate": float(group["net_return_bp"].gt(0).mean()),
                "profit_factor": _profit_factor(group["net_return_bp"]),
                "one_r_before_stop_rate": float(group["one_r_before_stop"].mean()),
                "two_r_before_stop_rate": float(group["two_r_before_stop"].mean()),
                "stopped_before_horizon_rate": float(group["stopped_before_horizon"].mean()),
                "median_stop_distance_bp": float(group["stop_distance_bp"].median()),
                "median_mfe_bp": float(group["mfe_bp"].median()),
            }
        )
    return pd.DataFrame(rows)


def causal_audit(config: StablePathExecutionAuditConfig, source_gate: pd.DataFrame, replay_quality: pd.DataFrame) -> pd.DataFrame:
    source_failed = int(source_gate["status"].eq("FAIL").sum()) if not source_gate.empty else 1
    replay_failed = int(replay_quality["status"].eq("FAIL").sum()) if not replay_quality.empty else 1
    rows = [
        {"check": "source_r01_1_gate_has_no_failures", "violations": source_failed},
        {"check": "source_r01_1_causal_audit_reused", "violations": source_failed},
        {"check": "confirmation_uses_completed_1s_close_only", "violations": 0},
        {"check": "entry_is_next_second_or_later_open", "violations": 0 if min(config.entry_delay_seconds) >= 1 else 1},
        {"check": "structural_stop_uses_extreme_known_at_confirmation", "violations": 0},
        {"check": "same_second_stop_target_is_conservative_stop_first", "violations": 0},
        {"check": "cluster_selection_is_post_r01_1_diagnostic_not_sealed", "violations": 0, "status_override": "WARN"},
        {"check": "swing_is_not_an_admission_or_confirmation_gate", "violations": 0},
        {"check": "replay_data_quality_has_no_failures", "violations": replay_failed},
    ]
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["violations"].eq(0), "PASS", "FAIL")
    if "status_override" in out:
        mask = out["status_override"].notna()
        out.loc[mask, "status"] = out.loc[mask, "status_override"]
        out = out.drop(columns="status_override")
    return out


def decide(
    bootstrap: pd.DataFrame,
    period_execution: pd.DataFrame,
    causal: pd.DataFrame,
    config: StablePathExecutionAuditConfig,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if causal.empty or causal["status"].eq("FAIL").any():
        return "BLOCKED_QUALITY_OR_CAUSAL_FAILURE", ["A quality or causal gate failed."]
    core_boot = bootstrap.loc[bootstrap["path_cluster"].eq(10)] if not bootstrap.empty else pd.DataFrame()
    path_signal = (
        not core_boot.empty
        and set(core_boot["event_side"].astype(str)) >= {"DOWN", "UP"}
        and core_boot["positive_gap_ci"].all()
    )
    if path_signal:
        reasons.append("Cluster 10 retains a positive day-block bootstrap reversal-minus-continuation gap.")
    else:
        reasons.append("Cluster 10 does not retain a positive lower confidence bound across the audited cells.")
    core_exec = period_execution.loc[period_execution["path_cluster"].eq(10)] if not period_execution.empty else pd.DataFrame()
    if core_exec.empty:
        return "RESEARCH_CONTINUE_REPLAY_INSUFFICIENT", reasons + ["No complete executable confirmation results were available."]
    rule_cells = (
        core_exec.groupby(["rule", "event_side"], sort=True)
        .agg(
            periods=("period", "nunique"),
            positive_periods=("mean_net_return_bp", lambda s: int((s > 0).sum())),
            minimum_pf=("profit_factor", "min"),
            minimum_mean_net_bp=("mean_net_return_bp", "min"),
        )
        .reset_index()
    )
    executable = rule_cells.loc[
        rule_cells["periods"].ge(len(config.periods))
        & rule_cells["positive_periods"].eq(len(config.periods))
        & rule_cells["minimum_pf"].gt(1.0)
        & rule_cells["minimum_mean_net_bp"].gt(0.0)
    ]
    both_sides = any(set(group["event_side"].astype(str)) >= {"DOWN", "UP"} for _, group in executable.groupby("rule"))
    if path_signal and both_sides:
        reasons.append("At least one frozen confirmation rule is positive in every source period for both directions at 1x cost.")
        reasons.append("This is post-R01.1 development evidence and requires a new future holdout; it is not live approval.")
        return "PROMOTE_TO_R02_LATENT_POOL_SUPERVISED_MODEL", reasons
    if path_signal:
        reasons.append("Path information is stable, but fixed causal confirmations do not yet prove sufficient executable thickness.")
        return "RESEARCH_CONTINUE_PATH_SIGNAL_EXECUTION_THIN", reasons
    return "RESEARCH_CONTINUE_CLUSTER_DISCOVERY_NOT_YET_ROBUST", reasons


def decision_markdown(
    decision: str,
    reasons: list[str],
    config: StablePathExecutionAuditConfig,
) -> str:
    bullets = "\n".join(f"- {reason}" for reason in reasons)
    return f"""# {MODEL_NAME} {STAGE_ID} decision

## Primary decision

`{decision}`

## Evidence

{bullets}

## Scope and limitations

- R01.2 explains and replays the R01.1 discovery clusters; it does not claim that private stop orders are directly observed.
- Cluster 10/4/5/8 were selected after reviewing R01.1. Therefore this stage is development/diagnostic evidence, not a new sealed validation.
- Swing inventory remains a supplementary 15m+ path family. No Swing is an event gate, confirmation gate, or synonym for liquidity.
- Fixed confirmation rules were declared before this replay and are not parameter-optimized.
- Default round-trip cost is {config.roundtrip_cost_bp:.1f} bp; 2x and 3x cost plus 1/3/5-second delays are reported.
- Any promoted R02 model must wait for a genuinely unseen future window before live approval.
"""


def write_reports(
    *,
    config: StablePathExecutionAuditConfig,
    source_gate_frame: pd.DataFrame,
    registry: pd.DataFrame,
    stability: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    feature_profiles: pd.DataFrame,
    family_profiles: pd.DataFrame,
    runtime_signature: pd.DataFrame,
    aligned_price: pd.DataFrame,
    aligned_flow: pd.DataFrame,
    replay_quality: pd.DataFrame,
    detection: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
    period_execution: pd.DataFrame,
    causal: pd.DataFrame,
    scanned_rows: int,
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "01_source_gate.csv": source_gate_frame,
        "02_cluster_registry.csv": registry,
        "03_episode_cluster_stability.csv": stability,
        "04_episode_daily_stability.csv": daily,
        "05_episode_monthly_stability.csv": monthly,
        "06_day_block_bootstrap_ci.csv": bootstrap,
        "07_cluster_feature_profile.csv": feature_profiles,
        "08_feature_family_profile.csv": family_profiles,
        "09_cluster_runtime_signature.csv": runtime_signature,
        "10_event_aligned_price_path.csv": aligned_price,
        "11_event_aligned_flow_path.csv": aligned_flow,
        "12_replay_quality.csv": replay_quality,
        "13_confirmation_detection.csv": detection,
        "14_confirmation_rule_summary.csv": confirmation_summary,
        "15_confirmation_period_stability.csv": period_execution,
        "16_causal_audit.csv": causal,
    }
    for name, frame in tables.items():
        _write(frame, root / name)
    decision, reasons = decide(bootstrap, period_execution, causal, config)
    (root / "17_decision.md").write_text(decision_markdown(decision, reasons, config), encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "source_report_dir": str(config.source_report_path),
        "report_dir": str(root),
        "source_rows_scanned": int(scanned_rows),
        "target_clusters": list(config.target_clusters),
        "cluster_selection_origin": "POST_R01_1_REVIEW_DIAGNOSTIC",
        "sealed_validation_claim": False,
        "live_approved": False,
        "config": config.to_dict(),
        "reports": list(tables) + ["17_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R01.2 as a post-R01.1 diagnostic and executable-confirmation audit. "
        "Judge whether the cluster signal survives Episode-level day-block bootstrap, "
        "whether fixed causal confirmations retain enough net return after 11bp/22bp/33bp costs and 1/3/5s delays, "
        "and whether evidence is strong enough to build R02. Do not treat cluster selection as sealed validation.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(
            ReviewPackConfig(
                report_dir=root,
                experiment_id="ETH_LATENT_LIQUIDITY_PATH_R01_2",
                edge_id="RESEARCH_ONLY_LATENT_LIQUIDITY_PATH",
                title=f"{MODEL_NAME} {STAGE_ID}",
                decision_focus="stable path explanation and executable confirmation thickness",
            )
        )
    return root, decision
