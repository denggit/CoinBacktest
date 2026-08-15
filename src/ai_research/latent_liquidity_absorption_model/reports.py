#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact report writer and commercial-gate decision for R01.3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, AbsorptionModelConfig


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def causal_audit(
    snapshots: pd.DataFrame,
    feature_columns: tuple[str, ...],
    source_gate: pd.DataFrame,
    replay_quality: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    config: AbsorptionModelConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_failures = int(source_gate.get("status", pd.Series(dtype=str)).astype(str).eq("FAIL").sum())
    rows.append({"check": "r01_1_source_gate", "value": source_failures, "status": "PASS" if source_failures == 0 else "FAIL"})
    if snapshots.empty:
        return pd.DataFrame(rows + [{"check": "snapshot_rows", "value": 0, "status": "FAIL"}])
    available = pd.to_datetime(snapshots["feature_available_time"], errors="coerce")
    decision = pd.to_datetime(snapshots["decision_time"], errors="coerce")
    entry = pd.to_datetime(snapshots["entry_time"], errors="coerce")
    rows.extend(
        [
            {"check": "snapshot_rows", "value": len(snapshots), "status": "PASS"},
            {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
            {"check": "entry_is_next_second_open", "value": int(((entry - decision) != pd.Timedelta(seconds=1)).sum()), "status": "PASS" if ((entry - decision) == pd.Timedelta(seconds=1)).all() else "FAIL"},
            {"check": "future_columns_excluded_from_features", "value": int(sum(name.startswith(("future_", "tradeable_", "barrier_result_")) for name in feature_columns)), "status": "PASS" if not any(name.startswith(("future_", "tradeable_", "barrier_result_")) for name in feature_columns) else "FAIL"},
            {"check": "direct_swing_feature_gate", "value": int(any("swing" in name.lower() or "unswept" in name.lower() for name in feature_columns)), "status": "PASS" if not any("swing" in name.lower() or "unswept" in name.lower() for name in feature_columns) else "WARN"},
            {"check": "periods_are_frozen", "value": ",".join(sorted(snapshots["period"].astype(str).unique())), "status": "PASS" if set(snapshots["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
            {"check": "threshold_uses_calibration_only", "value": int(threshold_frame.get("holdout_used_for_threshold", pd.Series(dtype=bool)).fillna(True).astype(bool).sum()), "status": "PASS" if not threshold_frame.empty and not threshold_frame["holdout_used_for_threshold"].astype(bool).any() else "FAIL"},
        ]
    )
    if not replay_quality.empty:
        requested = float(replay_quality["requested_events"].sum())
        complete = float(replay_quality["complete_events"].sum())
        rate = complete / requested if requested else 0.0
        rows.append({"check": "snapshot_replay_completion_rate", "value": rate, "status": "PASS" if rate >= 0.95 else "WARN"})
    return pd.DataFrame(rows)


def label_summary(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    return (
        snapshots.groupby(["period", "event_side", "path_cluster", "decision_offset_seconds"], sort=True)
        .agg(
            rows=("event_id", "size"),
            episodes=("event_id", "nunique"),
            absorption_rate=("absorption_complete_target", "mean"),
            tradeable_rate=("tradeable_before_stop_target", "mean"),
            mean_additional_extension_bp=("future_additional_extension_bp", "mean"),
            mean_remaining_mfe_bp=("future_favorable_mfe_bp", "mean"),
            mean_remaining_mae_bp=("future_adverse_mae_bp", "mean"),
        )
        .reset_index()
    )


def selected_cluster_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base = trades.loc[trades["entry_delay_seconds"].eq(1) & trades["cost_multiple"].eq(1.0)]
    if base.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in base.groupby(["selection_model", "period", "event_side", "path_cluster"], sort=True):
        values = group["net_return_bp"].dropna()
        gains = float(values.loc[values > 0].sum())
        losses = float(-values.loc[values < 0].sum())
        rows.append(
            {
                "selection_model": keys[0],
                "period": keys[1],
                "event_side": keys[2],
                "path_cluster": int(keys[3]),
                "trades": int(len(values)),
                "mean_net_bp": float(values.mean()) if len(values) else np.nan,
                "win_rate": float((values > 0).mean()) if len(values) else np.nan,
                "profit_factor": gains / losses if losses > 0 else np.inf if gains > 0 else np.nan,
                "mean_decision_offset_seconds": float(group["decision_offset_seconds"].mean()),
                "mean_selection_score": float(group["selection_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def decide(
    metrics: pd.DataFrame,
    trades_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    causal: pd.DataFrame,
    config: AbsorptionModelConfig,
) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].eq("FAIL").any():
        return "BLOCKED_R01_3_QUALITY_OR_CAUSAL_FAILURE", ["A source, replay, or causal gate failed."]
    if trades_summary.empty:
        trades_summary = pd.DataFrame(
            columns=[
                "selection_model", "period", "event_side", "entry_delay_seconds", "cost_multiple",
                "trades", "mean_net_bp", "profit_factor", "top10_removed_mean_net_bp",
            ]
        )
    if monthly.empty:
        monthly = pd.DataFrame(columns=["selection_model", "period", "event_side", "month", "sum_net_bp"])
    reasons: list[str] = []
    passed_sides: list[str] = []
    for side in ("DOWN", "UP"):
        holdout_full = metrics.loc[
            metrics["period"].eq(config.holdout_period)
            & metrics["event_side"].eq(side)
            & metrics["task"].eq("TRADEABLE_FULL")
        ]
        holdout_base = metrics.loc[
            metrics["period"].eq(config.holdout_period)
            & metrics["event_side"].eq(side)
            & metrics["task"].eq("TRADEABLE_BASELINE")
        ]
        base_exec = trades_summary.loc[
            trades_summary["selection_model"].eq("FULL")
            & trades_summary["event_side"].eq(side)
            & trades_summary["entry_delay_seconds"].eq(1)
            & trades_summary["cost_multiple"].eq(1.0)
            & trades_summary["period"].isin([config.calibration_period, config.holdout_period])
        ]
        stress = trades_summary.loc[
            trades_summary["selection_model"].eq("FULL")
            & trades_summary["event_side"].eq(side)
            & trades_summary["entry_delay_seconds"].eq(1)
            & trades_summary["cost_multiple"].eq(2.0)
            & trades_summary["period"].eq(config.holdout_period)
        ]
        if holdout_full.empty or holdout_base.empty or len(base_exec) < 2 or stress.empty:
            reasons.append(f"{side}: insufficient complete validation/holdout model or execution cells.")
            continue
        auc = float(holdout_full.iloc[0].get("roc_auc", np.nan))
        baseline_auc = float(holdout_base.iloc[0].get("roc_auc", np.nan))
        exec_ok = bool(
            (base_exec["trades"] >= 100).all()
            and (base_exec["mean_net_bp"] >= 3.0).all()
            and (base_exec["profit_factor"] >= 1.20).all()
            and (base_exec["top10_removed_mean_net_bp"] > 0.0).all()
        )
        stress_ok = bool(float(stress.iloc[0]["mean_net_bp"]) >= 0.0 and float(stress.iloc[0]["profit_factor"]) >= 1.0)
        side_monthly = monthly.loc[
            monthly["selection_model"].eq("FULL")
            & monthly["event_side"].eq(side)
            & monthly["period"].isin([config.calibration_period, config.holdout_period])
        ]
        monthly_cells = side_monthly.groupby("period", sort=True)["sum_net_bp"].agg(
            months="size", positive_month_rate=lambda values: float((values > 0).mean())
        ).reset_index()
        monthly_ok = bool(
            set(monthly_cells["period"].astype(str)) >= {config.calibration_period, config.holdout_period}
            and (monthly_cells["months"] >= 3).all()
            and (monthly_cells["positive_month_rate"] >= 0.60).all()
        )
        predictive_ok = bool(np.isfinite(auc) and np.isfinite(baseline_auc) and auc >= 0.58 and auc - baseline_auc >= 0.02)
        reasons.append(
            f"{side}: holdout AUC={auc:.3f}, baseline={baseline_auc:.3f}, "
            f"validation/holdout execution={'PASS' if exec_ok else 'FAIL'}, "
            f"positive-month gate={'PASS' if monthly_ok else 'FAIL'}, 2x cost={'PASS' if stress_ok else 'FAIL'}."
        )
        if predictive_ok and exec_ok and monthly_ok and stress_ok:
            passed_sides.append(side)
    if passed_sides:
        sides = "_AND_".join(passed_sides)
        reasons.append("R01.3 is still post-R01.1 development evidence; promotion means formal backtest only, not live approval.")
        return f"PROMOTE_{sides}_TO_R02_FORMAL_STRATEGY_BACKTEST", reasons
    predictive = metrics.loc[
        metrics["period"].eq(config.holdout_period) & metrics["task"].eq("TRADEABLE_FULL")
    ]
    if not predictive.empty and float(predictive["roc_auc"].max()) >= 0.55:
        reasons.append("The supervised absorption layer retains some prediction, but it does not clear the frozen commercial execution gate.")
    else:
        reasons.append("The absorption/remaining-space model does not retain sufficient holdout prediction.")
    reasons.append("This was the declared final commercial gate for this path family; do not continue parameter patching after failure.")
    return "STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE", reasons


def decision_markdown(decision: str, reasons: list[str], config: AbsorptionModelConfig) -> str:
    bullets = "\n".join(f"- {reason}" for reason in reasons)
    return f"""# {MODEL_NAME} {STAGE_ID} decision

## Primary decision

`{decision}`

## Evidence

{bullets}

## Frozen scope

- R01.3 trains only on `{config.train_period}`.
- The score threshold is frozen from `{config.calibration_period}` at q{int(config.selection_quantile * 100)}.
- `{config.holdout_period}` is evaluation-only inside this stage.
- Cluster 10/4/5/8 remain post-R01.1 discovery strata, so this is not a genuinely new sealed validation.
- Swing is not used as a direct R01.3 feature, admission rule, confirmation rule, or synonym for liquidity.
- Entry is next-second open; stop is the decision-time known extreme plus {config.structural_stop_buffer_bp:.1f} bp.
- Default round-trip cost is {config.roundtrip_cost_bp:.1f} bp; 2x/3x costs and 1/3/5-second delays are reported.
- Promotion permits only a formal strategy backtest and a later genuinely unseen validation. It does not permit live capital.
"""


def write_reports(
    *,
    config: AbsorptionModelConfig,
    source_gate: pd.DataFrame,
    replay_quality: pd.DataFrame,
    snapshots: pd.DataFrame,
    label_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    deciles: pd.DataFrame,
    importance: pd.DataFrame,
    thresholds: pd.DataFrame,
    selected_trades: pd.DataFrame,
    trade_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    causal: pd.DataFrame,
    source_rows_scanned: int,
    feature_columns: tuple[str, ...],
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    compact_trades = selected_trades.sort_values(["selection_model", "entry_time", "entry_delay_seconds", "cost_multiple"]).head(100_000)
    tables = {
        "01_source_gate.csv": source_gate,
        "02_snapshot_replay_quality.csv": replay_quality,
        "03_snapshot_label_summary.csv": label_frame,
        "04_model_metrics.csv": metrics,
        "05_score_decile_lift.csv": deciles,
        "06_feature_importance.csv": importance,
        "07_calibration_thresholds.csv": thresholds,
        "08_selected_trade_summary.csv": trade_summary,
        "09_selected_cluster_summary.csv": cluster_summary,
        "10_selected_monthly_summary.csv": monthly,
        "11_selected_trade_sample.csv": compact_trades,
        "12_causal_audit.csv": causal,
    }
    for name, frame in tables.items():
        _write(frame, root / name)
    decision, reasons = decide(metrics, trade_summary, monthly, causal, config)
    (root / "13_decision.md").write_text(decision_markdown(decision, reasons, config), encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "source_report_dir": str(config.source_report_path),
        "report_dir": str(root),
        "source_rows_scanned": int(source_rows_scanned),
        "snapshot_rows": int(len(snapshots)),
        "episodes": int(snapshots["event_id"].nunique()) if not snapshots.empty else 0,
        "feature_columns": list(feature_columns),
        "cluster_selection_origin": "POST_R01_1_REVIEW_DIAGNOSTIC",
        "sealed_validation_claim": False,
        "live_approved": False,
        "config": config.to_dict(),
        "reports": list(tables) + ["13_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R01.3 as the declared final commercial gate for the latent-liquidity path family. "
        "Check holdout prediction versus the cluster/side/checkpoint baseline, calibration-only score thresholds, "
        "first causal snapshot per Episode, 11/22/33bp costs, 1/3/5-second delay, top-10 removal and monthly concentration. "
        "Do not treat this post-R01.1 stage as sealed validation or live approval.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(
            ReviewPackConfig(
                report_dir=root,
                experiment_id="ETH_LATENT_LIQUIDITY_PATH_R01_3",
                edge_id="RESEARCH_ONLY_LATENT_LIQUIDITY_PATH",
                title=f"{MODEL_NAME} {STAGE_ID}",
                decision_focus="absorption completion, remaining space and commercial execution gate",
            )
        )
    return root, decision
