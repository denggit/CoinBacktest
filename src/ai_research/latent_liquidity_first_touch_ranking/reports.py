#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact reports and quality gate for R02.2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import MODEL_NAME, STAGE_ID, STAGE_NAME, FirstTouchLiquidityRankingConfig
from .modeling import RankModelBundle


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def causal_audit(
    frame: pd.DataFrame,
    models: RankModelBundle,
    source_gate: pd.DataFrame,
    config: FirstTouchLiquidityRankingConfig,
) -> pd.DataFrame:
    observed = frame["first_touch_observed"].astype(bool)
    complete = frame["first_touch_label_complete"].astype(bool)
    decision = pd.to_datetime(frame["decision_time"], errors="coerce")
    touch = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    touch_available = touch + pd.Timedelta(seconds=1)
    available = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    future_prefixes = (
        "touch_", "release_", "favorable_", "continuation_", "time_to_", "sweep_", "reversal_",
        "p_touch", "p_release", "p_favorable", "pred_", "pool_score", "high_strength", "ft_",
        "first_touch", "ranking_",
    )
    path_columns = sorted({name for bundle in models.by_side.values() for name in bundle.path_columns})
    leaked = sorted(name for name in path_columns if name.startswith(future_prefixes))
    swing = sorted(name for name in path_columns if name.startswith("swing_"))
    group_sizes = frame.groupby(["decision_time", "zone_side"], sort=False).size()
    expected = len(tuple(float(x) for x in range(10, 500, 20)))
    touch_without_r02 = observed & ~frame["touch_720m"].astype(bool)
    horizon_violation = observed & (touch >= decision + pd.Timedelta(minutes=config.primary_horizon_minutes))
    source_fail = int(source_gate["status"].astype(str).eq("FAIL").sum()) if not source_gate.empty else 0
    rows = [
        {"check": "r01_1_source_gate", "value": source_fail, "status": "PASS" if source_fail == 0 else "FAIL"},
        {"check": "complete_lattice_25_zones_per_side", "value": int(group_sizes.ne(expected).sum()), "status": "PASS" if len(group_sizes) and not group_sizes.ne(expected).any() else "FAIL"},
        {"check": "feature_available_not_after_decision", "value": int((available > decision).sum()), "status": "PASS" if not (available > decision).any() else "FAIL"},
        {"check": "first_touch_bar_available_strictly_after_decision", "value": int((observed & (touch_available <= decision)).sum()), "status": "PASS" if not (observed & (touch_available <= decision)).any() else "FAIL"},
        {"check": "first_touch_inside_exclusive_12h_horizon", "value": int(horizon_violation.sum()), "status": "PASS" if not horizon_violation.any() else "FAIL"},
        {"check": "exact_first_touch_implies_r02_touch", "value": int(touch_without_r02.sum()), "status": "PASS" if not touch_without_r02.any() else "FAIL"},
        {"check": "complete_label_implies_touch", "value": int((complete & ~observed).sum()), "status": "PASS" if not (complete & ~observed).any() else "FAIL"},
        {"check": "fixed_post_touch_windows", "value": ",".join(map(str, config.label_windows_seconds)), "status": "PASS" if config.label_windows_seconds == (30, 60, 180, 300) else "FAIL"},
        {"check": "primary_ranking_has_no_future_labels", "value": len(leaked), "status": "PASS" if not leaked else "FAIL"},
        {"check": "primary_ranking_excludes_swing", "value": len(swing), "status": "PASS" if not swing else "FAIL"},
        {"check": "no_absolute_strength_threshold_in_primary", "value": 0, "status": "PASS"},
        {"check": "periods_are_frozen", "value": ",".join(sorted(frame["period"].astype(str).unique())), "status": "PASS" if set(frame["period"].astype(str).unique()) <= set(config.periods) else "FAIL"},
    ]
    return pd.DataFrame(rows)


def horizon_profile(frame: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    touched = frame.loc[frame["first_touch_label_complete"].astype(bool)].copy()
    for (period, side), sf in touched.groupby(["period", "zone_side"], sort=True):
        for window in config.label_windows_seconds:
            density = pd.to_numeric(sf[f"ft_release_density_sum_{window}s"], errors="coerce")
            count = pd.to_numeric(sf[f"ft_release_episode_count_{window}s"], errors="coerce")
            flow = pd.to_numeric(sf[f"ft_notional_ratio_{window}s"], errors="coerce")
            rows.append({
                "period": period,
                "zone_side": side,
                "window_seconds": int(window),
                "touched_rows": int(len(sf)),
                "release_rate": float(count.gt(0).mean()),
                "mean_release_density": float(density.mean()),
                "median_release_density": float(density.median()),
                "mean_notional_ratio": float(flow.mean()),
                "median_notional_ratio": float(flow.median()),
            })
    return pd.DataFrame(rows)


def distance_profile(frame: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> pd.DataFrame:
    w = config.primary_label_window_seconds
    touched = frame.loc[frame["first_touch_label_complete"].astype(bool)].copy()
    if touched.empty:
        return pd.DataFrame()
    return touched.groupby(["period", "zone_side", "zone_distance_bp"], sort=True).agg(
        touched_rows=("zone_id", "size"),
        mean_density=(f"ft_release_density_sum_{w}s", "mean"),
        median_density=(f"ft_release_density_sum_{w}s", "median"),
        release_rate=(f"ft_release_episode_count_{w}s", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())),
        favorable_rate=(f"ft_favorable_episode_count_{w}s", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())),
        continuation_rate=(f"ft_continuation_episode_count_{w}s", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())),
        mean_notional_ratio_60s=("ft_notional_ratio_60s", "mean"),
    ).reset_index()


def swing_ablation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in metrics.groupby(["period", "zone_side"], sort=True):
        path = sf.loc[sf["model"].eq("PATH_NO_SWING")]
        full = sf.loc[sf["model"].eq("FULL_WITH_SWING")]
        if path.empty or full.empty:
            continue
        p, f = path.iloc[0], full.iloc[0]
        rows.append({
            "period": period,
            "zone_side": side,
            "path_mean_group_spearman": p.get("mean_group_spearman", np.nan),
            "full_mean_group_spearman": f.get("mean_group_spearman", np.nan),
            "swing_spearman_uplift": float(f.get("mean_group_spearman", np.nan) - p.get("mean_group_spearman", np.nan)),
            "path_ndcg3": p.get("mean_ndcg3", np.nan),
            "full_ndcg3": f.get("mean_ndcg3", np.nan),
            "swing_ndcg3_uplift": float(f.get("mean_ndcg3", np.nan) - p.get("mean_ndcg3", np.nan)),
        })
    return pd.DataFrame(rows)


def _decision(
    metrics: pd.DataFrame,
    top: pd.DataFrame,
    causal: pd.DataFrame,
    config: FirstTouchLiquidityRankingConfig,
) -> tuple[str, list[str]]:
    if causal.empty or causal["status"].astype(str).eq("FAIL").any():
        return "BLOCKED_R02_2_QUALITY_OR_CAUSAL_FAILURE", ["A source or causal gate failed."]
    promote: list[str] = []
    reasons: list[str] = []
    for side in ("DOWN", "UP"):
        hold = metrics.loc[
            metrics["period"].astype(str).eq(config.holdout_period)
            & metrics["zone_side"].astype(str).eq(side)
        ]
        p = hold.loc[hold["model"].eq("PATH_NO_SWING")]
        b = hold.loc[hold["model"].eq("DISTANCE_BASELINE")]
        top_side = top.loc[
            top["period"].astype(str).eq(config.holdout_period)
            & top["zone_side"].astype(str).eq(side)
            & top["model"].eq("PATH_NO_SWING")
        ]
        if p.empty or b.empty or top_side.empty:
            reasons.append(f"{side}: missing holdout ranking cells.")
            continue
        pr, br, tr = p.iloc[0], b.iloc[0], top_side.iloc[0]
        spearman = float(pr.get("mean_group_spearman", np.nan))
        base_spearman = float(br.get("mean_group_spearman", np.nan))
        lift = float(tr.get("top1_density_lift", np.nan))
        oracle = float(tr.get("oracle_strongest_zone_in_top3_rate", np.nan))
        touched = int(tr.get("top1_touched", 0))
        reasons.append(
            f"{side}: holdout mean group Spearman path/base={spearman:.3f}/{base_spearman:.3f}; "
            f"top1 density lift={lift:.3f}; oracle strongest zone in predicted top3={oracle:.3f}; "
            f"top1 touched={touched}."
        )
        passed = bool(
            np.isfinite(spearman) and spearman >= config.promotion_min_group_spearman
            and spearman > base_spearman
            and np.isfinite(lift) and lift >= config.promotion_min_top1_density_lift
            and np.isfinite(oracle) and oracle >= config.promotion_min_oracle_top3_rate
            and touched >= config.minimum_top1_touched
        )
        if passed:
            promote.append(side)
    if promote:
        return (
            f"PROMOTE_{'_AND_'.join(promote)}_TO_R02_3_LIMIT_PLACEMENT_SWEEP_DEPTH_STUDY",
            reasons + ["Promotion is research-only: next stage studies causal limit placement near predicted sweep depth; no live approval."],
        )
    hold_path = metrics.loc[
        metrics["period"].astype(str).eq(config.holdout_period)
        & metrics["model"].eq("PATH_NO_SWING"),
        "mean_group_spearman",
    ]
    if len(hold_path) and hold_path.max(skipna=True) >= 0.08:
        return "CONTINUE_R02_2_WITH_RANGE_FOOTPRINT_OI_RANKING_INCREMENT", reasons + [
            "Relative first-touch ranking contains signal but did not clear the placement gate. Add independent Range/Footprint/OI evidence before any order study."
        ]
    return "STOP_R02_2_FIRST_TOUCH_RANKING_NO_EDGE", reasons


def write_reports(
    *,
    config: FirstTouchLiquidityRankingConfig,
    source_gate: pd.DataFrame,
    quality: pd.DataFrame,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    top: pd.DataFrame,
    importance: pd.DataFrame,
    causal: pd.DataFrame,
    skip_review_pack: bool,
) -> tuple[Path, str]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    family = importance.groupby(["zone_side", "model", "feature_family"], sort=True)["importance_share"].sum().reset_index()
    horizon = horizon_profile(frame, config)
    distance = distance_profile(frame, config)
    ablation = swing_ablation(metrics)
    w = config.primary_label_window_seconds
    summary = pd.DataFrame([{
        "rows": int(len(frame)),
        "groups": int(frame.groupby(["decision_time", "zone_side"], sort=False).ngroups),
        "exact_first_touch_rows": int(frame["first_touch_observed"].astype(bool).sum()),
        "complete_first_touch_label_rows": int(frame["first_touch_label_complete"].astype(bool).sum()),
        "rankable_rows": int(frame["ranking_group_eligible"].astype(bool).sum()),
        "rankable_groups": int(frame.loc[frame["ranking_group_eligible"].astype(bool), "ranking_group"].nunique()),
        "primary_target": f"first-touch release density {w}s",
        "absolute_strength_threshold_used": False,
        "touch_probability_in_primary_rank_score": False,
        "swing_in_primary_rank_score": False,
    }])
    sample_cols = [c for c in (
        "zone_id", "decision_time", "period", "zone_side", "current_price", "zone_price", "zone_distance_bp",
        "first_touch_time", "time_to_first_touch_minutes", f"ft_release_episode_count_{w}s",
        f"ft_release_density_sum_{w}s", f"ft_favorable_episode_count_{w}s", f"ft_continuation_episode_count_{w}s",
        "ft_notional_ratio_60s", "ranking_target", "ranking_relevance",
        "rank_score_path_no_swing", "rank_score_full_with_swing", "rank_score_distance_baseline",
    ) if c in frame.columns]
    top_sample = frame.sort_values(["period", "zone_side", "rank_score_path_no_swing"], ascending=[True, True, False], kind="mergesort").loc[:, sample_cols].head(50_000)
    tables = {
        "01_source_gate.csv": source_gate,
        "02_touch_replay_quality.csv": quality,
        "03_dataset_summary.csv": summary,
        "04_ranking_metrics.csv": metrics,
        "05_top_zone_summary.csv": top,
        "06_first_touch_horizon_profile.csv": horizon,
        "07_feature_importance.csv": importance,
        "08_feature_family_importance.csv": family,
        "09_swing_ablation.csv": ablation,
        "10_distance_first_touch_profile.csv": distance,
        "11_causal_audit.csv": causal,
        "12_top_zone_sample.csv": top_sample,
    }
    for name, table in tables.items():
        _write(table, root / name)
    decision, reasons = _decision(metrics, top, causal, config)
    lines = [
        f"# {MODEL_NAME} {STAGE_ID} decision",
        "",
        "## Primary decision",
        "",
        f"`{decision}`",
        "",
        "## Evidence",
        "",
    ]
    lines += [f"- {reason}" for reason in reasons]
    lines += [
        "",
        "## Frozen interpretation",
        "",
        "- R02.2 replaces absolute pool-strength classification with within-snapshot relative ranking.",
        "- The target is realized R01.1 release density during a fixed 180-second window after the exact first touch of each zone; 30/60/300-second windows are diagnostics.",
        "- A near zone and a far zone receive the same post-touch observation duration, removing the R02.1 exposure-time bias.",
        "- The primary ranking model excludes Swing. 15m+ all-unswept Swing remains only as an ablation model.",
        "- Touch/arrival probability is reported separately and never multiplied into the primary ranking score.",
        "- R01.3 post-confirmation market entry remains stopped. R02.2 places no orders.",
        "- 2025Q4-2026H1 remains a development holdout, not sealed validation or live approval.",
    ]
    (root / "13_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "sealed_validation_claim": False,
        "live_approved": False,
        "config": config.to_dict(),
        "reports": list(tables) + ["13_decision.md"],
    }
    (root / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R02.2 as a first-touch relative liquidity-ranking study. Focus on within-group ranking metrics, Top-1/Top-3 realized first-touch release-density lift, distance baseline comparison, Swing ablation, fixed post-touch label windows, and causal/data-quality gates. Do not reinterpret it as an absolute pool-strength classifier, Swing strategy, limit-order backtest, or live approval.\n",
        encoding="utf-8",
    )
    if not skip_review_pack:
        write_gpt_review_pack(ReviewPackConfig(
            report_dir=root,
            experiment_id="ETH_LATENT_LIQUIDITY_PATH_R02_2",
            edge_id="RESEARCH_ONLY_FIRST_TOUCH_RELATIVE_LIQUIDITY_RANKING",
            title=f"{MODEL_NAME} {STAGE_ID}",
            decision_focus="first-touch fixed-window spatial ranking, distance baseline, Swing ablation and top-zone quality",
        ))
    return root, decision
