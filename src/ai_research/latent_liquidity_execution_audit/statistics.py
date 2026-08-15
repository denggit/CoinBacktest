#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Episode-level stability and path-profile diagnostics for R01.2."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import StablePathExecutionAuditConfig


def cluster_registry(config: StablePathExecutionAuditConfig, episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in config.target_clusters:
        subset = episodes.loc[episodes.get("path_cluster", pd.Series(dtype=int)).eq(cluster)] if not episodes.empty else pd.DataFrame()
        rows.append(
            {
                "path_cluster": int(cluster),
                "diagnostic_role": config.role_for_cluster(cluster),
                "selection_origin": "POST_R01_1_REVIEW",
                "selection_is_sealed_validation": False,
                "episode_rows_available": len(subset),
                "event_sides": ",".join(sorted(subset.get("event_side", pd.Series(dtype=str)).astype(str).unique())),
                "periods": ",".join(sorted(subset.get("period", pd.Series(dtype=str)).astype(str).unique())),
                "live_rule": False,
            }
        )
    return pd.DataFrame(rows)


def _outcome_rates(group: pd.DataFrame) -> dict[str, float]:
    outcome = group["outcome_type"].astype(str)
    return {
        "favorable_reversal_rate": float(group["favorable_reversal"].astype(bool).mean()) if len(group) else np.nan,
        "shallow_immediate_rate": float(outcome.eq("SHALLOW_IMMEDIATE_REVERSAL").mean()) if len(group) else np.nan,
        "deep_immediate_rate": float(outcome.eq("DEEP_IMMEDIATE_REVERSAL").mean()) if len(group) else np.nan,
        "extend_stabilize_rate": float(outcome.eq("EXTEND_STABILIZE_REVERSAL").mean()) if len(group) else np.nan,
        "accept_continuation_rate": float(outcome.eq("ACCEPT_CONTINUATION").mean()) if len(group) else np.nan,
        "mixed_rate": float(outcome.eq("MIXED_OR_UNRESOLVED").mean()) if len(group) else np.nan,
    }


def stability_scorecard(episodes: pd.DataFrame, config: StablePathExecutionAuditConfig) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in episodes.groupby(["path_cluster", "event_side", "period"], sort=True):
        cluster, side, period = keys
        rates = _outcome_rates(group)
        rows.append(
            {
                "path_cluster": int(cluster),
                "diagnostic_role": config.role_for_cluster(int(cluster)),
                "event_side": side,
                "period": period,
                "episodes": len(group),
                **rates,
                "reversal_minus_continuation": rates["favorable_reversal_rate"] - rates["accept_continuation_rate"],
                "mean_extension_bp": float(group["future_extension_bp"].mean()),
                "median_extension_bp": float(group["future_extension_bp"].median()),
                "mean_reversal_after_extreme_bp": float(group["future_reversal_after_extreme_bp"].mean()),
                "median_time_to_extreme_seconds": float(group["future_time_to_extreme_seconds"].median()),
                "minimum_episode_gate": "PASS" if len(group) >= config.minimum_period_episodes else "WARN",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    stable_rows: list[dict[str, object]] = []
    for (cluster, side), group in result.groupby(["path_cluster", "event_side"], sort=True):
        period_map = group.set_index("period")
        all_periods = all(period in period_map.index for period in config.periods)
        min_gap = float(group["reversal_minus_continuation"].min()) if len(group) else np.nan
        stable_rows.append(
            {
                "path_cluster": int(cluster),
                "event_side": side,
                "all_frozen_periods_present": all_periods,
                "positive_gap_every_period": bool(all_periods and (group["reversal_minus_continuation"] > 0).all()),
                "minimum_period_gap": min_gap,
            }
        )
    stable = pd.DataFrame(stable_rows)
    return result.merge(stable, on=["path_cluster", "event_side"], how="left", validate="many_to_one")


def _calendar_summary(episodes: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    frame = episodes.copy()
    frame["calendar_bucket"] = pd.to_datetime(frame["event_time"]).dt.to_period(frequency).astype(str)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["path_cluster", "event_side", "period", "calendar_bucket"], sort=True):
        cluster, side, period, bucket = keys
        rates = _outcome_rates(group)
        rows.append(
            {
                "path_cluster": int(cluster),
                "event_side": side,
                "period": period,
                "calendar_bucket": bucket,
                "episodes": len(group),
                **rates,
                "reversal_minus_continuation": rates["favorable_reversal_rate"] - rates["accept_continuation_rate"],
            }
        )
    return pd.DataFrame(rows)


def daily_stability(episodes: pd.DataFrame) -> pd.DataFrame:
    return _calendar_summary(episodes, "D")


def monthly_stability(episodes: pd.DataFrame) -> pd.DataFrame:
    return _calendar_summary(episodes, "M")


def block_bootstrap_ci(
    daily: pd.DataFrame,
    config: StablePathExecutionAuditConfig,
) -> pd.DataFrame:
    """Bootstrap calendar days, preserving within-day clustered episode dependence."""
    if daily.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(config.random_state)
    rows: list[dict[str, object]] = []
    for keys, group in daily.groupby(["path_cluster", "event_side", "period"], sort=True):
        cluster, side, period = keys
        group = group.loc[group["episodes"].gt(0)].reset_index(drop=True)
        if group.empty:
            continue
        weights = group["episodes"].to_numpy(dtype=float)
        favorable = group["favorable_reversal_rate"].to_numpy(dtype=float)
        continuation = group["accept_continuation_rate"].to_numpy(dtype=float)
        gaps = np.empty(config.bootstrap_repetitions, dtype=np.float64)
        favs = np.empty(config.bootstrap_repetitions, dtype=np.float64)
        conts = np.empty(config.bootstrap_repetitions, dtype=np.float64)
        for rep in range(config.bootstrap_repetitions):
            sampled = rng.integers(0, len(group), size=len(group))
            sampled_weights = weights[sampled]
            denominator = sampled_weights.sum()
            favs[rep] = np.sum(favorable[sampled] * sampled_weights) / denominator
            conts[rep] = np.sum(continuation[sampled] * sampled_weights) / denominator
            gaps[rep] = favs[rep] - conts[rep]
        rows.append(
            {
                "path_cluster": int(cluster),
                "event_side": side,
                "period": period,
                "calendar_days": len(group),
                "episodes": int(weights.sum()),
                "favorable_reversal_rate": float(np.average(favorable, weights=weights)),
                "accept_continuation_rate": float(np.average(continuation, weights=weights)),
                "gap_point_estimate": float(np.average(favorable - continuation, weights=weights)),
                "gap_ci_2p5": float(np.quantile(gaps, 0.025)),
                "gap_ci_50": float(np.quantile(gaps, 0.50)),
                "gap_ci_97p5": float(np.quantile(gaps, 0.975)),
                "favorable_ci_2p5": float(np.quantile(favs, 0.025)),
                "favorable_ci_97p5": float(np.quantile(favs, 0.975)),
                "continuation_ci_2p5": float(np.quantile(conts, 0.025)),
                "continuation_ci_97p5": float(np.quantile(conts, 0.975)),
                "positive_gap_ci": bool(np.quantile(gaps, 0.025) > 0.0),
            }
        )
    return pd.DataFrame(rows)


def feature_family(name: str) -> str:
    if name.startswith("unswept_"):
        return "SWING_INVENTORY_SUPPLEMENT"
    if "notional" in name or "turnover" in name or "trades" in name or "buy_share" in name or "delta" in name:
        return "FLOW_AND_TURNOVER"
    if "pressure_without_progress" in name or "impact_bp_per_million" in name:
        return "PRESSURE_AND_IMPACT_EFFICIENCY"
    if "overlap" in name or "residency" in name or "sign_changes" in name:
        return "PRICE_RESIDENCY_AND_CHOP"
    if "efficiency" in name or "travel" in name:
        return "PATH_EFFICIENCY"
    if "ret_" in name or "excursion" in name or "drawdown" in name or "rally" in name or "range" in name:
        return "PRICE_PATH_AND_RANGE"
    return "OTHER_PATH_CONTEXT"


def _safe_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.quantile(q)) if len(numeric) else np.nan


def cluster_feature_profiles(
    samples: dict[tuple[object, ...], pd.DataFrame],
    feature_columns: Iterable[str],
    config: StablePathExecutionAuditConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in config.target_clusters:
        for side in ("DOWN", "UP"):
            for period in config.periods:
                cluster_frame = samples.get((cluster, side, period), pd.DataFrame())
                base_frame = samples.get(("BASE", side, period), pd.DataFrame())
                if cluster_frame.empty or base_frame.empty:
                    continue
                for feature in feature_columns:
                    if feature not in cluster_frame or feature not in base_frame:
                        continue
                    c = pd.to_numeric(cluster_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    b = pd.to_numeric(base_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    c_nonnull = c.dropna()
                    b_nonnull = b.dropna()
                    if len(c_nonnull) < 20 or len(b_nonnull) < 20:
                        continue
                    b_q25 = _safe_quantile(b_nonnull, 0.25)
                    b_q75 = _safe_quantile(b_nonnull, 0.75)
                    robust_scale = (b_q75 - b_q25) / 1.349 if np.isfinite(b_q75 - b_q25) else np.nan
                    if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
                        continue
                    c_median = float(c_nonnull.median())
                    b_median = float(b_nonnull.median())
                    effect = (c_median - b_median) / robust_scale
                    rows.append(
                        {
                            "path_cluster": int(cluster),
                            "diagnostic_role": config.role_for_cluster(cluster),
                            "event_side": side,
                            "period": period,
                            "feature_family": feature_family(feature),
                            "feature": feature,
                            "cluster_sample_rows": len(c_nonnull),
                            "baseline_sample_rows": len(b_nonnull),
                            "cluster_median": c_median,
                            "baseline_median": b_median,
                            "baseline_q25": b_q25,
                            "baseline_q75": b_q75,
                            "robust_effect": float(effect),
                            "abs_robust_effect": float(abs(effect)),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["within_cluster_rank"] = result.groupby(
        ["path_cluster", "event_side", "period"], sort=False
    )["abs_robust_effect"].rank(method="first", ascending=False)
    return result.sort_values(
        ["path_cluster", "event_side", "period", "abs_robust_effect"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def feature_family_profiles(feature_profiles: pd.DataFrame) -> pd.DataFrame:
    if feature_profiles.empty:
        return pd.DataFrame()
    top = feature_profiles.loc[feature_profiles["within_cluster_rank"].le(40)].copy()
    return (
        top.groupby(["path_cluster", "event_side", "period", "feature_family"], sort=True)
        .agg(
            features=("feature", "size"),
            median_abs_robust_effect=("abs_robust_effect", "median"),
            max_abs_robust_effect=("abs_robust_effect", "max"),
            positive_effect_features=("robust_effect", lambda s: int((s > 0).sum())),
            negative_effect_features=("robust_effect", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(
            ["path_cluster", "event_side", "period", "max_abs_robust_effect"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
    )


def runtime_signature(feature_profiles: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Explain discovery clusters; this is not an executable classifier."""
    if feature_profiles.empty:
        return pd.DataFrame()
    selected = feature_profiles.loc[feature_profiles["within_cluster_rank"].le(top_n)].copy()
    selected["direction"] = np.where(selected["robust_effect"].ge(0), "ABOVE_BASELINE", "BELOW_BASELINE")
    selected["runtime_status"] = "EXPLANATORY_ONLY_CLUSTER_MODEL_NOT_A_LIVE_RULE"
    return selected[
        [
            "path_cluster",
            "diagnostic_role",
            "event_side",
            "period",
            "feature_family",
            "feature",
            "direction",
            "robust_effect",
            "cluster_median",
            "baseline_median",
            "runtime_status",
        ]
    ]
