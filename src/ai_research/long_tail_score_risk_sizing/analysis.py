#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score-tier attribution and policy qualification for R03.4.2.13."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ScoreRiskConfig


def build_tier_attribution(source_cycles: pd.DataFrame) -> pd.DataFrame:
    if source_cycles.empty:
        return pd.DataFrame()
    work = source_cycles.copy()
    work["is_win"] = work["cycle_return"].astype(float) > 0
    rows: list[dict[str, object]] = []
    for keys, frame in work.groupby(["fold_id", "delay_minutes", "cost_multiplier", "score_tier"], sort=True):
        pnl = frame["cycle_return"].astype(float).to_numpy()
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        rows.append({
            "fold_id": keys[0], "delay_minutes": int(keys[1]), "cost_multiplier": float(keys[2]), "score_tier": keys[3],
            "events": int(len(frame)), "mean_cycle_return": float(np.mean(pnl)), "median_cycle_return": float(np.median(pnl)),
            "sum_cycle_return": float(np.sum(pnl)), "win_rate": float(np.mean(pnl > 0)),
            "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
            "worst_cycle_return": float(np.min(pnl)), "hard_stop_share": float(frame["hard_stop_exit"].mean()),
            "soft_failure_share": float(frame["soft_failure_exit"].mean()),
        })
    return pd.DataFrame(rows)


def build_cross_year_order_audit(attribution: pd.DataFrame) -> pd.DataFrame:
    focus = attribution.loc[
        attribution["delay_minutes"].astype(int).eq(1)
        & np.isclose(attribution["cost_multiplier"].astype(float), 2.0)
    ].copy()
    if focus.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    tier_order = ["q70_to_q80", "q80_to_q90", "q90_plus"]
    for fold_id, frame in focus.groupby("fold_id"):
        means = frame.set_index("score_tier")["mean_cycle_return"].to_dict()
        values = [float(means.get(tier, np.nan)) for tier in tier_order]
        monotonic = bool(np.isfinite(values).all() and values[0] <= values[1] <= values[2])
        ranking = " > ".join(sorted(tier_order, key=lambda tier: means.get(tier, -np.inf), reverse=True))
        rows.append({"fold_id": fold_id, "monotonic_score_order": monotonic, "return_ranking": ranking, **{f"mean_{tier}": means.get(tier, np.nan) for tier in tier_order}})
    return pd.DataFrame(rows)


def policy_gate(summary: pd.DataFrame, config: ScoreRiskConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary.loc[
        summary["policy"].astype(str).eq("E100_equal_1R")
        & summary["delay_minutes"].astype(int).eq(1)
        & np.isclose(summary["cost_multiplier"].astype(float), 2.0)
    ].set_index("fold_id")
    rows: list[dict[str, object]] = []
    for policy in config.policies:
        primary = summary.loc[
            summary["policy"].astype(str).eq(policy.name)
            & summary["delay_minutes"].astype(int).eq(1)
            & np.isclose(summary["cost_multiplier"].astype(float), 2.0)
        ].copy()
        fold_checks: list[bool] = []
        retention: list[float] = []
        mdd_ratios: list[float] = []
        calmar_ratios: list[float] = []
        for row in primary.to_dict("records"):
            fold = str(row["fold_id"])
            if fold not in base.index:
                continue
            b = base.loc[fold]
            candidate_return = float(row["total_net_return"])
            base_return = float(b["total_net_return"])
            candidate_mdd = abs(float(row["max_drawdown"]))
            base_mdd = abs(float(b["max_drawdown"]))
            return_ratio = candidate_return / base_return if base_return > 0 else np.nan
            mdd_ratio = candidate_mdd / base_mdd if base_mdd > 0 else np.inf
            candidate_calmar = candidate_return / candidate_mdd if candidate_mdd > 0 else np.inf
            base_calmar = base_return / base_mdd if base_mdd > 0 else np.inf
            calmar_ratio = candidate_calmar / base_calmar if base_calmar > 0 else np.nan
            retention.append(return_ratio); mdd_ratios.append(mdd_ratio); calmar_ratios.append(calmar_ratio)
            top_increase = float(row["top10_profit_share"]) - float(b["top10_profit_share"])
            fold_checks.append(bool(
                candidate_return > 0
                and return_ratio >= config.minimum_return_retention_each_year
                and mdd_ratio <= config.maximum_mdd_multiple
                and candidate_mdd <= config.maximum_absolute_mdd
                and calmar_ratio >= config.minimum_calmar_improvement
                and float(row["total_return_without_top10"]) > 0
                and int(row["positive_quarters"]) >= config.minimum_positive_quarters_per_year
                and top_increase <= config.maximum_top10_profit_share_increase
                and float(row["max_risk_multiplier"]) <= config.maximum_candidate_tail_r + 1e-12
            ))
        stress_checks: list[bool] = []
        stress = summary.loc[summary["policy"].astype(str).eq(policy.name)]
        for fold in ("WF_2024", "WF_2025"):
            required = stress.loc[
                stress["fold_id"].astype(str).eq(fold)
                & stress["delay_minutes"].astype(int).isin(config.entry_delay_minutes)
                & stress["cost_multiplier"].astype(float).isin(config.cost_multipliers)
            ]
            stress_checks.append(bool(len(required) == 6 and (required["total_net_return"].astype(float) > 0).all()))
        candidate_total = float(primary["total_net_return"].sum())
        baseline_total = float(base["total_net_return"].sum())
        combined_ratio = candidate_total / baseline_total if baseline_total > 0 else np.nan
        rows.append({
            "policy": policy.name, "qualifying_candidate": policy.qualifying_candidate,
            "diagnostic_only": policy.diagnostic_only, "max_tail_r": policy.max_tail_r,
            "minimum_return_retention": min(retention) if retention else np.nan,
            "maximum_mdd_ratio": max(mdd_ratios) if mdd_ratios else np.nan,
            "minimum_calmar_ratio": min(calmar_ratios) if calmar_ratios else np.nan,
            "cross_year_total_return": candidate_total, "baseline_cross_year_total_return": baseline_total,
            "combined_return_ratio": combined_ratio,
            "fold_gate_pass": bool(len(fold_checks) == 2 and all(fold_checks)),
            "stress_gate_pass": bool(len(stress_checks) == 2 and all(stress_checks)),
            "pass_to_next_stage": bool(policy.qualifying_candidate and len(fold_checks) == 2 and all(fold_checks) and len(stress_checks) == 2 and all(stress_checks) and combined_ratio >= config.minimum_combined_return_ratio),
        })
    return pd.DataFrame(rows)
