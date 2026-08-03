#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summaries and pre-registered gates for R03.4.2.9."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DynamicRiskReleaseConfig


def protection_trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["fold_id", "protection_policy", "delay_minutes"]
    for values, group in trades.groupby(keys, sort=True):
        gross = group["gross_return"].astype(float).to_numpy()
        winners = gross[gross > 0]
        losers = gross[gross < 0]
        rows.append(
            {
                **dict(zip(keys, values)),
                "events": int(len(group)),
                "mean_gross_return": float(np.mean(gross)),
                "win_rate": float(np.mean(gross > 0)),
                "profit_factor_gross": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
                "hard_stop_share": float(group["hard_stop_triggered"].astype(bool).mean()),
                "disaster_exit_share": float(group["exit_reason"].astype(str).eq("disaster_stop").mean()),
                "median_holding_minutes": float(group["holding_minutes"].astype(float).median()),
                "mean_maximum_released_risk_fraction": float(group["maximum_released_risk_fraction"].astype(float).mean()),
                "full_release_share": float((group["maximum_released_risk_fraction"].astype(float) >= 1.0 - 1e-12).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_account_summary(
    simulation_rows: list[dict[str, object]],
    *,
    decisions: pd.DataFrame,
    candidate_counts: pd.DataFrame,
) -> pd.DataFrame:
    if not simulation_rows:
        return pd.DataFrame()
    summary = pd.DataFrame(simulation_rows)
    if not decisions.empty:
        decision_summary = (
            decisions.groupby(
                ["fold_id", "delay_minutes", "protection_policy", "dynamic_policy"],
                as_index=False,
            )
            .agg(
                selected_events=("action", lambda values: int((values == "ACCEPT").sum())),
                skipped_events=("action", lambda values: int((values == "SKIP").sum())),
                skip_single_baseline=("reason", lambda values: int((values == "single_tranche_baseline").sum())),
                skip_two_active=("reason", lambda values: int((values == "maximum_two_tranches").sum())),
                skip_insufficient_release=("reason", lambda values: int((values == "insufficient_enforceable_risk_release").sum())),
                skip_unhealthy=("reason", lambda values: int((values == "active_structure_not_healthy").sum())),
                skip_losing=("reason", lambda values: int((values == "active_position_still_losing").sum())),
                skip_missing_snapshot=("reason", lambda values: int((values == "missing_causal_release_snapshot").sum())),
            )
        )
        summary = summary.merge(
            decision_summary,
            on=["fold_id", "delay_minutes", "protection_policy", "dynamic_policy"],
            how="left",
        )
    summary = summary.merge(candidate_counts, on=["fold_id", "delay_minutes"], how="left")
    summary["coverage_ratio"] = summary["executed_tranches"] / summary["candidate_events"].clip(lower=1)
    summary["monthly_tranches"] = summary["executed_tranches"] / 12.0
    return summary.sort_values(
        ["protection_policy", "dynamic_policy", "fold_id", "delay_minutes", "cost_multiplier"]
    ).reset_index(drop=True)


def policy_gate(
    account_summary: pd.DataFrame,
    protection_summary: pd.DataFrame,
    config: DynamicRiskReleaseConfig,
) -> pd.DataFrame:
    if account_summary.empty:
        return pd.DataFrame()
    baseline = account_summary.loc[
        (account_summary["protection_policy"] == "S0_disaster_only")
        & (account_summary["dynamic_policy"] == "D0_single_1R")
    ].copy()
    baseline_lookup = {
        (str(row.fold_id), int(row.delay_minutes), float(row.cost_multiplier)): row
        for row in baseline.itertuples()
    }
    hard_stop_lookup = {
        (str(row.fold_id), str(row.protection_policy), int(row.delay_minutes)): float(row.hard_stop_share)
        for row in protection_summary.itertuples()
    }
    baseline_total = sum(
        float(baseline_lookup[(fold, 1, 2.0)].total_net_return)
        for fold in ("WF_2024", "WF_2025")
        if (fold, 1, 2.0) in baseline_lookup
    )

    protection_results: dict[str, dict[str, object]] = {}
    for stop_name in account_summary["protection_policy"].astype(str).unique():
        group = account_summary.loc[
            (account_summary["protection_policy"] == stop_name)
            & (account_summary["dynamic_policy"] == "D0_single_1R")
        ].copy()
        focus = group.loc[(group["delay_minutes"] == 1) & (group["cost_multiplier"] == 2.0)]
        stress = group.loc[group["cost_multiplier"].isin([2.0, 3.0])]
        checks: list[bool] = []
        retentions: list[float] = []
        mdd_multiples: list[float] = []
        for row in focus.itertuples():
            base = baseline_lookup.get((str(row.fold_id), 1, 2.0))
            if base is None:
                checks.append(False)
                continue
            retention = float(row.total_net_return / base.total_net_return) if float(base.total_net_return) > 0 else np.nan
            mdd_multiple = abs(float(row.max_drawdown)) / max(abs(float(base.max_drawdown)), 1e-12)
            retentions.append(retention)
            mdd_multiples.append(mdd_multiple)
            checks.append(
                bool(
                    float(row.total_net_return) > 0
                    and retention >= config.minimum_protection_return_retention
                    and mdd_multiple <= config.maximum_protection_mdd_multiple
                    and hard_stop_lookup.get((str(row.fold_id), stop_name, 1), 1.0) <= config.maximum_hard_stop_share
                    and float(row.total_return_without_top10) > 0
                )
            )
        stress_pass = bool(
            len(stress) == 12
            and (stress["total_net_return"].astype(float) > 0).all()
            and (stress["max_live_remaining_r"].astype(float) <= config.maximum_live_remaining_r + 0.02).all()
        )
        protection_results[stop_name] = {
            "pass": bool(len(checks) == 2 and all(checks) and stress_pass),
            "stress": stress_pass,
            "minimum_retention": float(min(retentions)) if retentions else np.nan,
            "maximum_mdd_multiple": float(max(mdd_multiples)) if mdd_multiples else np.nan,
        }

    rows: list[dict[str, object]] = []
    candidates = account_summary[["protection_policy", "dynamic_policy"]].drop_duplicates()
    for candidate in candidates.itertuples(index=False):
        stop_name = str(candidate.protection_policy)
        dynamic_name = str(candidate.dynamic_policy)
        if stop_name == "S0_disaster_only" and dynamic_name != "D0_single_1R":
            continue
        group = account_summary.loc[
            (account_summary["protection_policy"] == stop_name)
            & (account_summary["dynamic_policy"] == dynamic_name)
        ].copy()
        focus = group.loc[(group["delay_minutes"] == 1) & (group["cost_multiplier"] == 2.0)]
        stress = group.loc[group["cost_multiplier"].isin([2.0, 3.0])]
        fold_checks: list[bool] = []
        retentions: list[float] = []
        mdd_multiples: list[float] = []
        for row in focus.itertuples():
            base = baseline_lookup.get((str(row.fold_id), 1, 2.0))
            if base is None:
                fold_checks.append(False)
                continue
            retention = float(row.total_net_return / base.total_net_return) if float(base.total_net_return) > 0 else np.nan
            mdd_multiple = abs(float(row.max_drawdown)) / max(abs(float(base.max_drawdown)), 1e-12)
            retentions.append(retention)
            mdd_multiples.append(mdd_multiple)
            if dynamic_name == "D0_single_1R":
                fold_checks.append(bool(protection_results.get(stop_name, {}).get("pass", False)))
            else:
                fold_checks.append(
                    bool(
                        float(row.total_net_return) > 0
                        and retention >= config.minimum_dynamic_return_retention
                        and mdd_multiple <= config.maximum_dynamic_mdd_multiple
                        and float(row.coverage_ratio) >= config.minimum_coverage_ratio
                        and float(row.monthly_tranches) >= config.minimum_monthly_tranches
                        and int(row.positive_quarters) >= config.minimum_positive_quarters_per_year
                        and float(row.total_return_without_top10) > 0
                        and float(row.max_live_remaining_r) <= config.maximum_live_remaining_r + 0.02
                        and float(row.losing_second_add_share) <= config.maximum_losing_second_add_share
                        and float(row.broken_second_add_share) <= config.maximum_broken_second_add_share
                    )
                )
        stress_positive = bool(
            len(stress) == 12
            and (stress["total_net_return"].astype(float) > 0).all()
            and (stress["max_live_remaining_r"].astype(float) <= config.maximum_live_remaining_r + 0.02).all()
        )
        cross_year_total = float(focus["total_net_return"].sum())
        cross_year_improvement = bool(cross_year_total >= baseline_total - 1e-12)
        protection_pass = bool(protection_results.get(stop_name, {}).get("pass", False))
        if dynamic_name == "D0_single_1R":
            final_pass = protection_pass
        else:
            final_pass = bool(
                protection_pass
                and len(fold_checks) == 2
                and all(fold_checks)
                and stress_positive
                and cross_year_improvement
            )
        rows.append(
            {
                "protection_policy": stop_name,
                "dynamic_policy": dynamic_name,
                "minimum_return_retention": float(min(retentions)) if retentions else np.nan,
                "maximum_mdd_multiple": float(max(mdd_multiples)) if mdd_multiples else np.nan,
                "protection_gate_pass": protection_pass,
                "dynamic_fold_gate_pass": bool(len(fold_checks) == 2 and all(fold_checks)),
                "stress_gate_pass": stress_positive,
                "cross_year_total_return": cross_year_total,
                "baseline_cross_year_total_return": baseline_total,
                "cross_year_total_improvement": cross_year_improvement,
                "pass_to_next_stage": final_pass,
            }
        )
    return pd.DataFrame(rows).sort_values(["protection_policy", "dynamic_policy"]).reset_index(drop=True)
