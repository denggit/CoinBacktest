#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-policy attribution and qualification gates for R03.4.2.11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagedExecutionConfig


def attach_baseline_counterfactual(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return cycles.copy()
    keys = ["fold_id", "delay_minutes", "cost_multiplier", "event_id"]
    baseline = cycles.loc[cycles["policy"].astype(str).eq("P0_single_1R"), keys + ["cycle_return"]].copy()
    baseline = baseline.rename(columns={"cycle_return": "baseline_cycle_return"})
    result = cycles.merge(baseline, on=keys, how="left", validate="many_to_one")
    result["baseline_winner"] = result["baseline_cycle_return"].astype(float) > 0
    result["winner_turned_loser"] = result["baseline_winner"] & (result["cycle_return"].astype(float) < 0)
    result["incremental_cycle_return"] = (
        result["cycle_return"].astype(float) - result["baseline_cycle_return"].astype(float)
    )
    return result


def enrich_summaries(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    work = attach_baseline_counterfactual(cycles)
    grouped = (
        work.groupby(["fold_id", "policy", "delay_minutes", "cost_multiplier"], as_index=False)
        .agg(
            baseline_winners=("baseline_winner", "sum"),
            winners_turned_loser=("winner_turned_loser", "sum"),
            mean_incremental_cycle_return=("incremental_cycle_return", "mean"),
            median_incremental_cycle_return=("incremental_cycle_return", "median"),
        )
    )
    grouped["winner_to_loser_share"] = (
        grouped["winners_turned_loser"] / grouped["baseline_winners"].clip(lower=1)
    )
    return summary.merge(
        grouped,
        on=["fold_id", "policy", "delay_minutes", "cost_multiplier"],
        how="left",
        validate="one_to_one",
    )


def policy_gate(summary: pd.DataFrame, config: StagedExecutionConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary.loc[
        summary["policy"].astype(str).eq("P0_single_1R")
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
        retention: list[float] = []
        mdd_multiples: list[float] = []
        fold_checks: list[bool] = []
        for row in primary.to_dict("records"):
            fold_id = str(row["fold_id"])
            if fold_id not in base.index:
                continue
            baseline = base.loc[fold_id]
            base_return = float(baseline["total_net_return"])
            candidate_return = float(row["total_net_return"])
            return_ratio = candidate_return / base_return if base_return > 0 else np.nan
            base_mdd = abs(float(baseline["max_drawdown"]))
            candidate_mdd = abs(float(row["max_drawdown"]))
            mdd_multiple = candidate_mdd / base_mdd if base_mdd > 0 else np.inf
            retention.append(return_ratio)
            mdd_multiples.append(mdd_multiple)
            fold_checks.append(
                bool(
                    candidate_return > 0
                    and return_ratio >= config.minimum_return_retention_each_year
                    and mdd_multiple <= config.maximum_mdd_multiple
                    and float(row["total_return_without_top10"]) > 0
                    and int(row["positive_quarters"]) >= config.minimum_positive_quarters_per_year
                    and float(row.get("winner_to_loser_share", 0.0)) <= config.maximum_winner_to_loser_share
                    and float(row["max_hard_tail_r"])
                    <= min(policy.max_cycle_hard_r, config.maximum_account_tail_r)
                    + config.maximum_tail_r_tolerance
                    and float(row["max_notional_to_equity"])
                    <= min(policy.max_notional_to_equity, config.maximum_notional_to_equity)
                    + config.maximum_notional_tolerance
                    and float(row["addon_loss_share_of_base_profit"])
                    <= config.maximum_addon_loss_share_of_base_profit
                )
            )

        stress = summary.loc[summary["policy"].astype(str).eq(policy.name)].copy()
        stress_checks: list[bool] = []
        for fold_id in ("WF_2024", "WF_2025"):
            fold_stress = stress.loc[stress["fold_id"].astype(str).eq(fold_id)]
            required = fold_stress.loc[
                fold_stress["delay_minutes"].astype(int).isin(config.entry_delay_minutes)
                & fold_stress["cost_multiplier"].astype(float).isin(config.cost_multipliers)
            ]
            stress_checks.append(
                bool(
                    len(required) == len(config.entry_delay_minutes) * len(config.cost_multipliers)
                    and (required["total_net_return"].astype(float) > 0).all()
                )
            )
        candidate_total = float(primary["total_net_return"].sum())
        baseline_total = float(base["total_net_return"].sum())
        combined_ratio = candidate_total / baseline_total if baseline_total > 0 else np.nan
        rows.append(
            {
                "policy": policy.name,
                "minimum_return_retention": min(retention) if retention else np.nan,
                "maximum_mdd_multiple": max(mdd_multiples) if mdd_multiples else np.nan,
                "cross_year_total_return": candidate_total,
                "baseline_cross_year_total_return": baseline_total,
                "combined_return_ratio": combined_ratio,
                "fold_gate_pass": bool(len(fold_checks) == 2 and all(fold_checks)),
                "stress_gate_pass": bool(len(stress_checks) == 2 and all(stress_checks)),
                "pass_to_next_stage": bool(
                    policy.name != "P0_single_1R"
                    and len(fold_checks) == 2
                    and all(fold_checks)
                    and all(stress_checks)
                    and combined_ratio >= config.minimum_combined_return_ratio
                ),
            }
        )
    return pd.DataFrame(rows)
