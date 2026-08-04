#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attribution and qualification gates for R03.4.2.12."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData

from .config import TailCompressionConfig


def attach_baseline_counterfactual(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return cycles.copy()
    keys = ["fold_id", "delay_minutes", "cost_multiplier", "event_id"]
    baseline = cycles.loc[
        cycles["policy"].astype(str).eq("P0_single_1R"),
        keys + ["cycle_return"],
    ].rename(columns={"cycle_return": "baseline_cycle_return"})
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
    grouped["winner_to_loser_share"] = grouped["winners_turned_loser"] / grouped["baseline_winners"].clip(lower=1)
    return summary.merge(
        grouped,
        on=["fold_id", "policy", "delay_minutes", "cost_multiplier"],
        how="left",
        validate="one_to_one",
    )


def build_f1_attribution(
    source_cycles: pd.DataFrame,
    source_legs: pd.DataFrame,
    *,
    fold_id: str,
    path: MinutePathData,
    materiality: float,
) -> pd.DataFrame:
    """Separate F1 sizing leverage from its completed-close exit contribution."""

    keys = ["fold_id", "delay_minutes", "cost_multiplier", "event_id"]
    p0 = source_cycles.loc[
        source_cycles["fold_id"].astype(str).eq(fold_id)
        & source_cycles["policy"].astype(str).eq("P0_single_1R"),
        keys + ["entry_time", "source_exit_time", "cycle_return", "source_exit_reason"],
    ].copy()
    p0 = p0.rename(
        columns={
            "entry_time": "p0_entry_time",
            "source_exit_time": "p0_exit_time",
            "cycle_return": "p0_cycle_return",
            "source_exit_reason": "p0_exit_reason",
        }
    )
    f1 = source_cycles.loc[
        source_cycles["fold_id"].astype(str).eq(fold_id)
        & source_cycles["policy"].astype(str).eq("F1_soft_failure_1p5"),
        keys + ["cycle_return", "max_hard_tail_r", "soft_failure_exit"],
    ].copy()
    f1 = f1.rename(
        columns={
            "cycle_return": "f1_cycle_return",
            "max_hard_tail_r": "f1_hard_tail_r",
        }
    )
    base = p0.merge(f1, on=keys, how="inner", validate="one_to_one")

    legs = source_legs.loc[
        source_legs["fold_id"].astype(str).eq(fold_id)
        & source_legs["policy"].astype(str).eq("F1_soft_failure_1p5")
        & source_legs["tranche_role"].astype(str).eq("base"),
        keys + ["entry_price", "exit_time", "exit_price", "exit_reason"],
    ].copy()
    legs = legs.rename(
        columns={
            "entry_price": "f1_entry_price",
            "exit_time": "f1_exit_time",
            "exit_price": "f1_exit_price",
            "exit_reason": "f1_exit_reason",
        }
    )
    result = base.merge(legs, on=keys, how="left", validate="one_to_one")
    result["f1_1r_equivalent_return"] = (
        result["f1_cycle_return"].astype(float) / result["f1_hard_tail_r"].astype(float).clip(lower=1e-12)
    )
    result["f1_exit_edge_1r"] = result["f1_1r_equivalent_return"] - result["p0_cycle_return"].astype(float)

    post_max: list[float] = []
    post_min: list[float] = []
    recovered: list[bool] = []
    hit_15: list[bool] = []
    hit_20: list[bool] = []
    classes: list[str] = []
    for row in result.to_dict("records"):
        is_soft = bool(row.get("soft_failure_exit", False))
        entry_price = float(row.get("f1_entry_price", np.nan))
        soft_position = path.locate_exact(pd.Timestamp(row["f1_exit_time"])) if is_soft else None
        p0_exit_position = path.locate_exact(pd.Timestamp(row["p0_exit_time"])) if is_soft else None
        if is_soft and soft_position is not None and p0_exit_position is not None and p0_exit_position >= soft_position:
            segment_high = path.high[soft_position : p0_exit_position + 1]
            segment_low = path.low[soft_position : p0_exit_position + 1]
            mx = float(np.max(segment_high) / entry_price - 1.0)
            mn = float(np.min(segment_low) / entry_price - 1.0)
        else:
            mx = np.nan
            mn = np.nan
        post_max.append(mx)
        post_min.append(mn)
        recovered.append(bool(np.isfinite(mx) and mx > 0))
        hit_15.append(bool(np.isfinite(mn) and mn <= -0.015))
        hit_20.append(bool(np.isfinite(mn) and mn <= -0.02))

        p0_return = float(row["p0_cycle_return"])
        edge = float(row["f1_exit_edge_1r"])
        f1_equiv = float(row["f1_1r_equivalent_return"])
        if not is_soft:
            label = "SIZING_ONLY_SAME_EXIT"
        elif p0_return > 0 and f1_equiv <= 0:
            label = "WINNER_TO_LOSER_PREMATURE"
        elif p0_return > 0 and edge < -materiality:
            label = "PREMATURE_MISSED_RECOVERY"
        elif p0_return < 0 and edge > materiality:
            label = "EFFECTIVE_LOSS_REDUCTION"
        elif edge > materiality:
            label = "IMPROVED_EXIT"
        elif edge < -materiality:
            label = "WORSE_EXIT"
        else:
            label = "NEUTRAL_EXIT"
        classes.append(label)

    result["post_soft_max_return_from_entry"] = post_max
    result["post_soft_min_return_from_entry"] = post_min
    result["post_soft_recovered_above_entry"] = recovered
    result["post_soft_hit_1p5"] = hit_15
    result["post_soft_hit_2p0"] = hit_20
    result["attribution_class"] = classes
    return result


def summarize_f1_attribution(attribution: pd.DataFrame) -> pd.DataFrame:
    if attribution.empty:
        return pd.DataFrame()
    return (
        attribution.groupby(
            ["fold_id", "delay_minutes", "cost_multiplier", "attribution_class"],
            as_index=False,
        )
        .agg(
            events=("event_id", "count"),
            mean_p0_return=("p0_cycle_return", "mean"),
            mean_f1_1r_equivalent_return=("f1_1r_equivalent_return", "mean"),
            mean_exit_edge_1r=("f1_exit_edge_1r", "mean"),
            recovered_above_entry_share=("post_soft_recovered_above_entry", "mean"),
            later_hit_1p5_share=("post_soft_hit_1p5", "mean"),
            later_hit_2p0_share=("post_soft_hit_2p0", "mean"),
        )
    )


def policy_gate(summary: pd.DataFrame, config: TailCompressionConfig) -> pd.DataFrame:
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
                    and candidate_mdd <= config.maximum_absolute_mdd
                    and float(row["total_return_without_top10"]) > 0
                    and int(row["positive_quarters"]) >= config.minimum_positive_quarters_per_year
                    and float(row.get("winner_to_loser_share", 0.0)) <= config.maximum_winner_to_loser_share
                    and float(row["max_hard_tail_r"]) <= config.maximum_tail_r
                    and float(row["mean_base_notional_to_equity"]) >= config.minimum_mean_notional_to_equity
                    and float(row["worst_cycle_loss_r"]) <= config.maximum_worst_cycle_loss_r
                )
            )

        stress_checks: list[bool] = []
        policy_stress = summary.loc[summary["policy"].astype(str).eq(policy.name)]
        for fold_id in ("WF_2024", "WF_2025"):
            required = policy_stress.loc[
                policy_stress["fold_id"].astype(str).eq(fold_id)
                & policy_stress["delay_minutes"].astype(int).isin(config.entry_delay_minutes)
                & policy_stress["cost_multiplier"].astype(float).isin(config.cost_multipliers)
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
                "qualifying_candidate": bool(policy.qualifying_candidate),
                "declared_hard_tail_r": policy.declared_hard_tail_r,
                "minimum_return_retention": min(retention) if retention else np.nan,
                "maximum_mdd_multiple": max(mdd_multiples) if mdd_multiples else np.nan,
                "cross_year_total_return": candidate_total,
                "baseline_cross_year_total_return": baseline_total,
                "combined_return_ratio": combined_ratio,
                "fold_gate_pass": bool(len(fold_checks) == 2 and all(fold_checks)),
                "stress_gate_pass": bool(len(stress_checks) == 2 and all(stress_checks)),
                "pass_to_next_stage": bool(
                    policy.qualifying_candidate
                    and len(fold_checks) == 2
                    and all(fold_checks)
                    and len(stress_checks) == 2
                    and all(stress_checks)
                    and combined_ratio >= config.minimum_combined_return_ratio
                ),
            }
        )
    return pd.DataFrame(rows)
