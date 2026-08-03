#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance and robustness analysis for R03.4.2.7."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StructuralExitConfig


def profit_factor(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def maximum_drawdown(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return np.nan, np.nan
    equity = np.cumprod(1.0 + array)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = equity / peaks - 1.0
    return float(drawdown.min()), float(equity[-1] - 1.0)


def longest_losing_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def summarize(
    frame: pd.DataFrame,
    *,
    cost_multiplier: float,
    config: StructuralExitConfig,
) -> dict[str, object]:
    if frame.empty:
        return {"trades": 0, "mean_net_return": np.nan, "profit_factor": np.nan}
    work = frame.sort_values("entry_time").copy()
    work["net_return"] = work["gross_return"].astype(float) - (
        config.base_round_trip_cost * float(cost_multiplier)
    )
    net = work["net_return"].to_numpy(dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]
    mdd, total = maximum_drawdown(net)
    positive_profit = float(winners.sum())
    sorted_net = np.sort(net)[::-1]
    top_count = min(10, len(sorted_net))
    top_share = float(sorted_net[:top_count].sum() / positive_profit) if positive_profit > 0 else np.nan
    without_top = sorted_net[top_count:] if len(sorted_net) > top_count else np.empty(0)
    censored = work["is_censored"].astype(bool)
    return {
        "trades": int(len(work)),
        "resolved_trades": int((~censored).sum()),
        "censored_trades": int(censored.sum()),
        "censored_share": float(censored.mean()),
        "mean_gross_return": float(work["gross_return"].mean()),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": profit_factor(net),
        "mean_winner": float(winners.mean()) if len(winners) else np.nan,
        "mean_loser": float(losers.mean()) if len(losers) else np.nan,
        "payoff_ratio": float(winners.mean() / abs(losers.mean())) if len(winners) and len(losers) else np.nan,
        "mean_mfe": float(work["mfe"].mean()),
        "mean_mae": float(work["mae"].mean()),
        "median_holding_minutes": float(work["holding_minutes"].median()),
        "mean_holding_minutes": float(work["holding_minutes"].mean()),
        "p90_holding_minutes": float(work["holding_minutes"].quantile(0.90)),
        "total_compounded_return": total,
        "max_drawdown": mdd,
        "top10_profit_share": top_share,
        "mean_net_without_top10": float(without_top.mean()) if len(without_top) else np.nan,
        "longest_losing_streak": longest_losing_streak(net),
        "mean_state_transitions": float(work["state_transitions"].mean()),
        "mean_structure_breaks": float(work["structure_breaks"].mean()),
        "recovery_trade_share": float((work["recoveries"] > 0).mean()),
    }


def build_tables(
    trades: pd.DataFrame,
    *,
    overlap_audit: pd.DataFrame,
    config: StructuralExitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, overlap_audit
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    censor_rows: list[dict[str, object]] = []
    keys = ["fold_id", "policy", "policy_kind", "delay_minutes"]
    for values, group in trades.groupby(keys, sort=False):
        common = dict(zip(keys, values, strict=True))
        for metric_scope, metric_group in (
            ("all_positions_mark_to_market", group),
            ("resolved_only", group.loc[~group["is_censored"].astype(bool)]),
        ):
            for cost in config.cost_multipliers:
                summary_rows.append(
                    {
                        **common,
                        "metric_scope": metric_scope,
                        "cost_multiplier": float(cost),
                        **summarize(metric_group, cost_multiplier=cost, config=config),
                    }
                )
        for quarter, part in group.groupby("quarter", sort=True):
            for cost in (1.0, 2.0):
                period_rows.append(
                    {
                        **common,
                        "quarter": quarter,
                        "cost_multiplier": float(cost),
                        **summarize(part, cost_multiplier=cost, config=config),
                    }
                )
        for tier, part in group.groupby("score_tier", sort=True):
            for cost in (1.0, 2.0, 3.0):
                tier_rows.append(
                    {
                        **common,
                        "score_tier": tier,
                        "cost_multiplier": float(cost),
                        **summarize(part, cost_multiplier=cost, config=config),
                    }
                )
        for reason, part in group.groupby("exit_reason", sort=True):
            exit_rows.append(
                {
                    **common,
                    "exit_reason": reason,
                    "count": int(len(part)),
                    "share": float(len(part) / len(group)),
                    "mean_gross_return": float(part["gross_return"].mean()),
                    "median_holding_minutes": float(part["holding_minutes"].median()),
                    "mean_mfe": float(part["mfe"].mean()),
                    "mean_mae": float(part["mae"].mean()),
                }
            )
        censored = group.loc[group["is_censored"].astype(bool)]
        censor_rows.append(
            {
                **common,
                "trades": int(len(group)),
                "censored_trades": int(len(censored)),
                "censored_share": float(len(censored) / len(group)),
                "censored_mean_mark_return": float(censored["gross_return"].mean()) if len(censored) else np.nan,
                "censored_total_gross_return": float(censored["gross_return"].sum()) if len(censored) else 0.0,
                "data_gap_censored": int(censored["exit_reason"].str.contains("data_gap").sum()) if len(censored) else 0,
                "oos_end_censored": int(censored["exit_reason"].str.contains("oos_end").sum()) if len(censored) else 0,
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(period_rows),
        pd.DataFrame(tier_rows),
        pd.DataFrame(exit_rows),
        pd.DataFrame(censor_rows),
        overlap_audit,
    )


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    focus = summary.loc[
        (summary["metric_scope"] == "all_positions_mark_to_market")
        & (summary["delay_minutes"] == 1)
        & (summary["cost_multiplier"] == 2.0)
    ].copy()
    baseline = focus.loc[focus["policy"] == "fixed_6h"].set_index("fold_id")
    rows: list[dict[str, object]] = []
    for (fold_id, policy), group in focus.groupby(["fold_id", "policy"], sort=True):
        if policy == "fixed_6h" or fold_id not in baseline.index:
            continue
        row = group.iloc[0]
        base = baseline.loc[fold_id]
        base_mdd = abs(float(base["max_drawdown"]))
        policy_mdd = abs(float(row["max_drawdown"]))
        rows.append(
            {
                "fold_id": fold_id,
                "policy": policy,
                "policy_kind": row["policy_kind"],
                "trades": int(row["trades"]),
                "fixed_trades": int(base["trades"]),
                "trade_delta": int(row["trades"] - base["trades"]),
                "mean_net_return": float(row["mean_net_return"]),
                "fixed_mean_net_return": float(base["mean_net_return"]),
                "expectancy_delta": float(row["mean_net_return"] - base["mean_net_return"]),
                "profit_factor": float(row["profit_factor"]),
                "fixed_profit_factor": float(base["profit_factor"]),
                "total_compounded_return": float(row["total_compounded_return"]),
                "fixed_total_compounded_return": float(base["total_compounded_return"]),
                "total_return_delta": float(row["total_compounded_return"] - base["total_compounded_return"]),
                "profit_retention_ratio": float(
                    (1.0 + row["total_compounded_return"]) / (1.0 + base["total_compounded_return"])
                ) if np.isfinite(base["total_compounded_return"]) else np.nan,
                "max_drawdown": float(row["max_drawdown"]),
                "fixed_max_drawdown": float(base["max_drawdown"]),
                "relative_mdd_improvement": float((base_mdd - policy_mdd) / base_mdd) if base_mdd > 0 else np.nan,
                "censored_share": float(row["censored_share"]),
                "median_holding_minutes": float(row["median_holding_minutes"]),
            }
        )
    return pd.DataFrame(rows)


def stable_candidates(
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    comparison: pd.DataFrame,
    config: StructuralExitConfig,
) -> pd.DataFrame:
    if summary.empty or comparison.empty:
        return pd.DataFrame()
    required = {"WF_2024", "WF_2025"}
    focus = summary.loc[
        (summary["metric_scope"] == "all_positions_mark_to_market")
        & (summary["delay_minutes"] == 1)
        & (summary["cost_multiplier"] == 2.0)
        & (summary["policy_kind"] == "non_time_structural_candidate")
    ]
    rows: list[dict[str, object]] = []
    for policy, group in focus.groupby("policy", sort=True):
        by_fold = group.set_index("fold_id")
        complete = required.issubset(by_fold.index)
        if not complete:
            rows.append(
                {
                    "policy": policy,
                    "complete_2024_2025": False,
                    "positive_expectancy_2x_both_years": False,
                    "pf_2x_both_years": False,
                    "minimum_trades_both_years": False,
                    "mdd_pass": False,
                    "top10_concentration_pass": False,
                    "without_top10_pass": False,
                    "censoring_pass": False,
                    "delay_cost_stress_pass": False,
                    "positive_quarters": 0,
                    "base_robustness_pass": False,
                    "beats_fixed_total_profit_both_years": False,
                    "retains_profit_and_improves_mdd_both_years": False,
                    "passes_profit_upgrade": False,
                    "passes_risk_upgrade": False,
                    "mean_net_2x_2024": np.nan,
                    "mean_net_2x_2025": np.nan,
                    "pf_2x_2024": np.nan,
                    "pf_2x_2025": np.nan,
                    "mdd_2024": np.nan,
                    "mdd_2025": np.nan,
                    "censored_share_2024": np.nan,
                    "censored_share_2025": np.nan,
                }
            )
            continue
        ordered = [by_fold.loc[fold] for fold in ("WF_2024", "WF_2025")]
        quarters = periods.loc[
            (periods["policy"] == policy)
            & (periods["delay_minutes"] == 1)
            & (periods["cost_multiplier"] == 2.0)
        ]
        positive_quarters = int((quarters["mean_net_return"] > 0).sum())
        delay5 = summary.loc[
            (summary["policy"] == policy)
            & (summary["metric_scope"] == "all_positions_mark_to_market")
            & (summary["delay_minutes"] == 5)
            & (summary["cost_multiplier"] == 2.0)
        ].set_index("fold_id")
        cost3 = summary.loc[
            (summary["policy"] == policy)
            & (summary["metric_scope"] == "all_positions_mark_to_market")
            & (summary["delay_minutes"] == 1)
            & (summary["cost_multiplier"] == 3.0)
        ].set_index("fold_id")
        comp = comparison.loc[comparison["policy"] == policy].set_index("fold_id")
        positive = all(float(row["mean_net_return"]) > 0 for row in ordered)
        pf = all(float(row["profit_factor"]) >= config.minimum_pf_2x for row in ordered)
        trades = all(int(row["trades"]) >= config.minimum_trades_per_year for row in ordered)
        drawdown = all(abs(float(row["max_drawdown"])) <= config.maximum_mdd for row in ordered)
        concentration = all(float(row["top10_profit_share"]) <= config.maximum_top10_profit_share for row in ordered)
        without_top = all(float(row["mean_net_without_top10"]) > 0 for row in ordered)
        censoring = all(float(row["censored_share"]) <= config.maximum_censored_share for row in ordered)
        stress = (
            required.issubset(delay5.index)
            and required.issubset(cost3.index)
            and all(float(delay5.loc[fold, "mean_net_return"]) > 0 for fold in required)
            and all(float(cost3.loc[fold, "mean_net_return"]) > 0 for fold in required)
        )
        base_robust = bool(
            positive and pf and trades and drawdown and concentration and without_top
            and censoring and stress and positive_quarters >= config.minimum_positive_quarters
        )
        beats_profit = required.issubset(comp.index) and all(float(comp.loc[fold, "total_return_delta"]) > 0 for fold in required)
        risk_upgrade = required.issubset(comp.index) and all(
            float(comp.loc[fold, "profit_retention_ratio"]) >= config.minimum_profit_retention_vs_fixed
            and float(comp.loc[fold, "relative_mdd_improvement"]) >= config.minimum_relative_mdd_improvement
            for fold in required
        )
        rows.append(
            {
                "policy": policy,
                "complete_2024_2025": True,
                "positive_expectancy_2x_both_years": positive,
                "pf_2x_both_years": pf,
                "minimum_trades_both_years": trades,
                "mdd_pass": drawdown,
                "top10_concentration_pass": concentration,
                "without_top10_pass": without_top,
                "censoring_pass": censoring,
                "delay_cost_stress_pass": stress,
                "positive_quarters": positive_quarters,
                "base_robustness_pass": base_robust,
                "beats_fixed_total_profit_both_years": bool(beats_profit),
                "retains_profit_and_improves_mdd_both_years": bool(risk_upgrade),
                "passes_profit_upgrade": bool(base_robust and beats_profit),
                "passes_risk_upgrade": bool(base_robust and risk_upgrade),
                "mean_net_2x_2024": float(by_fold.loc["WF_2024", "mean_net_return"]),
                "mean_net_2x_2025": float(by_fold.loc["WF_2025", "mean_net_return"]),
                "pf_2x_2024": float(by_fold.loc["WF_2024", "profit_factor"]),
                "pf_2x_2025": float(by_fold.loc["WF_2025", "profit_factor"]),
                "mdd_2024": float(by_fold.loc["WF_2024", "max_drawdown"]),
                "mdd_2025": float(by_fold.loc["WF_2025", "max_drawdown"]),
                "censored_share_2024": float(by_fold.loc["WF_2024", "censored_share"]),
                "censored_share_2025": float(by_fold.loc["WF_2025", "censored_share"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_profit_upgrade", "passes_risk_upgrade", "base_robustness_pass", "mean_net_2x_2025", "mean_net_2x_2024"],
        ascending=[False, False, False, False, False],
    ) if rows else pd.DataFrame()
