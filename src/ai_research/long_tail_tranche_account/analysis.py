#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Account, coverage and pass-gate analysis for R03.4.2.8B."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_structural_exit.analysis import maximum_drawdown, profit_factor

from .config import TrancheAccountConfig


def fixed_6h_diagnostic_summary(fixed_6h: pd.DataFrame, config: TrancheAccountConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in fixed_6h.groupby(["fold_id", "delay_minutes"], sort=True):
        fold_id, delay = keys
        gross = group["gross_return"].astype(float).to_numpy()
        for multiplier in config.cost_multipliers:
            net = gross - config.base_round_trip_cost * float(multiplier)
            mdd, total = maximum_drawdown(net)
            rows.append(
                {
                    "benchmark": "q70_fixed_6h_all_signals_diagnostic",
                    "fold_id": fold_id,
                    "delay_minutes": int(delay),
                    "cost_multiplier": float(multiplier),
                    "signals": int(len(net)),
                    "mean_net_return": float(net.mean()) if len(net) else np.nan,
                    "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
                    "profit_factor": profit_factor(net),
                    "independent_compounded_return": float(total),
                    "independent_max_drawdown": float(mdd),
                }
            )
    return pd.DataFrame(rows)


def policy_coverage_summary(
    decisions: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    candidate_count: int,
) -> dict[str, object]:
    if decisions.empty:
        return {}
    accepted = decisions.loc[decisions["action"] == "ACCEPT"]
    second = trades.loc[trades.get("entry_role", pd.Series(index=trades.index, dtype=str)) == "secondary"] if not trades.empty else pd.DataFrame()
    return {
        "candidate_events": int(candidate_count),
        "selected_events": int(len(accepted)),
        "executed_events": int(len(trades)),
        "coverage_ratio": float(len(trades) / candidate_count) if candidate_count else np.nan,
        "monthly_tranches": float(len(trades) / 12.0),
        "primary_tranches": int((trades["entry_role"] == "primary").sum()) if not trades.empty else 0,
        "secondary_tranches": int((trades["entry_role"] == "secondary").sum()) if not trades.empty else 0,
        "secondary_share": float((trades["entry_role"] == "secondary").mean()) if not trades.empty else np.nan,
        "losing_active_second_add_share": float((second["active_current_return"] < 0).mean()) if not second.empty else 0.0,
        "dangerous_second_add_share": float(second["dangerous_second_add"].astype(bool).mean()) if not second.empty else 0.0,
        "score_up_price_down_second_share": float(second["score_up_price_down"].astype(bool).mean()) if not second.empty else 0.0,
        "skip_full_slots": int((decisions["reason"] == "risk_slots_full").sum()),
        "skip_protection": int((decisions["reason"] == "dangerous_or_broken_active_structure").sum()),
        "skip_missing_pair": int((decisions["reason"] == "missing_causal_pair_diagnostic").sum()),
    }


def build_account_summary(
    simulation_summaries: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> pd.DataFrame:
    summary = pd.DataFrame(simulation_summaries)
    coverage = pd.DataFrame(coverage_rows)
    if summary.empty:
        return summary
    if coverage.empty:
        return summary
    keys = ["fold_id", "policy", "delay_minutes", "cost_multiplier"]
    return summary.merge(coverage, on=keys, how="left", validate="one_to_one")


def concentration_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in trades.groupby(["fold_id", "policy", "delay_minutes", "cost_multiplier"], sort=True):
        fold_id, policy, delay, multiplier = keys
        pnl = group["net_pnl"].astype(float).sort_values(ascending=False)
        winners = pnl[pnl > 0]
        top = winners.head(10)
        rows.append(
            {
                "fold_id": fold_id,
                "policy": policy,
                "delay_minutes": int(delay),
                "cost_multiplier": float(multiplier),
                "tranches": int(len(group)),
                "top10_net_pnl": float(top.sum()),
                "top10_profit_share": float(top.sum() / winners.sum()) if winners.sum() > 0 else np.nan,
                "net_pnl_without_top10": float(pnl.sum() - top.sum()),
                "positive_without_top10": bool(pnl.sum() - top.sum() > 0),
            }
        )
    return pd.DataFrame(rows)


def policy_gate(account_summary: pd.DataFrame, config: TrancheAccountConfig) -> pd.DataFrame:
    """Require one unified policy to improve P0 without sacrificing coverage or risk."""

    if account_summary.empty:
        return pd.DataFrame()
    p0 = account_summary.loc[account_summary["policy"] == "P0_single_1R"].set_index(
        ["fold_id", "delay_minutes", "cost_multiplier"]
    )
    rows: list[dict[str, object]] = []
    for policy in [item.name for item in config.policies if item.name != "P0_single_1R"]:
        policy_frame = account_summary.loc[account_summary["policy"] == policy]
        row: dict[str, object] = {"policy": policy}
        all_primary_pass = True
        all_delay_stress_positive = True
        for fold_id in ("WF_2024", "WF_2025"):
            main = policy_frame.loc[
                (policy_frame["fold_id"] == fold_id)
                & (policy_frame["delay_minutes"] == 1)
                & (policy_frame["cost_multiplier"] == 2.0)
            ]
            stress3x = policy_frame.loc[
                (policy_frame["fold_id"] == fold_id)
                & (policy_frame["delay_minutes"] == 1)
                & (policy_frame["cost_multiplier"] == 3.0)
            ]
            if main.empty or stress3x.empty:
                row[f"fold_pass_{fold_id[-4:]}"] = False
                all_primary_pass = False
                continue
            value = main.iloc[0]
            stress_value = stress3x.iloc[0]
            p0_value = p0.loc[(fold_id, 1, 2.0)] if (fold_id, 1, 2.0) in p0.index else None
            improves_p0 = bool(p0_value is not None and value["total_net_return"] > p0_value["total_net_return"])
            fold_pass = bool(
                improves_p0
                and value["coverage_ratio"] >= config.minimum_coverage_ratio
                and value["executed_events"] >= config.minimum_tranches_per_year
                and value["monthly_tranches"] >= config.minimum_monthly_tranches
                and value["max_drawdown"] >= -config.maximum_account_drawdown
                and value["positive_quarters"] >= config.minimum_positive_quarters_per_year
                and value["total_return_without_top10"] > 0
                and value["max_slot_r"] <= config.maximum_allocated_r + 1e-9
                and value["dangerous_second_add_share"] <= config.maximum_dangerous_second_add_share
                and value["losing_active_second_add_share"] <= config.maximum_losing_second_add_share
                and stress_value["total_net_return"] > 0
            )
            year = fold_id[-4:]
            row.update(
                {
                    f"tranches_{year}": int(value["executed_events"]),
                    f"coverage_{year}": float(value["coverage_ratio"]),
                    f"return_2x_{year}": float(value["total_net_return"]),
                    f"p0_return_2x_{year}": float(p0_value["total_net_return"]) if p0_value is not None else np.nan,
                    f"return_3x_{year}": float(stress_value["total_net_return"]),
                    f"mdd_2x_{year}": float(value["max_drawdown"]),
                    f"positive_quarters_{year}": int(value["positive_quarters"]),
                    f"without_top10_{year}": float(value["total_return_without_top10"]),
                    f"dangerous_second_share_{year}": float(value["dangerous_second_add_share"]),
                    f"losing_second_share_{year}": float(value["losing_active_second_add_share"]),
                    f"fold_pass_{year}": fold_pass,
                }
            )
            all_primary_pass = all_primary_pass and fold_pass

            for delay in (3, 5):
                delay_row = policy_frame.loc[
                    (policy_frame["fold_id"] == fold_id)
                    & (policy_frame["delay_minutes"] == delay)
                    & (policy_frame["cost_multiplier"] == 3.0)
                ]
                delay_positive = bool(not delay_row.empty and delay_row.iloc[0]["total_net_return"] > 0)
                row[f"delay{delay}_3x_positive_{year}"] = delay_positive
                all_delay_stress_positive = all_delay_stress_positive and delay_positive
        row["primary_gate_pass"] = bool(all_primary_pass)
        row["delay_stress_pass"] = bool(all_delay_stress_positive)
        row["pass_to_entry_stop_research"] = bool(all_primary_pass and all_delay_stress_positive)
        rows.append(row)
    return pd.DataFrame(rows)
