#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analysis and pass gates for R03.4.2.8A."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_structural_exit.analysis import maximum_drawdown, profit_factor

from .config import TrancheEligibilityConfig


def _return_summary(values: pd.Series, *, cost: float) -> dict[str, object]:
    gross = values.astype(float).to_numpy()
    net = gross - float(cost)
    gains = net[net > 0]
    positive_profit = float(gains.sum())
    sorted_net = np.sort(net)[::-1]
    top_count = min(10, len(sorted_net))
    top_share = float(sorted_net[:top_count].sum() / positive_profit) if positive_profit > 0 else np.nan
    mdd, total = maximum_drawdown(net)
    return {
        "signals": int(len(net)),
        "mean_net_return": float(np.mean(net)) if len(net) else np.nan,
        "median_net_return": float(np.median(net)) if len(net) else np.nan,
        "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
        "profit_factor": profit_factor(net),
        "total_compounded_return": total,
        "max_drawdown_diagnostic": mdd,
        "top10_profit_share": top_share,
        "mean_net_without_top10": float(sorted_net[top_count:].mean()) if len(sorted_net) > top_count else np.nan,
    }


def occupancy_summary(event_audit: pd.DataFrame) -> pd.DataFrame:
    if event_audit.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in event_audit.groupby(["fold_id", "delay_minutes"], sort=True):
        fold_id, delay = keys
        candidate = int(group["candidate_events"].sum())
        complete = int(group["complete_events"].sum())
        executed = int(group["executed_events"].sum())
        occupied = int(group["occupied_events"].sum())
        rows.append(
            {
                "fold_id": fold_id,
                "delay_minutes": int(delay),
                "candidate_events": candidate,
                "complete_events": complete,
                "executed_events": executed,
                "occupied_events": occupied,
                "occupied_share_of_complete": float(occupied / complete) if complete else np.nan,
            }
        )
    return pd.DataFrame(rows)



def baseline_summary(
    baseline_trades: pd.DataFrame,
    standalone_outcomes: pd.DataFrame,
    config: TrancheEligibilityConfig,
) -> pd.DataFrame:
    """Summarize the two frozen references required by the stage contract."""

    rows: list[dict[str, object]] = []
    sources: list[tuple[str, pd.DataFrame]] = []
    if not baseline_trades.empty:
        sources.append(("P0_failed_reclaim_single_position", baseline_trades))
    if not standalone_outcomes.empty:
        fixed = standalone_outcomes.loc[standalone_outcomes["standalone_outcome"] == "fixed_6h"]
        if not fixed.empty:
            sources.append(("q70_fixed_6h_diagnostic", fixed))
    for baseline, frame in sources:
        for keys, group in frame.groupby(["fold_id", "delay_minutes"], sort=True):
            fold_id, delay = keys
            for multiplier in config.cost_multipliers:
                rows.append(
                    {
                        "baseline": baseline,
                        "fold_id": fold_id,
                        "delay_minutes": int(delay),
                        "cost_multiplier": float(multiplier),
                        **_return_summary(
                            group["gross_return"].dropna(),
                            cost=config.base_round_trip_cost * float(multiplier),
                        ),
                    }
                )
    return pd.DataFrame(rows)

def class_summary(atlas: pd.DataFrame, config: TrancheEligibilityConfig) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["fold_id", "delay_minutes", "signal_class", "eligible_for_tranche_simulation"]
    for values, group in atlas.groupby(keys, sort=True):
        common = dict(zip(keys, values, strict=True))
        for outcome in ("fixed6h_gross_return", "standalone_failed_reclaim_gross_return"):
            available = group.loc[group[outcome].notna()]
            for multiplier in config.cost_multipliers:
                rows.append(
                    {
                        **common,
                        "outcome": outcome,
                        "cost_multiplier": float(multiplier),
                        **_return_summary(
                            available[outcome],
                            cost=config.base_round_trip_cost * float(multiplier),
                        ),
                        "mean_current_return_vs_root": float(group["current_return_vs_root"].mean()),
                        "losing_position_share": float((group["current_return_vs_root"] < 0).mean()),
                        "mean_released_risk_fraction": float(group["released_risk_fraction"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def quarter_summary(atlas: pd.DataFrame, config: TrancheEligibilityConfig) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    work = atlas.loc[atlas["eligible_for_tranche_simulation"].astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["quarter"] = pd.to_datetime(work["new_entry_time"]).dt.to_period("Q").astype(str)
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(["fold_id", "delay_minutes", "quarter"], sort=True):
        fold_id, delay, quarter = keys
        rows.append(
            {
                "fold_id": fold_id,
                "delay_minutes": int(delay),
                "quarter": quarter,
                **_return_summary(
                    group["fixed6h_gross_return"].dropna(),
                    cost=config.base_round_trip_cost * 2.0,
                ),
            }
        )
    return pd.DataFrame(rows)


def score_tier_summary(atlas: pd.DataFrame, config: TrancheEligibilityConfig) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in atlas.groupby(["fold_id", "delay_minutes", "score_tier", "signal_class"], sort=True):
        fold_id, delay, tier, signal_class = keys
        rows.append(
            {
                "fold_id": fold_id,
                "delay_minutes": int(delay),
                "score_tier": tier,
                "signal_class": signal_class,
                **_return_summary(
                    group["fixed6h_gross_return"].dropna(),
                    cost=config.base_round_trip_cost * 2.0,
                ),
            }
        )
    return pd.DataFrame(rows)


def score_price_diagnostic(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    work = atlas.copy()
    work["score_up_price_down"] = (work["score_delta_vs_root"] > 0) & (work["price_return_at_new_entry"] < 0)
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(["fold_id", "delay_minutes", "signal_class"], sort=True):
        fold_id, delay, signal_class = keys
        rows.append(
            {
                "fold_id": fold_id,
                "delay_minutes": int(delay),
                "signal_class": signal_class,
                "signals": int(len(group)),
                "score_up_price_down_count": int(group["score_up_price_down"].sum()),
                "score_up_price_down_share": float(group["score_up_price_down"].mean()),
                "mean_score_delta": float(group["score_delta_vs_root"].mean()),
                "mean_price_return_at_new_entry": float(group["price_return_at_new_entry"].mean()),
            }
        )
    return pd.DataFrame(rows)


def risk_release_distribution(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in atlas.groupby(["fold_id", "delay_minutes", "signal_class"], sort=True):
        fold_id, delay, signal_class = keys
        released = group["released_risk_fraction"].astype(float)
        rows.append(
            {
                "fold_id": fold_id,
                "delay_minutes": int(delay),
                "signal_class": signal_class,
                "signals": int(len(group)),
                "released_risk_q10": float(released.quantile(0.10)),
                "released_risk_median": float(released.median()),
                "released_risk_q90": float(released.quantile(0.90)),
                "candidate_stop_above_entry_share": float((group["candidate_hard_stop_return"] >= 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def tranche_gate(
    atlas: pd.DataFrame,
    quarters: pd.DataFrame,
    config: TrancheEligibilityConfig,
) -> pd.DataFrame:
    """Gate only whether a separate P2/P3 simulation is justified."""

    rows: list[dict[str, object]] = []
    required_folds = {"WF_2024", "WF_2025"}
    for delay in config.entry_delay_minutes:
        eligible = atlas.loc[
            (atlas["delay_minutes"] == delay)
            & atlas["eligible_for_tranche_simulation"].astype(bool)
        ]
        fold_set = set(eligible["fold_id"].unique())
        row: dict[str, object] = {
            "delay_minutes": int(delay),
            "complete_2024_2025": fold_set == required_folds,
        }
        all_pass = fold_set == required_folds
        for fold_id in sorted(required_folds):
            fold = eligible.loc[eligible["fold_id"] == fold_id]
            combined = _return_summary(
                fold["fixed6h_gross_return"].dropna(),
                cost=config.base_round_trip_cost * 2.0,
            )
            combined_3x = _return_summary(
                fold["fixed6h_gross_return"].dropna(),
                cost=config.base_round_trip_cost * 3.0,
            )
            structural = _return_summary(
                fold["standalone_failed_reclaim_gross_return"].dropna(),
                cost=config.base_round_trip_cost * 2.0,
            )
            structural_3x = _return_summary(
                fold["standalone_failed_reclaim_gross_return"].dropna(),
                cost=config.base_round_trip_cost * 3.0,
            )
            positive_quarters = 0
            if not quarters.empty:
                q = quarters.loc[
                    (quarters["fold_id"] == fold_id)
                    & (quarters["delay_minutes"] == delay)
                ]
                positive_quarters = int((q["mean_net_return"] > 0).sum())
            losing_share = float((fold["current_return_vs_root"] < 0).mean()) if len(fold) else np.nan
            prefix = fold_id[-4:]
            row.update(
                {
                    f"eligible_signals_{prefix}": int(len(fold)),
                    f"mean_net_2x_{prefix}": combined["mean_net_return"],
                    f"pf_2x_{prefix}": combined["profit_factor"],
                    f"mean_net_3x_{prefix}": combined_3x["mean_net_return"],
                    f"top10_share_{prefix}": combined["top10_profit_share"],
                    f"mean_net_without_top10_{prefix}": combined["mean_net_without_top10"],
                    f"structural_mean_net_2x_{prefix}": structural["mean_net_return"],
                    f"structural_pf_2x_{prefix}": structural["profit_factor"],
                    f"structural_mean_net_3x_{prefix}": structural_3x["mean_net_return"],
                    f"structural_mean_net_without_top10_{prefix}": structural["mean_net_without_top10"],
                    f"positive_quarters_{prefix}": positive_quarters,
                    f"eligible_losing_share_{prefix}": losing_share,
                }
            )
            fold_pass = bool(
                len(fold) >= config.minimum_eligible_events_per_year
                and np.isfinite(combined["mean_net_return"])
                and float(combined["mean_net_return"]) > 0
                and np.isfinite(combined["profit_factor"])
                and float(combined["profit_factor"]) >= config.minimum_pf_2x
                and np.isfinite(combined_3x["mean_net_return"])
                and float(combined_3x["mean_net_return"]) > 0
                and positive_quarters >= config.minimum_positive_quarters_per_year
                and np.isfinite(combined["top10_profit_share"])
                and float(combined["top10_profit_share"]) <= config.maximum_top10_profit_share
                and np.isfinite(combined["mean_net_without_top10"])
                and float(combined["mean_net_without_top10"]) > 0
                and np.isfinite(structural["mean_net_return"])
                and float(structural["mean_net_return"]) > 0
                and np.isfinite(structural["profit_factor"])
                and float(structural["profit_factor"]) >= config.minimum_pf_2x
                and np.isfinite(structural_3x["mean_net_return"])
                and float(structural_3x["mean_net_return"]) > 0
                and np.isfinite(structural["mean_net_without_top10"])
                and float(structural["mean_net_without_top10"]) > 0
                and np.isfinite(losing_share)
                and losing_share <= config.maximum_eligible_losing_position_share
            )
            row[f"fold_pass_{prefix}"] = fold_pass
            all_pass = all_pass and fold_pass
        row["pass_to_tranche_simulation"] = bool(all_pass)
        rows.append(row)
    return pd.DataFrame(rows)
