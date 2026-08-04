#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metrics and pre-registered gates for the 2026 sealed holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SealedHoldoutConfig


def profit_factor(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def summarize_fixed_diagnostic(trades: pd.DataFrame, config: SealedHoldoutConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for delay, group in trades.groupby("delay_minutes", sort=True):
        for cost in config.cost_multipliers:
            net = group["gross_return"].astype(float).to_numpy() - config.base_round_trip_cost * float(cost)
            winners = net[net > 0]
            losers = net[net < 0]
            rows.append(
                {
                    "fold_id": "WF_2026_SEALED",
                    "delay_minutes": int(delay),
                    "cost_multiplier": float(cost),
                    "signals": int(len(group)),
                    "mean_net_return": float(np.mean(net)) if len(net) else np.nan,
                    "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
                    "profit_factor": profit_factor(net),
                    "mean_mfe": float(group["mfe"].mean()),
                    "mean_mae": float(group["mae"].mean()),
                    "mean_winner": float(np.mean(winners)) if len(winners) else np.nan,
                    "mean_loser": float(np.mean(losers)) if len(losers) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def enrich_account_summary(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    selected: pd.DataFrame,
    config: SealedHoldoutConfig,
) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    result = summary.copy()
    span_months = max(
        1.0,
        (pd.Timestamp(config.holdout_end) - pd.Timestamp(config.holdout_start)).total_seconds()
        / (365.2425 / 12.0 * 86400.0),
    )
    extra: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        subset = cycles.loc[
            cycles["delay_minutes"].astype(int).eq(int(row["delay_minutes"]))
            & np.isclose(cycles["cost_multiplier"].astype(float), float(row["cost_multiplier"]))
        ].copy()
        source = selected.loc[selected["delay_minutes"].astype(int).eq(int(row["delay_minutes"]))].copy()
        if not subset.empty and "source_is_censored" in subset.columns:
            censored = int(subset["source_is_censored"].fillna(False).astype(bool).sum())
        elif not source.empty and "is_censored" in source.columns:
            censored = int(source["is_censored"].astype(bool).sum())
        else:
            censored = 0
        holds = (
            (pd.to_datetime(subset["exit_time"]) - pd.to_datetime(subset["entry_time"])).dt.total_seconds() / 3600.0
            if not subset.empty
            else pd.Series(dtype=float)
        )
        extra.append(
            {
                "delay_minutes": int(row["delay_minutes"]),
                "cost_multiplier": float(row["cost_multiplier"]),
                "trades_per_month": float(len(subset) / span_months),
                "censored_cycles": censored,
                "censored_share": float(censored / len(subset)) if len(subset) else 0.0,
                "mean_hold_hours": float(holds.mean()) if len(holds) else np.nan,
                "median_hold_hours": float(holds.median()) if len(holds) else np.nan,
                "p90_hold_hours": float(holds.quantile(0.90)) if len(holds) else np.nan,
                "max_hold_hours": float(holds.max()) if len(holds) else np.nan,
            }
        )
    return result.merge(pd.DataFrame(extra), on=["delay_minutes", "cost_multiplier"], how="left")


def period_returns(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows_m: list[dict[str, object]] = []
    rows_q: list[dict[str, object]] = []
    for (delay, cost), group in daily.groupby(["delay_minutes", "cost_multiplier"], sort=True):
        ordered = group.sort_values("date").copy()
        ordered.index = pd.to_datetime(ordered["date"])
        series = ordered["equity"].astype(float)
        for freq, rows, kind in (("ME", rows_m, "month"), ("QE", rows_q, "quarter")):
            ends = series.resample(freq).last()
            changes = ends.pct_change()
            if len(ends):
                changes.iloc[0] = ends.iloc[0] - 1.0
            for timestamp, value in changes.items():
                rows.append(
                    {
                        "delay_minutes": int(delay),
                        "cost_multiplier": float(cost),
                        "period_kind": kind,
                        "period": str(timestamp.to_period("M" if kind == "month" else "Q")),
                        "return": float(value),
                    }
                )
    return pd.DataFrame(rows_m), pd.DataFrame(rows_q)


def build_gate(
    summary: pd.DataFrame,
    score_audit: pd.DataFrame,
    seal_check: dict[str, object],
    config: SealedHoldoutConfig,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    anchor = summary.loc[
        summary["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & np.isclose(summary["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
    ]
    all_cells = len(summary) == len(config.entry_delay_minutes) * len(config.cost_multipliers)
    if len(anchor) != 1:
        return pd.DataFrame([{"check": "anchor_cell_unique", "pass": False, "value": len(anchor), "threshold": 1, "gate_class": "hard"}])
    row = anchor.iloc[0]
    records: list[dict[str, object]] = [
        {"check": "seal_unchanged", "pass": bool(seal_check.get("unchanged", False)), "value": seal_check.get("status"), "threshold": "PASS", "gate_class": "hard"},
        {"check": "all_frozen_cells_complete", "pass": bool(all_cells), "value": int(len(summary)), "threshold": int(len(config.entry_delay_minutes) * len(config.cost_multipliers)), "gate_class": "hard"},
        {"check": "feature_schema_matches_history", "pass": bool(score_audit.get("feature_schema_matches_history", pd.Series([False])).iloc[0]), "value": score_audit.get("feature_schema_hash", pd.Series([""])).iloc[0], "threshold": score_audit.get("historical_feature_schema_hash", pd.Series([""])).iloc[0], "gate_class": "hard"},
        {"check": "minimum_holdout_trades", "pass": int(row["executed_cycles"]) >= config.minimum_executed_cycles, "value": int(row["executed_cycles"]), "threshold": config.minimum_executed_cycles, "gate_class": "hard"},
        {"check": "anchor_positive_return", "pass": float(row["total_net_return"]) > config.minimum_anchor_total_return, "value": float(row["total_net_return"]), "threshold": config.minimum_anchor_total_return, "gate_class": "hard"},
        {"check": "anchor_profit_factor", "pass": float(row["profit_factor"]) >= config.minimum_anchor_profit_factor, "value": float(row["profit_factor"]), "threshold": config.minimum_anchor_profit_factor, "gate_class": "hard"},
        {"check": "anchor_mdd", "pass": abs(float(row["max_drawdown"])) <= config.maximum_anchor_mdd, "value": float(row["max_drawdown"]), "threshold": config.maximum_anchor_mdd, "gate_class": "hard"},
        {"check": "all_cost_delay_cells_profitable", "pass": bool((summary["total_net_return"].astype(float) > 0).all()), "value": float(summary["total_net_return"].astype(float).min()), "threshold": 0.0, "gate_class": "hard"},
        {"check": "stress_mdd", "pass": bool((summary["max_drawdown"].astype(float).abs() <= config.maximum_stress_mdd).all()), "value": float(summary["max_drawdown"].astype(float).abs().max()), "threshold": config.maximum_stress_mdd, "gate_class": "hard"},
        {"check": "worst_net_r", "pass": bool((summary["worst_cycle_loss_r"].astype(float) <= config.maximum_worst_net_r).all()), "value": float(summary["worst_cycle_loss_r"].astype(float).max()), "threshold": config.maximum_worst_net_r, "gate_class": "hard"},
        {"check": "censored_cycles", "pass": int(row.get("censored_cycles", 0)) <= config.maximum_censored_cycles, "value": int(row.get("censored_cycles", 0)), "threshold": config.maximum_censored_cycles, "gate_class": "hard"},
        {"check": "positive_months", "pass": int(row["positive_months"]) >= config.minimum_positive_months, "value": int(row["positive_months"]), "threshold": config.minimum_positive_months, "gate_class": "quality"},
        {"check": "positive_quarters", "pass": int(row["positive_quarters"]) >= config.minimum_positive_quarters, "value": int(row["positive_quarters"]), "threshold": config.minimum_positive_quarters, "gate_class": "quality"},
        {"check": "top10_concentration", "pass": float(row["top10_profit_share"]) <= config.maximum_top10_profit_share, "value": float(row["top10_profit_share"]), "threshold": config.maximum_top10_profit_share, "gate_class": "quality"},
        {"check": "return_without_top10", "pass": float(row["total_return_without_top10"]) >= config.minimum_return_without_top10, "value": float(row["total_return_without_top10"]), "threshold": config.minimum_return_without_top10, "gate_class": "quality"},
    ]
    return pd.DataFrame(records)


def extended_account_summary(source_scenarios: pd.DataFrame, holdout_summary: pd.DataFrame, config: SealedHoldoutConfig) -> pd.DataFrame:
    if source_scenarios.empty or holdout_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for holdout in holdout_summary.to_dict("records"):
        source = source_scenarios.loc[
            source_scenarios["delay_minutes"].astype(int).eq(int(holdout["delay_minutes"]))
            & np.isclose(source_scenarios["cost_multiplier"].astype(float), float(holdout["cost_multiplier"]))
        ]
        if len(source) != 1:
            continue
        prior = source.iloc[0]
        final_equity = float(prior["final_equity"]) * float(holdout["final_equity"])
        rows.append(
            {
                "delay_minutes": int(holdout["delay_minutes"]),
                "cost_multiplier": float(holdout["cost_multiplier"]),
                "oos_start": "2024-01-01",
                "oos_end": config.holdout_end,
                "trades": int(prior["trades"]) + int(holdout["executed_cycles"]),
                "final_equity": final_equity,
                "total_return": final_equity - 1.0,
                "note": "live-style stitched walk-forward diagnostic; 2026 uses a newly fitted pre-2026 champion under the frozen recipe",
            }
        )
    return pd.DataFrame(rows)
