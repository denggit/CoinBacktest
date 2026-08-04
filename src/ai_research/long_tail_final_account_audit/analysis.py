#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Account, concentration, lot-size and deployment analysis for R03.4.2.15."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .config import FinalAccountAuditConfig

FOLD_ORDER = ("WF_2024", "WF_2025")


def _longest_streak(values: pd.Series, predicate: Callable[[float], bool]) -> int:
    longest = current = 0
    for value in pd.to_numeric(values, errors="coerce").fillna(0.0):
        if predicate(float(value)):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _month_returns(daily: pd.DataFrame) -> pd.Series:
    equity = daily.set_index("date")["continuous_equity"].sort_index()
    month_end = equity.resample("ME").last()
    returns = month_end.pct_change(fill_method=None)
    if not returns.empty:
        returns.iloc[0] = float(month_end.iloc[0]) - 1.0
    return returns


def _quarter_returns(daily: pd.DataFrame) -> pd.Series:
    equity = daily.set_index("date")["continuous_equity"].sort_index()
    quarter_end = equity.resample("QE").last()
    returns = quarter_end.pct_change(fill_method=None)
    if not returns.empty:
        returns.iloc[0] = float(quarter_end.iloc[0]) - 1.0
    return returns


def _drawdown_duration_days(daily: pd.DataFrame) -> int:
    equity = pd.to_numeric(daily["continuous_equity"], errors="coerce")
    underwater = equity < equity.cummax() - 1e-12
    longest = current = 0
    for value in underwater.fillna(False):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return int(longest)


def build_continuous_scenarios(
    cycles: pd.DataFrame,
    daily_equity: pd.DataFrame,
    source_summary: pd.DataFrame,
    config: FinalAccountAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_cycles = cycles.loc[cycles["policy"].astype(str).eq(config.source_policy)].copy()
    source_daily = daily_equity.loc[daily_equity["policy"].astype(str).eq(config.source_policy)].copy()
    source_summary = source_summary.loc[source_summary["policy"].astype(str).eq(config.source_policy)].copy()

    all_cycles: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    scenario_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    quarter_rows: list[dict[str, object]] = []

    for delay in config.entry_delay_minutes:
        for cost in config.cost_multipliers:
            scenario = source_cycles.loc[
                source_cycles["delay_minutes"].astype(int).eq(delay)
                & source_cycles["cost_multiplier"].astype(float).eq(cost)
            ].sort_values(["entry_time", "event_id"]).copy()
            if scenario.empty:
                continue

            equity = 1.0
            starts: list[float] = []
            pnls: list[float] = []
            ends: list[float] = []
            for row in scenario.itertuples(index=False):
                cycle_return = float(row.cycle_return)
                starts.append(equity)
                pnl = equity * cycle_return
                equity += pnl
                pnls.append(pnl)
                ends.append(equity)
            scenario["continuous_equity_start"] = starts
            scenario["continuous_net_pnl"] = pnls
            scenario["continuous_equity_end"] = ends
            all_cycles.append(scenario)

            scenario_daily = source_daily.loc[
                source_daily["delay_minutes"].astype(int).eq(delay)
                & source_daily["cost_multiplier"].astype(float).eq(cost)
            ].copy()
            scaled_parts: list[pd.DataFrame] = []
            fold_scale = 1.0
            for fold in FOLD_ORDER:
                part = scenario_daily.loc[scenario_daily["fold_id"].astype(str).eq(fold)].sort_values("date").copy()
                if part.empty:
                    continue
                part["continuous_equity"] = pd.to_numeric(part["equity"], errors="coerce") * fold_scale
                scaled_parts.append(part)
                fold_summary = source_summary.loc[
                    source_summary["fold_id"].astype(str).eq(fold)
                    & source_summary["delay_minutes"].astype(int).eq(delay)
                    & source_summary["cost_multiplier"].astype(float).eq(cost)
                ]
                if fold_summary.empty:
                    fold_scale = float(part["continuous_equity"].iloc[-1])
                else:
                    fold_scale *= float(fold_summary.iloc[0]["final_equity"])
            continuous_daily = pd.concat(scaled_parts, ignore_index=True).sort_values("date")
            continuous_daily["continuous_drawdown"] = (
                continuous_daily["continuous_equity"] / continuous_daily["continuous_equity"].cummax() - 1.0
            )
            all_daily.append(continuous_daily)

            month_returns = _month_returns(continuous_daily)
            quarter_returns = _quarter_returns(continuous_daily)
            month_rows.extend(
                {"delay_minutes": delay, "cost_multiplier": cost, "month": str(index.to_period("M")), "return": float(value)}
                for index, value in month_returns.items()
            )
            quarter_rows.extend(
                {"delay_minutes": delay, "cost_multiplier": cost, "quarter": str(index.to_period("Q")), "return": float(value)}
                for index, value in quarter_returns.items()
            )

            positive_pnl = scenario.loc[scenario["continuous_net_pnl"] > 0, "continuous_net_pnl"]
            negative_pnl = -scenario.loc[scenario["continuous_net_pnl"] < 0, "continuous_net_pnl"]
            profit_factor = float(positive_pnl.sum() / negative_pnl.sum()) if float(negative_pnl.sum()) > 0 else float("inf")
            top10 = scenario.nlargest(10, "continuous_net_pnl")
            top10_share = float(top10["continuous_net_pnl"].sum() / positive_pnl.sum()) if float(positive_pnl.sum()) > 0 else float("nan")
            top_ids = set(top10.index)
            no_top_equity = 1.0
            for index, row in scenario.iterrows():
                no_top_equity *= 1.0 + (0.0 if index in top_ids else float(row["cycle_return"]))

            hold_hours = (scenario["exit_time"] - scenario["entry_time"]).dt.total_seconds() / 3600.0
            entry_gaps = scenario["entry_time"].sort_values().diff().dt.total_seconds() / 3600.0
            source_mdd = source_summary.loc[
                source_summary["delay_minutes"].astype(int).eq(delay)
                & source_summary["cost_multiplier"].astype(float).eq(cost),
                "max_drawdown",
            ]
            # Daily output is end-of-day; retain the worse minute-marked annual source MDD.
            max_drawdown = min(float(continuous_daily["continuous_drawdown"].min()), float(source_mdd.min()))
            years = max((pd.Timestamp(config.oos_end) - pd.Timestamp(config.oos_start)).days / 365.25, 1.0)
            final_equity = float(scenario["continuous_equity_end"].iloc[-1])
            scenario_rows.append(
                {
                    "delay_minutes": delay,
                    "cost_multiplier": cost,
                    "trades": int(len(scenario)),
                    "trades_per_month": float(len(scenario) / 24.0),
                    "final_equity": final_equity,
                    "total_return": final_equity - 1.0,
                    "cagr": final_equity ** (1.0 / years) - 1.0,
                    "max_drawdown": max_drawdown,
                    "calmar": (final_equity ** (1.0 / years) - 1.0) / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
                    "win_rate": float((scenario["cycle_return"] > 0).mean()),
                    "profit_factor": profit_factor,
                    "mean_cycle_return": float(scenario["cycle_return"].mean()),
                    "worst_cycle_return": float(scenario["cycle_return"].min()),
                    "worst_net_r": float(abs(scenario["cycle_return"].min()) / config.account_risk_fraction_per_full_r),
                    "positive_months": int((month_returns > 0).sum()),
                    "months": int(len(month_returns)),
                    "positive_quarters": int((quarter_returns > 0).sum()),
                    "quarters": int(len(quarter_returns)),
                    "best_month": float(month_returns.max()),
                    "worst_month": float(month_returns.min()),
                    "mean_hold_hours": float(hold_hours.mean()),
                    "median_hold_hours": float(hold_hours.median()),
                    "p90_hold_hours": float(hold_hours.quantile(0.90)),
                    "max_hold_hours": float(hold_hours.max()),
                    "max_entry_gap_hours": float(entry_gaps.max()),
                    "longest_losing_streak": _longest_streak(scenario["cycle_return"], lambda value: value <= 0),
                    "longest_winning_streak": _longest_streak(scenario["cycle_return"], lambda value: value > 0),
                    "max_drawdown_duration_days": _drawdown_duration_days(continuous_daily),
                    "top10_profit_share": top10_share,
                    "total_return_without_top10": no_top_equity - 1.0,
                    "hard_stop_share": float(scenario["hard_stop_exit"].astype(bool).mean()),
                    "soft_failure_share": float(scenario["soft_failure_exit"].astype(bool).mean()),
                    "censored_share": float(scenario["exit_reason"].astype(str).str.contains("censored", case=False).mean()),
                }
            )

    cycles_out = pd.concat(all_cycles, ignore_index=True) if all_cycles else pd.DataFrame()
    daily_out = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    return cycles_out, daily_out, pd.DataFrame(scenario_rows), pd.DataFrame(month_rows), pd.DataFrame(quarter_rows)


def build_lot_size_audit(
    legs: pd.DataFrame,
    config: FinalAccountAuditConfig,
    *,
    live_price_risk_fraction: float | None = None,
) -> pd.DataFrame:
    anchor = legs.loc[
        legs["policy"].astype(str).eq(config.source_policy)
        & legs["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & legs["cost_multiplier"].astype(float).eq(config.anchor_cost_multiplier)
    ].copy()
    profiles = [("research_1pct_price_budget", config.account_risk_fraction_per_full_r)]
    if live_price_risk_fraction is not None:
        profiles.append(("conservative_live_budget", float(live_price_risk_fraction)))
    rows: list[dict[str, object]] = []
    for profile, price_risk_fraction in profiles:
        target_notional_multiple = price_risk_fraction / config.hard_stop_distance
        for equity in config.initial_equity_tiers:
            contract_notional = pd.to_numeric(anchor["entry_price"], errors="coerce") * config.contract_value_base
            target_notional = equity * target_notional_multiple
            contracts = np.floor(target_notional / contract_notional)
            actual_notional = contracts * contract_notional
            actual_risk = actual_notional / equity * config.hard_stop_distance
            tradable = contracts >= config.minimum_contracts
            rows.append(
                {
                    "sizing_profile": profile,
                    "target_price_risk_fraction": price_risk_fraction,
                    "initial_equity_usdt": equity,
                    "target_notional_multiple": target_notional_multiple,
                    "untradable_share": float((~tradable).mean()),
                    "mean_contracts": float(contracts.mean()),
                    "median_contracts": float(contracts.median()),
                    "mean_actual_notional_multiple": float((actual_notional / equity).mean()),
                    "mean_actual_price_risk_fraction": float(actual_risk.mean()),
                    "minimum_actual_price_risk_fraction_when_tradable": float(actual_risk.loc[tradable].min()) if tradable.any() else float("nan"),
                    "maximum_actual_price_risk_fraction": float(actual_risk.max()),
                    "mean_sizing_efficiency": float((actual_notional / target_notional).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_risk_reserve_audit(scenarios: pd.DataFrame, config: FinalAccountAuditConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in scenarios.to_dict("records"):
        worst_r = float(row["worst_net_r"])
        reserve_scalar = min(1.0, 1.0 / max(worst_r, 1.0))
        rows.append(
            {
                "delay_minutes": int(row["delay_minutes"]),
                "cost_multiplier": float(row["cost_multiplier"]),
                "observed_worst_net_r_at_1pct_price_budget": worst_r,
                "maximum_price_risk_budget_for_1pct_net_tail": config.account_risk_fraction_per_full_r * reserve_scalar,
                "recommended_live_price_risk_budget": min(0.0090, config.account_risk_fraction_per_full_r * reserve_scalar),
                "fee_slippage_reserve": config.account_risk_fraction_per_full_r - min(0.0090, config.account_risk_fraction_per_full_r * reserve_scalar),
            }
        )
    return pd.DataFrame(rows)


def build_model_governance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cadence": "continuous", "action": "live inference", "promotion": "none", "rule": "serve one immutable champion model/version until an explicit release"},
            {"cadence": "daily", "action": "data and feature health", "promotion": "none", "rule": "alert on missing bars, schema drift, stale features and inference failures"},
            {"cadence": "monthly", "action": "performance and drift audit", "promotion": "none", "rule": "review score distribution, q70 frequency, calibration, MAE, PF, costs and regime drift"},
            {"cadence": "monthly optional", "action": "shadow candidate retrain", "promotion": "forbidden automatically", "rule": "candidate may train on expanding/rolling causal data with embargo; champion remains live"},
            {"cadence": "quarterly or event-driven", "action": "release gate", "promotion": "manual explicit", "rule": "promote only after frozen OOS, stress, shadow and rollback gates; otherwise keep champion"},
            {"cadence": "event-driven", "action": "emergency rollback", "promotion": "rollback only", "rule": "schema break, persistent drift or risk-control failure reverts to last known-good artifact"},
        ]
    )


def build_live_state_contract() -> pd.DataFrame:
    rows = [
        ("model_version", "immutable artifact id and training cutoff"),
        ("feature_schema_hash", "exact multi-timeframe causal feature schema"),
        ("q70_threshold", "frozen calibration threshold for the deployed artifact"),
        ("decision_time", "completed 15m decision timestamp"),
        ("score", "opening score produced by the deployed model"),
        ("entry_order_id", "next observable 1m-open execution/order identity"),
        ("entry_price", "actual fill used for all stop distances"),
        ("initial_equity", "equity snapshot used for risk sizing"),
        ("hard_stop_price", "real exchange-side 2% protection"),
        ("hard_stop_order_id", "recoverable exchange order identity"),
        ("soft_failure_armed", "whether adverse excursion reached 1.5%"),
        ("soft_failure_confirmation_time", "completed 15m close that confirms failure"),
        ("structure_break_level", "causal failed_reclaim structure floor"),
        ("lower_high_seen", "failed-reclaim state flag"),
        ("last_completed_structure_bar", "restart-safe 15m state timestamp"),
        ("exit_order_id", "active/filled structural exit identity"),
        ("position_state", "flat/open/exit_pending/recovery_required"),
    ]
    return pd.DataFrame(rows, columns=["field", "purpose"])


def build_gate(scenarios: pd.DataFrame, risk: pd.DataFrame, config: FinalAccountAuditConfig) -> pd.DataFrame:
    anchor = scenarios.loc[
        scenarios["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
        & scenarios["cost_multiplier"].astype(float).eq(config.anchor_cost_multiplier)
    ]
    stress = scenarios.loc[scenarios["cost_multiplier"].astype(float).eq(max(config.cost_multipliers))]
    checks = [
        ("complete_scenario_grid", len(scenarios) == len(config.entry_delay_minutes) * len(config.cost_multipliers)),
        ("anchor_return", not anchor.empty and float(anchor.iloc[0]["total_return"]) >= config.minimum_anchor_total_return),
        ("anchor_mdd", not anchor.empty and abs(float(anchor.iloc[0]["max_drawdown"])) <= config.maximum_anchor_mdd),
        ("anchor_pf", not anchor.empty and float(anchor.iloc[0]["profit_factor"]) >= config.minimum_anchor_profit_factor),
        ("monthly_stability", not anchor.empty and int(anchor.iloc[0]["positive_months"]) >= config.minimum_positive_months),
        ("quarterly_stability", not anchor.empty and int(anchor.iloc[0]["positive_quarters"]) >= config.minimum_positive_quarters),
        ("losing_streak", not anchor.empty and int(anchor.iloc[0]["longest_losing_streak"]) <= config.maximum_losing_streak),
        ("drawdown_duration", not anchor.empty and int(anchor.iloc[0]["max_drawdown_duration_days"]) <= config.maximum_drawdown_duration_days),
        ("top10_robustness", not anchor.empty and float(anchor.iloc[0]["total_return_without_top10"]) > config.minimum_return_without_top10 and float(anchor.iloc[0]["top10_profit_share"]) <= config.maximum_top10_profit_share),
        ("net_tail", not anchor.empty and float(anchor.iloc[0]["worst_net_r"]) <= config.maximum_anchor_worst_net_r),
        ("stress_profit", not stress.empty and bool((stress["total_return"] > 0).all())),
        ("stress_mdd", not stress.empty and bool((stress["max_drawdown"].abs() <= config.maximum_stress_mdd).all())),
        ("risk_reserve_defined", not risk.empty and bool((risk["recommended_live_price_risk_budget"] > 0).all())),
    ]
    return pd.DataFrame([{"check": name, "pass": bool(status)} for name, status in checks])
