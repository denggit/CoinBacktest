#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact account replay for fixed C2 exits under score-tier risk multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.research_common.progress import ProgressReporter

from .config import ScoreRiskConfig, ScoreRiskPolicy


@dataclass(frozen=True)
class ScoreRiskSimulation:
    cycles: pd.DataFrame
    legs: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    rejections: pd.DataFrame


def _fee(units: float, price: float, side_cost: float) -> float:
    return abs(float(units) * float(price)) * float(side_cost)


def _fold_dates(fold_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    year = 2024 if str(fold_id) == "WF_2024" else 2025
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")


def simulate_score_risk_account(
    source_cycles: pd.DataFrame,
    source_legs: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: ScoreRiskPolicy,
    delay_minutes: int,
    cost_multiplier: float,
    config: ScoreRiskConfig,
    progress: bool = True,
) -> ScoreRiskSimulation:
    cycle_work = source_cycles.loc[
        source_cycles["fold_id"].astype(str).eq(fold_id)
        & source_cycles["delay_minutes"].astype(int).eq(int(delay_minutes))
        & np.isclose(source_cycles["cost_multiplier"].astype(float), float(cost_multiplier))
    ].copy()
    leg_work = source_legs.loc[
        source_legs["fold_id"].astype(str).eq(fold_id)
        & source_legs["delay_minutes"].astype(int).eq(int(delay_minutes))
        & np.isclose(source_legs["cost_multiplier"].astype(float), float(cost_multiplier))
    ].copy()
    leg_columns = ["event_id", "entry_time", "exit_time", "entry_price", "exit_price", "exit_reason"]
    work = cycle_work.drop(columns=[c for c in ("entry_time", "exit_time", "exit_reason") if c in cycle_work.columns]).merge(
        leg_work[leg_columns], on="event_id", how="inner", validate="one_to_one"
    )
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False])
    if work.empty:
        return ScoreRiskSimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    side_cost = float(config.base_round_trip_cost * float(cost_multiplier) / 2.0)
    equity = float(config.initial_equity)
    peak = equity
    max_drawdown = 0.0
    cycle_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    daily_map: dict[pd.Timestamp, dict[str, object]] = {}
    risk_map = policy.risk_map

    reporter = ProgressReporter(
        f"[R03.4.2.13 {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        len(work), every=max(1, len(work) // 100), enabled=progress,
    )
    for number, event in enumerate(work.to_dict("records"), start=1):
        event_id = str(event["event_id"])
        tier = str(event["score_tier"])
        multiplier = float(risk_map[tier])
        entry_time = pd.Timestamp(event["entry_time"])
        exit_time = pd.Timestamp(event["exit_time"])
        entry_position = path.locate_exact(entry_time)
        exit_position = path.locate_exact(exit_time)
        if entry_position is None or exit_position is None or exit_position < entry_position:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "minute_path_missing"})
            reporter.update(number)
            continue
        entry_price = float(event["entry_price"])
        exit_price = float(event["exit_price"])
        hard_distance = float(event["hard_stop_distance"])
        equity_start = equity
        risk_budget = equity_start * config.account_risk_fraction_per_full_r * multiplier
        units = risk_budget / max(entry_price * hard_distance, 1e-12)
        entry_fee = _fee(units, entry_price, side_cost)
        cash = equity_start - entry_fee
        cycle_mdd = 0.0
        for position in range(entry_position, exit_position + 1):
            timestamp = pd.Timestamp(path.index[position])
            marked_price = exit_price if position == exit_position else float(path.close[position])
            exit_fee_preview = _fee(units, exit_price, side_cost) if position == exit_position else 0.0
            marked_equity = cash + units * (marked_price - entry_price) - exit_fee_preview
            peak = max(peak, marked_equity)
            drawdown = marked_equity / peak - 1.0 if peak > 0 else -1.0
            max_drawdown = min(max_drawdown, drawdown)
            cycle_mdd = min(cycle_mdd, drawdown)
            daily_map[timestamp.normalize()] = {
                "date": timestamp.normalize(),
                "equity": float(marked_equity),
                "drawdown": float(drawdown),
                "active_tranches": 0 if position == exit_position else 1,
                "risk_multiplier": 0.0 if position == exit_position else multiplier,
                "notional_to_equity": 0.0 if position == exit_position else float(units * float(path.close[position]) / max(marked_equity, 1e-12)),
            }
        exit_fee = _fee(units, exit_price, side_cost)
        gross = units * (exit_price - entry_price)
        net = gross - entry_fee - exit_fee
        equity = equity_start + net
        cycle_return = net / max(equity_start, 1e-12)
        cycle_rows.append({
            "event_id": event_id,
            "fold_id": fold_id,
            "policy": policy.name,
            "delay_minutes": int(delay_minutes),
            "cost_multiplier": float(cost_multiplier),
            "decision_time": pd.Timestamp(event["decision_time"]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_reason": str(event["exit_reason"]),
            "score": float(event["score"]),
            "score_percentile": float(event["score_percentile"]),
            "score_tier": tier,
            "risk_multiplier": multiplier,
            "equity_start": equity_start,
            "equity_end": equity,
            "cycle_net_pnl": net,
            "cycle_return": cycle_return,
            "hard_stop_distance": hard_distance,
            "base_notional_to_equity": units * entry_price / max(equity_start, 1e-12),
            "max_tail_r": multiplier,
            "cycle_max_drawdown": cycle_mdd,
            "hard_stop_exit": bool(event["hard_stop_exit"]),
            "soft_failure_exit": bool(event["soft_failure_exit"]),
        })
        leg_rows.append({
            "event_id": event_id, "fold_id": fold_id, "policy": policy.name,
            "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier),
            "score_tier": tier, "risk_multiplier": multiplier,
            "entry_time": entry_time, "exit_time": exit_time,
            "entry_price": entry_price, "exit_price": exit_price, "units": units,
            "entry_fee": entry_fee, "exit_fee": exit_fee, "gross_pnl": gross,
            "net_pnl": net, "exit_reason": str(event["exit_reason"]),
        })
        reporter.update(number)
    reporter.close()

    start, end = _fold_dates(fold_id)
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="D")
    filled: list[dict[str, object]] = []
    last_equity = config.initial_equity
    last_peak = config.initial_equity
    for date in calendar:
        row = daily_map.get(date)
        if row is not None:
            last_equity = float(row["equity"])
            last_peak = max(last_peak, last_equity)
        else:
            row = {"date": date, "equity": last_equity, "drawdown": last_equity / last_peak - 1.0, "active_tranches": 0, "risk_multiplier": 0.0, "notional_to_equity": 0.0}
        filled.append({**row, "fold_id": fold_id, "policy": policy.name, "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier)})

    cycles = pd.DataFrame(cycle_rows)
    legs = pd.DataFrame(leg_rows)
    daily = pd.DataFrame(filled)
    rejections = pd.DataFrame(rejection_rows)
    pnl = cycles["cycle_net_pnl"].to_numpy(float) if not cycles.empty else np.array([])
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    top = np.sort(winners)[::-1][:min(10, len(winners))]
    top_share = float(top.sum() / winners.sum()) if len(winners) and winners.sum() > 0 else np.nan
    without_top10 = float(equity - config.initial_equity - top.sum())
    positive_months = positive_quarters = 0
    if not daily.empty:
        series = daily.set_index("date")["equity"]
        month_end = series.resample("ME").last()
        quarter_end = series.resample("QE").last()
        monthly = month_end.pct_change().dropna()
        quarterly = quarter_end.pct_change().dropna()
        if len(month_end):
            monthly = pd.concat([pd.Series([month_end.iloc[0] / config.initial_equity - 1.0], index=[month_end.index[0]]), monthly])
        if len(quarter_end):
            quarterly = pd.concat([pd.Series([quarter_end.iloc[0] / config.initial_equity - 1.0], index=[quarter_end.index[0]]), quarterly])
        positive_months = int((monthly > 0).sum())
        positive_quarters = int((quarterly > 0).sum())
    summary = {
        "fold_id": fold_id, "policy": policy.name, "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier),
        "executed_cycles": int(len(cycles)), "final_equity": float(equity), "total_net_return": float(equity / config.initial_equity - 1.0),
        "max_drawdown": float(max_drawdown), "win_rate": float(np.mean(pnl > 0)) if len(pnl) else np.nan,
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_cycle_return": float(cycles["cycle_return"].mean()) if not cycles.empty else np.nan,
        "worst_cycle_return": float(cycles["cycle_return"].min()) if not cycles.empty else np.nan,
        "worst_cycle_loss_r": float(abs(min(cycles["cycle_return"].min() / config.account_risk_fraction_per_full_r, 0.0))) if not cycles.empty else np.nan,
        "top10_profit_share": top_share, "total_return_without_top10": without_top10,
        "hard_stop_exits": int(cycles["hard_stop_exit"].sum()) if not cycles.empty else 0,
        "soft_failure_exits": int(cycles["soft_failure_exit"].sum()) if not cycles.empty else 0,
        "mean_risk_multiplier": float(cycles["risk_multiplier"].mean()) if not cycles.empty else 0.0,
        "max_risk_multiplier": float(cycles["risk_multiplier"].max()) if not cycles.empty else 0.0,
        "mean_base_notional_to_equity": float(cycles["base_notional_to_equity"].mean()) if not cycles.empty else 0.0,
        "max_base_notional_to_equity": float(cycles["base_notional_to_equity"].max()) if not cycles.empty else 0.0,
        "positive_months": positive_months, "positive_quarters": positive_quarters,
        "runtime_rejections": int(len(rejections)),
    }
    return ScoreRiskSimulation(cycles, legs, daily, summary, rejections)
