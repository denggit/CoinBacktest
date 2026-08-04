#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal entry timing selection and frozen-C2 account replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_soft_failure_tail_compression.config import TailCompressionConfig, TailCompressionPolicy
from src.ai_research.long_tail_soft_failure_tail_compression.simulator import _simulate_cycle
from src.ai_research.long_tail_structural_exit.config import StructuralPolicy
from src.ai_research.long_tail_structural_exit.simulator import simulate_structural_event
from src.ai_research.long_tail_structural_exit.structure import build_event_bars
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.research_common.progress import ProgressReporter

from .config import EntryTimingConfig, EntryTimingPolicy


@dataclass(frozen=True)
class EntryTimingSimulation:
    decisions: pd.DataFrame
    cycles: pd.DataFrame
    legs: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    rejections: pd.DataFrame


def _entry_position_for_policy(
    event: dict[str, object],
    signal_timeline: pd.DataFrame,
    *,
    path: MinutePathData,
    policy: EntryTimingPolicy,
    delay_minutes: int,
) -> tuple[int | None, dict[str, object]]:
    decision = pd.Timestamp(event["decision_time"])
    base_position = path.locate_exact(decision + pd.Timedelta(minutes=delay_minutes))
    if base_position is None:
        return None, {"trigger_reason": "missing_immediate_open"}
    base_price = float(path.open[base_position])
    base_atr = float(path.prior_atr_60[base_position])
    base_atr = base_atr if np.isfinite(base_atr) and base_atr > 0 else base_price * 0.0025
    root_score = float(event["score"])

    if policy.mode == "immediate":
        return base_position, {"trigger_reason": "immediate_q70", "fallback_used": False, "trigger_score": root_score, "trigger_score_delta": 0.0, "immediate_entry_price": base_price}

    deadline = decision + pd.Timedelta(minutes=policy.max_wait_minutes)
    if "decision_time" in signal_timeline.columns:
        later = signal_timeline.loc[(signal_timeline["decision_time"] > decision) & (signal_timeline["decision_time"] <= deadline)].sort_values("decision_time")
    else:
        later = pd.DataFrame()
    if policy.mode in {"score_rise", "score_rise_no_chase"}:
        for row in later.to_dict("records"):
            score = float(row["score"])
            if score <= root_score + policy.minimum_score_increase:
                continue
            candidate_position = path.locate_exact(pd.Timestamp(row["decision_time"]) + pd.Timedelta(minutes=delay_minutes))
            if candidate_position is None:
                continue
            if policy.mode == "score_rise_no_chase":
                candidate_price = float(path.open[candidate_position])
                if candidate_price > base_price + policy.maximum_chase_atr * base_atr:
                    continue
            return candidate_position, {"trigger_reason": policy.mode, "fallback_used": False, "trigger_score": score, "trigger_score_delta": score - root_score, "immediate_entry_price": base_price}

    if policy.mode == "pullback_reclaim":
        arm_price = base_price * (1.0 - policy.pullback_fraction)
        reclaim_price = base_price * (1.0 - policy.reclaim_tolerance_fraction)
        deadline_position = path.locate_exact(deadline + pd.Timedelta(minutes=delay_minutes))
        if deadline_position is None:
            deadline_position = min(base_position + policy.max_wait_minutes, len(path.index) - 1)
        armed = False
        for position in range(base_position, max(base_position + 1, deadline_position + 1)):
            if float(path.low[position]) <= arm_price:
                armed = True
            timestamp = pd.Timestamp(path.index[position])
            # A five-minute bar ending at minute 4/9/14/... becomes available at the next open.
            if armed and timestamp.minute % 5 == 4 and float(path.close[position]) >= reclaim_price:
                entry_position = position + max(1, delay_minutes)
                if entry_position < len(path.index):
                    return entry_position, {"trigger_reason": "pullback_reclaim_5m_close", "fallback_used": False, "trigger_score": root_score, "trigger_score_delta": 0.0, "immediate_entry_price": base_price}

    fallback_position = path.locate_exact(deadline + pd.Timedelta(minutes=delay_minutes))
    if fallback_position is None:
        return None, {"trigger_reason": "missing_fallback_open"}
    return fallback_position, {"trigger_reason": "bounded_wait_fallback", "fallback_used": True, "trigger_score": root_score, "trigger_score_delta": 0.0, "immediate_entry_price": base_price}


def _soft_timeline(path: MinutePathData, entry_position: int, exit_position: int, event_id: str, fold_id: str, delay_minutes: int, structural_config: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bars = build_event_bars(path, entry_position=entry_position, end_position=exit_position, config=structural_config)
    if bars is None:
        return pd.DataFrame()
    for bar_index in range(int(bars.entry_bar_index), len(bars.close)):
        close_position = int(bars.close_position[bar_index])
        effective_position = close_position + 1
        if effective_position > exit_position or effective_position >= len(path.index):
            continue
        rows.append({"event_id": event_id, "fold_id": fold_id, "delay_minutes": delay_minutes, "effective_time": pd.Timestamp(path.index[effective_position]), "current_close": float(bars.close[bar_index])})
    return pd.DataFrame(rows)


def _early_path_metrics(path: MinutePathData, entry_position: int, exit_position: int, entry_price: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in (15, 30, 60, 120):
        right = min(exit_position, entry_position + horizon - 1)
        lows = path.low[entry_position : right + 1]
        highs = path.high[entry_position : right + 1]
        result[f"mae_{horizon}m"] = float(np.min(lows) / entry_price - 1.0) if len(lows) else np.nan
        result[f"mfe_{horizon}m"] = float(np.max(highs) / entry_price - 1.0) if len(highs) else np.nan
    lows = path.low[entry_position : exit_position + 1]
    highs = path.high[entry_position : exit_position + 1]
    result["full_mae"] = float(np.min(lows) / entry_price - 1.0)
    result["full_mfe"] = float(np.max(highs) / entry_price - 1.0)
    return result


def simulate_entry_timing_account(
    events: pd.DataFrame,
    all_signals: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: EntryTimingPolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: EntryTimingConfig,
    progress: bool = True,
) -> EntryTimingSimulation:
    work = events.loc[events["fold_id"].astype(str).eq(fold_id) & events["delay_minutes"].astype(int).eq(delay_minutes)].copy().sort_values(["decision_time", "score"], ascending=[True, False])
    signals = all_signals.loc[all_signals["fold_id"].astype(str).eq(fold_id) & all_signals["delay_minutes"].astype(int).eq(delay_minutes)].copy().sort_values("decision_time")
    if work.empty:
        return EntryTimingSimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    structural_config = TrancheEligibilityConfig().structural_config()
    structural_policy = StructuralPolicy(name="failed_reclaim", exit_on_failed_reclaim=True)
    c2_policy = TailCompressionPolicy("C2_replay", "fixed", sizing_stop_distance=config.hard_stop_distance, hard_stop_distance=config.hard_stop_distance, soft_failure_distance=config.soft_failure_distance)
    c2_config = TailCompressionConfig(account_risk_fraction_per_full_r=config.account_risk_fraction_per_full_r, initial_equity=config.initial_equity, policies=(c2_policy,))
    side_cost = config.base_round_trip_cost * cost_multiplier / 2.0

    equity = config.initial_equity
    peak = equity
    max_drawdown = 0.0
    occupied_until = pd.Timestamp(test_start) - pd.Timedelta(minutes=1)
    decision_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    daily_map: dict[pd.Timestamp, dict[str, object]] = {}
    rejection_rows: list[dict[str, object]] = []
    reporter = ProgressReporter(f"[R03.4.2.14 {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]", len(work), every=max(1, len(work)//100), enabled=progress)

    for number, event in enumerate(work.to_dict("records"), start=1):
        event_id = str(event["event_id"])
        decision_time = pd.Timestamp(event["decision_time"])
        if decision_time < occupied_until:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "occupied_by_prior_alternative"})
            reporter.update(number)
            continue
        entry_position, meta = _entry_position_for_policy(event, signals, path=path, policy=policy, delay_minutes=delay_minutes)
        if entry_position is None:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": str(meta.get("trigger_reason"))})
            reporter.update(number)
            continue
        entry_time = pd.Timestamp(path.index[entry_position])
        source_exit_time = pd.Timestamp(event["exit_time"])
        if entry_time >= source_exit_time:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "entry_after_frozen_source_exit"})
            reporter.update(number)
            continue

        replay_event = EventCandidate(event_id=event_id, decision_time_ns=int(path.timestamps_ns[entry_position] - pd.Timedelta(minutes=1).value), score=float(event["score"]), signal_quantile=float(event["signal_quantile"]))
        structural = simulate_structural_event(replay_event, fold_id=fold_id, policy=structural_policy, delay_minutes=1, percentile=float(event["score_percentile"]), path=path, oos_end_ns=int(pd.Timestamp(test_end).value), config=structural_config)
        if structural is None:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "structural_replay_unavailable"})
            reporter.update(number)
            continue
        structural_values = structural.to_dict()
        structural_exit_position = path.locate_exact(pd.Timestamp(structural_values["exit_time"]))
        if structural_exit_position is None or structural_exit_position <= entry_position:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "invalid_structural_exit"})
            reporter.update(number)
            continue
        soft_timeline = _soft_timeline(path, entry_position, structural_exit_position, event_id, fold_id, delay_minutes, structural_config)
        synthetic = {
            **event,
            "entry_time": entry_time,
            "entry_price": float(path.open[entry_position]),
            "exit_time": pd.Timestamp(structural_values["exit_time"]),
            "exit_price": float(structural_values["exit_price"]),
            "exit_reason": str(structural_values["exit_reason"]),
        }
        try:
            equity_end, peak, cycle_mdd, cycle, leg, _actions, daily = _simulate_cycle(synthetic, soft_timeline, path=path, policy=c2_policy, config=c2_config, side_cost=side_cost, equity_start=equity, global_peak=peak)
        except Exception as exc:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": f"{type(exc).__name__}: {exc}"})
            reporter.update(number)
            continue
        exit_position = path.locate_exact(pd.Timestamp(cycle["exit_time"]))
        if exit_position is None:
            rejection_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "reason": "missing_final_exit"})
            reporter.update(number)
            continue
        metrics = _early_path_metrics(path, entry_position, exit_position, float(synthetic["entry_price"]))
        wait_minutes = int((entry_time - decision_time) / pd.Timedelta(minutes=1))
        immediate_price = float(meta.get("immediate_entry_price", synthetic["entry_price"]))
        decision_rows.append({"event_id": event_id, "fold_id": fold_id, "policy": policy.name, "delay_minutes": delay_minutes, "decision_time": decision_time, "entry_time": entry_time, "wait_minutes": wait_minutes, "trigger_reason": meta.get("trigger_reason"), "fallback_used": bool(meta.get("fallback_used", False)), "root_score": float(event["score"]), "trigger_score": float(meta.get("trigger_score", event["score"])), "trigger_score_delta": float(meta.get("trigger_score_delta", 0.0)), "immediate_entry_price": immediate_price, "actual_entry_price": float(synthetic["entry_price"]), "entry_price_improvement": immediate_price / float(synthetic["entry_price"]) - 1.0})
        cycle_rows.append({**cycle, **metrics, "policy": policy.name, "cost_multiplier": cost_multiplier, "wait_minutes": wait_minutes, "trigger_reason": meta.get("trigger_reason"), "fallback_used": bool(meta.get("fallback_used", False)), "entry_price_improvement": immediate_price / float(synthetic["entry_price"]) - 1.0})
        leg_rows.append({**leg, "fold_id": fold_id, "policy": policy.name, "delay_minutes": delay_minutes, "cost_multiplier": cost_multiplier})
        equity = equity_end
        occupied_until = pd.Timestamp(cycle["exit_time"])
        max_drawdown = min(max_drawdown, cycle_mdd)
        for date, row in daily.items():
            daily_map[date] = {**row, "fold_id": fold_id, "policy": policy.name, "delay_minutes": delay_minutes, "cost_multiplier": cost_multiplier}
        reporter.update(number)
    reporter.close()

    calendar = pd.date_range(pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end).normalize(), freq="D")
    daily_rows: list[dict[str, object]] = []
    last_equity = config.initial_equity
    last_peak = config.initial_equity
    for date in calendar:
        row = daily_map.get(date)
        if row is not None:
            last_equity = float(row["equity"]); last_peak = max(last_peak, last_equity); daily_rows.append(row)
        else:
            daily_rows.append({"date": date, "equity": last_equity, "drawdown": last_equity / last_peak - 1.0 if last_peak > 0 else -1.0, "active_tranches": 0, "hard_tail_r": 0.0, "notional_to_equity": 0.0, "fold_id": fold_id, "policy": policy.name, "delay_minutes": delay_minutes, "cost_multiplier": cost_multiplier})

    cycles = pd.DataFrame(cycle_rows); legs = pd.DataFrame(leg_rows); decisions = pd.DataFrame(decision_rows); daily = pd.DataFrame(daily_rows); rejections = pd.DataFrame(rejection_rows)
    pnl = cycles["cycle_net_pnl"].astype(float).to_numpy() if not cycles.empty else np.array([], dtype=float)
    winners = pnl[pnl > 0]; losers = pnl[pnl < 0]
    top = np.sort(winners)[::-1][: min(10, len(winners))]
    top_share = float(top.sum()/winners.sum()) if len(winners) and winners.sum() > 0 else np.nan
    without_top = float(equity - config.initial_equity - top.sum())
    positive_months = positive_quarters = 0
    if not daily.empty:
        series = daily.set_index("date")["equity"]
        month_end = series.resample("ME").last(); quarter_end = series.resample("QE").last()
        monthly = month_end.pct_change().dropna(); quarterly = quarter_end.pct_change().dropna()
        if len(month_end): monthly = pd.concat([pd.Series([month_end.iloc[0]/config.initial_equity-1], index=[month_end.index[0]]), monthly])
        if len(quarter_end): quarterly = pd.concat([pd.Series([quarter_end.iloc[0]/config.initial_equity-1], index=[quarter_end.index[0]]), quarterly])
        positive_months = int((monthly > 0).sum()); positive_quarters = int((quarterly > 0).sum())
    summary = {
        "fold_id": fold_id, "policy": policy.name, "delay_minutes": delay_minutes, "cost_multiplier": cost_multiplier,
        "candidate_cycles": int(len(work)), "executed_cycles": int(len(cycles)), "coverage_ratio": float(len(cycles)/len(work)) if len(work) else 0.0,
        "final_equity": float(equity), "total_net_return": float(equity/config.initial_equity-1), "max_drawdown": float(max_drawdown),
        "win_rate": float(np.mean(pnl>0)) if len(pnl) else np.nan, "profit_factor": float(winners.sum()/abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_cycle_return": float(cycles["cycle_return"].mean()) if not cycles.empty else np.nan, "worst_cycle_return": float(cycles["cycle_return"].min()) if not cycles.empty else np.nan,
        "top10_profit_share": top_share, "total_return_without_top10": without_top,
        "hard_stop_share": float(cycles["hard_stop_exit"].mean()) if not cycles.empty else 0.0, "soft_failure_share": float(cycles["soft_failure_exit"].mean()) if not cycles.empty else 0.0,
        "mean_mae_60m": float(cycles["mae_60m"].mean()) if not cycles.empty else np.nan, "median_mae_60m": float(cycles["mae_60m"].median()) if not cycles.empty else np.nan,
        "mean_full_mae": float(cycles["full_mae"].mean()) if not cycles.empty else np.nan, "mean_wait_minutes": float(cycles["wait_minutes"].mean()) if not cycles.empty else np.nan,
        "fallback_share": float(cycles["fallback_used"].mean()) if not cycles.empty else 0.0, "mean_entry_price_improvement": float(cycles["entry_price_improvement"].mean()) if not cycles.empty else np.nan,
        "positive_months": positive_months, "positive_quarters": positive_quarters, "runtime_rejections": int(len(rejections)),
    }
    return EntryTimingSimulation(decisions, cycles, legs, daily, summary, rejections)
