#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Account simulator for real one-R initial-tail compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.research_common.progress import ProgressReporter

from .config import TailCompressionConfig, TailCompressionPolicy


@dataclass(frozen=True)
class TailCompressionSimulation:
    cycles: pd.DataFrame
    legs: pd.DataFrame
    actions: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    runtime_rejections: pd.DataFrame


def _fee(units: float, price: float, side_cost: float) -> float:
    return float(abs(units) * price * side_cost)


def _effective_updates(timeline: pd.DataFrame, path: MinutePathData) -> dict[int, list[dict[str, object]]]:
    updates: dict[int, list[dict[str, object]]] = {}
    if timeline.empty:
        return updates
    for row in timeline.sort_values("effective_time").to_dict("records"):
        position = path.locate_exact(pd.Timestamp(row["effective_time"]))
        if position is not None:
            updates.setdefault(int(position), []).append(row)
    return updates


def _policy_distances(
    policy: TailCompressionPolicy,
    *,
    path: MinutePathData,
    entry_position: int,
    entry_price: float,
) -> tuple[float, float, float, float]:
    atr = float(path.prior_atr_60[entry_position])
    atr_pct = atr / entry_price if np.isfinite(atr) and atr > 0 else np.nan
    if policy.mode == "adaptive":
        raw = atr_pct * float(policy.adaptive_atr_multiple) if np.isfinite(atr_pct) else policy.adaptive_max_distance
        hard = float(np.clip(raw, policy.adaptive_min_distance, policy.adaptive_max_distance))
        sizing = hard
        soft = hard * float(policy.adaptive_soft_fraction) if policy.adaptive_soft_fraction > 0 else 0.0
    else:
        sizing = float(policy.sizing_stop_distance)
        hard = float(policy.hard_stop_distance)
        soft = float(policy.soft_failure_distance)
    return sizing, hard, soft, atr_pct


def _close_leg(
    *,
    event_id: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    units: float,
    side_cost: float,
    entry_fee: float,
    exit_reason: str,
    sizing_distance: float,
    hard_distance: float,
    soft_distance: float,
) -> dict[str, object]:
    gross = units * (exit_price - entry_price)
    exit_fee = _fee(units, exit_price, side_cost)
    return {
        "event_id": event_id,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "units": float(units),
        "entry_fee": float(entry_fee),
        "exit_fee": float(exit_fee),
        "gross_pnl": float(gross),
        "net_pnl": float(gross - entry_fee - exit_fee),
        "exit_reason": exit_reason,
        "sizing_stop_distance": float(sizing_distance),
        "hard_stop_distance": float(hard_distance),
        "soft_failure_distance": float(soft_distance),
    }


def _simulate_cycle(
    event: dict[str, object],
    timeline: pd.DataFrame,
    *,
    path: MinutePathData,
    policy: TailCompressionPolicy,
    config: TailCompressionConfig,
    side_cost: float,
    equity_start: float,
    global_peak: float,
) -> tuple[float, float, float, dict[str, object], dict[str, object], list[dict[str, object]], dict[pd.Timestamp, dict[str, object]]]:
    event_id = str(event["event_id"])
    entry_time = pd.Timestamp(event["entry_time"])
    source_exit_time = pd.Timestamp(event["exit_time"])
    entry_position = path.locate_exact(entry_time)
    source_exit_position = path.locate_exact(source_exit_time)
    if entry_position is None or source_exit_position is None or source_exit_position < entry_position:
        raise RuntimeError(f"minute path missing for {event_id}")

    entry_price = float(event["entry_price"])
    source_exit_price = float(event["exit_price"])
    sizing_distance, hard_distance, soft_distance, atr_pct = _policy_distances(
        policy,
        path=path,
        entry_position=entry_position,
        entry_price=entry_price,
    )
    cycle_budget = equity_start * float(config.account_risk_fraction_per_full_r)
    units = cycle_budget / max(entry_price * sizing_distance, 1e-12)
    hard_stop_price = entry_price * (1.0 - hard_distance)
    entry_fee = _fee(units, entry_price, side_cost)
    cash = equity_start - entry_fee
    updates = _effective_updates(timeline, path)
    peak = max(global_peak, equity_start)
    max_drawdown = 0.0
    daily: dict[pd.Timestamp, dict[str, object]] = {}
    actions: list[dict[str, object]] = [
        {
            "event_id": event_id,
            "action_time": entry_time,
            "action": "OPEN_BASE",
            "reason": "q70_next_open",
            "sizing_stop_distance": sizing_distance,
            "hard_stop_distance": hard_distance,
            "soft_failure_distance": soft_distance,
            "notional_to_equity": units * entry_price / max(equity_start, 1e-12),
        }
    ]

    exit_time = source_exit_time
    exit_price = source_exit_price
    exit_reason = str(event["exit_reason"])
    hard_stop_exit = False
    soft_failure_exit = False

    for position in range(entry_position, source_exit_position + 1):
        timestamp = pd.Timestamp(path.index[position])
        open_price = float(path.open[position])
        low_price = float(path.low[position])
        close_price = float(path.close[position])

        # The frozen source exit is a next-open action. It has precedence over
        # the rest of that minute's intrabar path.
        if position == source_exit_position:
            exit_time = timestamp
            exit_price = source_exit_price
            exit_reason = str(event["exit_reason"])
        else:
            # Exchange-side gap through the real hard stop occurs before an
            # active strategy can submit a completed-bar soft exit.
            if open_price <= hard_stop_price and policy.mode not in {"baseline", "reference"}:
                exit_time = timestamp
                exit_price = open_price
                exit_reason = "real_hard_stop_gap"
                hard_stop_exit = True
            else:
                soft_now = bool(
                    soft_distance > 0
                    and position in updates
                    and any(
                        float(update.get("current_close", np.inf))
                        <= entry_price * (1.0 - soft_distance)
                        for update in updates[position]
                    )
                )
                if soft_now:
                    exit_time = timestamp
                    exit_price = open_price
                    exit_reason = "soft_failure_confirmed_close"
                    soft_failure_exit = True
                elif low_price <= hard_stop_price and policy.mode not in {"baseline", "reference"}:
                    exit_time = timestamp
                    exit_price = hard_stop_price
                    exit_reason = "real_hard_stop_intrabar"
                    hard_stop_exit = True

        marked_price = exit_price if timestamp == exit_time else close_price
        exit_fee_preview = _fee(units, exit_price, side_cost) if timestamp == exit_time else 0.0
        marked_equity = cash + units * (marked_price - entry_price) - exit_fee_preview
        peak = max(peak, marked_equity)
        drawdown = marked_equity / peak - 1.0 if peak > 0 else -1.0
        max_drawdown = min(max_drawdown, drawdown)
        daily[timestamp.normalize()] = {
            "date": timestamp.normalize(),
            "equity": float(marked_equity),
            "drawdown": float(drawdown),
            "active_tranches": 0 if timestamp == exit_time else 1,
            "hard_tail_r": 0.0 if timestamp == exit_time else float(hard_distance / sizing_distance),
            "notional_to_equity": 0.0 if timestamp == exit_time else float(units * close_price / max(marked_equity, 1e-12)),
        }
        if timestamp == exit_time:
            break

    leg = _close_leg(
        event_id=event_id,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        units=units,
        side_cost=side_cost,
        entry_fee=entry_fee,
        exit_reason=exit_reason,
        sizing_distance=sizing_distance,
        hard_distance=hard_distance,
        soft_distance=soft_distance,
    )
    equity_end = equity_start + float(leg["net_pnl"])
    actions.append(
        {
            "event_id": event_id,
            "action_time": exit_time,
            "action": "EXIT_BASE",
            "reason": exit_reason,
            "sizing_stop_distance": sizing_distance,
            "hard_stop_distance": hard_distance,
            "soft_failure_distance": soft_distance,
            "notional_to_equity": 0.0,
        }
    )
    hard_tail_r = hard_distance / sizing_distance
    cycle_return = (equity_end - equity_start) / max(equity_start, 1e-12)
    cycle = {
        "event_id": event_id,
        "fold_id": str(event["fold_id"]),
        "policy": policy.name,
        "delay_minutes": int(event["delay_minutes"]),
        "decision_time": pd.Timestamp(event["decision_time"]),
        "entry_time": entry_time,
        "source_exit_time": source_exit_time,
        "source_exit_reason": str(event["exit_reason"]),
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "score": float(event["score"]),
        "signal_quantile": float(event["signal_quantile"]),
        "equity_start": float(equity_start),
        "equity_end": float(equity_end),
        "cycle_net_pnl": float(leg["net_pnl"]),
        "cycle_return": float(cycle_return),
        "sizing_stop_distance": sizing_distance,
        "hard_stop_distance": hard_distance,
        "soft_failure_distance": soft_distance,
        "prior_atr_60_pct": atr_pct,
        "base_notional_to_equity": float(units * entry_price / max(equity_start, 1e-12)),
        "max_hard_tail_r": float(hard_tail_r),
        "cycle_max_drawdown": float(max_drawdown),
        "hard_stop_exit": hard_stop_exit,
        "soft_failure_exit": soft_failure_exit,
    }
    return equity_end, peak, max_drawdown, cycle, leg, actions, daily


def simulate_tail_compression_account(
    events: pd.DataFrame,
    timelines: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: TailCompressionPolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: TailCompressionConfig,
    progress: bool = True,
) -> TailCompressionSimulation:
    work = events.loc[
        events["fold_id"].astype(str).eq(fold_id)
        & events["delay_minutes"].astype(int).eq(int(delay_minutes))
    ].copy()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False])
    if work.empty:
        return TailCompressionSimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    timeline_work = timelines.loc[
        timelines["fold_id"].astype(str).eq(fold_id)
        & timelines["delay_minutes"].astype(int).eq(int(delay_minutes))
    ].copy() if not timelines.empty else pd.DataFrame()
    timeline_groups = {
        key: frame for key, frame in timeline_work.groupby("event_id", sort=False)
    } if not timeline_work.empty else {}

    side_cost = float(config.base_round_trip_cost * cost_multiplier / 2.0)
    equity = float(config.initial_equity)
    peak = equity
    max_drawdown = 0.0
    cycle_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    daily_map: dict[pd.Timestamp, dict[str, object]] = {}

    reporter = ProgressReporter(
        f"[R03.4.2.12 {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        len(work),
        every=max(1, len(work) // 100),
        enabled=progress,
    )
    for number, event in enumerate(work.to_dict("records"), start=1):
        event_id = str(event["event_id"])
        try:
            equity, peak, cycle_mdd, cycle, leg, actions, daily = _simulate_cycle(
                event,
                timeline_groups.get(event_id, pd.DataFrame()),
                path=path,
                policy=policy,
                config=config,
                side_cost=side_cost,
                equity_start=equity,
                global_peak=peak,
            )
        except Exception as exc:
            rejection_rows.append(
                {
                    "event_id": event_id,
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            reporter.update(number)
            continue
        max_drawdown = min(max_drawdown, cycle_mdd)
        cycle_rows.append({**cycle, "cost_multiplier": float(cost_multiplier)})
        leg_rows.append({
            **leg,
            "fold_id": fold_id,
            "policy": policy.name,
            "delay_minutes": int(delay_minutes),
            "cost_multiplier": float(cost_multiplier),
        })
        action_rows.extend({
            **row,
            "fold_id": fold_id,
            "policy": policy.name,
            "delay_minutes": int(delay_minutes),
            "cost_multiplier": float(cost_multiplier),
        } for row in actions)
        rejection_rows.extend([])
        for date, row in daily.items():
            daily_map[date] = {
                **row,
                "fold_id": fold_id,
                "policy": policy.name,
                "delay_minutes": int(delay_minutes),
                "cost_multiplier": float(cost_multiplier),
            }
        reporter.update(number)
    reporter.close()

    calendar = pd.date_range(pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end).normalize(), freq="D")
    filled_daily: list[dict[str, object]] = []
    last_equity = float(config.initial_equity)
    last_peak = float(config.initial_equity)
    for date in calendar:
        row = daily_map.get(date)
        if row is not None:
            last_equity = float(row["equity"])
            last_peak = max(last_peak, last_equity)
            filled_daily.append(row)
        else:
            filled_daily.append(
                {
                    "date": date,
                    "equity": last_equity,
                    "drawdown": last_equity / last_peak - 1.0 if last_peak > 0 else -1.0,
                    "active_tranches": 0,
                    "hard_tail_r": 0.0,
                    "notional_to_equity": 0.0,
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                }
            )

    cycles = pd.DataFrame(cycle_rows)
    legs = pd.DataFrame(leg_rows)
    actions = pd.DataFrame(action_rows)
    daily = pd.DataFrame(filled_daily)
    rejections = pd.DataFrame(rejection_rows)
    pnl = cycles["cycle_net_pnl"].astype(float).to_numpy() if not cycles.empty else np.array([], dtype=float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    top = np.sort(winners)[::-1][: min(10, len(winners))]
    top_share = float(top.sum() / winners.sum()) if len(winners) and winners.sum() > 0 else np.nan
    without_top10 = float(equity - config.initial_equity - top.sum())

    positive_months = 0
    positive_quarters = 0
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

    returns_r = cycles["cycle_return"].astype(float) / config.account_risk_fraction_per_full_r if not cycles.empty else pd.Series(dtype=float)
    summary: dict[str, Any] = {
        "fold_id": fold_id,
        "policy": policy.name,
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "candidate_cycles": int(len(work)),
        "executed_cycles": int(len(cycles)),
        "final_equity": float(equity),
        "total_net_return": float(equity / config.initial_equity - 1.0),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else np.nan,
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_cycle_return": float(cycles["cycle_return"].mean()) if not cycles.empty else np.nan,
        "mean_losing_cycle_return": float(cycles.loc[cycles["cycle_return"] < 0, "cycle_return"].mean()) if not cycles.empty else np.nan,
        "worst_cycle_return": float(cycles["cycle_return"].min()) if not cycles.empty else np.nan,
        "worst_cycle_loss_r": float(abs(min(returns_r.min(), 0.0))) if len(returns_r) else np.nan,
        "top10_profit_share": top_share,
        "total_return_without_top10": without_top10,
        "hard_stop_exits": int(cycles["hard_stop_exit"].sum()) if not cycles.empty else 0,
        "soft_failure_exits": int(cycles["soft_failure_exit"].sum()) if not cycles.empty else 0,
        "source_exits": int((~cycles["hard_stop_exit"] & ~cycles["soft_failure_exit"]).sum()) if not cycles.empty else 0,
        "hard_stop_share": float(cycles["hard_stop_exit"].mean()) if not cycles.empty else 0.0,
        "soft_failure_share": float(cycles["soft_failure_exit"].mean()) if not cycles.empty else 0.0,
        "max_hard_tail_r": float(cycles["max_hard_tail_r"].max()) if not cycles.empty else 0.0,
        "mean_hard_stop_distance": float(cycles["hard_stop_distance"].mean()) if not cycles.empty else np.nan,
        "median_hard_stop_distance": float(cycles["hard_stop_distance"].median()) if not cycles.empty else np.nan,
        "mean_base_notional_to_equity": float(cycles["base_notional_to_equity"].mean()) if not cycles.empty else 0.0,
        "max_base_notional_to_equity": float(cycles["base_notional_to_equity"].max()) if not cycles.empty else 0.0,
        "positive_months": positive_months,
        "positive_quarters": positive_quarters,
        "runtime_rejections": int(len(rejections)),
    }
    return TailCompressionSimulation(cycles, legs, actions, daily, summary, rejections)
