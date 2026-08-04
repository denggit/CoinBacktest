#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event-local account simulation for staged entry and asymmetric pyramiding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.research_common.progress import ProgressReporter

from .config import StagedExecutionConfig, StagedExecutionPolicy


@dataclass(frozen=True)
class StagedExecutionSimulation:
    cycles: pd.DataFrame
    legs: pd.DataFrame
    actions: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    runtime_rejections: pd.DataFrame


def exact_position(path: MinutePathData, timestamp: pd.Timestamp) -> int | None:
    value = int(pd.Timestamp(timestamp).value)
    position = int(np.searchsorted(path.timestamps_ns, value, side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != value:
        return None
    return position


def _fee(units: float, price: float, side_cost: float) -> float:
    return float(abs(units) * price * side_cost)


def _marked_equity(cash: float, tranches: list[dict[str, object]], price: float) -> float:
    return float(
        cash
        + sum(float(item["units"]) * (price - float(item["entry_price"])) for item in tranches)
    )


def _hard_risk_dollars(tranches: list[dict[str, object]]) -> float:
    return float(
        sum(
            float(item["units"])
            * max(float(item["entry_price"]) - float(item["hard_stop_price"]), 0.0)
            for item in tranches
        )
    )


def _notional(tranches: list[dict[str, object]], price: float) -> float:
    return float(sum(float(item["units"]) * price for item in tranches))


def _close_tranche(
    item: dict[str, object],
    *,
    price: float,
    timestamp: pd.Timestamp,
    side_cost: float,
    reason: str,
) -> dict[str, object]:
    units = float(item["units"])
    gross = units * (price - float(item["entry_price"]))
    exit_fee = _fee(units, price, side_cost)
    entry_fee = float(item["entry_fee"])
    return {
        "cycle_event_id": str(item["cycle_event_id"]),
        "tranche_id": str(item["tranche_id"]),
        "tranche_role": str(item["tranche_role"]),
        "entry_time": pd.Timestamp(item["entry_time"]),
        "exit_time": pd.Timestamp(timestamp),
        "entry_price": float(item["entry_price"]),
        "exit_price": float(price),
        "units": units,
        "allocated_r": float(item["allocated_r"]),
        "hard_stop_distance": float(item["hard_stop_distance"]),
        "hard_stop_price": float(item["hard_stop_price"]),
        "gross_pnl": float(gross),
        "entry_fee": entry_fee,
        "exit_fee": float(exit_fee),
        "net_pnl": float(gross - entry_fee - exit_fee),
        "exit_reason": reason,
    }


def _structure_updates_for_event(
    timeline: pd.DataFrame,
    *,
    path: MinutePathData,
) -> dict[int, list[dict[str, object]]]:
    updates: dict[int, list[dict[str, object]]] = {}
    if timeline.empty:
        return updates
    for row in timeline.to_dict("records"):
        position = exact_position(path, pd.Timestamp(row["effective_time"]))
        if position is not None:
            updates.setdefault(int(position), []).append(row)
    return updates


def _clamped_n_pct(path: MinutePathData, entry_position: int, entry_price: float, config: StagedExecutionConfig) -> float:
    atr = float(path.prior_atr_60[entry_position])
    raw = atr / entry_price if np.isfinite(atr) and atr > 0 else config.minimum_n_pct
    return float(np.clip(raw, config.minimum_n_pct, config.maximum_n_pct))


def _add_stop_distance(n_pct: float, policy: StagedExecutionPolicy, config: StagedExecutionConfig) -> float:
    return float(
        np.clip(
            n_pct * float(policy.add_stop_n),
            config.minimum_add_stop_pct,
            config.maximum_add_stop_pct,
        )
    )


def _attempt_add(
    *,
    cycle_event_id: str,
    role_index: int,
    entry_position: int,
    entry_price: float,
    allocated_r: float,
    stop_distance: float,
    cash: float,
    active: list[dict[str, object]],
    cycle_budget: float,
    marked_equity: float,
    policy: StagedExecutionPolicy,
    config: StagedExecutionConfig,
    side_cost: float,
) -> tuple[float, dict[str, object] | None, str, float]:
    desired_risk = float(allocated_r) * cycle_budget
    hard_stop_price = entry_price * (1.0 - stop_distance)
    risk_per_unit = max(entry_price - hard_stop_price, 1e-12)
    desired_units = desired_risk / risk_per_unit

    current_hard_risk = _hard_risk_dollars(active)
    allowed_risk = max(float(policy.max_cycle_hard_r) * cycle_budget - current_hard_risk, 0.0)
    desired_units = min(desired_units, allowed_risk / risk_per_unit)

    current_notional = _notional(active, entry_price)
    allowed_notional = max(
        min(policy.max_notional_to_equity, config.maximum_notional_to_equity) * marked_equity
        - current_notional,
        0.0,
    )
    desired_units = min(desired_units, allowed_notional / max(entry_price, 1e-12))
    actual_risk = desired_units * risk_per_unit
    actual_r = actual_risk / max(cycle_budget, 1e-12)
    if actual_r + 1e-12 < config.minimum_executed_add_r:
        return cash, None, "insufficient_risk_or_notional_capacity", 0.0

    if policy.require_profit_cover:
        unrealized = sum(
            float(item["units"]) * (entry_price - float(item["entry_price"])) for item in active
        )
        if unrealized + 1e-12 < actual_risk:
            return cash, None, "unrealized_profit_does_not_cover_add_risk", 0.0

    entry_fee = _fee(desired_units, entry_price, side_cost)
    cash -= entry_fee
    item = {
        "cycle_event_id": cycle_event_id,
        "tranche_id": f"{cycle_event_id}:add{role_index}",
        "tranche_role": f"add_{role_index}",
        "entry_position": int(entry_position),
        "entry_time": pd.NaT,
        "entry_price": float(entry_price),
        "units": float(desired_units),
        "allocated_r": float(actual_r),
        "hard_stop_distance": float(stop_distance),
        "hard_stop_price": float(hard_stop_price),
        "entry_fee": float(entry_fee),
    }
    return cash, item, "accepted", float(actual_r)


def _simulate_one_cycle(
    event: dict[str, object],
    timeline: pd.DataFrame,
    *,
    path: MinutePathData,
    policy: StagedExecutionPolicy,
    config: StagedExecutionConfig,
    side_cost: float,
    equity_start: float,
    global_peak: float,
) -> tuple[
    float,
    float,
    float,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[pd.Timestamp, dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    event_id = str(event["event_id"])
    entry_position = exact_position(path, pd.Timestamp(event["entry_time"]))
    exit_position = exact_position(path, pd.Timestamp(event["exit_time"]))
    if entry_position is None or exit_position is None or exit_position < entry_position:
        raise RuntimeError(f"missing path for {event_id}")

    entry_price = float(event["entry_price"])
    source_exit_price = float(event["exit_price"])
    cycle_budget = equity_start * float(config.account_risk_fraction_per_full_r)
    n_pct = _clamped_n_pct(path, entry_position, entry_price, config)
    add_stop_distance = _add_stop_distance(n_pct, policy, config)

    base_sizing_distance = float(policy.base_sizing_stop_distance)
    base_units = float(policy.base_r) * cycle_budget / max(entry_price * base_sizing_distance, 1e-12)
    base_stop = entry_price * (1.0 - config.disaster_stop_distance)
    base_entry_fee = _fee(base_units, entry_price, side_cost)
    cash = equity_start - base_entry_fee
    active: list[dict[str, object]] = [
        {
            "cycle_event_id": event_id,
            "tranche_id": f"{event_id}:base",
            "tranche_role": "base",
            "entry_position": int(entry_position),
            "entry_time": pd.Timestamp(event["entry_time"]),
            "entry_price": entry_price,
            "units": base_units,
            "allocated_r": float(policy.base_r),
            "hard_stop_distance": float(config.disaster_stop_distance),
            "hard_stop_price": float(base_stop),
            "entry_fee": float(base_entry_fee),
        }
    ]
    legs: list[dict[str, object]] = []
    actions: list[dict[str, object]] = [
        {
            "cycle_event_id": event_id,
            "action_time": pd.Timestamp(event["entry_time"]),
            "action": "OPEN_BASE",
            "reason": "q70_next_open",
            "allocated_r": float(policy.base_r),
            "n_pct": n_pct,
            "notional_to_equity": float(base_units * entry_price / max(equity_start, 1e-12)),
        }
    ]
    rejections: list[dict[str, object]] = []
    updates = _structure_updates_for_event(timeline, path=path)
    state = "HEALTHY"
    pending_failed_reclaim = False
    pullback_armed = False
    pending_add_index: int | None = None
    add_count = 0
    max_hard_r = _hard_risk_dollars(active) / max(cycle_budget, 1e-12)
    max_notional_ratio = base_units * entry_price / max(equity_start, 1e-12)
    peak = max(global_peak, equity_start)
    max_drawdown = 0.0
    daily: dict[pd.Timestamp, dict[str, object]] = {}

    for position in range(entry_position, exit_position + 1):
        timestamp = pd.Timestamp(path.index[position])
        open_price = float(path.open[position])
        low_price = float(path.low[position])
        close_price = float(path.close[position])

        for update in updates.get(position, []):
            state = str(update.get("state", state))
            pending_failed_reclaim = bool(update.get("pending_failed_reclaim_exit", False))

        # A pending add was decided on the previous completed minute and executes now.
        if pending_add_index is not None and position < exit_position:
            add_number = int(pending_add_index) + 1
            allocated_r = float(policy.add_r[pending_add_index])
            marked_before = _marked_equity(cash, active, open_price)
            cash, item, reason, actual_r = _attempt_add(
                cycle_event_id=event_id,
                role_index=add_number,
                entry_position=position,
                entry_price=open_price,
                allocated_r=allocated_r,
                stop_distance=add_stop_distance,
                cash=cash,
                active=active,
                cycle_budget=cycle_budget,
                marked_equity=marked_before,
                policy=policy,
                config=config,
                side_cost=side_cost,
            )
            if item is None:
                rejections.append(
                    {
                        "cycle_event_id": event_id,
                        "time": timestamp,
                        "add_number": add_number,
                        "reason": reason,
                    }
                )
            else:
                item["entry_time"] = timestamp
                active.append(item)
                add_count += 1
                actions.append(
                    {
                        "cycle_event_id": event_id,
                        "action_time": timestamp,
                        "action": "OPEN_ADD",
                        "reason": policy.mode,
                        "allocated_r": actual_r,
                        "n_pct": n_pct,
                        "notional_to_equity": float(
                            _notional(active, open_price) / max(_marked_equity(cash, active, open_price), 1e-12)
                        ),
                    }
                )
            pending_add_index = None

        # Independent add-on stops. The base is deliberately not tightened.
        survivors: list[dict[str, object]] = []
        for item in active:
            if str(item["tranche_role"]) == "base":
                survivors.append(item)
                continue
            stop = float(item["hard_stop_price"])
            fill: float | None = None
            if open_price <= stop:
                fill = open_price
            elif low_price <= stop:
                fill = stop
            if fill is None:
                survivors.append(item)
                continue
            leg = _close_tranche(
                item,
                price=fill,
                timestamp=timestamp,
                side_cost=side_cost,
                reason="independent_add_stop",
            )
            cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
            legs.append(leg)
            actions.append(
                {
                    "cycle_event_id": event_id,
                    "action_time": timestamp,
                    "action": "STOP_ADD",
                    "reason": "independent_add_stop",
                    "allocated_r": float(item["allocated_r"]),
                    "n_pct": n_pct,
                    "notional_to_equity": np.nan,
                }
            )
        active = survivors

        # Soft-failure sizing experiment: threshold is confirmed by a completed
        # structure bar and executes at its next one-minute open. Intrabar wicks
        # alone do not exit the base.
        soft_failure_now = bool(
            policy.mode == "soft_failure"
            and position < exit_position
            and position in updates
            and any(
                float(update.get("current_close", np.inf))
                <= entry_price * (1.0 - float(policy.soft_failure_distance))
                for update in updates[position]
            )
        )
        if soft_failure_now:
            for item in active:
                leg = _close_tranche(
                    item,
                    price=open_price,
                    timestamp=timestamp,
                    side_cost=side_cost,
                    reason="soft_failure_confirmed_close",
                )
                cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
                legs.append(leg)
            active = []
            actions.append(
                {
                    "cycle_event_id": event_id,
                    "action_time": timestamp,
                    "action": "SOFT_FAILURE_EXIT",
                    "reason": "completed_structure_close_below_operating_threshold",
                    "allocated_r": 0.0,
                    "n_pct": n_pct,
                    "notional_to_equity": 0.0,
                }
            )

        if position == exit_position and active:
            for item in active:
                leg = _close_tranche(
                    item,
                    price=source_exit_price,
                    timestamp=timestamp,
                    side_cost=side_cost,
                    reason=str(event["exit_reason"]),
                )
                cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
                legs.append(leg)
            active = []

        # Decide add-on after this minute closes; execute next minute open.
        if active and pending_add_index is None and add_count < len(policy.add_r) and position < exit_position:
            healthy = state == "HEALTHY" and not pending_failed_reclaim
            base_open_return = close_price / entry_price - 1.0
            trigger = False
            trigger_reason = ""
            if policy.mode == "staged_dual_path":
                if low_price <= entry_price * (1.0 - config.staged_pullback_arm_n * n_pct):
                    pullback_armed = True
                if healthy and pullback_armed and close_price >= entry_price:
                    trigger = True
                    trigger_reason = "pullback_reclaim"
                elif healthy and close_price >= entry_price * (1.0 + policy.trigger_n[add_count] * n_pct):
                    trigger = True
                    trigger_reason = "continuation_fallback"
            elif policy.mode in {"turtle", "pyramid"}:
                threshold = entry_price * (1.0 + policy.trigger_n[add_count] * n_pct)
                if healthy and close_price >= threshold and base_open_return > 0:
                    trigger = True
                    trigger_reason = "causal_N_advance"
            if trigger and policy.require_profit_cover:
                desired_risk = float(policy.add_r[add_count]) * cycle_budget
                visible_unrealized = sum(
                    float(item["units"]) * (close_price - float(item["entry_price"]))
                    for item in active
                )
                trigger = bool(visible_unrealized + 1e-12 >= desired_risk)
            if trigger:
                pending_add_index = add_count
                actions.append(
                    {
                        "cycle_event_id": event_id,
                        "action_time": timestamp,
                        "action": "ARM_ADD",
                        "reason": trigger_reason,
                        "allocated_r": float(policy.add_r[add_count]),
                        "n_pct": n_pct,
                        "notional_to_equity": float(
                            _notional(active, close_price) / max(_marked_equity(cash, active, close_price), 1e-12)
                        ),
                    }
                )

        equity = _marked_equity(cash, active, close_price)
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else -1.0
        max_drawdown = min(max_drawdown, drawdown)
        hard_r = _hard_risk_dollars(active) / max(cycle_budget, 1e-12) if active else 0.0
        notional_ratio = _notional(active, close_price) / max(equity, 1e-12) if active else 0.0
        max_hard_r = max(max_hard_r, hard_r)
        max_notional_ratio = max(max_notional_ratio, notional_ratio)
        daily[timestamp.normalize()] = {
            "date": timestamp.normalize(),
            "equity": equity,
            "drawdown": drawdown,
            "active_tranches": len(active),
            "hard_tail_r": hard_r,
            "notional_to_equity": notional_ratio,
        }
        if not active and position < exit_position:
            break

    cycle_net = cash - equity_start
    leg_frame = pd.DataFrame(legs)
    base_net = float(leg_frame.loc[leg_frame["tranche_role"].eq("base"), "net_pnl"].sum()) if not leg_frame.empty else 0.0
    add_net = float(leg_frame.loc[~leg_frame["tranche_role"].eq("base"), "net_pnl"].sum()) if not leg_frame.empty else 0.0
    cycle = {
        "event_id": event_id,
        "fold_id": str(event["fold_id"]),
        "policy": policy.name,
        "delay_minutes": int(event["delay_minutes"]),
        "decision_time": pd.Timestamp(event["decision_time"]),
        "entry_time": pd.Timestamp(event["entry_time"]),
        "source_exit_time": pd.Timestamp(event["exit_time"]),
        "source_exit_reason": str(event["exit_reason"]),
        "score": float(event["score"]),
        "signal_quantile": float(event["signal_quantile"]),
        "equity_start": float(equity_start),
        "equity_end": float(cash),
        "cycle_net_pnl": float(cycle_net),
        "cycle_return": float(cycle_net / max(equity_start, 1e-12)),
        "base_net_pnl": base_net,
        "add_net_pnl": add_net,
        "add_count": int(add_count),
        "n_pct": n_pct,
        "base_notional_to_equity": float(base_units * entry_price / max(equity_start, 1e-12)),
        "max_hard_tail_r": float(max_hard_r),
        "max_notional_to_equity": float(max_notional_ratio),
        "cycle_max_drawdown": float(max_drawdown),
        "final_exit_time": pd.Timestamp(leg_frame["exit_time"].max()) if not leg_frame.empty else pd.Timestamp(event["entry_time"]),
        "soft_failure_exit": bool((leg_frame.get("exit_reason", pd.Series(dtype=str)) == "soft_failure_confirmed_close").any()) if not leg_frame.empty else False,
        "addon_stop_count": int((leg_frame.get("exit_reason", pd.Series(dtype=str)) == "independent_add_stop").sum()) if not leg_frame.empty else 0,
    }
    return cash, peak, max_drawdown, legs, actions, daily, cycle, rejections


def simulate_staged_execution_account(
    events: pd.DataFrame,
    timelines: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: StagedExecutionPolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: StagedExecutionConfig,
    progress: bool = True,
) -> StagedExecutionSimulation:
    """Simulate frozen P0 cycles with optional staged and add-on execution."""

    work = events.loc[
        events["fold_id"].astype(str).eq(fold_id)
        & events["delay_minutes"].astype(int).eq(int(delay_minutes))
    ].copy()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False])
    if work.empty:
        return StagedExecutionSimulation(
            cycles=pd.DataFrame(),
            legs=pd.DataFrame(),
            actions=pd.DataFrame(),
            daily_equity=pd.DataFrame(),
            summary={},
            runtime_rejections=pd.DataFrame(),
        )

    if timelines.empty:
        timeline_work = pd.DataFrame()
    else:
        timeline_work = timelines.loc[
            timelines["fold_id"].astype(str).eq(fold_id)
            & timelines["delay_minutes"].astype(int).eq(int(delay_minutes))
        ].copy()
    timeline_groups = (
        {key: frame for key, frame in timeline_work.groupby("event_id", sort=False)}
        if not timeline_work.empty
        else {}
    )

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
        f"[R03.4.2.11 {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        len(work),
        every=max(1, len(work) // 100),
        enabled=progress,
    )
    for number, event in enumerate(work.to_dict("records"), start=1):
        event_id = str(event["event_id"])
        event_timeline = timeline_groups.get(event_id, pd.DataFrame())
        try:
            (
                equity,
                peak,
                cycle_mdd,
                legs,
                actions,
                daily,
                cycle,
                rejections,
            ) = _simulate_one_cycle(
                event,
                event_timeline,
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
        leg_rows.extend(
            {
                **row,
                "fold_id": fold_id,
                "policy": policy.name,
                "delay_minutes": int(delay_minutes),
                "cost_multiplier": float(cost_multiplier),
            }
            for row in legs
        )
        action_rows.extend(
            {
                **row,
                "fold_id": fold_id,
                "policy": policy.name,
                "delay_minutes": int(delay_minutes),
                "cost_multiplier": float(cost_multiplier),
            }
            for row in actions
        )
        rejection_rows.extend(
            {
                **row,
                "fold_id": fold_id,
                "policy": policy.name,
                "delay_minutes": int(delay_minutes),
                "cost_multiplier": float(cost_multiplier),
            }
            for row in rejections
        )
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

    # Fill idle calendar days so month/quarter metrics use a continuous account curve.
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
            drawdown = last_equity / last_peak - 1.0 if last_peak > 0 else -1.0
            filled_daily.append(
                {
                    "date": date,
                    "equity": last_equity,
                    "drawdown": drawdown,
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
    add_net = float(cycles["add_net_pnl"].sum()) if not cycles.empty else 0.0
    base_positive = float(cycles.loc[cycles["base_net_pnl"] > 0, "base_net_pnl"].sum()) if not cycles.empty else 0.0

    positive_months = 0
    positive_quarters = 0
    if not daily.empty:
        series = daily.set_index("date")["equity"]
        month_end = series.resample("ME").last()
        quarter_end = series.resample("QE").last()
        monthly = month_end.pct_change().dropna()
        quarterly = quarter_end.pct_change().dropna()
        if len(month_end):
            monthly = pd.concat([
                pd.Series([month_end.iloc[0] / config.initial_equity - 1.0], index=[month_end.index[0]]),
                monthly,
            ])
        if len(quarter_end):
            quarterly = pd.concat([
                pd.Series([quarter_end.iloc[0] / config.initial_equity - 1.0], index=[quarter_end.index[0]]),
                quarterly,
            ])
        positive_months = int((monthly > 0).sum())
        positive_quarters = int((quarterly > 0).sum())

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
        "top10_profit_share": top_share,
        "total_return_without_top10": without_top10,
        "cycles_with_add": int((cycles["add_count"] > 0).sum()) if not cycles.empty else 0,
        "add_tranches": int(cycles["add_count"].sum()) if not cycles.empty else 0,
        "add_net_pnl": add_net,
        "addon_loss_share_of_base_profit": float(max(-add_net, 0.0) / max(base_positive, 1e-12)),
        "soft_failure_exits": int(cycles["soft_failure_exit"].sum()) if not cycles.empty else 0,
        "addon_stop_count": int(cycles["addon_stop_count"].sum()) if not cycles.empty else 0,
        "max_hard_tail_r": float(cycles["max_hard_tail_r"].max()) if not cycles.empty else 0.0,
        "max_notional_to_equity": float(cycles["max_notional_to_equity"].max()) if not cycles.empty else 0.0,
        "mean_base_notional_to_equity": float(cycles["base_notional_to_equity"].mean()) if not cycles.empty else 0.0,
        "positive_months": positive_months,
        "positive_quarters": positive_quarters,
        "runtime_rejections": int(len(rejections)),
    }
    return StagedExecutionSimulation(
        cycles=cycles,
        legs=legs,
        actions=actions,
        daily_equity=daily,
        summary=summary,
        runtime_rejections=rejections,
    )
