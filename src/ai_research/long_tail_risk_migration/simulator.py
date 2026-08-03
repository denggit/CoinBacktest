#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minute-marked account simulation for partial de-risking and risk migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.research_common.progress import ProgressReporter

from .config import MigrationPolicy, RiskMigrationConfig
from .structure import exact_position


@dataclass(frozen=True)
class RiskMigrationSimulation:
    trades: pd.DataFrame
    legs: pd.DataFrame
    daily_equity: pd.DataFrame
    decisions: pd.DataFrame
    actions: pd.DataFrame
    summary: dict[str, Any]
    runtime_rejections: pd.DataFrame


def _marked_equity(cash: float, active: dict[str, dict[str, object]], price: float) -> float:
    return float(
        cash
        + sum(float(item["units"]) * (price - float(item["entry_price"])) for item in active.values())
    )


def _risk_dollars(item: dict[str, object]) -> float:
    return float(item["units"]) * max(
        float(item["entry_price"]) - float(item["disaster_stop_price"]),
        0.0,
    )


def _reduce_position(
    item: dict[str, object],
    *,
    units_to_close: float,
    price: float,
    timestamp: pd.Timestamp,
    side_cost: float,
    reason: str,
    leg_type: str,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    current_units = float(item["units"])
    units_to_close = float(np.clip(units_to_close, 0.0, current_units))
    if units_to_close <= 1e-15:
        return item, {}
    fraction = units_to_close / current_units
    entry_fee_alloc = float(item["remaining_entry_fee"]) * fraction
    gross_pnl = units_to_close * (price - float(item["entry_price"]))
    exit_fee = units_to_close * price * side_cost
    net_pnl = gross_pnl - entry_fee_alloc - exit_fee
    leg = {
        "event_id": str(item["event_id"]),
        "entry_role": str(item["entry_role"]),
        "leg_type": leg_type,
        "exit_reason": reason,
        "entry_time": pd.Timestamp(item["entry_time"]),
        "exit_time": pd.Timestamp(timestamp),
        "entry_price": float(item["entry_price"]),
        "exit_price": float(price),
        "units": units_to_close,
        "gross_pnl": float(gross_pnl),
        "entry_fee": float(entry_fee_alloc),
        "exit_fee": float(exit_fee),
        "net_pnl": float(net_pnl),
        "initial_risk_dollars": float(item["initial_risk_dollars"]),
    }
    remaining_units = current_units - units_to_close
    remaining_fee = float(item["remaining_entry_fee"]) - entry_fee_alloc
    if remaining_units <= 1e-15:
        return None, leg
    updated = dict(item)
    updated["units"] = float(remaining_units)
    updated["remaining_entry_fee"] = float(max(remaining_fee, 0.0))
    updated["partial_close_count"] = int(item.get("partial_close_count", 0)) + 1
    return updated, leg


def _aggregate_event_trades(legs: pd.DataFrame, opened: pd.DataFrame) -> pd.DataFrame:
    if legs.empty or opened.empty:
        return pd.DataFrame()
    grouped = (
        legs.groupby("event_id", as_index=False)
        .agg(
            exit_time=("exit_time", "max"),
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            total_exit_fee=("exit_fee", "sum"),
            allocated_entry_fee=("entry_fee", "sum"),
            leg_count=("leg_type", "size"),
            partial_leg_count=("leg_type", lambda values: int((values == "partial_reduce").sum())),
            migration_release_leg_count=("leg_type", lambda values: int((values == "migration_release").sum())),
            final_exit_reason=("exit_reason", "last"),
        )
    )
    result = opened.merge(grouped, on="event_id", how="inner", suffixes=("", "_closed"))
    result["realized_r"] = result["net_pnl"] / result["initial_risk_dollars"].clip(lower=1e-12)
    exit_column = "exit_time_closed" if "exit_time_closed" in result.columns else "exit_time"
    result["holding_minutes"] = (
        pd.to_datetime(result[exit_column]) - pd.to_datetime(result["entry_time"])
    ).dt.total_seconds() / 60.0
    result["exit_time"] = pd.to_datetime(result[exit_column])
    if exit_column != "exit_time":
        result = result.drop(columns=[exit_column])
    return result


def simulate_risk_migration_account(
    structural: pd.DataFrame,
    timelines: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: MigrationPolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: RiskMigrationConfig,
    progress: bool = True,
) -> RiskMigrationSimulation:
    """Simulate one unified policy with a fixed one-R risk budget per cycle."""

    work = structural.loc[structural["delay_minutes"].astype(int) == int(delay_minutes)].copy()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    if work.empty:
        return RiskMigrationSimulation(
            trades=pd.DataFrame(),
            legs=pd.DataFrame(),
            daily_equity=pd.DataFrame(),
            decisions=pd.DataFrame(),
            actions=pd.DataFrame(),
            summary={},
            runtime_rejections=pd.DataFrame(),
        )

    start_position = exact_position(path, pd.Timestamp(test_start))
    end_position = exact_position(path, pd.Timestamp(test_end).floor("min"))
    if start_position is None or end_position is None or end_position <= start_position:
        raise RuntimeError(f"minute path does not exactly cover {fold_id}")

    entry_map: dict[int, list[dict[str, object]]] = {}
    exit_map: dict[int, list[dict[str, object]]] = {}
    update_map: dict[int, list[dict[str, object]]] = {}
    runtime_rejections: list[dict[str, object]] = []
    for row in work.to_dict("records"):
        entry_position = exact_position(path, pd.Timestamp(row["entry_time"]))
        exit_position = exact_position(path, pd.Timestamp(row["exit_time"]))
        if entry_position is None or exit_position is None or exit_position < entry_position:
            runtime_rejections.append({"event_id": row["event_id"], "reason": "missing_entry_or_exit_path"})
            continue
        row["entry_position"] = int(entry_position)
        row["exit_position"] = int(exit_position)
        entry_map.setdefault(int(entry_position), []).append(row)
        exit_map.setdefault(int(exit_position), []).append(row)
    if not timelines.empty:
        timeline_work = timelines.loc[timelines["delay_minutes"].astype(int) == int(delay_minutes)].copy()
        for row in timeline_work.to_dict("records"):
            position = exact_position(path, pd.Timestamp(row["effective_time"]))
            if position is not None:
                update_map.setdefault(int(position), []).append(row)

    cash = float(config.initial_equity)
    active: dict[str, dict[str, object]] = {}
    cycle_budget_dollars: float | None = None
    opened_rows: list[dict[str, object]] = []
    legs: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    side_cost = float(config.base_round_trip_cost * cost_multiplier / 2.0)
    full_risk_fraction = float(config.account_risk_fraction_per_full_r)
    peak_equity = cash
    max_drawdown = 0.0
    max_cycle_r = 0.0
    max_notional_to_equity = 0.0
    risk_observation_sum = 0.0
    exposure_observation_sum = 0.0
    observations = 0

    reporter = ProgressReporter(
        f"[R03.4.2.10 {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        end_position - start_position + 1,
        every=max(1, (end_position - start_position + 1) // 100),
        enabled=progress,
    )

    for position in range(start_position, end_position + 1):
        open_price = float(path.open[position])
        close_price = float(path.close[position])
        timestamp = pd.Timestamp(path.index[position])

        # Completed 15m structure becomes actionable at this minute open.
        for update in update_map.get(position, []):
            event_id = str(update["event_id"])
            item = active.get(event_id)
            if item is None:
                continue
            item["latest_structure_state"] = str(update["state"])
            item["latest_structure_return"] = float(update["current_return"])
            item["latest_pending_failed_reclaim"] = bool(update["pending_failed_reclaim_exit"])
            item["latest_proven_structure"] = bool(update["proven_structure"])
            item["latest_structure_close_time"] = pd.Timestamp(update["structure_close_time"])

            eligible_partial = bool(
                policy.partial_reduce_fraction > 0.0
                and not bool(item.get("soft_break_reduced", False))
                and bool(update["entered_broken_this_bar"])
                and bool(update["proven_structure"])
                and not bool(update["pending_failed_reclaim_exit"])
                and position < int(item["exit_position"])
                and open_price >= float(item["entry_price"])
            )
            if eligible_partial:
                before_units = float(item["units"])
                close_units = before_units * float(policy.partial_reduce_fraction)
                updated, leg = _reduce_position(
                    item,
                    units_to_close=close_units,
                    price=open_price,
                    timestamp=timestamp,
                    side_cost=side_cost,
                    reason="soft_structure_first_break",
                    leg_type="partial_reduce",
                )
                if leg:
                    cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
                    legs.append({**leg, "fold_id": fold_id, "policy": policy.name, "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier)})
                    actions.append(
                        {
                            "fold_id": fold_id,
                            "policy": policy.name,
                            "event_id": event_id,
                            "action_time": timestamp,
                            "action": "PARTIAL_REDUCE",
                            "reason": "first_proven_soft_break_non_losing",
                            "units_before": before_units,
                            "units_after": float(updated["units"]) if updated is not None else 0.0,
                            "released_cycle_r": float(
                                close_units
                                * (float(item["entry_price"]) - float(item["disaster_stop_price"]))
                                / max(float(cycle_budget_dollars or 0.0), 1e-12)
                            ),
                        }
                    )
                if updated is None:
                    active.pop(event_id, None)
                else:
                    updated["soft_break_reduced"] = True
                    active[event_id] = updated

        # Preserve frozen overlap convention: entries are considered before a
        # source exit at the same minute open.
        for row in entry_map.get(position, []):
            event_id = str(row["event_id"])
            decision = {
                "fold_id": fold_id,
                "policy": policy.name,
                "event_id": event_id,
                "decision_time": pd.Timestamp(row["decision_time"]),
                "entry_time": pd.Timestamp(row["entry_time"]),
                "active_count": int(len(active)),
                "action": "SKIP",
                "reason": "",
                "entry_role": "",
                "allocated_cycle_r": 0.0,
                "root_event_id": "",
                "root_state": "",
                "root_open_return": np.nan,
                "root_was_reduced": False,
            }
            if event_id in active:
                decision["reason"] = "event_already_active"
                decisions.append(decision)
                continue
            marked_equity = _marked_equity(cash, active, open_price)
            if not active:
                cycle_budget_dollars = full_risk_fraction * marked_equity
                risk_dollars = float(cycle_budget_dollars)
                role = "primary"
            else:
                if len(active) >= config.maximum_virtual_tranches:
                    decision["reason"] = "maximum_two_tranches"
                    decisions.append(decision)
                    continue
                if not policy.allow_migration:
                    decision["reason"] = "single_position_policy"
                    decisions.append(decision)
                    continue
                if cycle_budget_dollars is None or cycle_budget_dollars <= 0:
                    decision["reason"] = "missing_cycle_budget"
                    decisions.append(decision)
                    continue
                root = min(active.values(), key=lambda item: pd.Timestamp(item["entry_time"]))
                root_id = str(root["event_id"])
                root_state = str(root.get("latest_structure_state", "UNKNOWN"))
                root_pending = bool(root.get("latest_pending_failed_reclaim", False))
                root_open_return = float(open_price / float(root["entry_price"]) - 1.0)
                decision.update(
                    {
                        "root_event_id": root_id,
                        "root_state": root_state,
                        "root_open_return": root_open_return,
                    }
                )
                if root_state != "HEALTHY" or root_pending:
                    decision["reason"] = "root_structure_not_healthy"
                    decisions.append(decision)
                    continue
                if root_open_return < 0.0:
                    decision["reason"] = "root_position_losing"
                    decisions.append(decision)
                    continue

                current_risk = sum(_risk_dollars(item) for item in active.values())
                free_capacity = max(float(cycle_budget_dollars) - current_risk, 0.0)
                target = float(policy.migration_target_r) * float(cycle_budget_dollars)
                required_release = max(target - free_capacity, 0.0)
                if required_release > 1e-12:
                    root_current_units = float(root["units"])
                    root_min_units = float(root["original_units"]) * float(config.minimum_root_remaining_fraction)
                    max_reducible_units = max(root_current_units - root_min_units, 0.0)
                    per_unit_risk = max(float(root["entry_price"]) - float(root["disaster_stop_price"]), 1e-12)
                    units_to_close = min(max_reducible_units, required_release / per_unit_risk)
                    if units_to_close > 1e-15:
                        updated, leg = _reduce_position(
                            root,
                            units_to_close=units_to_close,
                            price=open_price,
                            timestamp=timestamp,
                            side_cost=side_cost,
                            reason="q70_risk_migration_release",
                            leg_type="migration_release",
                        )
                        if leg:
                            cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
                            legs.append({**leg, "fold_id": fold_id, "policy": policy.name, "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier)})
                            decision["root_was_reduced"] = True
                            actions.append(
                                {
                                    "fold_id": fold_id,
                                    "policy": policy.name,
                                    "event_id": root_id,
                                    "new_event_id": event_id,
                                    "action_time": timestamp,
                                    "action": "MIGRATION_RELEASE",
                                    "reason": "fund_new_q70_without_increasing_cycle_risk",
                                    "units_before": root_current_units,
                                    "units_after": float(updated["units"]) if updated is not None else 0.0,
                                    "released_cycle_r": float(units_to_close * per_unit_risk / max(float(cycle_budget_dollars), 1e-12)),
                                }
                            )
                        if updated is None:
                            active.pop(root_id, None)
                        else:
                            active[root_id] = updated
                current_risk = sum(_risk_dollars(item) for item in active.values())
                free_capacity = max(float(cycle_budget_dollars) - current_risk, 0.0)
                risk_dollars = min(target, free_capacity)
                if risk_dollars + 1e-12 < float(config.minimum_migration_r) * float(cycle_budget_dollars):
                    decision["reason"] = "insufficient_real_released_capacity"
                    decisions.append(decision)
                    continue
                role = "secondary"

            entry_price = float(row["entry_price"])
            disaster_stop = entry_price * (1.0 - float(config.disaster_stop_distance))
            units = risk_dollars / max(entry_price - disaster_stop, 1e-12)
            entry_fee = units * entry_price * side_cost
            cash -= entry_fee
            item = {
                **row,
                "entry_role": role,
                "units": float(units),
                "original_units": float(units),
                "entry_fee": float(entry_fee),
                "remaining_entry_fee": float(entry_fee),
                "initial_risk_dollars": float(risk_dollars),
                "disaster_stop_price": float(disaster_stop),
                "soft_break_reduced": False,
                "partial_close_count": 0,
                "latest_structure_state": "UNKNOWN",
                "latest_structure_return": np.nan,
                "latest_pending_failed_reclaim": False,
                "latest_proven_structure": False,
                "latest_structure_close_time": pd.NaT,
            }
            active[event_id] = item
            opened_rows.append(
                {
                    "event_id": event_id,
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                    "entry_role": role,
                    "decision_time": pd.Timestamp(row["decision_time"]),
                    "entry_time": pd.Timestamp(row["entry_time"]),
                    "entry_price": entry_price,
                    "score": float(row["score"]),
                    "signal_quantile": float(row["signal_quantile"]),
                    "initial_risk_dollars": float(risk_dollars),
                    "allocated_cycle_r": float(risk_dollars / max(float(cycle_budget_dollars or risk_dollars), 1e-12)),
                    "source_exit_time": pd.Timestamp(row["exit_time"]),
                    "source_exit_reason": str(row["exit_reason"]),
                }
            )
            decision.update(
                {
                    "action": "ACCEPT",
                    "reason": "new_cycle_full_1R" if role == "primary" else "risk_migrated_to_new_q70",
                    "entry_role": role,
                    "allocated_cycle_r": float(risk_dollars / max(float(cycle_budget_dollars or risk_dollars), 1e-12)),
                }
            )
            decisions.append(decision)

        for row in exit_map.get(position, []):
            event_id = str(row["event_id"])
            item = active.pop(event_id, None)
            if item is None:
                continue
            updated, leg = _reduce_position(
                item,
                units_to_close=float(item["units"]),
                price=float(row["exit_price"]),
                timestamp=timestamp,
                side_cost=side_cost,
                reason=str(row["exit_reason"]),
                leg_type="final_exit",
            )
            if leg:
                cash += float(leg["gross_pnl"]) - float(leg["exit_fee"])
                legs.append({**leg, "fold_id": fold_id, "policy": policy.name, "delay_minutes": int(delay_minutes), "cost_multiplier": float(cost_multiplier)})
            if updated is not None:
                raise RuntimeError("final exit did not close the complete remaining tranche")

        if not active:
            cycle_budget_dollars = None

        equity = _marked_equity(cash, active, close_price)
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0 if peak_equity > 0 else -1.0
        max_drawdown = min(max_drawdown, drawdown)
        cycle_r = (
            sum(_risk_dollars(item) for item in active.values()) / max(float(cycle_budget_dollars or 0.0), 1e-12)
            if active
            else 0.0
        )
        exposure = sum(float(item["units"]) * close_price for item in active.values()) / max(equity, 1e-12)
        max_cycle_r = max(max_cycle_r, cycle_r)
        max_notional_to_equity = max(max_notional_to_equity, exposure)
        risk_observation_sum += cycle_r
        exposure_observation_sum += exposure
        observations += 1

        is_last_of_day = position == end_position or path.index[position + 1].date() != timestamp.date()
        if is_last_of_day:
            daily_rows.append(
                {
                    "date": timestamp.normalize(),
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                    "equity": float(equity),
                    "drawdown": float(drawdown),
                    "active_tranches": int(len(active)),
                    "cycle_allocated_r": float(cycle_r),
                    "notional_to_equity": float(exposure),
                }
            )
        reporter.update(position - start_position + 1)
    reporter.close()

    if active:
        raise RuntimeError(f"{policy.name} has active tranches after {fold_id}: {sorted(active)}")

    legs_frame = pd.DataFrame(legs)
    opened_frame = pd.DataFrame(opened_rows)
    trades = _aggregate_event_trades(legs_frame, opened_frame)
    daily = pd.DataFrame(daily_rows)
    decisions_frame = pd.DataFrame(decisions)
    actions_frame = pd.DataFrame(actions)
    final_equity = float(daily["equity"].iloc[-1]) if not daily.empty else cash
    pnl = trades["net_pnl"].astype(float).to_numpy() if not trades.empty else np.array([], dtype=float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    top = np.sort(winners)[::-1][: min(10, len(winners))]
    top_share = float(top.sum() / winners.sum()) if len(winners) and winners.sum() > 0 else np.nan
    without_top10 = float(final_equity - config.initial_equity - top.sum())

    migration_decisions = decisions_frame.loc[decisions_frame.get("entry_role", pd.Series(dtype=str)).astype(str).eq("secondary")] if not decisions_frame.empty else pd.DataFrame()
    accepted_secondary = migration_decisions.loc[migration_decisions["action"].astype(str).eq("ACCEPT")] if not migration_decisions.empty else pd.DataFrame()
    losing_share = float((accepted_secondary["root_open_return"].astype(float) < 0).mean()) if not accepted_secondary.empty else 0.0
    broken_share = float(accepted_secondary["root_state"].astype(str).ne("HEALTHY").mean()) if not accepted_secondary.empty else 0.0

    summary: dict[str, Any] = {
        "fold_id": fold_id,
        "policy": policy.name,
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "candidate_events": int(len(work)),
        "executed_tranches": int(len(trades)),
        "primary_tranches": int((trades["entry_role"] == "primary").sum()) if not trades.empty else 0,
        "secondary_tranches": int((trades["entry_role"] == "secondary").sum()) if not trades.empty else 0,
        "coverage_ratio": float(len(trades) / max(len(work), 1)),
        "monthly_tranches": float(len(trades) / 12.0),
        "final_equity": final_equity,
        "total_net_return": float(final_equity / config.initial_equity - 1.0),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else np.nan,
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_net_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "top10_profit_share": top_share,
        "total_return_without_top10": without_top10,
        "max_cycle_allocated_r": float(max_cycle_r),
        "mean_cycle_allocated_r": float(risk_observation_sum / max(observations, 1)),
        "max_notional_to_equity": float(max_notional_to_equity),
        "mean_notional_to_equity": float(exposure_observation_sum / max(observations, 1)),
        "partial_reduce_actions": int((actions_frame.get("action", pd.Series(dtype=str)) == "PARTIAL_REDUCE").sum()) if not actions_frame.empty else 0,
        "migration_release_actions": int((actions_frame.get("action", pd.Series(dtype=str)) == "MIGRATION_RELEASE").sum()) if not actions_frame.empty else 0,
        "losing_migration_share": losing_share,
        "broken_migration_share": broken_share,
        "runtime_rejections": int(len(runtime_rejections)),
        "positive_months": 0,
        "positive_quarters": 0,
    }
    if not daily.empty:
        equity_series = daily.set_index("date")["equity"]
        month_end = equity_series.resample("ME").last()
        quarter_end = equity_series.resample("QE").last()
        monthly = month_end.pct_change().dropna()
        quarterly = quarter_end.pct_change().dropna()
        if len(month_end):
            monthly = pd.concat([pd.Series([month_end.iloc[0] / config.initial_equity - 1.0], index=[month_end.index[0]]), monthly])
        if len(quarter_end):
            quarterly = pd.concat([pd.Series([quarter_end.iloc[0] / config.initial_equity - 1.0], index=[quarter_end.index[0]]), quarterly])
        summary["positive_months"] = int((monthly > 0).sum())
        summary["positive_quarters"] = int((quarterly > 0).sum())

    return RiskMigrationSimulation(
        trades=trades,
        legs=legs_frame,
        daily_equity=daily,
        decisions=decisions_frame,
        actions=actions_frame,
        summary=summary,
        runtime_rejections=pd.DataFrame(runtime_rejections),
    )
