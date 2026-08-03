#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal pair diagnostics, virtual-slot selection and account simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_tranche_eligibility.simulator import (
    classify_occupied_signal,
    failed_reclaim_snapshots,
)

from src.research_common.progress import ProgressReporter

from .config import TrancheAccountConfig, TranchePolicy


@dataclass(frozen=True)
class PolicySelection:
    accepted: pd.DataFrame
    decisions: pd.DataFrame


@dataclass(frozen=True)
class AccountSimulation:
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    runtime_rejections: pd.DataFrame


def _event_candidate(row: pd.Series) -> EventCandidate:
    return EventCandidate(
        event_id=str(row["event_id"]),
        decision_time_ns=int(pd.Timestamp(row["decision_time"]).value),
        score=float(row["score"]),
        signal_quantile=float(row["signal_quantile"]),
    )


def build_pair_diagnostics(
    structural: pd.DataFrame,
    *,
    delay_minutes: int,
    path: MinutePathData,
    config: TrancheAccountConfig,
    progress: bool = True,
) -> pd.DataFrame:
    """Build causal diagnostics for every overlapping root/new-event pair.

    R03.4.2.8A only snapshots events blocked by its single-position baseline.
    A two-slot policy can leave a formerly secondary tranche active after the
    original root exits, so 2.8B must rebuild pair diagnostics for every
    possible active root. No future bar after the new decision is used.
    """

    work = structural.loc[structural["delay_minutes"] == int(delay_minutes)].copy()
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    eligibility_config = config.eligibility_config()
    structural_config = eligibility_config.structural_config()
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter(
        f"[R03.4.2.8B pair d{delay_minutes}]",
        len(work),
        every=max(1, len(work) // 100),
        enabled=progress,
    )

    entry_ns = pd.to_datetime(work["entry_time"]).astype("int64").to_numpy()
    for root_index, root in work.iterrows():
        root_exit = pd.Timestamp(root["exit_time"])
        right = int(np.searchsorted(entry_ns, int(root_exit.value), side="right"))
        if right <= root_index + 1:
            reporter.update(root_index + 1)
            continue
        candidates = work.iloc[root_index + 1 : right]
        candidates = candidates.loc[pd.to_datetime(candidates["decision_time"]) > pd.Timestamp(root["decision_time"])]
        if candidates.empty:
            reporter.update(root_index + 1)
            continue
        observation_times = tuple(int(pd.Timestamp(value).value) for value in candidates["decision_time"].tolist())
        snapshots = failed_reclaim_snapshots(
            _event_candidate(root),
            delay_minutes=int(delay_minutes),
            observation_times_ns=observation_times,
            path=path,
            end_time_ns=int(root_exit.value),
            config=structural_config,
        )
        for _, new in candidates.iterrows():
            observation_ns = int(pd.Timestamp(new["decision_time"]).value)
            snapshot = snapshots.get(observation_ns)
            if snapshot is None:
                continue
            base: dict[str, object] = {
                "fold_id": str(root["fold_id"]),
                "delay_minutes": int(delay_minutes),
                "root_event_id": str(root["event_id"]),
                "event_id": str(new["event_id"]),
                "root_decision_time": pd.Timestamp(root["decision_time"]),
                "decision_time": pd.Timestamp(new["decision_time"]),
                "root_entry_time": pd.Timestamp(root["entry_time"]),
                "new_entry_time": pd.Timestamp(new["entry_time"]),
                "root_exit_time": root_exit,
                "root_entry_price": float(root["entry_price"]),
                "new_entry_price": float(new["entry_price"]),
                "root_score": float(root["score"]),
                "new_score": float(new["score"]),
                "score_delta_vs_root": float(new["score"] - root["score"]),
                "price_return_at_new_entry": float(new["entry_price"] / root["entry_price"] - 1.0),
                **snapshot,
            }
            signal_class, class_reason, strict_eligible = classify_occupied_signal(base, config=eligibility_config)
            base["signal_class"] = signal_class
            base["class_reason"] = class_reason
            base["strict_2_8a_eligible"] = bool(strict_eligible)
            base["score_up_price_down"] = bool(
                float(base["score_delta_vs_root"]) > 0
                and float(base["price_return_at_new_entry"]) < 0
            )
            base["protected_policy_block"] = bool(
                signal_class == "dangerous_average_down"
                or str(base.get("state", "UNKNOWN")) == "BROKEN"
                or bool(base.get("pending_failed_reclaim_exit", False))
            )
            rows.append(base)
        reporter.update(root_index + 1)
    reporter.close()
    return pd.DataFrame(rows)


def _pair_lookup(pair_diagnostics: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if pair_diagnostics.empty:
        return {}
    return {
        (str(row["root_event_id"]), str(row["event_id"])): row
        for row in pair_diagnostics.to_dict("records")
    }


def select_policy_trades(
    structural: pd.DataFrame,
    *,
    policy: TranchePolicy,
    pair_diagnostics: pd.DataFrame,
) -> PolicySelection:
    """Select at most two independent failed-reclaim tranches causally."""

    policy.validate()
    work = structural.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    pair_lookup = _pair_lookup(pair_diagnostics)
    active: dict[str, dict[str, object]] = {}
    accepted_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for event in work.to_dict("records"):
        entry_time = pd.Timestamp(event["entry_time"])
        # Equality remains occupied to match the frozen P0 overlap convention.
        active = {
            slot: row
            for slot, row in active.items()
            if pd.Timestamp(row["exit_time"]) >= entry_time
        }
        active_rows = list(active.values())
        active_count = len(active_rows)
        root = active_rows[0] if active_count == 1 else None
        pair = pair_lookup.get((str(root["event_id"]), str(event["event_id"]))) if root is not None else None

        decision: dict[str, object] = {
            "policy": policy.name,
            "event_id": str(event["event_id"]),
            "fold_id": str(event["fold_id"]),
            "delay_minutes": int(event["delay_minutes"]),
            "decision_time": pd.Timestamp(event["decision_time"]),
            "entry_time": entry_time,
            "active_tranches_before": int(active_count),
            "active_root_event_id": str(root["event_id"]) if root is not None else "",
            "pair_diagnostic_available": pair is not None,
            "signal_class_vs_active": pair.get("signal_class", "") if pair else "",
            "active_current_return": float(pair.get("current_return_vs_root", np.nan)) if pair else np.nan,
            "score_up_price_down": bool(pair.get("score_up_price_down", False)) if pair else False,
            "protected_policy_block": bool(pair.get("protected_policy_block", False)) if pair else False,
        }

        if active_count >= policy.max_tranches:
            decision.update({"action": "SKIP", "reason": "risk_slots_full", "slot": "", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue

        free_slots = [slot for slot in ("A", "B") if slot not in active]
        if policy.max_tranches == 1:
            free_slots = [slot for slot in free_slots if slot == "A"]
        slot = free_slots[0] if free_slots else ""
        if not slot:
            decision.update({"action": "SKIP", "reason": "no_free_slot", "slot": "", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue

        if policy.protection_gate and active_count == 1:
            if pair is None:
                decision.update({"action": "SKIP", "reason": "missing_causal_pair_diagnostic", "slot": slot, "risk_weight_r": 0.0})
                decision_rows.append(decision)
                continue
            if bool(pair.get("protected_policy_block", False)):
                decision.update({"action": "SKIP", "reason": "dangerous_or_broken_active_structure", "slot": slot, "risk_weight_r": 0.0})
                decision_rows.append(decision)
                continue

        risk_weight = float(policy.slot_a_r if slot == "A" else policy.slot_b_r)
        if risk_weight <= 0:
            decision.update({"action": "SKIP", "reason": "zero_risk_slot", "slot": slot, "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue

        accepted = {
            **event,
            "policy": policy.name,
            "slot": slot,
            "risk_weight_r": risk_weight,
            "entry_role": "primary" if active_count == 0 else "secondary",
            "active_root_event_id": str(root["event_id"]) if root is not None else "",
            "signal_class_vs_active": pair.get("signal_class", "") if pair else "",
            "active_current_return": float(pair.get("current_return_vs_root", np.nan)) if pair else np.nan,
            "score_up_price_down": bool(pair.get("score_up_price_down", False)) if pair else False,
            "dangerous_second_add": bool(pair.get("signal_class") == "dangerous_average_down") if pair else False,
            "protected_policy_block": bool(pair.get("protected_policy_block", False)) if pair else False,
        }
        active[slot] = accepted
        accepted_rows.append(accepted)
        decision.update({"action": "ACCEPT", "reason": "slot_available", "slot": slot, "risk_weight_r": risk_weight})
        decision_rows.append(decision)

    return PolicySelection(pd.DataFrame(accepted_rows), pd.DataFrame(decision_rows))


def _exact_position(path: MinutePathData, timestamp: pd.Timestamp) -> int | None:
    value = int(pd.Timestamp(timestamp).value)
    position = int(np.searchsorted(path.timestamps_ns, value, side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != value:
        return None
    return position


def simulate_account(
    accepted: pd.DataFrame,
    *,
    path: MinutePathData,
    fold_id: str,
    policy: TranchePolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: TrancheAccountConfig,
    progress: bool = True,
) -> AccountSimulation:
    """Run a minute-marked account with independent virtual tranche exits."""

    if accepted.empty:
        return AccountSimulation(pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())
    work = accepted.copy().sort_values(["entry_time", "decision_time", "event_id"]).reset_index(drop=True)
    start_position = _exact_position(path, pd.Timestamp(test_start))
    end_position = _exact_position(path, pd.Timestamp(test_end).floor("min"))
    if start_position is None or end_position is None or end_position <= start_position:
        raise RuntimeError(f"minute path does not exactly cover {fold_id}")

    entry_map: dict[int, list[dict[str, object]]] = {}
    exit_map: dict[int, list[dict[str, object]]] = {}
    missing_rows: list[dict[str, object]] = []
    for row in work.to_dict("records"):
        entry_position = _exact_position(path, pd.Timestamp(row["entry_time"]))
        exit_position = _exact_position(path, pd.Timestamp(row["exit_time"]))
        if entry_position is None or exit_position is None or exit_position < entry_position:
            missing_rows.append({"event_id": row["event_id"], "reason": "missing_entry_or_exit_path"})
            continue
        row["entry_position"] = int(entry_position)
        row["exit_position"] = int(exit_position)
        entry_map.setdefault(int(entry_position), []).append(row)
        exit_map.setdefault(int(exit_position), []).append(row)

    cash = float(config.initial_equity)
    active: dict[str, dict[str, object]] = {}
    closed_rows: list[dict[str, object]] = []
    runtime_rejections: list[dict[str, object]] = list(missing_rows)
    daily_rows: list[dict[str, object]] = []
    peak_equity = float(config.initial_equity)
    max_drawdown = 0.0
    max_allocated_r = 0.0
    max_slot_r = 0.0
    max_notional_to_equity = 0.0
    risk_sum = 0.0
    exposure_sum = 0.0
    observation_count = 0
    side_cost = float(config.base_round_trip_cost * cost_multiplier / 2.0)
    account_reporter = ProgressReporter(
        f"[R03.4.2.8B {fold_id} {policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        end_position - start_position + 1,
        every=max(1, (end_position - start_position + 1) // 100),
        enabled=progress,
    )
    full_risk_fraction = float(config.account_risk_fraction_per_full_r)

    for position in range(start_position, end_position + 1):
        open_price = float(path.open[position])
        close_price = float(path.close[position])

        # Preserve the frozen overlap convention: an entry at the exact exit
        # timestamp still sees the old tranche as active. Entries therefore run
        # before exits at the same minute open.
        for row in entry_map.get(position, []):
            marked_equity = cash + sum(
                float(item["units"]) * (open_price - float(item["entry_price"]))
                for item in active.values()
            )
            active_risk_dollars = sum(float(item["risk_dollars"]) for item in active.values())
            cap_dollars = max(0.0, config.maximum_allocated_r * full_risk_fraction * marked_equity)
            target_risk = float(row["risk_weight_r"]) * full_risk_fraction * marked_equity
            risk_dollars = min(target_risk, max(0.0, cap_dollars - active_risk_dollars))
            if not np.isfinite(risk_dollars) or risk_dollars <= 1e-12:
                runtime_rejections.append(
                    {
                        "event_id": row["event_id"],
                        "fold_id": fold_id,
                        "policy": policy.name,
                        "delay_minutes": delay_minutes,
                        "cost_multiplier": cost_multiplier,
                        "entry_time": row["entry_time"],
                        "reason": "runtime_account_risk_cap",
                    }
                )
                continue
            entry_price = float(row["entry_price"])
            units = risk_dollars / (entry_price * config.disaster_stop_distance)
            notional = units * entry_price
            entry_fee = notional * side_cost
            cash -= entry_fee
            active[str(row["event_id"])] = {
                **row,
                "units": float(units),
                "notional": float(notional),
                "risk_dollars": float(risk_dollars),
                "entry_fee": float(entry_fee),
                "entry_equity": float(marked_equity),
                "actual_allocated_r": float(risk_dollars / max(full_risk_fraction * marked_equity, 1e-12)),
            }

        for row in exit_map.get(position, []):
            event_id = str(row["event_id"])
            item = active.pop(event_id, None)
            if item is None:
                continue
            exit_price = float(row["exit_price"])
            gross_pnl = float(item["units"]) * (exit_price - float(item["entry_price"]))
            exit_fee = float(item["units"]) * exit_price * side_cost
            net_pnl = gross_pnl - float(item["entry_fee"]) - exit_fee
            # Entry fee was already deducted from cash. Add gross PnL and only
            # the exit fee now so cash ends at initial cash + complete net PnL.
            cash += gross_pnl - exit_fee
            closed_rows.append(
                {
                    **{key: value for key, value in item.items() if key not in {"entry_position", "exit_position"}},
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "exit_fee": float(exit_fee),
                    "realized_r": float(net_pnl / max(float(item["risk_dollars"]), 1e-12)),
                    "loss_exceeded_allocated_risk": bool(net_pnl < -float(item["risk_dollars"])),
                }
            )

        equity = cash + sum(
            float(item["units"]) * (close_price - float(item["entry_price"]))
            for item in active.values()
        )
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0 if peak_equity > 0 else -1.0
        max_drawdown = min(max_drawdown, drawdown)
        allocated_r = sum(float(item["risk_dollars"]) for item in active.values()) / max(full_risk_fraction * equity, 1e-12)
        notional_exposure = sum(float(item["units"]) * close_price for item in active.values()) / max(equity, 1e-12)
        slot_r = sum(float(item["risk_weight_r"]) for item in active.values())
        max_allocated_r = max(max_allocated_r, allocated_r)
        max_slot_r = max(max_slot_r, slot_r)
        max_notional_to_equity = max(max_notional_to_equity, notional_exposure)
        risk_sum += allocated_r
        exposure_sum += notional_exposure
        observation_count += 1

        timestamp = path.index[position]
        is_last_of_day = position == end_position or path.index[position + 1].date() != timestamp.date()
        if is_last_of_day:
            daily_rows.append(
                {
                    "date": pd.Timestamp(timestamp).normalize(),
                    "fold_id": fold_id,
                    "policy": policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                    "equity": float(equity),
                    "drawdown": float(drawdown),
                    "active_tranches": int(len(active)),
                    "allocated_r": float(allocated_r),
                    "notional_to_equity": float(notional_exposure),
                }
            )
        account_reporter.update(position - start_position + 1)
    account_reporter.close()

    if active:
        raise RuntimeError(f"{policy.name} has active tranches after {fold_id} test end: {sorted(active)}")

    trades = pd.DataFrame(closed_rows)
    daily = pd.DataFrame(daily_rows)
    final_equity = float(daily["equity"].iloc[-1]) if not daily.empty else cash
    net_pnls = trades["net_pnl"].astype(float).to_numpy() if not trades.empty else np.array([], dtype=float)
    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls < 0]
    positive_profit = float(winners.sum())
    top_count = min(10, len(winners))
    top_winners = np.sort(winners)[::-1][:top_count]
    top_share = float(top_winners.sum() / positive_profit) if positive_profit > 0 else np.nan
    without_top10 = float(final_equity - config.initial_equity - top_winners.sum())

    summary = {
        "fold_id": fold_id,
        "policy": policy.name,
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "accepted_tranches": int(len(work)),
        "executed_tranches": int(len(trades)),
        "runtime_rejections": int(len(runtime_rejections)),
        "final_equity": final_equity,
        "total_net_return": float(final_equity / config.initial_equity - 1.0),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(np.mean(net_pnls > 0)) if len(net_pnls) else np.nan,
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_net_pnl": float(net_pnls.mean()) if len(net_pnls) else np.nan,
        "median_realized_r": float(trades["realized_r"].median()) if not trades.empty else np.nan,
        "top10_profit_share": top_share,
        "total_return_without_top10": without_top10,
        "positive_months": 0,
        "positive_quarters": 0,
        "max_allocated_r": float(max_allocated_r),
        "max_slot_r": float(max_slot_r),
        "mean_allocated_r": float(risk_sum / max(observation_count, 1)),
        "max_notional_to_equity": float(max_notional_to_equity),
        "mean_notional_to_equity": float(exposure_sum / max(observation_count, 1)),
        "risk_breach_count": int((trades["loss_exceeded_allocated_risk"] == True).sum()) if not trades.empty else 0,
        "censored_tranches": int(trades["is_censored"].astype(bool).sum()) if not trades.empty and "is_censored" in trades.columns else 0,
        "censored_share": float(trades["is_censored"].astype(bool).mean()) if not trades.empty and "is_censored" in trades.columns else 0.0,
    }
    if not daily.empty:
        equity = daily.set_index("date")["equity"]
        month_end = equity.resample("ME").last()
        quarter_end = equity.resample("QE").last()
        monthly_returns = month_end.pct_change().dropna()
        quarterly_returns = quarter_end.pct_change().dropna()
        # Include the first period against initial equity.
        if len(month_end):
            monthly_returns = pd.concat([pd.Series([month_end.iloc[0] / config.initial_equity - 1.0], index=[month_end.index[0]]), monthly_returns])
        if len(quarter_end):
            quarterly_returns = pd.concat([pd.Series([quarter_end.iloc[0] / config.initial_equity - 1.0], index=[quarter_end.index[0]]), quarterly_returns])
        summary["positive_months"] = int((monthly_returns > 0).sum())
        summary["positive_quarters"] = int((quarterly_returns > 0).sum())
    return AccountSimulation(
        trades=trades,
        daily_equity=daily,
        summary=summary,
        runtime_rejections=pd.DataFrame(runtime_rejections),
    )
