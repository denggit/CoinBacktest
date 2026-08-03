#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dynamic risk-release selection and minute-marked account simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.research_common.progress import ProgressReporter

from .config import DynamicReleasePolicy, DynamicRiskReleaseConfig
from .protection import exact_position


@dataclass(frozen=True)
class DynamicSelection:
    accepted: pd.DataFrame
    decisions: pd.DataFrame


@dataclass(frozen=True)
class DynamicAccountSimulation:
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    summary: dict[str, Any]
    runtime_rejections: pd.DataFrame


def build_release_pair_diagnostics(
    trades: pd.DataFrame,
    states: pd.DataFrame,
    *,
    protection_policy: str,
    delay_minutes: int,
) -> pd.DataFrame:
    """Map every overlapping new q70 event to the active root's latest causal state."""

    work = trades.loc[
        (trades["protection_policy"] == protection_policy)
        & (trades["delay_minutes"].astype(int) == int(delay_minutes))
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    state_work = states.loc[
        (states["protection_policy"] == protection_policy)
        & (states["delay_minutes"].astype(int) == int(delay_minutes))
    ].copy()
    state_groups = {
        str(event_id): group.sort_values("structure_close_time").reset_index(drop=True)
        for event_id, group in state_work.groupby("event_id", sort=False)
    }
    entry_ns = pd.to_datetime(work["entry_time"]).astype("int64").to_numpy()
    rows: list[dict[str, object]] = []
    for root_index, root in work.iterrows():
        root_exit = pd.Timestamp(root["exit_time"])
        right = int(np.searchsorted(entry_ns, int(root_exit.value), side="right"))
        if right <= root_index + 1:
            continue
        group = state_groups.get(str(root["event_id"]))
        if group is None or group.empty:
            continue
        state_ns = pd.to_datetime(group["structure_close_time"]).astype("int64").to_numpy()
        for _, new in work.iloc[root_index + 1 : right].iterrows():
            decision_time = pd.Timestamp(new["decision_time"])
            if decision_time <= pd.Timestamp(root["decision_time"]):
                continue
            position = int(np.searchsorted(state_ns, int(decision_time.value), side="right")) - 1
            if position < 0:
                continue
            snapshot = group.iloc[position]
            rows.append(
                {
                    "fold_id": str(root["fold_id"]),
                    "delay_minutes": int(delay_minutes),
                    "protection_policy": protection_policy,
                    "root_event_id": str(root["event_id"]),
                    "event_id": str(new["event_id"]),
                    "root_entry_time": pd.Timestamp(root["entry_time"]),
                    "root_exit_time": root_exit,
                    "decision_time": decision_time,
                    "new_entry_time": pd.Timestamp(new["entry_time"]),
                    "root_entry_price": float(root["entry_price"]),
                    "new_entry_price": float(new["entry_price"]),
                    "structure_close_time": pd.Timestamp(snapshot["structure_close_time"]),
                    "state": str(snapshot["state"]),
                    "pending_failed_reclaim_exit": bool(snapshot["pending_failed_reclaim_exit"]),
                    "current_return": float(snapshot["current_return"]),
                    "stop_price": float(snapshot["stop_price"]),
                    "stop_return_vs_entry": float(snapshot["stop_return_vs_entry"]),
                    "remaining_initial_risk_fraction": float(snapshot["remaining_initial_risk_fraction"]),
                    "released_risk_fraction": float(snapshot["released_risk_fraction"]),
                    "stop_at_or_above_entry": bool(snapshot["stop_at_or_above_entry"]),
                    "score_delta": float(new["score"] - root["score"]),
                    "price_return_vs_root_entry": float(new["entry_price"] / root["entry_price"] - 1.0),
                    "score_up_price_down": bool(
                        float(new["score"] - root["score"]) > 0
                        and float(new["entry_price"] / root["entry_price"] - 1.0) < 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pair_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if frame.empty:
        return {}
    return {
        (str(row["root_event_id"]), str(row["event_id"])): row
        for row in frame.to_dict("records")
    }


def select_dynamic_trades(
    trades: pd.DataFrame,
    *,
    protection_policy: str,
    dynamic_policy: DynamicReleasePolicy,
    pair_diagnostics: pd.DataFrame,
) -> DynamicSelection:
    """Keep primary entries at 1R and fund only a causal second tranche from released risk."""

    dynamic_policy.validate()
    work = trades.loc[trades["protection_policy"] == protection_policy].copy()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    lookup = _pair_lookup(pair_diagnostics)
    active: dict[str, dict[str, object]] = {}
    accepted_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for event in work.to_dict("records"):
        entry_time = pd.Timestamp(event["entry_time"])
        active = {
            event_id: row
            for event_id, row in active.items()
            if pd.Timestamp(row["exit_time"]) >= entry_time
        }
        active_rows = list(active.values())
        active_count = len(active_rows)
        decision: dict[str, object] = {
            "fold_id": str(event["fold_id"]),
            "delay_minutes": int(event["delay_minutes"]),
            "protection_policy": protection_policy,
            "dynamic_policy": dynamic_policy.name,
            "event_id": str(event["event_id"]),
            "decision_time": pd.Timestamp(event["decision_time"]),
            "entry_time": entry_time,
            "active_tranches_before": int(active_count),
        }

        if active_count == 0:
            accepted = {
                **event,
                "dynamic_policy": dynamic_policy.name,
                "risk_weight_r": 1.0,
                "entry_role": "primary",
                "active_root_event_id": "",
                "released_account_r_at_entry": 0.0,
                "active_remaining_r_at_entry": 0.0,
                "active_current_return": np.nan,
                "active_state": "",
                "active_pending_failed_reclaim": False,
            }
            active[str(event["event_id"])] = accepted
            accepted_rows.append(accepted)
            decision.update({"action": "ACCEPT", "reason": "primary_full_1R", "risk_weight_r": 1.0})
            decision_rows.append(decision)
            continue

        if active_count >= 2:
            decision.update({"action": "SKIP", "reason": "maximum_two_tranches", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue
        if not dynamic_policy.allow_secondary:
            decision.update({"action": "SKIP", "reason": "single_tranche_baseline", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue

        root = active_rows[0]
        pair = lookup.get((str(root["event_id"]), str(event["event_id"])))
        if pair is None:
            decision.update({"action": "SKIP", "reason": "missing_causal_release_snapshot", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue
        state = str(pair.get("state", "UNKNOWN"))
        pending = bool(pair.get("pending_failed_reclaim_exit", False))
        current_return = float(pair.get("current_return", np.nan))
        released_fraction = float(pair.get("released_risk_fraction", 0.0))
        root_weight = float(root["risk_weight_r"])
        released_account_r = float(np.clip(root_weight * released_fraction, 0.0, root_weight))
        remaining_account_r = float(root_weight - released_account_r)
        decision.update(
            {
                "active_root_event_id": str(root["event_id"]),
                "active_state": state,
                "active_pending_failed_reclaim": pending,
                "active_current_return": current_return,
                "active_stop_price": float(pair.get("stop_price", np.nan)),
                "released_risk_fraction": released_fraction,
                "released_account_r": released_account_r,
                "active_remaining_r": remaining_account_r,
                "score_up_price_down": bool(pair.get("score_up_price_down", False)),
            }
        )
        if dynamic_policy.require_healthy_state and (state != "HEALTHY" or pending):
            decision.update({"action": "SKIP", "reason": "active_structure_not_healthy", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue
        if dynamic_policy.require_non_losing_active and (not np.isfinite(current_return) or current_return < 0.0):
            decision.update({"action": "SKIP", "reason": "active_position_still_losing", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue
        risk_weight = min(float(dynamic_policy.max_secondary_r), released_account_r)
        if risk_weight + 1e-12 < float(dynamic_policy.minimum_release_r):
            decision.update({"action": "SKIP", "reason": "insufficient_enforceable_risk_release", "risk_weight_r": 0.0})
            decision_rows.append(decision)
            continue

        accepted = {
            **event,
            "dynamic_policy": dynamic_policy.name,
            "risk_weight_r": float(risk_weight),
            "entry_role": "secondary",
            "active_root_event_id": str(root["event_id"]),
            "released_account_r_at_entry": released_account_r,
            "active_remaining_r_at_entry": remaining_account_r,
            "active_current_return": current_return,
            "active_state": state,
            "active_pending_failed_reclaim": pending,
            "score_up_price_down": bool(pair.get("score_up_price_down", False)),
        }
        active[str(event["event_id"])] = accepted
        accepted_rows.append(accepted)
        decision.update({"action": "ACCEPT", "reason": "funded_by_enforceable_release", "risk_weight_r": float(risk_weight)})
        decision_rows.append(decision)

    return DynamicSelection(pd.DataFrame(accepted_rows), pd.DataFrame(decision_rows))


def simulate_dynamic_account(
    accepted: pd.DataFrame,
    *,
    stop_updates: pd.DataFrame,
    path: MinutePathData,
    fold_id: str,
    protection_policy: str,
    dynamic_policy: DynamicReleasePolicy,
    delay_minutes: int,
    cost_multiplier: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: DynamicRiskReleaseConfig,
    progress: bool = True,
) -> DynamicAccountSimulation:
    """Minute-marked account using live stop risk rather than static slot sums."""

    if accepted.empty:
        return DynamicAccountSimulation(pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())
    work = accepted.copy().sort_values(["entry_time", "decision_time", "event_id"]).reset_index(drop=True)
    start_position = exact_position(path, pd.Timestamp(test_start))
    end_position = exact_position(path, pd.Timestamp(test_end).floor("min"))
    if start_position is None or end_position is None or end_position <= start_position:
        raise RuntimeError(f"minute path does not exactly cover {fold_id}")

    entry_map: dict[int, list[dict[str, object]]] = {}
    exit_map: dict[int, list[dict[str, object]]] = {}
    runtime_rejections: list[dict[str, object]] = []
    accepted_ids = set(work["event_id"].astype(str))
    update_map: dict[int, list[dict[str, object]]] = {}
    update_work = stop_updates.loc[
        (stop_updates["protection_policy"] == protection_policy)
        & (stop_updates["delay_minutes"].astype(int) == int(delay_minutes))
        & (stop_updates["event_id"].astype(str).isin(accepted_ids))
    ].copy()
    for update in update_work.to_dict("records"):
        position = exact_position(path, pd.Timestamp(update["effective_time"]))
        if position is not None:
            update_map.setdefault(int(position), []).append(update)

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

    cash = float(config.initial_equity)
    active: dict[str, dict[str, object]] = {}
    closed_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    peak_equity = cash
    max_drawdown = 0.0
    max_live_remaining_r = 0.0
    max_initial_r_sum = 0.0
    max_notional_to_equity = 0.0
    mean_live_r_sum = 0.0
    mean_notional_sum = 0.0
    observations = 0
    side_cost = float(config.base_round_trip_cost * cost_multiplier / 2.0)
    full_risk_fraction = float(config.account_risk_fraction_per_full_r)
    reporter = ProgressReporter(
        f"[R03.4.2.9 {fold_id} {protection_policy}/{dynamic_policy.name} d{delay_minutes} c{cost_multiplier:g}]",
        end_position - start_position + 1,
        every=max(1, (end_position - start_position + 1) // 100),
        enabled=progress,
    )

    for position in range(start_position, end_position + 1):
        open_price = float(path.open[position])
        close_price = float(path.close[position])

        # A stop confirmed by the previous completed structure bar is live at
        # this minute open before any new q70 entry is considered.
        for update in update_map.get(position, []):
            item = active.get(str(update["event_id"]))
            if item is not None:
                item["current_stop_price"] = max(float(item["current_stop_price"]), float(update["stop_price"]))

        for row in entry_map.get(position, []):
            marked_equity = cash + sum(
                float(item["units"]) * (open_price - float(item["entry_price"]))
                for item in active.values()
            )
            live_risk_dollars = sum(
                float(item["units"]) * max(float(item["entry_price"]) - float(item["current_stop_price"]), 0.0)
                for item in active.values()
            )
            cap_dollars = max(0.0, config.maximum_live_remaining_r * full_risk_fraction * marked_equity)
            target_risk = float(row["risk_weight_r"]) * full_risk_fraction * marked_equity
            risk_dollars = min(target_risk, max(0.0, cap_dollars - live_risk_dollars))
            if not np.isfinite(risk_dollars) or risk_dollars <= 1e-12:
                runtime_rejections.append(
                    {
                        "event_id": row["event_id"],
                        "fold_id": fold_id,
                        "protection_policy": protection_policy,
                        "dynamic_policy": dynamic_policy.name,
                        "delay_minutes": int(delay_minutes),
                        "cost_multiplier": float(cost_multiplier),
                        "entry_time": row["entry_time"],
                        "reason": "runtime_live_risk_cap",
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
                "current_stop_price": float(row["initial_stop_price"]),
            }

        # Preserve the prior overlap convention: a same-minute new entry still
        # saw the old tranche as active when selected.
        for row in exit_map.get(position, []):
            event_id = str(row["event_id"])
            item = active.pop(event_id, None)
            if item is None:
                continue
            exit_price = float(row["exit_price"])
            gross_pnl = float(item["units"]) * (exit_price - float(item["entry_price"]))
            exit_fee = float(item["units"]) * exit_price * side_cost
            net_pnl = gross_pnl - float(item["entry_fee"]) - exit_fee
            cash += gross_pnl - exit_fee
            closed_rows.append(
                {
                    **{key: value for key, value in item.items() if key not in {"entry_position", "exit_position"}},
                    "fold_id": fold_id,
                    "protection_policy": protection_policy,
                    "dynamic_policy": dynamic_policy.name,
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
        live_risk_dollars = sum(
            float(item["units"]) * max(float(item["entry_price"]) - float(item["current_stop_price"]), 0.0)
            for item in active.values()
        )
        live_r = live_risk_dollars / max(full_risk_fraction * equity, 1e-12)
        initial_r_sum = sum(float(item["actual_allocated_r"]) for item in active.values())
        notional_to_equity = sum(float(item["units"]) * close_price for item in active.values()) / max(equity, 1e-12)
        max_live_remaining_r = max(max_live_remaining_r, live_r)
        max_initial_r_sum = max(max_initial_r_sum, initial_r_sum)
        max_notional_to_equity = max(max_notional_to_equity, notional_to_equity)
        mean_live_r_sum += live_r
        mean_notional_sum += notional_to_equity
        observations += 1

        timestamp = path.index[position]
        is_last_of_day = position == end_position or path.index[position + 1].date() != timestamp.date()
        if is_last_of_day:
            daily_rows.append(
                {
                    "date": pd.Timestamp(timestamp).normalize(),
                    "fold_id": fold_id,
                    "protection_policy": protection_policy,
                    "dynamic_policy": dynamic_policy.name,
                    "delay_minutes": int(delay_minutes),
                    "cost_multiplier": float(cost_multiplier),
                    "equity": float(equity),
                    "drawdown": float(drawdown),
                    "active_tranches": int(len(active)),
                    "live_remaining_r": float(live_r),
                    "initial_r_sum": float(initial_r_sum),
                    "notional_to_equity": float(notional_to_equity),
                }
            )
        reporter.update(position - start_position + 1)
    reporter.close()

    if active:
        raise RuntimeError(f"active tranches remain after {fold_id}: {sorted(active)}")

    trades_frame = pd.DataFrame(closed_rows)
    daily = pd.DataFrame(daily_rows)
    final_equity = float(daily["equity"].iloc[-1]) if not daily.empty else cash
    net = trades_frame["net_pnl"].astype(float).to_numpy() if not trades_frame.empty else np.array([], dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]
    positive_profit = float(winners.sum())
    top = np.sort(winners)[::-1][: min(10, len(winners))]
    top_share = float(top.sum() / positive_profit) if positive_profit > 0 else np.nan
    without_top10 = float(final_equity - config.initial_equity - top.sum())
    secondary = trades_frame.loc[trades_frame["entry_role"] == "secondary"] if not trades_frame.empty else pd.DataFrame()

    summary: dict[str, Any] = {
        "fold_id": fold_id,
        "protection_policy": protection_policy,
        "dynamic_policy": dynamic_policy.name,
        "delay_minutes": int(delay_minutes),
        "cost_multiplier": float(cost_multiplier),
        "accepted_tranches": int(len(work)),
        "executed_tranches": int(len(trades_frame)),
        "runtime_rejections": int(len(runtime_rejections)),
        "final_equity": final_equity,
        "total_net_return": float(final_equity / config.initial_equity - 1.0),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
        "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.inf,
        "mean_net_pnl": float(net.mean()) if len(net) else np.nan,
        "top10_profit_share": top_share,
        "total_return_without_top10": without_top10,
        "max_live_remaining_r": float(max_live_remaining_r),
        "max_initial_r_sum": float(max_initial_r_sum),
        "mean_live_remaining_r": float(mean_live_r_sum / max(observations, 1)),
        "max_notional_to_equity": float(max_notional_to_equity),
        "mean_notional_to_equity": float(mean_notional_sum / max(observations, 1)),
        "risk_breach_count": int((trades_frame["loss_exceeded_allocated_risk"] == True).sum()) if not trades_frame.empty else 0,
        "primary_tranches": int((trades_frame["entry_role"] == "primary").sum()) if not trades_frame.empty else 0,
        "secondary_tranches": int((trades_frame["entry_role"] == "secondary").sum()) if not trades_frame.empty else 0,
        "secondary_share": float((trades_frame["entry_role"] == "secondary").mean()) if not trades_frame.empty else 0.0,
        "losing_second_add_share": float((secondary["active_current_return"].astype(float) < 0).mean()) if not secondary.empty else 0.0,
        "broken_second_add_share": float(
            ((secondary["active_state"].astype(str) != "HEALTHY") | secondary["active_pending_failed_reclaim"].astype(bool)).mean()
        ) if not secondary.empty else 0.0,
        "mean_secondary_r": float(secondary["risk_weight_r"].astype(float).mean()) if not secondary.empty else 0.0,
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
    return DynamicAccountSimulation(
        trades=trades_frame,
        daily_equity=daily,
        summary=summary,
        runtime_rejections=pd.DataFrame(runtime_rejections),
    )
