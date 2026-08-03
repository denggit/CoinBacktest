#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal occupancy mapping and failed-reclaim state snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_structural_exit.config import StructuralExitConfig
from src.ai_research.long_tail_structural_exit.structure import (
    build_event_bars,
    confirmed_pivots,
    latest_pivot,
    structure_buffer,
)

from .config import TrancheEligibilityConfig


@dataclass(frozen=True)
class OccupancyResult:
    executed: pd.DataFrame
    occupied: pd.DataFrame


def build_occupancy_map(raw_failed_reclaim: pd.DataFrame) -> OccupancyResult:
    """Apply the frozen one-position overlap rule and retain root ownership."""

    if raw_failed_reclaim.empty:
        return OccupancyResult(pd.DataFrame(), pd.DataFrame())
    work = raw_failed_reclaim.sort_values(
        ["entry_time", "decision_time", "score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    executed_rows: list[dict[str, object]] = []
    occupied_rows: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    for row in work.to_dict("records"):
        entry_time = pd.Timestamp(row["entry_time"])
        if active is not None and entry_time <= pd.Timestamp(active["exit_time"]):
            occupied_rows.append(
                {
                    "event_id": row["event_id"],
                    "root_event_id": active["event_id"],
                    "entry_time": entry_time,
                    "decision_time": pd.Timestamp(row["decision_time"]),
                    "root_entry_time": pd.Timestamp(active["entry_time"]),
                    "root_exit_time": pd.Timestamp(active["exit_time"]),
                    "root_exit_reason": active["exit_reason"],
                }
            )
            continue
        active = row
        executed_rows.append(row)
    return OccupancyResult(pd.DataFrame(executed_rows), pd.DataFrame(occupied_rows))


def _exact_position(path: MinutePathData, timestamp_ns: int) -> int | None:
    position = int(np.searchsorted(path.timestamps_ns, int(timestamp_ns), side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != int(timestamp_ns):
        return None
    return position


def failed_reclaim_snapshots(
    root_event: EventCandidate,
    *,
    delay_minutes: int,
    observation_times_ns: tuple[int, ...],
    path: MinutePathData,
    end_time_ns: int,
    config: StructuralExitConfig,
) -> dict[int, dict[str, object]]:
    """Replay the frozen state machine and snapshot only completed structure bars.

    The function is diagnostic. ``candidate_hard_stop_price`` is the causal
    level that a later tranche-risk study could choose to enforce. It is not
    executed here and therefore no risk-release claim is made in this stage.
    """

    if not observation_times_ns:
        return {}
    entry_ns = int(root_event.decision_time_ns + pd.Timedelta(minutes=delay_minutes).value)
    entry_position = _exact_position(path, entry_ns)
    end_position = _exact_position(path, int(end_time_ns))
    if entry_position is None or end_position is None or end_position <= entry_position:
        return {}
    bars = build_event_bars(path, entry_position=entry_position, end_position=end_position, config=config)
    if bars is None:
        return {}
    pivots = confirmed_pivots(
        bars,
        left_bars=config.pivot_left_bars,
        right_bars=config.pivot_right_bars,
    )
    pivot_by_confirmation: dict[int, list[object]] = {}
    for pivot in pivots:
        pivot_by_confirmation.setdefault(pivot.confirmation_index, []).append(pivot)

    entry_price = float(path.open[entry_position])
    entry_bar = int(bars.entry_bar_index)
    known_low = latest_pivot(pivots, kind="low", confirmed_by=entry_bar - 1)
    floor = float(known_low.price) if known_low is not None else float(path.prior_low_180[entry_position])
    if not np.isfinite(floor) or floor >= entry_price:
        floor = float(path.prior_low_60[entry_position])
    disaster_stop = entry_price * (1.0 + config.disaster_stop_return)
    if not np.isfinite(floor) or floor >= entry_price:
        floor = disaster_stop
    initial_floor = float(floor)
    initial_risk_distance = max(entry_price - disaster_stop, np.finfo(float).eps)

    state = "HEALTHY"
    break_level = np.nan
    break_low = np.nan
    break_bar = -1
    lower_high_seen = False
    structure_breaks = 0
    recoveries = 0
    last_recovery_bar = -1
    max_high = entry_price
    min_low = entry_price
    post_entry_lows: list[tuple[int, int, float]] = []
    requested = sorted(set(int(value) for value in observation_times_ns))
    snapshots: dict[int, dict[str, object]] = {}
    request_position = 0
    last_payload: dict[str, object] | None = None

    for bar_index in range(entry_bar, len(bars.close)):
        close_position = int(bars.close_position[bar_index])
        execution_position = close_position + 1
        if execution_position > end_position:
            break
        close_time_ns = int(bars.close_time_ns[bar_index])
        # For off-boundary observations (notably 3m/5m delay stress), use the
        # latest structure bar that was already complete. Never use this bar
        # before its own close timestamp.
        while request_position < len(requested) and requested[request_position] < close_time_ns:
            observation_ns = requested[request_position]
            if last_payload is not None:
                snapshots[observation_ns] = {
                    **last_payload,
                    "observation_time": pd.Timestamp(observation_ns, unit="ns"),
                }
            request_position += 1

        close = float(bars.close[bar_index])
        high = float(bars.high[bar_index])
        low = float(bars.low[bar_index])
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        atr = float(bars.atr[bar_index])
        buffer_value = structure_buffer(price=close, atr=atr, config=config)

        for pivot in pivot_by_confirmation.get(bar_index, []):
            if pivot.kind == "high":
                if (
                    state == "BROKEN"
                    and pivot.pivot_index > break_bar
                    and float(pivot.price) < float(break_level) - buffer_value
                ):
                    lower_high_seen = True
            else:
                pivot_price = float(pivot.price)
                if pivot.pivot_index >= entry_bar:
                    post_entry_lows.append((pivot.pivot_index, pivot.confirmation_index, pivot_price))
                if state != "BROKEN" and pivot_price > floor and pivot_price < close:
                    floor = pivot_price

        pending_failed_reclaim_exit = False
        if state != "BROKEN" and close < floor - buffer_value:
            state = "BROKEN"
            structure_breaks += 1
            break_level = floor
            break_low = low
            break_bar = bar_index
            lower_high_seen = False
        elif state == "BROKEN":
            previous_break_low = float(break_low)
            if close > break_level + buffer_value:
                state = "HEALTHY"
                recoveries += 1
                last_recovery_bar = bar_index
                lower_high_seen = False
                break_low = np.nan
                break_bar = -1
            else:
                if lower_high_seen and close < break_level:
                    pending_failed_reclaim_exit = True
                break_low = min(previous_break_low, low)

        candidate_stop = max(disaster_stop, float(floor) - buffer_value)
        remaining_loss_r = float(np.clip((entry_price - candidate_stop) / initial_risk_distance, 0.0, 1.0))
        released_risk = 1.0 - remaining_loss_r
        latest_low = post_entry_lows[-1] if post_entry_lows else None
        previous_low = post_entry_lows[-2] if len(post_entry_lows) >= 2 else None
        latest_low_confirmation_ns = (
            int(bars.close_time_ns[latest_low[1]]) if latest_low is not None else None
        )
        structure_age_minutes = (
            float((close_time_ns - latest_low_confirmation_ns) / pd.Timedelta(minutes=1).value)
            if latest_low_confirmation_ns is not None
            else np.nan
        )
        latest_low_after_recovery = bool(latest_low is not None and latest_low[0] > last_recovery_bar)
        higher_low_confirmed = bool(
            latest_low is not None
            and (
                latest_low[2] > initial_floor + buffer_value
                if previous_low is None
                else latest_low[2] > previous_low[2] + buffer_value
            )
        )

        current_payload = {
            "state": state,
            "pending_failed_reclaim_exit": bool(pending_failed_reclaim_exit),
            "current_close": close,
            "current_return_vs_root": float(close / entry_price - 1.0),
            "mfe_to_signal": float(max_high / entry_price - 1.0),
            "mae_to_signal": float(min_low / entry_price - 1.0),
            "active_floor": float(floor),
            "initial_floor": initial_floor,
            "floor_raised": bool(floor > initial_floor + buffer_value),
            "break_level": float(break_level) if np.isfinite(break_level) else np.nan,
            "break_low": float(break_low) if np.isfinite(break_low) else np.nan,
            "structure_breaks": int(structure_breaks),
            "recoveries": int(recoveries),
            "lower_high_seen": bool(lower_high_seen),
            "latest_post_entry_low": float(latest_low[2]) if latest_low is not None else np.nan,
            "previous_post_entry_low": float(previous_low[2]) if previous_low is not None else np.nan,
            "latest_structure_confirmation_time": (
                pd.Timestamp(latest_low_confirmation_ns, unit="ns")
                if latest_low_confirmation_ns is not None
                else pd.NaT
            ),
            "structure_age_minutes": structure_age_minutes,
            "higher_low_confirmed": higher_low_confirmed,
            "latest_low_after_recovery": latest_low_after_recovery,
            "independent_structure_confirmed": bool(latest_low is not None),
            "candidate_hard_stop_price": float(candidate_stop),
            "candidate_hard_stop_return": float(candidate_stop / entry_price - 1.0),
            "remaining_initial_loss_r": remaining_loss_r,
            "released_risk_fraction": float(released_risk),
        }
        last_payload = current_payload
        if request_position < len(requested) and requested[request_position] == close_time_ns:
            observation_ns = requested[request_position]
            snapshots[observation_ns] = {
                **current_payload,
                "observation_time": pd.Timestamp(observation_ns, unit="ns"),
            }
            request_position += 1
        if request_position >= len(requested) or pending_failed_reclaim_exit:
            break

    while request_position < len(requested) and last_payload is not None:
        observation_ns = requested[request_position]
        if observation_ns > int(end_time_ns):
            break
        snapshots[observation_ns] = {
            **last_payload,
            "observation_time": pd.Timestamp(observation_ns, unit="ns"),
        }
        request_position += 1
    return snapshots


def classify_occupied_signal(
    row: dict[str, object],
    *,
    config: TrancheEligibilityConfig,
) -> tuple[str, str, bool]:
    """Classify from information available at the new signal decision time."""

    state = str(row.get("state", "UNKNOWN"))
    current_return = float(row.get("current_return_vs_root", np.nan))
    score_delta = float(row.get("score_delta_vs_root", np.nan))
    released = float(row.get("released_risk_fraction", np.nan))
    pending_exit = bool(row.get("pending_failed_reclaim_exit", False))
    independent_structure = bool(row.get("independent_structure_confirmed", False))
    structure_age = float(row.get("structure_age_minutes", np.nan))
    fresh_structure = bool(
        independent_structure
        and np.isfinite(structure_age)
        and 0.0 <= structure_age <= config.maximum_structure_age_minutes
    )
    floor_raised = bool(row.get("floor_raised", False))
    higher_low = bool(row.get("higher_low_confirmed", False))
    recoveries = int(row.get("recoveries", 0))
    low_after_recovery = bool(row.get("latest_low_after_recovery", False))

    if pending_exit or state == "BROKEN":
        return "dangerous_average_down", "failed_reclaim_process_active", False
    if current_return < 0 and score_delta > 0 and not floor_raised:
        return "dangerous_average_down", "score_up_price_down_without_protection", False
    if current_return < 0 and (not np.isfinite(released) or released < config.minimum_released_risk_fraction):
        return "dangerous_average_down", "losing_position_without_meaningful_risk_release", False

    risk_released = bool(np.isfinite(released) and released >= config.minimum_released_risk_fraction)
    if (
        state == "HEALTHY"
        and current_return > 0
        and fresh_structure
        and floor_raised
        and higher_low
        and risk_released
        and recoveries == 0
    ):
        return "healthy_trend", "profitable_higher_low_with_candidate_risk_release", True
    if (
        state == "HEALTHY"
        and recoveries > 0
        and fresh_structure
        and floor_raised
        and low_after_recovery
        and risk_released
    ):
        return "recovered_structure", "reclaim_completed_then_new_confirmed_low", True
    return "ambiguous_no_add", "insufficient_causal_structure_or_risk_release", False
