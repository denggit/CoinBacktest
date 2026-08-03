#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal structural hard-protection paths for R03.4.2.9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_structural_exit.config import StructuralExitConfig
from src.ai_research.long_tail_structural_exit.structure import (
    build_event_bars,
    confirmed_pivots,
    latest_pivot,
    structure_buffer,
)

from .config import ProtectionPolicy


@dataclass(frozen=True)
class ProtectionSimulation:
    trade: dict[str, object]
    updates: pd.DataFrame
    states: pd.DataFrame


def exact_position(path: MinutePathData, timestamp: pd.Timestamp) -> int | None:
    value = int(pd.Timestamp(timestamp).value)
    position = int(np.searchsorted(path.timestamps_ns, value, side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != value:
        return None
    return position


def candidate_stop_from_levels(
    *,
    policy: ProtectionPolicy,
    disaster_stop: float,
    structural_levels: list[float],
    buffer_value: float,
) -> float:
    """Return a causal stop from levels already confirmed by this bar close."""

    policy.validate()
    if policy.mode == "disaster_only":
        return float(disaster_stop)
    if policy.mode == "latest_confirmed":
        level = structural_levels[-1] if structural_levels else np.nan
    else:
        level = structural_levels[-2] if len(structural_levels) >= 2 else np.nan
    if not np.isfinite(level):
        return float(disaster_stop)
    return float(max(disaster_stop, float(level) - float(buffer_value)))


def stop_fill_price(*, open_price: float, low_price: float, stop_price: float) -> float | None:
    """Conservative long-stop fill: gap at open, otherwise at the stop."""

    if open_price <= stop_price:
        return float(open_price)
    if low_price <= stop_price:
        return float(stop_price)
    return None


def _initial_structure(
    *,
    path: MinutePathData,
    entry_position: int,
    bars,
    pivots,
    structural_config: StructuralExitConfig,
) -> tuple[float, float]:
    entry_price = float(path.open[entry_position])
    entry_bar = int(bars.entry_bar_index)
    known_low = latest_pivot(pivots, kind="low", confirmed_by=entry_bar - 1)
    floor = float(known_low.price) if known_low is not None else float(path.prior_low_180[entry_position])
    if not np.isfinite(floor) or floor >= entry_price:
        floor = float(path.prior_low_60[entry_position])
    disaster_stop = entry_price * (1.0 + structural_config.disaster_stop_return)
    if not np.isfinite(floor) or floor >= entry_price:
        floor = disaster_stop
    return float(floor), float(disaster_stop)


def _baseline_trade(row: dict[str, object], policy: ProtectionPolicy) -> dict[str, object]:
    return {
        **row,
        "protection_policy": policy.name,
        "source_exit_time": pd.Timestamp(row["exit_time"]),
        "source_exit_price": float(row["exit_price"]),
        "source_exit_reason": str(row["exit_reason"]),
        "hard_stop_triggered": False,
        "hard_stop_price_at_exit": np.nan,
        "initial_stop_price": float(row["entry_price"]) * 0.97,
        "maximum_stop_price": float(row["entry_price"]) * 0.97,
        "maximum_released_risk_fraction": 0.0,
    }


def simulate_protection_event(
    row: pd.Series | dict[str, object],
    *,
    path: MinutePathData,
    policy: ProtectionPolicy,
    structural_config: StructuralExitConfig,
) -> ProtectionSimulation:
    """Overlay one enforceable hard-stop path on the frozen failed-reclaim exit.

    Structure bars are left-anchored on the event entry. A stop learned from a
    completed structure bar becomes active only at the next one-minute open.
    The existing failed-reclaim exit remains unchanged unless the hard stop
    executes earlier.
    """

    policy.validate()
    source = dict(row)
    entry_time = pd.Timestamp(source["entry_time"])
    source_exit_time = pd.Timestamp(source["exit_time"])
    entry_position = exact_position(path, entry_time)
    source_exit_position = exact_position(path, source_exit_time)
    if entry_position is None or source_exit_position is None or source_exit_position < entry_position:
        raise RuntimeError(f"missing path for event {source.get('event_id')}")

    entry_price = float(source["entry_price"])
    disaster_stop = entry_price * (1.0 + structural_config.disaster_stop_return)
    baseline = _baseline_trade(source, policy)
    initial_update = {
        "event_id": str(source["event_id"]),
        "fold_id": str(source["fold_id"]),
        "delay_minutes": int(source["delay_minutes"]),
        "protection_policy": policy.name,
        "decision_time": pd.Timestamp(source["decision_time"]),
        "effective_time": entry_time,
        "effective_position": int(entry_position),
        "stop_price": float(disaster_stop),
        "stop_return_vs_entry": float(disaster_stop / entry_price - 1.0),
        "source": "disaster_floor",
    }

    bars = build_event_bars(
        path,
        entry_position=entry_position,
        end_position=source_exit_position,
        config=structural_config,
    )
    if bars is None or policy.mode == "disaster_only":
        return ProtectionSimulation(baseline, pd.DataFrame([initial_update]), pd.DataFrame())

    pivots = confirmed_pivots(
        bars,
        left_bars=structural_config.pivot_left_bars,
        right_bars=structural_config.pivot_right_bars,
    )
    pivot_by_confirmation: dict[int, list[object]] = {}
    for pivot in pivots:
        pivot_by_confirmation.setdefault(pivot.confirmation_index, []).append(pivot)

    initial_floor, disaster_stop = _initial_structure(
        path=path,
        entry_position=entry_position,
        bars=bars,
        pivots=pivots,
        structural_config=structural_config,
    )
    entry_bar = int(bars.entry_bar_index)
    entry_atr = float(bars.atr[entry_bar - 1]) if entry_bar > 0 else np.nan
    if not np.isfinite(entry_atr):
        entry_atr = float(path.prior_atr_60[entry_position])
    initial_buffer = structure_buffer(price=entry_price, atr=entry_atr, config=structural_config)

    structural_levels: list[float] = [float(initial_floor)]
    buffered_stop_levels: list[float] = [float(max(disaster_stop, initial_floor - initial_buffer))]
    if policy.mode == "latest_confirmed":
        current_stop = float(buffered_stop_levels[-1])
    else:
        current_stop = float(disaster_stop)
    initial_update["stop_price"] = float(current_stop)
    initial_update["stop_return_vs_entry"] = float(current_stop / entry_price - 1.0)
    initial_update["source"] = "initial_confirmed_structure" if current_stop > disaster_stop else "disaster_floor"
    updates: list[dict[str, object]] = [initial_update]
    states: list[dict[str, object]] = []

    state = "HEALTHY"
    floor = float(initial_floor)
    break_level = np.nan
    break_low = np.nan
    break_bar = -1
    lower_high_seen = False
    max_stop = float(current_stop)

    for bar_index in range(entry_bar, len(bars.close)):
        close_position = int(bars.close_position[bar_index])
        effective_position = close_position + 1
        if effective_position > source_exit_position:
            break
        close = float(bars.close[bar_index])
        low = float(bars.low[bar_index])
        atr = float(bars.atr[bar_index])
        buffer_value = structure_buffer(price=close, atr=atr, config=structural_config)
        raised_this_bar = False

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
                if state != "BROKEN" and pivot_price > floor and pivot_price < close:
                    floor = pivot_price
                    structural_levels.append(pivot_price)
                    raised_this_bar = True

        pending_failed_reclaim_exit = False
        if state != "BROKEN" and close < floor - buffer_value:
            state = "BROKEN"
            break_level = floor
            break_low = low
            break_bar = bar_index
            lower_high_seen = False
        elif state == "BROKEN":
            previous_break_low = float(break_low)
            if close > break_level + buffer_value:
                state = "HEALTHY"
                lower_high_seen = False
                break_low = np.nan
                break_bar = -1
            else:
                if lower_high_seen and close < break_level:
                    pending_failed_reclaim_exit = True
                break_low = min(previous_break_low, low)

        candidate = current_stop
        if raised_this_bar:
            buffered_stop_levels.append(float(max(disaster_stop, structural_levels[-1] - buffer_value)))
            if policy.mode == "latest_confirmed":
                candidate = buffered_stop_levels[-1]
            elif policy.mode == "lagged_confirmed" and len(buffered_stop_levels) >= 2:
                candidate = buffered_stop_levels[-2]
        if candidate > current_stop + max(entry_price * 1e-10, 1e-12):
            current_stop = float(candidate)
            max_stop = max(max_stop, current_stop)
            updates.append(
                {
                    "event_id": str(source["event_id"]),
                    "fold_id": str(source["fold_id"]),
                    "delay_minutes": int(source["delay_minutes"]),
                    "protection_policy": policy.name,
                    "decision_time": pd.Timestamp(source["decision_time"]),
                    "effective_time": pd.Timestamp(path.index[effective_position]),
                    "effective_position": int(effective_position),
                    "stop_price": float(current_stop),
                    "stop_return_vs_entry": float(current_stop / entry_price - 1.0),
                    "source": "raised_confirmed_floor",
                }
            )

        remaining = float(np.clip((entry_price - current_stop) / max(entry_price - disaster_stop, 1e-12), 0.0, 1.0))
        states.append(
            {
                "event_id": str(source["event_id"]),
                "fold_id": str(source["fold_id"]),
                "delay_minutes": int(source["delay_minutes"]),
                "protection_policy": policy.name,
                "structure_close_time": pd.Timestamp(bars.close_time_ns[bar_index], unit="ns"),
                "effective_time": pd.Timestamp(path.index[effective_position]),
                "state": state,
                "pending_failed_reclaim_exit": bool(pending_failed_reclaim_exit),
                "current_close": close,
                "current_return": float(close / entry_price - 1.0),
                "active_floor": float(floor),
                "structural_level_count": int(len(structural_levels)),
                "stop_price": float(current_stop),
                "stop_return_vs_entry": float(current_stop / entry_price - 1.0),
                "remaining_initial_risk_fraction": remaining,
                "released_risk_fraction": float(1.0 - remaining),
                "stop_at_or_above_entry": bool(current_stop >= entry_price),
            }
        )

    update_frame = pd.DataFrame(updates).sort_values(["effective_position", "stop_price"]).drop_duplicates(
        ["effective_position"], keep="last"
    )
    active_stop = float(disaster_stop)
    update_rows = update_frame.to_dict("records")
    update_index = 0
    hard_exit_position: int | None = None
    hard_exit_price: float | None = None
    hard_stop_at_exit: float | None = None
    for position in range(entry_position, source_exit_position + 1):
        while update_index < len(update_rows) and int(update_rows[update_index]["effective_position"]) <= position:
            active_stop = max(active_stop, float(update_rows[update_index]["stop_price"]))
            update_index += 1
        fill = stop_fill_price(
            open_price=float(path.open[position]),
            low_price=float(path.low[position]),
            stop_price=active_stop,
        )
        if fill is not None:
            hard_exit_position = int(position)
            hard_exit_price = float(fill)
            hard_stop_at_exit = float(active_stop)
            break

    use_hard = hard_exit_position is not None and hard_exit_position < source_exit_position
    if use_hard:
        exit_position = int(hard_exit_position)
        exit_price = float(hard_exit_price)
        exit_reason = "structural_hard_protection"
        hard_triggered = True
    else:
        exit_position = int(source_exit_position)
        exit_price = float(source["exit_price"])
        exit_reason = str(source["exit_reason"])
        hard_triggered = False

    highs = np.asarray(path.high[entry_position : exit_position + 1], dtype=float)
    lows = np.asarray(path.low[entry_position : exit_position + 1], dtype=float)
    gross_return = float(exit_price / entry_price - 1.0)
    trade = {
        **source,
        "protection_policy": policy.name,
        "source_exit_time": source_exit_time,
        "source_exit_price": float(source["exit_price"]),
        "source_exit_reason": str(source["exit_reason"]),
        "exit_time": pd.Timestamp(path.index[exit_position]),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "mfe": float(np.max(highs) / entry_price - 1.0),
        "mae": float(np.min(lows) / entry_price - 1.0),
        "holding_minutes": int(exit_position - entry_position + 1),
        "hard_stop_triggered": hard_triggered,
        "hard_stop_price_at_exit": float(hard_stop_at_exit) if hard_stop_at_exit is not None else np.nan,
        "initial_stop_price": float(update_frame.iloc[0]["stop_price"]),
        "maximum_stop_price": float(max_stop),
        "maximum_released_risk_fraction": float(
            np.clip(1.0 - (entry_price - max_stop) / max(entry_price - disaster_stop, 1e-12), 0.0, 1.0)
        ),
    }
    if not update_frame.empty:
        update_frame = update_frame.loc[pd.to_datetime(update_frame["effective_time"]) <= pd.Timestamp(trade["exit_time"])].copy()
    state_frame = pd.DataFrame(states)
    if not state_frame.empty:
        state_frame = state_frame.loc[pd.to_datetime(state_frame["effective_time"]) <= pd.Timestamp(trade["exit_time"])].copy()
    return ProtectionSimulation(trade=trade, updates=update_frame, states=state_frame)
