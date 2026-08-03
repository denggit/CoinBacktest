#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal soft-structure timeline used for partial reductions and migration."""

from __future__ import annotations

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


def exact_position(path: MinutePathData, timestamp: pd.Timestamp) -> int | None:
    value = int(pd.Timestamp(timestamp).value)
    position = int(np.searchsorted(path.timestamps_ns, value, side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != value:
        return None
    return position


def build_soft_structure_timeline(
    source: dict[str, object],
    *,
    path: MinutePathData,
    config: StructuralExitConfig,
) -> pd.DataFrame:
    """Return every causally available structural state until the frozen exit.

    This replays the frozen Failed-Reclaim state semantics but does not create
    a hard Pivot stop and does not alter the source exit. Each 15m structure
    close becomes actionable only at the next 1m open.
    """

    entry_position = exact_position(path, pd.Timestamp(source["entry_time"]))
    exit_position = exact_position(path, pd.Timestamp(source["exit_time"]))
    if entry_position is None or exit_position is None or exit_position < entry_position:
        return pd.DataFrame()

    bars = build_event_bars(
        path,
        entry_position=entry_position,
        end_position=exit_position,
        config=config,
    )
    if bars is None:
        return pd.DataFrame()

    pivots = confirmed_pivots(
        bars,
        left_bars=config.pivot_left_bars,
        right_bars=config.pivot_right_bars,
    )
    pivot_by_confirmation: dict[int, list[object]] = {}
    for pivot in pivots:
        pivot_by_confirmation.setdefault(int(pivot.confirmation_index), []).append(pivot)

    entry_price = float(source["entry_price"])
    entry_bar = int(bars.entry_bar_index)
    known_low = latest_pivot(pivots, kind="low", confirmed_by=entry_bar - 1)
    floor = float(known_low.price) if known_low is not None else float(path.prior_low_180[entry_position])
    if not np.isfinite(floor) or floor >= entry_price:
        floor = float(path.prior_low_60[entry_position])
    disaster_stop = entry_price * (1.0 + float(config.disaster_stop_return))
    if not np.isfinite(floor) or floor >= entry_price:
        floor = disaster_stop
    initial_floor = float(floor)

    state = "HEALTHY"
    break_level = np.nan
    break_low = np.nan
    break_bar = -1
    lower_high_seen = False
    structure_breaks = 0
    recoveries = 0
    last_recovery_bar = -1
    post_entry_lows: list[tuple[int, int, float]] = []
    rows: list[dict[str, object]] = []

    for bar_index in range(entry_bar, len(bars.close)):
        close_position = int(bars.close_position[bar_index])
        effective_position = close_position + 1
        if effective_position > exit_position:
            break

        previous_state = state
        close = float(bars.close[bar_index])
        low = float(bars.low[bar_index])
        atr = float(bars.atr[bar_index])
        buffer_value = structure_buffer(price=close, atr=atr, config=config)

        for pivot in pivot_by_confirmation.get(bar_index, []):
            if pivot.kind == "high":
                if (
                    state == "BROKEN"
                    and int(pivot.pivot_index) > break_bar
                    and float(pivot.price) < float(break_level) - buffer_value
                ):
                    lower_high_seen = True
            else:
                pivot_price = float(pivot.price)
                if int(pivot.pivot_index) >= entry_bar:
                    post_entry_lows.append((int(pivot.pivot_index), int(pivot.confirmation_index), pivot_price))
                if state != "BROKEN" and pivot_price > floor and pivot_price < close:
                    floor = pivot_price

        pending_failed_reclaim_exit = False
        recovered_this_bar = False
        entered_broken_this_bar = False
        if state != "BROKEN" and close < floor - buffer_value:
            state = "BROKEN"
            entered_broken_this_bar = True
            structure_breaks += 1
            break_level = floor
            break_low = low
            break_bar = bar_index
            lower_high_seen = False
        elif state == "BROKEN":
            previous_break_low = float(break_low)
            if close > break_level + buffer_value:
                state = "HEALTHY"
                recovered_this_bar = True
                recoveries += 1
                last_recovery_bar = bar_index
                lower_high_seen = False
                break_low = np.nan
                break_bar = -1
            else:
                if lower_high_seen and close < break_level:
                    pending_failed_reclaim_exit = True
                break_low = min(previous_break_low, low)

        latest_low = post_entry_lows[-1] if post_entry_lows else None
        previous_low = post_entry_lows[-2] if len(post_entry_lows) >= 2 else None
        higher_low_confirmed = bool(
            latest_low is not None
            and (
                latest_low[2] > initial_floor + buffer_value
                if previous_low is None
                else latest_low[2] > previous_low[2] + buffer_value
            )
        )
        floor_raised = bool(floor > initial_floor + buffer_value)
        proven_structure = bool(floor_raised and higher_low_confirmed)
        effective_time = pd.Timestamp(int(path.timestamps_ns[effective_position]), unit="ns")
        structure_close_time = pd.Timestamp(int(bars.close_time_ns[bar_index]), unit="ns")
        rows.append(
            {
                "event_id": str(source["event_id"]),
                "fold_id": str(source["fold_id"]),
                "delay_minutes": int(source["delay_minutes"]),
                "decision_time": pd.Timestamp(source["decision_time"]),
                "entry_time": pd.Timestamp(source["entry_time"]),
                "source_exit_time": pd.Timestamp(source["exit_time"]),
                "structure_close_time": structure_close_time,
                "effective_time": effective_time,
                "effective_position": int(effective_position),
                "previous_state": previous_state,
                "state": state,
                "entered_broken_this_bar": entered_broken_this_bar,
                "recovered_this_bar": recovered_this_bar,
                "pending_failed_reclaim_exit": bool(pending_failed_reclaim_exit),
                "current_close": close,
                "current_return": float(close / entry_price - 1.0),
                "active_floor": float(floor),
                "initial_floor": initial_floor,
                "floor_raised": floor_raised,
                "higher_low_confirmed": higher_low_confirmed,
                "proven_structure": proven_structure,
                "structure_breaks": int(structure_breaks),
                "recoveries": int(recoveries),
                "lower_high_seen": bool(lower_high_seen),
                "latest_low_after_recovery": bool(latest_low is not None and latest_low[0] > last_recovery_bar),
                "break_level": float(break_level) if np.isfinite(break_level) else np.nan,
                "break_low": float(break_low) if np.isfinite(break_low) else np.nan,
            }
        )
        if pending_failed_reclaim_exit:
            break

    return pd.DataFrame(rows)


def build_candidate_pair_snapshots(
    structural: pd.DataFrame,
    timelines: pd.DataFrame,
    *,
    delay_minutes: int,
) -> pd.DataFrame:
    """Map each overlapping q70 candidate to the active root's latest state."""

    work = structural.loc[structural["delay_minutes"].astype(int) == int(delay_minutes)].copy()
    if work.empty or timelines.empty:
        return pd.DataFrame()
    work = work.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).reset_index(drop=True)
    timeline_groups = {
        str(event_id): group.sort_values("structure_close_time").reset_index(drop=True)
        for event_id, group in timelines.groupby("event_id", sort=False)
    }
    entry_ns = pd.to_datetime(work["entry_time"]).astype("int64").to_numpy()
    rows: list[dict[str, object]] = []
    for root_index, root in work.iterrows():
        root_exit = pd.Timestamp(root["exit_time"])
        right = int(np.searchsorted(entry_ns, int(root_exit.value), side="right"))
        if right <= root_index + 1:
            continue
        timeline = timeline_groups.get(str(root["event_id"]))
        if timeline is None or timeline.empty:
            continue
        time_ns = pd.to_datetime(timeline["structure_close_time"]).astype("int64").to_numpy()
        for _, new in work.iloc[root_index + 1 : right].iterrows():
            decision_time = pd.Timestamp(new["decision_time"])
            if decision_time <= pd.Timestamp(root["decision_time"]):
                continue
            position = int(np.searchsorted(time_ns, int(decision_time.value), side="right")) - 1
            if position < 0:
                continue
            snapshot = timeline.iloc[position]
            rows.append(
                {
                    "fold_id": str(root["fold_id"]),
                    "delay_minutes": int(delay_minutes),
                    "root_event_id": str(root["event_id"]),
                    "event_id": str(new["event_id"]),
                    "root_decision_time": pd.Timestamp(root["decision_time"]),
                    "decision_time": decision_time,
                    "root_entry_time": pd.Timestamp(root["entry_time"]),
                    "new_entry_time": pd.Timestamp(new["entry_time"]),
                    "root_exit_time": root_exit,
                    "root_entry_price": float(root["entry_price"]),
                    "new_entry_price": float(new["entry_price"]),
                    "root_score": float(root["score"]),
                    "new_score": float(new["score"]),
                    "score_delta_vs_root": float(new["score"] - root["score"]),
                    "price_return_at_new_entry": float(new["entry_price"] / root["entry_price"] - 1.0),
                    "snapshot_structure_close_time": pd.Timestamp(snapshot["structure_close_time"]),
                    "snapshot_effective_time": pd.Timestamp(snapshot["effective_time"]),
                    "state": str(snapshot["state"]),
                    "pending_failed_reclaim_exit": bool(snapshot["pending_failed_reclaim_exit"]),
                    "current_return": float(snapshot["current_return"]),
                    "floor_raised": bool(snapshot["floor_raised"]),
                    "higher_low_confirmed": bool(snapshot["higher_low_confirmed"]),
                    "proven_structure": bool(snapshot["proven_structure"]),
                    "recoveries": int(snapshot["recoveries"]),
                }
            )
    return pd.DataFrame(rows)
