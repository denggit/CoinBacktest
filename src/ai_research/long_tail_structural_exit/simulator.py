#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal structural state-machine simulation without a holding-time exit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate

from .config import StructuralExitConfig, StructuralPolicy
from .structure import build_event_bars, confirmed_pivots, latest_pivot, structure_buffer


@dataclass(frozen=True)
class StructuralTrade:
    values: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.values)


def score_tier(percentile: float) -> str:
    value = float(percentile)
    if value >= 0.90:
        return "q90_plus"
    if value >= 0.80:
        return "q80_to_q90"
    return "q70_to_q80"


def _exact_position(path: MinutePathData, timestamp_ns: int) -> int | None:
    position = int(np.searchsorted(path.timestamps_ns, int(timestamp_ns), side="left"))
    if position >= len(path.timestamps_ns) or int(path.timestamps_ns[position]) != int(timestamp_ns):
        return None
    return position


def _contiguous_end(path: MinutePathData, start_position: int, maximum_position: int) -> tuple[int, str]:
    maximum_position = min(int(maximum_position), len(path.timestamps_ns) - 1)
    if maximum_position <= start_position:
        return int(start_position), "oos_end"
    expected = int(pd.Timedelta(minutes=1).value)
    diffs = np.diff(path.timestamps_ns[start_position : maximum_position + 1])
    gaps = np.flatnonzero(diffs != expected)
    if len(gaps):
        return int(start_position + gaps[0]), "data_gap"
    return maximum_position, "oos_end"


def _first_disaster_execution(
    path: MinutePathData,
    *,
    entry_position: int,
    end_position: int,
    entry_price: float,
    config: StructuralExitConfig,
) -> int | None:
    stop_price = float(entry_price) * (1.0 + float(config.disaster_stop_return))
    lows = np.asarray(path.low[entry_position : end_position + 1], dtype=float)
    breaches = np.flatnonzero(lows <= stop_price)
    if not len(breaches):
        return None
    breach_position = int(entry_position + breaches[0])
    execution = breach_position + 1
    return execution if execution <= end_position else None


def _trade_values(
    *,
    event: EventCandidate,
    fold_id: str,
    policy: str,
    policy_kind: str,
    delay_minutes: int,
    percentile: float,
    path: MinutePathData,
    entry_position: int,
    exit_position: int,
    exit_reason: str,
    is_censored: bool,
    state_transitions: int,
    structure_breaks: int,
    recoveries: int,
    profit_guard_activated: bool,
    active_floor_at_exit: float | None,
    exit_at_close: bool = False,
) -> StructuralTrade:
    entry_price = float(path.open[entry_position])
    exit_price = float(path.close[exit_position]) if (is_censored or exit_at_close) else float(path.open[exit_position])
    highs = np.asarray(path.high[entry_position : exit_position + 1], dtype=float)
    lows = np.asarray(path.low[entry_position : exit_position + 1], dtype=float)
    gross = exit_price / entry_price - 1.0
    entry_time = pd.Timestamp(int(path.timestamps_ns[entry_position]), unit="ns")
    exit_time = pd.Timestamp(int(path.timestamps_ns[exit_position]), unit="ns")
    return StructuralTrade(
        {
            "event_id": event.event_id,
            "fold_id": fold_id,
            "scope": "broad_q70",
            "policy": policy,
            "policy_kind": policy_kind,
            "decision_time": pd.Timestamp(event.decision_time_ns, unit="ns"),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "delay_minutes": int(delay_minutes),
            "signal_quantile": float(event.signal_quantile),
            "score": float(event.score),
            "score_percentile": float(percentile),
            "score_tier": score_tier(percentile),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": float(gross),
            "mfe": float(np.max(highs) / entry_price - 1.0),
            "mae": float(np.min(lows) / entry_price - 1.0),
            "holding_minutes": int(exit_position - entry_position + 1),
            "exit_reason": exit_reason,
            "is_censored": bool(is_censored),
            "state_transitions": int(state_transitions),
            "structure_breaks": int(structure_breaks),
            "recoveries": int(recoveries),
            "profit_guard_activated": bool(profit_guard_activated),
            "active_floor_at_exit": active_floor_at_exit,
            "year": int(entry_time.year),
            "quarter": str(entry_time.to_period("Q")),
            "month": str(entry_time.to_period("M")),
        }
    )


def simulate_fixed_diagnostic(
    event: EventCandidate,
    *,
    fold_id: str,
    policy: str,
    delay_minutes: int,
    percentile: float,
    path: MinutePathData,
    config: StructuralExitConfig,
    disaster_protected: bool,
) -> StructuralTrade | None:
    entry_ns = int(event.decision_time_ns + pd.Timedelta(minutes=delay_minutes).value)
    entry_position = _exact_position(path, entry_ns)
    if entry_position is None:
        return None
    exit_position = entry_position + config.diagnostic_horizon_hours * 60 - 1
    if exit_position >= len(path.timestamps_ns):
        return None
    expected_exit = entry_ns + int(pd.Timedelta(minutes=config.diagnostic_horizon_hours * 60 - 1).value)
    if int(path.timestamps_ns[exit_position]) != expected_exit:
        return None
    if disaster_protected:
        execution = _first_disaster_execution(
            path,
            entry_position=entry_position,
            end_position=exit_position,
            entry_price=float(path.open[entry_position]),
            config=config,
        )
        if execution is not None:
            exit_position = execution
            reason = "disaster_stop"
        else:
            reason = "fixed_6h_diagnostic"
    else:
        reason = "fixed_6h_diagnostic"
    return _trade_values(
        event=event,
        fold_id=fold_id,
        policy=policy,
        policy_kind="diagnostic_time_baseline",
        delay_minutes=delay_minutes,
        percentile=percentile,
        path=path,
        entry_position=entry_position,
        exit_position=exit_position,
        exit_reason=reason,
        is_censored=False,
        state_transitions=0,
        structure_breaks=0,
        recoveries=0,
        profit_guard_activated=False,
        active_floor_at_exit=None,
        exit_at_close=(reason == "fixed_6h_diagnostic"),
    )


def _latest_two_highs(pivots: list[tuple[int, float]]) -> tuple[float | None, float | None]:
    if len(pivots) < 2:
        return None, None
    return float(pivots[-2][1]), float(pivots[-1][1])


def simulate_structural_event(
    event: EventCandidate,
    *,
    fold_id: str,
    policy: StructuralPolicy,
    delay_minutes: int,
    percentile: float,
    path: MinutePathData,
    oos_end_ns: int,
    config: StructuralExitConfig,
) -> StructuralTrade | None:
    """Simulate one event until a structural exit or right-censoring boundary.

    There is deliberately no maximum holding duration. OOS-end/data-gap marks
    are reporting censoring events, not strategy exits.
    """

    entry_ns = int(event.decision_time_ns + pd.Timedelta(minutes=delay_minutes).value)
    entry_position = _exact_position(path, entry_ns)
    if entry_position is None:
        return None
    oos_position = int(np.searchsorted(path.timestamps_ns, int(oos_end_ns), side="right")) - 1
    if oos_position <= entry_position:
        return None
    end_position, censor_reason = _contiguous_end(path, entry_position, oos_position)
    if end_position - entry_position + 1 < config.diagnostic_horizon_hours * 60:
        # Keep one common q70 pool with at least the frozen six-hour baseline available.
        return None
    bars = build_event_bars(path, entry_position=entry_position, end_position=end_position, config=config)
    if bars is None:
        return None
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
    if not np.isfinite(floor) or floor >= entry_price:
        floor = entry_price * (1.0 + config.disaster_stop_return)

    entry_atr = float(bars.atr[entry_bar - 1]) if entry_bar > 0 else np.nan
    if not np.isfinite(entry_atr):
        entry_atr = float(path.prior_atr_60[entry_position])
    entry_atr_pct = entry_atr / entry_price if np.isfinite(entry_atr) and entry_price > 0 else 0.0
    profit_activation = max(
        float(config.profit_activation_return),
        float(config.profit_activation_atr_multiple) * entry_atr_pct,
    )

    disaster_execution = _first_disaster_execution(
        path,
        entry_position=entry_position,
        end_position=end_position,
        entry_price=entry_price,
        config=config,
    )

    state = "HEALTHY"
    break_level = np.nan
    break_low = np.nan
    break_bar = -1
    lower_high_seen = False
    profit_floor = np.nan
    max_high = entry_price
    state_transitions = 0
    structure_breaks = 0
    recoveries = 0
    profit_guard_activated = False
    confirmed_highs: list[tuple[int, float]] = []
    structural_exit_position: int | None = None
    structural_reason: str | None = None

    for bar_index in range(entry_bar, len(bars.close)):
        close_position = int(bars.close_position[bar_index])
        execution_position = close_position + 1
        if execution_position > end_position:
            break
        close = float(bars.close[bar_index])
        high = float(bars.high[bar_index])
        low = float(bars.low[bar_index])
        max_high = max(max_high, high)
        current_return = close / entry_price - 1.0
        mfe = max_high / entry_price - 1.0
        atr = float(bars.atr[bar_index])
        buffer_value = structure_buffer(price=close, atr=atr, config=config)

        for pivot in pivot_by_confirmation.get(bar_index, []):
            if pivot.kind == "high":
                if pivot.pivot_index >= entry_bar:
                    confirmed_highs.append((pivot.pivot_index, float(pivot.price)))
                if (
                    state == "BROKEN"
                    and pivot.pivot_index > break_bar
                    and float(pivot.price) < float(break_level) - buffer_value
                ):
                    lower_high_seen = True
            else:
                pivot_price = float(pivot.price)
                if pivot.pivot_index >= entry_bar and pivot_price > entry_price:
                    profit_floor = max(profit_floor, pivot_price) if np.isfinite(profit_floor) else pivot_price
                if state != "BROKEN" and pivot_price > floor and pivot_price < close:
                    floor = pivot_price

        if not profit_guard_activated and mfe >= profit_activation:
            profit_guard_activated = True
            state_transitions += 1

        if state != "BROKEN" and close < floor - buffer_value:
            state = "BROKEN"
            state_transitions += 1
            structure_breaks += 1
            break_level = floor
            break_low = low
            break_bar = bar_index
            lower_high_seen = False
        elif state == "BROKEN":
            previous_break_low = float(break_low)
            if close > break_level + buffer_value:
                state = "HEALTHY"
                state_transitions += 1
                recoveries += 1
                lower_high_seen = False
                break_low = np.nan
                break_bar = -1
            else:
                if policy.exit_on_failed_reclaim and lower_high_seen and close < break_level:
                    structural_exit_position = execution_position
                    structural_reason = "failed_reclaim_below_structure"
                    break
                if lower_high_seen and low < previous_break_low - buffer_value:
                    structural_exit_position = execution_position
                    structural_reason = "confirmed_lower_high_lower_low"
                    break
                break_low = min(previous_break_low, low)

        if policy.enable_profit_guard and profit_guard_activated:
            giveback = mfe - current_return
            giveback_gate = max(
                float(config.minimum_peak_giveback_return),
                float(config.peak_giveback_fraction) * mfe,
            )
            previous_high, latest_high = _latest_two_highs(confirmed_highs)
            lower_high = (
                previous_high is not None
                and latest_high is not None
                and latest_high < previous_high - buffer_value
            )
            start = max(entry_bar, bar_index - config.profit_guard_declining_bars + 1)
            recent_closes = bars.close[start : bar_index + 1]
            declining = (
                len(recent_closes) >= config.profit_guard_declining_bars
                and bool(np.all(np.diff(recent_closes) < 0))
            )
            broke_profit_floor = np.isfinite(profit_floor) and close < profit_floor - buffer_value
            if giveback >= giveback_gate and (broke_profit_floor or (lower_high and declining)):
                structural_exit_position = execution_position
                structural_reason = "profit_giveback_structure_break"
                break

    candidates = [
        (position, reason)
        for position, reason in (
            (disaster_execution, "disaster_stop"),
            (structural_exit_position, structural_reason),
        )
        if position is not None and reason is not None
    ]
    if candidates:
        exit_position, exit_reason = min(candidates, key=lambda item: int(item[0]))
        is_censored = False
    else:
        exit_position = end_position
        exit_reason = f"censored_{censor_reason}_mark_to_market"
        is_censored = True

    return _trade_values(
        event=event,
        fold_id=fold_id,
        policy=policy.name,
        policy_kind="non_time_structural_candidate",
        delay_minutes=delay_minutes,
        percentile=percentile,
        path=path,
        entry_position=entry_position,
        exit_position=int(exit_position),
        exit_reason=str(exit_reason),
        is_censored=is_censored,
        state_transitions=state_transitions,
        structure_breaks=structure_breaks,
        recoveries=recoveries,
        profit_guard_activated=profit_guard_activated,
        active_floor_at_exit=float(floor) if np.isfinite(floor) else None,
    )


def enforce_non_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    work = frame.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).copy()
    keep: list[int] = []
    last_exit: pd.Timestamp | None = None
    skipped = 0
    for index, row in work.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        if last_exit is not None and entry <= last_exit:
            skipped += 1
            continue
        keep.append(index)
        last_exit = pd.Timestamp(row["exit_time"])
    return work.loc[keep].sort_values("entry_time").reset_index(drop=True), int(skipped)
