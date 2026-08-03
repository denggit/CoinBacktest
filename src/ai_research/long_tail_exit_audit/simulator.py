#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal event construction and one-minute exit simulation for R03.4.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.modeling import select_episode_peaks

from .config import ExitRecipe, LongTailExitAuditConfig
from .data import MinutePathData


@dataclass(frozen=True)
class ScoreTimeline:
    decision_times_ns: np.ndarray
    scores: np.ndarray
    calibration_thresholds: dict[float, float]

    def threshold(self, quantile: float) -> float:
        try:
            return float(self.calibration_thresholds[float(quantile)])
        except KeyError as exc:
            raise KeyError(f"missing score threshold q={quantile}") from exc

    def latest_score_at_or_before(self, timestamp_ns: int) -> float:
        position = int(np.searchsorted(self.decision_times_ns, int(timestamp_ns), side="right")) - 1
        if position < 0:
            return np.nan
        return float(self.scores[position])


@dataclass(frozen=True)
class EventCandidate:
    event_id: str
    decision_time_ns: int
    score: float
    signal_quantile: float


@dataclass(frozen=True)
class StructuralStop:
    stop_price: float
    risk_pct: float
    raw_structure_price: float
    buffer_value: float


@dataclass(frozen=True)
class SimulatedTrade:
    event_id: str
    decision_time_ns: int
    entry_time_ns: int
    exit_time_ns: int
    signal_quantile: float
    score: float
    recipe: str
    delay_minutes: int
    entry_price: float
    exit_price: float
    stop_price: float | None
    initial_risk_pct: float | None
    gross_return: float
    mfe: float
    mae: float
    realized_r: float | None
    holding_minutes: int
    exit_reason: str
    renewal_count: int
    maximum_score_after_entry: float
    minimum_score_after_entry: float

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "decision_time": pd.Timestamp(self.decision_time_ns, unit="ns"),
            "entry_time": pd.Timestamp(self.entry_time_ns, unit="ns"),
            "exit_time": pd.Timestamp(self.exit_time_ns, unit="ns"),
            "signal_quantile": self.signal_quantile,
            "score": self.score,
            "recipe": self.recipe,
            "delay_minutes": self.delay_minutes,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_price": self.stop_price,
            "initial_risk_pct": self.initial_risk_pct,
            "gross_return": self.gross_return,
            "mfe": self.mfe,
            "mae": self.mae,
            "realized_r": self.realized_r,
            "holding_minutes": self.holding_minutes,
            "exit_reason": self.exit_reason,
            "renewal_count": self.renewal_count,
            "maximum_score_after_entry": self.maximum_score_after_entry,
            "minimum_score_after_entry": self.minimum_score_after_entry,
        }


def build_event_candidates(
    timeline: ScoreTimeline,
    *,
    signal_quantile: float,
    config: LongTailExitAuditConfig,
) -> tuple[EventCandidate, ...]:
    threshold = timeline.threshold(signal_quantile)
    valid = np.isfinite(timeline.scores)
    signal = valid & (timeline.scores >= threshold)
    positions = select_episode_peaks(
        timeline.decision_times_ns,
        timeline.scores,
        signal,
        merge_gap_minutes=config.episode_merge_gap_minutes,
        cooldown_hours=config.independent_event_cooldown_hours,
    )
    return tuple(
        EventCandidate(
            event_id=f"q{int(signal_quantile * 100)}_{int(timeline.decision_times_ns[pos])}",
            decision_time_ns=int(timeline.decision_times_ns[pos]),
            score=float(timeline.scores[pos]),
            signal_quantile=float(signal_quantile),
        )
        for pos in positions
    )


def structural_stop_at(
    path: MinutePathData,
    entry_position: int,
    entry_price: float,
    lookback_minutes: int,
    config: LongTailExitAuditConfig,
) -> StructuralStop | None:
    if lookback_minutes == 60:
        structure = float(path.prior_low_60[entry_position])
    elif lookback_minutes == 180:
        structure = float(path.prior_low_180[entry_position])
    else:
        raise ValueError(f"unsupported structural lookback: {lookback_minutes}")
    atr = float(path.prior_atr_60[entry_position])
    if not np.isfinite(structure) or not np.isfinite(atr) or entry_price <= 0:
        return None
    buffer_value = max(entry_price * config.structural_buffer_bps / 10_000.0, atr * config.atr_buffer_multiple)
    raw_stop = structure - buffer_value
    minimum_risk, maximum_risk = config.stop_bounds(lookback_minutes)
    minimum_stop = entry_price * (1.0 - minimum_risk)
    stop_price = min(raw_stop, minimum_stop)
    risk_pct = (entry_price - stop_price) / entry_price
    if not np.isfinite(risk_pct) or risk_pct <= 0 or risk_pct > maximum_risk:
        return None
    return StructuralStop(
        stop_price=float(stop_price),
        risk_pct=float(risk_pct),
        raw_structure_price=float(structure),
        buffer_value=float(buffer_value),
    )


def _scheduled_model_exit_ns(
    *,
    event: EventCandidate,
    entry_time_ns: int,
    recipe: ExitRecipe,
    timeline: ScoreTimeline,
) -> tuple[int | None, str | None, int, float, float]:
    """Return earliest causal model exit, reason, renewals, max score and min score.

    A decision at time ``t`` can only execute at the open of minute ``t+1``.
    Checkpoints are anchored to the original decision time, not to future price.
    """

    if recipe.renewal_quantile is None and recipe.invalidation_quantile is None:
        return None, None, 0, np.nan, np.nan
    times = np.asarray(timeline.decision_times_ns, dtype=np.int64)
    scores = np.asarray(timeline.scores, dtype=float)
    start = int(np.searchsorted(times, event.decision_time_ns, side="right"))
    safety_end = event.decision_time_ns + int(pd.Timedelta(hours=recipe.safety_cap_hours).value)
    stop = int(np.searchsorted(times, safety_end, side="right"))
    renewal_threshold = (
        timeline.threshold(recipe.renewal_quantile) if recipe.renewal_quantile is not None else None
    )
    invalidation_threshold = (
        timeline.threshold(recipe.invalidation_quantile) if recipe.invalidation_quantile is not None else None
    )
    checkpoint_step = int(pd.Timedelta(hours=recipe.checkpoint_hours).value)
    next_checkpoint = event.decision_time_ns + checkpoint_step
    minimum_invalidation_time = entry_time_ns + int(
        pd.Timedelta(minutes=recipe.minimum_invalidation_hold_minutes).value
    )
    invalid_count = 0
    renewals = 0
    observed: list[float] = []
    one_minute = int(pd.Timedelta(minutes=1).value)

    for position in range(start, stop):
        decision_ns = int(times[position])
        score = float(scores[position])
        if not np.isfinite(score):
            continue
        observed.append(score)
        if invalidation_threshold is not None and decision_ns >= minimum_invalidation_time:
            invalid_count = invalid_count + 1 if score < invalidation_threshold else 0
            if invalid_count >= recipe.invalidation_confirmations:
                return (
                    decision_ns + one_minute,
                    "model_invalidation",
                    renewals,
                    float(np.max(observed)),
                    float(np.min(observed)),
                )
        while decision_ns >= next_checkpoint:
            if renewal_threshold is not None:
                if score < renewal_threshold:
                    return (
                        decision_ns + one_minute,
                        "model_not_renewed",
                        renewals,
                        float(np.max(observed)),
                        float(np.min(observed)),
                    )
                renewals += 1
            next_checkpoint += checkpoint_step
    if observed:
        return None, None, renewals, float(np.max(observed)), float(np.min(observed))
    return None, None, renewals, np.nan, np.nan


def simulate_event(
    *,
    event: EventCandidate,
    recipe: ExitRecipe,
    delay_minutes: int,
    path: MinutePathData,
    timeline: ScoreTimeline,
    config: LongTailExitAuditConfig,
) -> SimulatedTrade | None:
    entry_time_ns = event.decision_time_ns + int(pd.Timedelta(minutes=delay_minutes).value)
    entry_position = int(np.searchsorted(path.timestamps_ns, entry_time_ns, side="left"))
    if entry_position >= len(path.timestamps_ns) or int(path.timestamps_ns[entry_position]) != entry_time_ns:
        return None
    entry_price = float(path.open[entry_position])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    if recipe.is_time_baseline:
        exit_position = entry_position + config.primary_horizon_hours * 60 - 1
        if exit_position >= len(path.timestamps_ns):
            return None
        exit_price = float(path.close[exit_position])
        highs = path.high[entry_position : exit_position + 1]
        lows = path.low[entry_position : exit_position + 1]
        gross = exit_price / entry_price - 1.0
        return SimulatedTrade(
            event_id=event.event_id,
            decision_time_ns=event.decision_time_ns,
            entry_time_ns=entry_time_ns,
            exit_time_ns=int(path.timestamps_ns[exit_position]),
            signal_quantile=event.signal_quantile,
            score=event.score,
            recipe=recipe.name,
            delay_minutes=delay_minutes,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=None,
            initial_risk_pct=None,
            gross_return=float(gross),
            mfe=float(np.max(highs) / entry_price - 1.0),
            mae=float(1.0 - np.min(lows) / entry_price),
            realized_r=None,
            holding_minutes=int(exit_position - entry_position + 1),
            exit_reason="fixed_6h_close",
            renewal_count=0,
            maximum_score_after_entry=np.nan,
            minimum_score_after_entry=np.nan,
        )

    stop = structural_stop_at(path, entry_position, entry_price, recipe.stop_lookback_minutes, config)
    if stop is None:
        return None
    risk_value = entry_price - stop.stop_price
    target_price = entry_price + recipe.take_profit_r * risk_value if recipe.take_profit_r is not None else None
    model_exit_ns, model_exit_reason, renewals, maximum_score, minimum_score = _scheduled_model_exit_ns(
        event=event,
        entry_time_ns=entry_time_ns,
        recipe=recipe,
        timeline=timeline,
    )
    safety_end_ns = entry_time_ns + int(pd.Timedelta(hours=recipe.safety_cap_hours).value)
    safety_position = int(np.searchsorted(path.timestamps_ns, safety_end_ns, side="left"))
    if safety_position >= len(path.timestamps_ns):
        return None
    final_position = safety_position
    peak_price = entry_price
    trailing_stop: float | None = None
    path_high = entry_price
    path_low = entry_price
    exit_position: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    for position in range(entry_position, final_position + 1):
        timestamp_ns = int(path.timestamps_ns[position])
        open_price = float(path.open[position])
        high_price = float(path.high[position])
        low_price = float(path.low[position])
        close_price = float(path.close[position])

        if model_exit_ns is not None and timestamp_ns >= model_exit_ns:
            exit_position = position
            exit_price = open_price
            exit_reason = str(model_exit_reason)
            break
        if timestamp_ns >= safety_end_ns:
            exit_position = position
            exit_price = open_price
            exit_reason = "safety_time_cap"
            break

        effective_stop = stop.stop_price if trailing_stop is None else max(stop.stop_price, trailing_stop)
        stop_hit = low_price <= effective_stop
        target_hit = target_price is not None and high_price >= target_price
        # Conservative same-minute ordering: if both are touched, the stop wins.
        if stop_hit:
            exit_position = position
            exit_price = min(open_price, effective_stop) if open_price < effective_stop else effective_stop
            exit_reason = "trailing_stop" if trailing_stop is not None and effective_stop > stop.stop_price else "hard_stop"
            path_high = max(path_high, high_price)
            path_low = min(path_low, low_price)
            break
        if target_hit:
            exit_position = position
            exit_price = float(target_price)
            exit_reason = "take_profit"
            path_high = max(path_high, high_price)
            path_low = min(path_low, low_price)
            break

        path_high = max(path_high, high_price)
        path_low = min(path_low, low_price)
        peak_price = max(peak_price, high_price)
        # A trail activated by this minute's high is usable only from the next minute.
        if recipe.trail_activation_r is not None and peak_price >= entry_price + recipe.trail_activation_r * risk_value:
            candidate = peak_price - float(recipe.trail_giveback_r) * risk_value
            trailing_stop = candidate if trailing_stop is None else max(trailing_stop, candidate)
        _ = close_price

    if exit_position is None or exit_price is None or exit_reason is None:
        return None
    gross = exit_price / entry_price - 1.0
    mfe = path_high / entry_price - 1.0
    mae = 1.0 - path_low / entry_price
    return SimulatedTrade(
        event_id=event.event_id,
        decision_time_ns=event.decision_time_ns,
        entry_time_ns=entry_time_ns,
        exit_time_ns=int(path.timestamps_ns[exit_position]),
        signal_quantile=event.signal_quantile,
        score=event.score,
        recipe=recipe.name,
        delay_minutes=delay_minutes,
        entry_price=entry_price,
        exit_price=float(exit_price),
        stop_price=stop.stop_price,
        initial_risk_pct=stop.risk_pct,
        gross_return=float(gross),
        mfe=float(mfe),
        mae=float(mae),
        realized_r=float(gross / stop.risk_pct),
        holding_minutes=int(exit_position - entry_position),
        exit_reason=exit_reason,
        renewal_count=renewals,
        maximum_score_after_entry=maximum_score,
        minimum_score_after_entry=minimum_score,
    )


def simulate_sequential_events(
    *,
    events: tuple[EventCandidate, ...],
    recipe: ExitRecipe,
    delay_minutes: int,
    path: MinutePathData,
    timeline: ScoreTimeline,
    config: LongTailExitAuditConfig,
) -> tuple[list[SimulatedTrade], dict[str, int]]:
    trades: list[SimulatedTrade] = []
    skipped_overlap = 0
    skipped_invalid = 0
    last_exit_ns = -1
    for event in events:
        entry_ns = event.decision_time_ns + int(pd.Timedelta(minutes=delay_minutes).value)
        if entry_ns <= last_exit_ns:
            skipped_overlap += 1
            continue
        trade = simulate_event(
            event=event,
            recipe=recipe,
            delay_minutes=delay_minutes,
            path=path,
            timeline=timeline,
            config=config,
        )
        if trade is None:
            skipped_invalid += 1
            continue
        trades.append(trade)
        last_exit_ns = trade.exit_time_ns
    return trades, {
        "candidate_events": len(events),
        "executed_trades": len(trades),
        "skipped_overlap": skipped_overlap,
        "skipped_invalid_or_missing": skipped_invalid,
    }
