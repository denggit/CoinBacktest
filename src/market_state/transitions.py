#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal hysteresis, state stabilization and segment construction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.market_state.models import MarketStateSegment


def stabilize_labels(raw_labels: Iterable[str], min_state_bars: int) -> tuple[list[str], list[int]]:
    """Backward-compatible V0 debounce helper."""

    stable, ages, _, _ = stabilize_labels_detailed(
        raw_labels,
        confirm_bars=min_state_bars,
        min_duration_bars=1,
    )
    return stable, ages


def stabilize_labels_detailed(
    raw_labels: Iterable[str],
    *,
    confirm_bars: int,
    min_duration_bars: int,
    immediate_labels: set[str] | None = None,
) -> tuple[list[str], list[int], list[str], list[float]]:
    """Debounce categorical labels without looking ahead.

    A new candidate must persist for ``confirm_bars`` and the current state must
    have lived for ``min_duration_bars``.  Labels in ``immediate_labels`` can
    switch immediately after the first valid observation (used for shocks).
    """

    confirm = max(1, int(confirm_bars))
    minimum_age = max(1, int(min_duration_bars))
    immediate = immediate_labels or set()
    stable: list[str] = []
    ages: list[int] = []
    candidates: list[str] = []
    progresses: list[float] = []
    current = "warmup"
    candidate = "warmup"
    candidate_count = 0
    age = 0

    for raw in raw_labels:
        value = str(raw or "warmup")
        if value == "warmup":
            current = "warmup"
            candidate = "warmup"
            candidate_count = 0
            age = 0
        elif current == "warmup":
            current = value
            candidate = value
            candidate_count = 0
            age = 1
        elif value == current:
            candidate = current
            candidate_count = 0
            age += 1
        else:
            if value == candidate:
                candidate_count += 1
            else:
                candidate = value
                candidate_count = 1
            can_switch = candidate in immediate or (
                candidate_count >= confirm and age >= minimum_age
            )
            if can_switch:
                current = candidate
                age = 1
                candidate_count = 0
            else:
                age += 1
        stable.append(current)
        ages.append(age)
        candidates.append(candidate if candidate_count > 0 else current)
        progresses.append(0.0 if candidate_count <= 0 else min(1.0, candidate_count / confirm))
    return stable, ages, candidates, progresses


def stabilize_direction_scores(
    scores: Iterable[float],
    ready: Iterable[bool],
    *,
    enter_threshold: float,
    exit_threshold: float,
    confirm_bars: int,
    min_duration_bars: int,
) -> tuple[list[str], list[str], list[int], list[str], list[float]]:
    """Three-state directional hysteresis for up / balanced / down.

    Entering a directional state requires the wider enter threshold.  Exiting
    only requires crossing the narrower exit threshold.  Balanced can enter a
    direction after confirmation immediately; established directional states
    also respect the minimum duration before switching.
    """

    enter = float(enter_threshold)
    exit_ = float(exit_threshold)
    confirm = max(1, int(confirm_bars))
    minimum_age = max(1, int(min_duration_bars))

    raw_states: list[str] = []
    stable: list[str] = []
    ages: list[int] = []
    candidates: list[str] = []
    progresses: list[float] = []
    current = "warmup"
    candidate = "warmup"
    candidate_count = 0
    age = 0

    for score_value, is_ready in zip(scores, ready):
        score = float(score_value) if score_value is not None else np.nan
        if not bool(is_ready) or not np.isfinite(score):
            raw = "warmup"
        elif score >= enter:
            raw = "up"
        elif score <= -enter:
            raw = "down"
        else:
            raw = "balanced"
        raw_states.append(raw)

        if raw == "warmup":
            current = "warmup"
            candidate = "warmup"
            candidate_count = 0
            age = 0
        else:
            if current == "warmup":
                current = "balanced"
                age = 0

            if current == "balanced":
                if score >= enter:
                    target = "up"
                elif score <= -enter:
                    target = "down"
                else:
                    target = "balanced"
                required_age = 1
            elif current == "up":
                if score <= -enter:
                    target = "down"
                elif score <= exit_:
                    target = "balanced"
                else:
                    target = "up"
                required_age = minimum_age
            else:  # down
                if score >= enter:
                    target = "up"
                elif score >= -exit_:
                    target = "balanced"
                else:
                    target = "down"
                required_age = minimum_age

            if target == current:
                candidate = current
                candidate_count = 0
                age += 1
            else:
                if target == candidate:
                    candidate_count += 1
                else:
                    candidate = target
                    candidate_count = 1
                if candidate_count >= confirm and age + 1 >= required_age:
                    current = candidate
                    candidate_count = 0
                    age = 1
                else:
                    age += 1

        stable.append(current)
        ages.append(age)
        candidates.append(candidate if candidate_count > 0 else current)
        progresses.append(0.0 if candidate_count <= 0 else min(1.0, candidate_count / confirm))

    return raw_states, stable, ages, candidates, progresses



def stabilize_volatility_scores(
    volatility_z: Iterable[float],
    activity_z: Iterable[float],
    return_z: Iterable[float],
    ready: Iterable[bool],
    *,
    quiet_enter: float,
    quiet_exit: float,
    expand_enter: float,
    expand_exit: float,
    shock_enter: float,
    shock_exit: float,
    confirm_bars: int,
    min_duration_bars: int,
) -> tuple[list[str], list[str], list[int], list[str], list[float]]:
    """Causal volatility state machine with asymmetric enter/exit thresholds."""

    confirm = max(1, int(confirm_bars))
    minimum_age = max(1, int(min_duration_bars))
    raw_states: list[str] = []
    stable: list[str] = []
    ages: list[int] = []
    candidates: list[str] = []
    progresses: list[float] = []
    current = "warmup"
    candidate = "warmup"
    candidate_count = 0
    age = 0

    for vol_value, activity_value, ret_value, is_ready in zip(
        volatility_z, activity_z, return_z, ready
    ):
        vol = float(vol_value) if vol_value is not None else np.nan
        activity = float(activity_value) if activity_value is not None else np.nan
        ret = float(ret_value) if ret_value is not None else np.nan
        if not bool(is_ready) or not np.isfinite(vol) or not np.isfinite(activity):
            raw = "warmup"
            current = "warmup"
            candidate = "warmup"
            candidate_count = 0
            age = 0
        else:
            shock_metric = max(vol, ret if np.isfinite(ret) else vol)
            if shock_metric >= shock_enter:
                raw = "shock"
            elif vol >= expand_enter:
                raw = "expansion"
            elif vol <= quiet_enter and activity <= quiet_enter:
                raw = "dormant"
            elif vol <= quiet_enter:
                raw = "compression"
            else:
                raw = "normal"

            if current == "warmup":
                current = raw
                age = 1
            else:
                if current == "shock":
                    if shock_metric >= shock_exit:
                        target = "shock"
                    elif vol >= expand_exit:
                        target = "expansion"
                    elif vol <= quiet_enter:
                        target = "dormant" if activity <= quiet_enter else "compression"
                    else:
                        target = "normal"
                elif current == "expansion":
                    if shock_metric >= shock_enter:
                        target = "shock"
                    elif vol >= expand_exit:
                        target = "expansion"
                    elif vol <= quiet_enter:
                        target = "dormant" if activity <= quiet_enter else "compression"
                    else:
                        target = "normal"
                elif current in {"compression", "dormant"}:
                    if shock_metric >= shock_enter:
                        target = "shock"
                    elif vol >= expand_enter:
                        target = "expansion"
                    elif vol <= quiet_exit:
                        target = "dormant" if activity <= quiet_enter else "compression"
                    else:
                        target = "normal"
                else:
                    if shock_metric >= shock_enter:
                        target = "shock"
                    elif vol >= expand_enter:
                        target = "expansion"
                    elif vol <= quiet_enter:
                        target = "dormant" if activity <= quiet_enter else "compression"
                    else:
                        target = "normal"

                if target == current:
                    candidate = current
                    candidate_count = 0
                    age += 1
                else:
                    if target == candidate:
                        candidate_count += 1
                    else:
                        candidate = target
                        candidate_count = 1
                    immediate = target == "shock"
                    if immediate or (candidate_count >= confirm and age + 1 >= minimum_age):
                        current = target
                        candidate_count = 0
                        age = 1
                    else:
                        age += 1

        raw_states.append(raw)
        stable.append(current)
        ages.append(age)
        candidates.append(candidate if candidate_count > 0 else current)
        progresses.append(0.0 if candidate_count <= 0 else min(1.0, candidate_count / confirm))

    return raw_states, stable, ages, candidates, progresses


def build_state_segments(frame: pd.DataFrame) -> tuple[MarketStateSegment, ...]:
    if frame is None or frame.empty or "primary_state" not in frame:
        return ()
    labels = frame["primary_state"].astype(str)
    change = labels.ne(labels.shift(1)).cumsum()
    segments: list[MarketStateSegment] = []

    def finite_mean(series: pd.Series) -> float | None:
        values = pd.to_numeric(series, errors="coerce")
        return None if values.notna().sum() == 0 else float(values.mean())

    def finite_max(series: pd.Series) -> float | None:
        values = pd.to_numeric(series, errors="coerce")
        return None if values.notna().sum() == 0 else float(values.max())

    for _, group in frame.groupby(change, sort=False):
        if group.empty:
            continue
        first = group.iloc[0]
        state = str(first["primary_state"])
        if state.startswith("warmup"):
            continue
        segments.append(
            MarketStateSegment(
                start_timestamp=pd.Timestamp(group.index[0]),
                end_timestamp=pd.Timestamp(group.index[-1]),
                start_available_time=pd.Timestamp(group["available_time"].iloc[0]),
                end_available_time=pd.Timestamp(group["available_time"].iloc[-1]),
                primary_state=state,
                trend_state=str(first["trend_state"]),
                volatility_state=str(first["volatility_state"]),
                bars=int(len(group)),
                mean_trend_score=finite_mean(group["trend_score"]),
                mean_orderliness_score=finite_mean(group["orderliness_score"]),
                mean_volatility_score=finite_mean(group["volatility_score"]),
                mean_activity_score=finite_mean(group["activity_score"]),
                max_volatility_z=finite_max(group["volatility_z"]),
            )
        )
    return tuple(segments)
