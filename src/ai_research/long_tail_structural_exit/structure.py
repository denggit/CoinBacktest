#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal event-relative bars and confirmed swing structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData

from .config import StructuralExitConfig


@dataclass(frozen=True)
class EventBars:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    close_time_ns: np.ndarray
    close_position: np.ndarray
    atr: np.ndarray
    entry_bar_index: int


@dataclass(frozen=True)
class ConfirmedPivot:
    kind: str
    pivot_index: int
    confirmation_index: int
    price: float


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous = np.concatenate([[np.nan], close[:-1]])
    values = np.column_stack(
        [
            high - low,
            np.abs(high - previous),
            np.abs(low - previous),
        ]
    )
    values[0, 1:] = np.nan
    return np.nanmax(values, axis=1)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window, min_periods=window).mean().to_numpy(dtype=float)


def build_event_bars(
    path: MinutePathData,
    *,
    entry_position: int,
    end_position: int,
    config: StructuralExitConfig,
) -> EventBars | None:
    """Build fixed-width bars with entry on an exact bar boundary.

    The bar sequence starts ``pre_entry_history_minutes`` before entry. Each
    post-entry state decision occurs only after a full bar has closed.
    """

    history = int(config.pre_entry_history_minutes)
    width = int(config.structure_bar_minutes)
    start = int(entry_position - history)
    if start < 0 or end_position < entry_position:
        return None
    available = int(end_position - start + 1)
    complete_rows = available - (available % width)
    if complete_rows < history + width:
        return None
    stop = start + complete_rows
    timestamps = np.asarray(path.timestamps_ns[start:stop], dtype=np.int64)
    expected_step = int(pd.Timedelta(minutes=1).value)
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) == expected_step):
        return None

    shape = (-1, width)
    open_ = np.asarray(path.open[start:stop], dtype=float).reshape(shape)[:, 0]
    high = np.asarray(path.high[start:stop], dtype=float).reshape(shape).max(axis=1)
    low = np.asarray(path.low[start:stop], dtype=float).reshape(shape).min(axis=1)
    close = np.asarray(path.close[start:stop], dtype=float).reshape(shape)[:, -1]
    close_pos = np.arange(start + width - 1, stop, width, dtype=np.int64)
    close_ns = np.asarray(path.timestamps_ns[close_pos], dtype=np.int64)
    tr = _true_range(high, low, close)
    atr = _rolling_mean(tr, int(config.atr_window_bars))
    entry_bar = history // width
    return EventBars(
        open=open_,
        high=high,
        low=low,
        close=close,
        close_time_ns=close_ns,
        close_position=close_pos,
        atr=atr,
        entry_bar_index=entry_bar,
    )


def confirmed_pivots(
    bars: EventBars,
    *,
    left_bars: int,
    right_bars: int,
) -> tuple[ConfirmedPivot, ...]:
    """Return pivots with the bar index at which each pivot becomes knowable."""

    pivots: list[ConfirmedPivot] = []
    total = len(bars.close)
    for pivot_index in range(left_bars, total - right_bars):
        left = pivot_index - left_bars
        right = pivot_index + right_bars + 1
        low_window = bars.low[left:right]
        high_window = bars.high[left:right]
        low_value = float(bars.low[pivot_index])
        high_value = float(bars.high[pivot_index])
        other_lows = np.delete(low_window, left_bars)
        other_highs = np.delete(high_window, left_bars)
        confirmation = pivot_index + right_bars
        if np.isfinite(low_value) and np.all(low_value < other_lows):
            pivots.append(ConfirmedPivot("low", pivot_index, confirmation, low_value))
        if np.isfinite(high_value) and np.all(high_value > other_highs):
            pivots.append(ConfirmedPivot("high", pivot_index, confirmation, high_value))
    return tuple(sorted(pivots, key=lambda item: (item.confirmation_index, item.pivot_index, item.kind)))


def pivots_confirmed_by(
    pivots: tuple[ConfirmedPivot, ...],
    confirmation_index: int,
) -> tuple[ConfirmedPivot, ...]:
    return tuple(pivot for pivot in pivots if pivot.confirmation_index <= confirmation_index)


def latest_pivot(
    pivots: tuple[ConfirmedPivot, ...],
    *,
    kind: str,
    confirmed_by: int,
    before_or_at_pivot: int | None = None,
) -> ConfirmedPivot | None:
    candidates = [
        pivot
        for pivot in pivots
        if pivot.kind == kind
        and pivot.confirmation_index <= confirmed_by
        and (before_or_at_pivot is None or pivot.pivot_index <= before_or_at_pivot)
    ]
    return max(candidates, key=lambda item: item.pivot_index) if candidates else None


def structure_buffer(
    *,
    price: float,
    atr: float,
    config: StructuralExitConfig,
) -> float:
    bps = float(price) * float(config.minimum_structure_buffer_bps) / 10_000.0
    atr_buffer = float(atr) * float(config.structure_buffer_atr_multiple) if np.isfinite(atr) else 0.0
    return float(max(bps, atr_buffer))
