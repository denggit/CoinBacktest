#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal timestamp utilities for multi-timeframe market-state contexts."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([smhdSMHD])\s*$")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Parse project-style timeframes such as 1s, 5m, 4H and 1D."""

    match = _TIMEFRAME_RE.match(str(timeframe))
    if not match:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("timeframe amount must be positive")
    unit = match.group(2).lower()
    unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    return pd.Timedelta(**{unit_map[unit]: amount})


def available_time_index(
    index: pd.Index,
    *,
    bar_duration: pd.Timedelta | str | None,
    timestamp_semantics: str,
) -> pd.DatetimeIndex:
    """Return when each row becomes observable.

    ``bar_start`` timestamps are shifted by the full bar duration.  ``bar_end``
    and ``available`` timestamps are already observable and are not shifted.
    """

    semantics = str(timestamp_semantics).strip().lower()
    if semantics not in {"bar_start", "bar_end", "available"}:
        raise ValueError("timestamp_semantics must be bar_start/bar_end/available")
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if semantics == "bar_start":
        if bar_duration is None:
            raise ValueError("bar_duration is required for bar_start timestamps")
        delta = pd.Timedelta(bar_duration)
        if delta <= pd.Timedelta(0):
            raise ValueError("bar_duration must be positive")
        return idx + delta
    return idx


def shift_context_to_available_time(
    context: pd.DataFrame,
    *,
    bar_duration: pd.Timedelta | str,
) -> pd.DataFrame:
    """Shift a left-labeled context frame to its first causal use time."""

    out = context.copy()
    out.index = available_time_index(
        out.index,
        bar_duration=bar_duration,
        timestamp_semantics="bar_start",
    )
    out.index.name = "available_time"
    return out.sort_index()


def causal_merge_context(
    primary: pd.DataFrame,
    context: pd.DataFrame,
    *,
    context_bar_duration: pd.Timedelta | str,
    suffix: str = "_ctx",
    tolerance: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    """Backward-asof merge using context *available* time, never bar start."""

    if primary is None or primary.empty:
        return primary.copy()
    left = primary.copy().sort_index()
    left.index = pd.DatetimeIndex(pd.to_datetime(left.index))
    right = shift_context_to_available_time(context, bar_duration=context_bar_duration)
    if right.empty:
        return left
    right = right.rename(columns={column: f"{column}{suffix}" for column in right.columns})
    merged = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=None if tolerance is None else pd.Timedelta(tolerance),
    )
    merged.index.name = left.index.name
    return merged
