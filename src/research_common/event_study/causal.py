#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal alignment and audit helpers for research code."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def ensure_datetime_index(df: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    """Return a sorted copy with a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a pandas DatetimeIndex")
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicate timestamps")
    return out


def add_available_time_index(context: pd.DataFrame, timeframe: str | pd.Timedelta) -> pd.DataFrame:
    """Shift a closed-bar context frame from bar-start index to available-time index."""
    ctx = ensure_datetime_index(context, name="context")
    delta = pd.Timedelta(timeframe)
    if delta <= pd.Timedelta(0):
        raise ValueError("timeframe must be positive")
    out = ctx.copy()
    out.index = out.index + delta
    out.index.name = ctx.index.name or "available_time"
    return out


def causal_align_context(
    primary: pd.DataFrame,
    context: pd.DataFrame,
    *,
    timeframe: str | pd.Timedelta,
    suffix: str = "_ctx",
    direction: str = "backward",
) -> pd.DataFrame:
    """Merge high-timeframe context onto a primary axis by available_time.

    The context index is assumed to be bar-start time and is shifted by
    `timeframe` before merge_asof. This prevents left-label high timeframe bars
    from leaking into lower timeframe signal rows before the bar has closed.
    """
    base = ensure_datetime_index(primary, name="primary")
    ctx = add_available_time_index(context, timeframe=timeframe)
    merged = pd.merge_asof(
        base.sort_index(),
        ctx.sort_index(),
        left_index=True,
        right_index=True,
        direction=direction,
        suffixes=("", suffix),
    )
    return merged


def audit_context_available_times(
    events: pd.DataFrame,
    *,
    signal_time_col: str = "signal_time",
    context_available_time_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Audit that every context available_time is <= signal_time."""
    if signal_time_col not in events.columns:
        raise KeyError(f"events is missing signal_time_col: {signal_time_col}")
    signal_time = pd.to_datetime(events[signal_time_col], errors="coerce")
    out = pd.DataFrame(index=events.index)
    out["signal_time"] = signal_time
    flags: list[pd.Series] = []
    for col in context_available_time_cols:
        if col not in events.columns:
            flag = pd.Series(False, index=events.index)
            out[f"missing_{col}"] = True
        else:
            ctx_time = pd.to_datetime(events[col], errors="coerce")
            flag = ctx_time.notna() & signal_time.notna() & (ctx_time > signal_time)
            out[col] = ctx_time
            out[f"{col}_flag"] = flag.astype(bool)
        flags.append(flag.astype(bool))
    out["context_available_time_flag"] = pd.concat(flags, axis=1).any(axis=1) if flags else False
    return out


def audit_next_open_entries(events: pd.DataFrame, *, signal_time_col: str = "signal_time", entry_time_col: str = "entry_time") -> pd.DataFrame:
    """Audit that event entries occur strictly after the signal timestamp."""
    if signal_time_col not in events.columns:
        raise KeyError(f"events is missing signal_time_col: {signal_time_col}")
    if entry_time_col not in events.columns:
        raise KeyError(f"events is missing entry_time_col: {entry_time_col}")
    signal_time = pd.to_datetime(events[signal_time_col], errors="coerce")
    entry_time = pd.to_datetime(events[entry_time_col], errors="coerce")
    out = pd.DataFrame(index=events.index)
    out["signal_time"] = signal_time
    out["entry_time"] = entry_time
    out["entry_after_signal_flag"] = entry_time.notna() & signal_time.notna() & (entry_time > signal_time)
    out["entry_not_after_signal_flag"] = ~out["entry_after_signal_flag"]
    return out
