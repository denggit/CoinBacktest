#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal alignment primitives for the RL market-agent dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.market_state.causal_alignment import available_time_index


@dataclass(frozen=True)
class AlignmentResult:
    features: pd.DataFrame
    source_available_time: pd.Series


def normalize_datetime_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if out.isna().any():
        raise ValueError(f"{name} contains invalid timestamps")
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def make_decision_index(start: str | pd.Timestamp, end: str | pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    step = pd.Timedelta(interval)
    if step <= pd.Timedelta(0):
        raise ValueError("interval must be positive")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    # The state at t is based only on bars whose available_time <= t.  The
    # execution/reference price later uses the 1m bar opening exactly at t.
    first = start_ts.ceil(interval)
    last = end_ts.floor(interval)
    if last < first:
        return pd.DatetimeIndex([], name="decision_time")
    return pd.date_range(first, last, freq=interval, name="decision_time")


def align_left_labeled_bars(
    decision_index: pd.DatetimeIndex,
    feature_frame: pd.DataFrame,
    *,
    bar_duration: pd.Timedelta | str,
    tolerance: pd.Timedelta | str | None = None,
) -> AlignmentResult:
    """Backward-asof align left-labeled closed bars by their available time."""

    decisions = normalize_datetime_index(decision_index, name="decision_index")
    if feature_frame is None or feature_frame.empty:
        empty = pd.DataFrame(index=decisions, columns=list(getattr(feature_frame, "columns", [])), dtype=float)
        available = pd.Series(pd.NaT, index=decisions, dtype="datetime64[ns]", name="source_available_time")
        return AlignmentResult(empty, available)

    right = feature_frame.copy()
    right.index = normalize_datetime_index(right.index, name="feature_frame")
    right = right[~right.index.duplicated(keep="last")].sort_index()
    available = available_time_index(
        right.index,
        bar_duration=pd.Timedelta(bar_duration),
        timestamp_semantics="bar_start",
    )
    right = right.copy()
    right["__source_available_time"] = available
    right.index = available
    right.index.name = "available_time"

    left = pd.DataFrame(index=decisions)
    aligned = pd.merge_asof(
        left.sort_index(),
        right.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
        tolerance=None if tolerance is None else pd.Timedelta(tolerance),
    )
    source_available = pd.to_datetime(aligned.pop("__source_available_time"), errors="coerce")
    source_available.name = "source_available_time"
    aligned.index.name = "decision_time"
    source_available.index = aligned.index
    return AlignmentResult(aligned, source_available)


def align_available_events(
    decision_index: pd.DatetimeIndex,
    event_features: pd.DataFrame,
    *,
    available_time_col: str | None = None,
) -> AlignmentResult:
    """Align event data whose timestamp already means first observable time."""

    decisions = normalize_datetime_index(decision_index, name="decision_index")
    if event_features is None or event_features.empty:
        columns = [c for c in getattr(event_features, "columns", []) if c != available_time_col]
        empty = pd.DataFrame(index=decisions, columns=columns, dtype=float)
        available = pd.Series(pd.NaT, index=decisions, dtype="datetime64[ns]", name="source_available_time")
        return AlignmentResult(empty, available)

    right = event_features.copy()
    if available_time_col is None:
        idx = normalize_datetime_index(right.index, name="event_features")
    else:
        if available_time_col not in right.columns:
            raise KeyError(f"event_features missing {available_time_col!r}")
        idx = normalize_datetime_index(right[available_time_col], name=available_time_col)
        right = right.drop(columns=[available_time_col])
    right.index = idx
    # Multiple events may become available at the same instant.  The caller is
    # expected to have aggregated them; keep last as a deterministic safeguard.
    right = right[~right.index.duplicated(keep="last")].sort_index()
    right["__source_available_time"] = right.index

    aligned = pd.merge_asof(
        pd.DataFrame(index=decisions),
        right,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )
    source_available = pd.to_datetime(aligned.pop("__source_available_time"), errors="coerce")
    aligned.index.name = "decision_time"
    source_available.index = aligned.index
    source_available.name = "source_available_time"
    return AlignmentResult(aligned, source_available)


def causal_audit(decision_index: pd.DatetimeIndex, source_available_time: pd.Series) -> dict[str, float | int | bool | None]:
    decisions = normalize_datetime_index(decision_index, name="decision_index")
    available = pd.to_datetime(source_available_time, errors="coerce")
    available = pd.Series(available.to_numpy(), index=decisions)
    valid = available.notna()
    delta = pd.Series(np.nan, index=decisions, dtype=float)
    if valid.any():
        delta.loc[valid] = (
            available.loc[valid].to_numpy(dtype="datetime64[ns]")
            - decisions[valid.to_numpy()].to_numpy(dtype="datetime64[ns]")
        ).astype("timedelta64[ns]").astype(np.int64) / 1e9
    violations = int((delta > 1e-9).sum())
    max_delta = None if not valid.any() else float(np.nanmax(delta.to_numpy(dtype=float)))
    return {
        "rows_checked": int(valid.sum()),
        "future_visibility_violations": violations,
        "max_available_minus_decision_seconds": max_delta,
        "passed": violations == 0,
    }
