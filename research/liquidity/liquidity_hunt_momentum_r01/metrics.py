#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Return summaries, chronological splits and causal audit helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

def profit_factor(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else np.nan)


def summarize_returns(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    if frame is None or frame.empty or value_col not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_cols), observed=True, dropna=False) if group_cols else [((), frame)]
    for key, part in grouped:
        values = pd.to_numeric(part[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_cols, keys, strict=False)}
        row.update(
            {
                "metric": value_col,
                "events": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "win_rate": float(np.mean(values > 0)),
                "profit_factor": profit_factor(values),
                "p10": float(np.quantile(values, 0.10)),
                "p90": float(np.quantile(values, 0.90)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def chronological_split_labels(times: pd.Series) -> pd.Series:
    """60/20/20 chronological split; no shuffling and no target-based choice."""

    ts = pd.to_datetime(times)
    if ts.empty:
        return pd.Series(dtype=object)
    unique = np.sort(ts.dropna().unique())
    if len(unique) == 0:
        return pd.Series("unknown", index=times.index)
    train_cut = unique[min(len(unique) - 1, int(len(unique) * 0.60))]
    valid_cut = unique[min(len(unique) - 1, int(len(unique) * 0.80))]
    labels = np.where(ts <= train_cut, "train_60", np.where(ts <= valid_cut, "validation_20", "holdout_20"))
    return pd.Series(labels, index=times.index, dtype=object)


def _flag_array(values: pd.Series) -> np.ndarray:
    """Coerce nullable/object flags without pandas silent downcasting."""

    nullable = pd.Series(pd.array(values, dtype="boolean"), index=values.index)
    return nullable.fillna(False).to_numpy(dtype=bool)


def build_causal_audit(events: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    cols = [
        "event_id",
        "range_tag",
        "mode",
        "stage",
        "side_name",
        "signal_time",
        "book_available_time",
        "book_available_after_signal_flag",
        "book_context_missing_flag",
        "footprint_missing_flag",
    ]
    audit = events[[c for c in cols if c in events.columns]].copy().reset_index(drop=True)
    audit["signal_time"] = pd.to_datetime(audit["signal_time"], errors="coerce")
    if "book_available_time" in audit:
        audit["book_available_time"] = pd.to_datetime(audit["book_available_time"], errors="coerce")
        audit["book_available_after_signal_flag"] = audit["book_available_time"] > audit["signal_time"]
    if trades is not None and not trades.empty:
        entry = trades[["event_id", "variant", "entry_time", "entry_not_after_signal_flag", "same_bar_both_hit_flag"]].copy()
        audit = audit.merge(entry, on="event_id", how="left")
    # Missing optional context is a data-quality limitation, not evidence of
    # lookahead.  Keep it visible without inflating the causal-failure count.
    missing_flags = [
        c
        for c in ("book_context_missing_flag", "footprint_missing_flag")
        if c in audit.columns
    ]
    audit["data_missing_flag"] = False
    for col in missing_flags:
        audit["data_missing_flag"] |= _flag_array(audit[col])

    non_causal_flags = {
        "book_context_missing_flag",
        "footprint_missing_flag",
        "data_missing_flag",
        "same_bar_both_hit_flag",
    }
    causal_flags = [
        c for c in audit.columns if c.endswith("flag") and c not in non_causal_flags
    ]
    audit["causal_fail_flag"] = False
    for col in causal_flags:
        audit["causal_fail_flag"] |= _flag_array(audit[col])
    return audit
