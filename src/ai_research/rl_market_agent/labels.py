#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forward-path labels for supervised opportunity learning.

Labels may use future bars because they are outcomes, never state features.
Execution/reference entry is the open of the 1m bar beginning at decision_time;
the state itself only sees source bars available at or before that instant.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import LabelSpec


def _forward_rolling(series: pd.Series, window: int, op: str) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    reverse = pd.to_numeric(series, errors="coerce").iloc[::-1]
    roller = reverse.rolling(window, min_periods=window)
    if op == "max":
        out = roller.max()
    elif op == "min":
        out = roller.min()
    else:
        raise ValueError(f"unsupported op: {op}")
    return out.iloc[::-1]


def build_forward_path_labels(
    one_minute_bars: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    horizons_minutes: Iterable[int],
) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in one_minute_bars.columns]
    if missing:
        raise ValueError(f"1m path frame missing columns: {missing}")
    bars = one_minute_bars.copy()
    bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index, errors="coerce"))
    bars = bars.loc[~bars.index.isna()]
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    for column in required:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    entry = bars["open"].where(bars["open"] > 0)
    out = pd.DataFrame(index=bars.index)
    out["entry_price"] = entry
    for raw_h in horizons_minutes:
        h = int(raw_h)
        if h <= 0:
            raise ValueError("horizons must be positive")
        high = _forward_rolling(bars["high"], h, "max")
        low = _forward_rolling(bars["low"], h, "min")
        final_close = bars["close"].shift(-(h - 1))
        out[f"h{h}__final_return"] = final_close / entry - 1.0
        out[f"h{h}__long_mfe"] = high / entry - 1.0
        out[f"h{h}__long_mae"] = low / entry - 1.0
        out[f"h{h}__short_mfe"] = 1.0 - low / entry
        out[f"h{h}__short_mae"] = 1.0 - high / entry
        out[f"h{h}__path_width"] = (high - low) / entry

    decision = pd.DatetimeIndex(pd.to_datetime(decision_index))
    selected = out.reindex(decision)
    selected.index.name = "decision_time"
    return selected.replace([np.inf, -np.inf], np.nan)


def label_specs(horizons_minutes: Iterable[int]) -> list[LabelSpec]:
    specs = [LabelSpec("entry_price", 0, "Open price of the 1m execution bar at decision_time.")]
    for raw_h in horizons_minutes:
        h = int(raw_h)
        specs.extend(
            [
                LabelSpec(f"h{h}__final_return", h, f"Close return from entry open through minute {h}."),
                LabelSpec(f"h{h}__long_mfe", h, f"Maximum favorable long excursion over the next {h} minutes."),
                LabelSpec(f"h{h}__long_mae", h, f"Maximum adverse long excursion over the next {h} minutes (<=0)."),
                LabelSpec(f"h{h}__short_mfe", h, f"Maximum favorable short excursion over the next {h} minutes."),
                LabelSpec(f"h{h}__short_mae", h, f"Maximum adverse short excursion over the next {h} minutes (<=0)."),
                LabelSpec(f"h{h}__path_width", h, f"High-low path width divided by entry over the next {h} minutes."),
            ]
        )
    return specs


def label_names(horizons_minutes: Iterable[int]) -> list[str]:
    return [spec.name for spec in label_specs(horizons_minutes)]
