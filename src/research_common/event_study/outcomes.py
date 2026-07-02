#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forward outcome labels for event-study research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .causal import ensure_datetime_index
from .models import CostConfig


def normalize_side(side: pd.Series) -> pd.Series:
    """Normalize common side encodings to +1 long / -1 short."""
    def convert(value: object) -> int:
        if isinstance(value, str):
            v = value.strip().upper()
            if v in {"LONG", "BUY", "UP", "L", "1"}:
                return 1
            if v in {"SHORT", "SELL", "DOWN", "S", "-1"}:
                return -1
            return 0
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
        if numeric > 0:
            return 1
        if numeric < 0:
            return -1
        return 0

    return side.map(convert).astype(int)


def signed_close_to_close_return(close: pd.Series, side: pd.Series, horizon: int) -> pd.Series:
    """Signed close-to-future-close return from the signal bar close."""
    close_num = pd.to_numeric(close, errors="coerce")
    future = close_num.shift(-int(horizon))
    return (future / close_num - 1.0) * side.astype(float)


def signed_next_open_return(bars: pd.DataFrame, side: pd.Series, horizon: int, *, entry_delay_bars: int = 1) -> pd.Series:
    """Signed next-open-to-future-close return.

    `horizon` is counted from the signal bar, matching a closed-bar signal then
    next-bar-open execution convention. horizon=1 means entry at next open and
    exit at the next bar close.
    """
    frame = ensure_datetime_index(bars, name="bars")
    next_open = pd.to_numeric(frame["open"], errors="coerce").shift(-int(entry_delay_bars))
    future_close = pd.to_numeric(frame["close"], errors="coerce").shift(-int(horizon))
    return (future_close / next_open - 1.0) * side.astype(float)


def cost_adjust_return(gross_return: pd.Series, cost: CostConfig) -> pd.Series:
    """Subtract round-trip cost from a signed gross-return label."""
    return pd.to_numeric(gross_return, errors="coerce") - float(cost.round_trip_cost_pct)


def forward_mfe_mae(
    bars: pd.DataFrame,
    side: pd.Series,
    horizon: int,
    *,
    entry_delay_bars: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Return signed MFE and MAE from next open over the forward horizon.

    MFE is non-negative when a favorable excursion exists. MAE is non-positive
    when an adverse excursion exists. Both are measured in simple return units.
    """
    frame = ensure_datetime_index(bars, name="bars")
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    sig = side.astype(int).to_numpy()
    mfe = np.full(len(frame), np.nan, dtype=float)
    mae = np.full(len(frame), np.nan, dtype=float)
    h = int(horizon)
    delay = int(entry_delay_bars)
    for i in range(len(frame)):
        direction = sig[i]
        if direction == 0:
            continue
        entry_pos = i + delay
        if entry_pos >= len(frame):
            continue
        end = min(len(frame), i + h + 1)
        if end <= entry_pos:
            continue
        entry = opens[entry_pos]
        if not np.isfinite(entry) or entry <= 0:
            continue
        high_path = highs[entry_pos:end]
        low_path = lows[entry_pos:end]
        if direction == 1:
            path_high = np.nanmax(high_path) if high_path.size else np.nan
            path_low = np.nanmin(low_path) if low_path.size else np.nan
            mfe[i] = path_high / entry - 1.0 if np.isfinite(path_high) else np.nan
            mae[i] = path_low / entry - 1.0 if np.isfinite(path_low) else np.nan
        else:
            path_low = np.nanmin(low_path) if low_path.size else np.nan
            path_high = np.nanmax(high_path) if high_path.size else np.nan
            mfe[i] = entry / path_low - 1.0 if np.isfinite(path_low) and path_low > 0 else np.nan
            mae[i] = entry / path_high - 1.0 if np.isfinite(path_high) and path_high > 0 else np.nan
    return pd.Series(mfe, index=frame.index), pd.Series(mae, index=frame.index)


def first_touch_outcome(
    bars: pd.DataFrame,
    side: pd.Series,
    *,
    target_pct: float,
    stop_pct: float,
    horizon: int,
    entry_delay_bars: int = 1,
    same_bar_policy: str = "conservative",
) -> pd.DataFrame:
    """Label whether target or stop is touched first after a next-open entry.

    same_bar_policy='conservative' marks same-bar target+stop as stop. This is
    deliberately pessimistic for OHLC bars where intrabar path is unknowable.
    """
    if same_bar_policy not in {"conservative", "target", "unknown"}:
        raise ValueError("same_bar_policy must be conservative, target, or unknown")
    frame = ensure_datetime_index(bars, name="bars")
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    sig = side.astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    idx = frame.index
    for i, direction in enumerate(sig):
        entry_pos = i + int(entry_delay_bars)
        if direction == 0 or entry_pos >= len(frame):
            rows.append({"touch_result": "NO_EVENT", "touch_bars": np.nan, "same_bar_both_hit_flag": False})
            continue
        entry = opens[entry_pos]
        if not np.isfinite(entry) or entry <= 0:
            rows.append({"touch_result": "NO_ENTRY", "touch_bars": np.nan, "same_bar_both_hit_flag": False})
            continue
        if direction == 1:
            target_price = entry * (1.0 + float(target_pct))
            stop_price = entry * (1.0 - float(stop_pct))
        else:
            target_price = entry * (1.0 - float(target_pct))
            stop_price = entry * (1.0 + float(stop_pct))
        result = "TIMEOUT"
        touch_bars = np.nan
        both_flag = False
        end = min(len(frame), i + int(horizon) + 1)
        for pos in range(entry_pos, end):
            if direction == 1:
                hit_target = highs[pos] >= target_price
                hit_stop = lows[pos] <= stop_price
            else:
                hit_target = lows[pos] <= target_price
                hit_stop = highs[pos] >= stop_price
            if hit_target and hit_stop:
                both_flag = True
                if same_bar_policy == "target":
                    result = "TARGET"
                elif same_bar_policy == "unknown":
                    result = "BOTH_UNKNOWN"
                else:
                    result = "STOP"
                touch_bars = pos - i
                break
            if hit_stop:
                result = "STOP"
                touch_bars = pos - i
                break
            if hit_target:
                result = "TARGET"
                touch_bars = pos - i
                break
        rows.append({"touch_result": result, "touch_bars": touch_bars, "same_bar_both_hit_flag": both_flag})
    return pd.DataFrame(rows, index=idx)
