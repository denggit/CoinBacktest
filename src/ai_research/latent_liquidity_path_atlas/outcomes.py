#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Future-only micro path labels for liquidity-release candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LatentLiquidityPathAtlasConfig


def _directional_path_metrics(
    path: pd.DataFrame,
    side: str,
    event_price: float,
    config: LatentLiquidityPathAtlasConfig,
) -> dict[str, object]:
    high = path["high"].to_numpy(dtype=float)
    low = path["low"].to_numpy(dtype=float)
    close = path["close"].to_numpy(dtype=float)
    if side == "DOWN":
        directional = (event_price - low) / event_price * 1e4
        opposite = (high - np.minimum.accumulate(low)) / np.minimum.accumulate(low) * 1e4
        trough_pos = int(np.argmax(directional))
        extreme_price = float(low[trough_pos])
        rebound_after_extreme = (np.max(high[trough_pos:]) - extreme_price) / extreme_price * 1e4
        closes_beyond = close < event_price
    else:
        directional = (high - event_price) / event_price * 1e4
        opposite = (np.maximum.accumulate(high) - low) / np.maximum.accumulate(high) * 1e4
        trough_pos = int(np.argmax(directional))
        extreme_price = float(high[trough_pos])
        rebound_after_extreme = (extreme_price - np.min(low[trough_pos:])) / extreme_price * 1e4
        closes_beyond = close > event_price

    extension_bp = float(np.max(directional))
    time_to_extreme = int(np.argmax(directional)) + 1
    immediate_end = min(config.immediate_reversal_seconds, len(path))
    extended_end = min(config.extended_reversal_seconds, len(path))
    immediate_reversal = float(np.max(opposite[:immediate_end]))
    if trough_pos < extended_end:
        if side == "DOWN":
            extended_reversal = float((np.max(high[trough_pos:extended_end]) - extreme_price) / extreme_price * 1e4)
        else:
            extended_reversal = float((extreme_price - np.min(low[trough_pos:extended_end])) / extreme_price * 1e4)
    else:
        extended_reversal = 0.0
    acceptance_fraction_60s = float(np.mean(closes_beyond[: min(60, len(path))]))

    stability_start = min(len(path) - 1, trough_pos + 1)
    stability_end = min(len(path), stability_start + config.stabilization_seconds)
    if side == "DOWN":
        stable_after_extreme = bool(np.min(low[stability_start:stability_end]) >= extreme_price * (1.0 - 0.0002))
    else:
        stable_after_extreme = bool(np.max(high[stability_start:stability_end]) <= extreme_price * (1.0 + 0.0002))

    if extension_bp <= config.shallow_extension_bp and immediate_reversal >= config.immediate_reversal_bp:
        outcome = "SHALLOW_IMMEDIATE_REVERSAL"
    elif (
        extension_bp > config.shallow_extension_bp
        and time_to_extreme <= min(15, config.immediate_reversal_seconds)
        and immediate_reversal >= config.extended_reversal_bp
    ):
        outcome = "DEEP_IMMEDIATE_REVERSAL"
    elif (
        extension_bp >= config.extended_min_extension_bp
        and time_to_extreme <= config.extended_reversal_seconds
        and stable_after_extreme
        and extended_reversal >= config.extended_reversal_bp
    ):
        outcome = "EXTEND_STABILIZE_REVERSAL"
    elif (
        extension_bp >= config.continuation_extension_bp
        and acceptance_fraction_60s >= config.continuation_acceptance_fraction
        and extended_reversal < config.extended_reversal_bp
    ):
        outcome = "ACCEPT_CONTINUATION"
    else:
        outcome = "MIXED_OR_UNRESOLVED"

    result: dict[str, object] = {
        "future_extension_bp": extension_bp,
        "future_time_to_extreme_seconds": time_to_extreme,
        "future_immediate_reversal_bp": immediate_reversal,
        "future_reversal_after_extreme_bp": float(extended_reversal),
        "future_acceptance_fraction_60s": acceptance_fraction_60s,
        "future_stable_after_extreme": stable_after_extreme,
        "outcome_type": outcome,
        "favorable_reversal": outcome in {"SHALLOW_IMMEDIATE_REVERSAL", "DEEP_IMMEDIATE_REVERSAL", "EXTEND_STABILIZE_REVERSAL"},
    }
    for horizon in (5, 15, 30, 60, 180, 300, 600):
        if horizon > len(path):
            continue
        end = horizon
        if side == "DOWN":
            result[f"future_same_direction_extension_{horizon}s_bp"] = float(np.max((event_price - low[:end]) / event_price * 1e4))
            result[f"future_opposite_excursion_{horizon}s_bp"] = float(np.max((high[:end] - event_price) / event_price * 1e4))
            result[f"future_close_return_{horizon}s_bp"] = float((close[end - 1] / event_price - 1.0) * 1e4)
        else:
            result[f"future_same_direction_extension_{horizon}s_bp"] = float(np.max((high[:end] - event_price) / event_price * 1e4))
            result[f"future_opposite_excursion_{horizon}s_bp"] = float(np.max((event_price - low[:end]) / event_price * 1e4))
            result[f"future_close_return_{horizon}s_bp"] = float((close[end - 1] / event_price - 1.0) * 1e4)
    return result


def attach_outcomes(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    timestamps = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    one_second = int(pd.Timedelta(seconds=1).value)
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        event_time = pd.Timestamp(event.event_time)
        event_ns = int(event_time.value)
        pos = int(np.searchsorted(timestamps, event_ns, side="left"))
        start = pos + 1
        stop = start + config.post_label_seconds
        if pos >= len(bars) or stop > len(bars):
            continue
        expected = event_ns + np.arange(1, config.post_label_seconds + 1, dtype=np.int64) * one_second
        actual = timestamps[start:stop]
        if len(actual) != len(expected) or not np.array_equal(actual, expected):
            continue
        path = bars.iloc[start:stop]
        if int(path["unsafe_gap"].max()) != 0:
            continue
        event_row = bars.iloc[pos]
        event_price = float(event_row["low"] if event.event_side == "DOWN" else event_row["high"])
        metrics = _directional_path_metrics(path, str(event.event_side), event_price, config)
        metrics.update(
            {
                "event_id": event.event_id,
                "event_time": event_time,
                "event_side": event.event_side,
                "label_start_time": event_time + pd.Timedelta(seconds=1),
                "label_end_time": event_time + pd.Timedelta(seconds=config.post_label_seconds),
                "event_reference_price": event_price,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)
