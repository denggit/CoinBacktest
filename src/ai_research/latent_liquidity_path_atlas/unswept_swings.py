#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementary 15m+ unswept Swing inventory.

This module deliberately does *not* define liquidity events.  It only records
all causally confirmed 15m/30m/1H/4H/1D Swing High/Low levels and keeps each
level active until its first true sweep.  Old levels are never discarded merely
because of age; age, distance and cross-timeframe confluence are model inputs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import aggregate_timeframe, normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .config import LatentLiquidityPathAtlasConfig
from .time_axis import as_datetime_ns

_REQUIRED_LEVEL_COLUMNS = (
    "level_id",
    "level_side",
    "source_timeframe",
    "source_timeframe_min",
    "pivot_time",
    "level_price",
    "initial_available_time",
    "sweep_available_time",
)


def _pivot_mask(values: np.ndarray, order: int, *, side: str) -> np.ndarray:
    n = len(values)
    order = int(order)
    if n < order * 2 + 1:
        return np.zeros(n, dtype=bool)
    mask = np.isfinite(values)
    mask[:order] = False
    mask[n - order :] = False
    for lag in range(1, order + 1):
        left = np.empty(n, dtype=float)
        right = np.empty(n, dtype=float)
        left[:lag] = np.nan
        left[lag:] = values[:-lag]
        right[-lag:] = np.nan
        right[:-lag] = values[lag:]
        if side == "LOW":
            mask &= values < left
            mask &= values <= right
        else:
            mask &= values > left
            mask &= values >= right
    return mask


def _build_timeframe_levels(
    minute_bars: pd.DataFrame,
    timeframe: str,
    minutes: int,
    confirmation_order: int,
) -> pd.DataFrame:
    htf = aggregate_timeframe(minute_bars, minutes=int(minutes))
    if htf.empty:
        return pd.DataFrame()
    order = int(confirmation_order)
    delta = pd.Timedelta(minutes=int(minutes))
    rows: list[pd.DataFrame] = []
    for side, column in (("LOW", "low"), ("HIGH", "high")):
        values = pd.to_numeric(htf[column], errors="coerce").to_numpy(dtype=float)
        positions = np.flatnonzero(_pivot_mask(values, order, side=side))
        if not len(positions):
            continue
        level_price = values[positions]
        bar_range = (
            pd.to_numeric(htf["high"], errors="coerce").to_numpy(dtype=float)[positions]
            - pd.to_numeric(htf["low"], errors="coerce").to_numpy(dtype=float)[positions]
        )
        close = pd.to_numeric(htf["close"], errors="coerce").to_numpy(dtype=float)[positions]
        notional = pd.to_numeric(htf.get("notional", np.nan), errors="coerce")
        if isinstance(notional, pd.Series):
            notional_value = notional.iloc[positions].to_numpy(dtype=float)
            notional_baseline = (
                notional.shift(1).rolling(20, min_periods=5).median().iloc[positions].to_numpy(dtype=float)
            )
            notional_ratio = np.divide(
                notional_value,
                notional_baseline,
                out=np.full(len(positions), np.nan),
                where=np.isfinite(notional_baseline) & (notional_baseline > 0),
            )
        else:
            notional_value = np.full(len(positions), np.nan)
            notional_ratio = np.full(len(positions), np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "level_side": side,
                    "source_timeframe": str(timeframe),
                    "source_timeframe_min": int(minutes),
                    "pivot_time": htf.index[positions],
                    "level_price": level_price,
                    # An order-k pivot is known only after the kth right bar closes.
                    "initial_available_time": htf.index[positions] + (order + 1) * delta,
                    "pivot_range_bp": np.divide(
                        bar_range,
                        close,
                        out=np.full(len(positions), np.nan),
                        where=np.isfinite(close) & (close > 0),
                    )
                    * 1e4,
                    "pivot_notional": notional_value,
                    "pivot_notional_vs_past20": notional_ratio,
                    "confirmation_order": order,
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["initial_available_time", "source_timeframe_min", "pivot_time", "level_side"],
        kind="mergesort",
    )


def build_unswept_swing_lifecycle(
    minute_bars: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Build sparse lifecycle rows for every confirmed 15m+ Swing level."""
    config.validate()
    bars = normalize_primary_bars(minute_bars)
    parts = [
        _build_timeframe_levels(
            bars,
            timeframe=name,
            minutes=int(minutes),
            confirmation_order=config.swing_confirmation_order,
        )
        for name, minutes in config.swing_timeframes
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(columns=_REQUIRED_LEVEL_COLUMNS)
    levels = pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["initial_available_time", "source_timeframe_min", "pivot_time", "level_side"],
        kind="mergesort",
    ).reset_index(drop=True)
    levels.insert(0, "level_id", np.arange(1, len(levels) + 1, dtype=np.int64))

    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    bar_available = (pd.DatetimeIndex(bars.index) + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]")
    bar_start = pd.DatetimeIndex(bars.index)
    epsilon = float(config.swing_sweep_epsilon_bp) / 1e4
    sweep_times: list[pd.Timestamp | pd.NaT] = []
    sweep_prices: list[float] = []
    active_positions: list[int] = []
    for row in levels.itertuples(index=False):
        available = np.datetime64(pd.Timestamp(row.initial_available_time), "ns")
        start = int(np.searchsorted(bar_available, available, side="left"))
        active_positions.append(start)
        if start >= len(bars):
            sweep_times.append(pd.NaT)
            sweep_prices.append(np.nan)
            continue
        level_price = float(row.level_price)
        if row.level_side == "LOW":
            pos = low_index.first_leq(start, len(bars) - 1, level_price * (1.0 - epsilon))
            sweep_price = low[pos] if pos >= 0 else np.nan
        else:
            pos = high_index.first_geq(start, len(bars) - 1, level_price * (1.0 + epsilon))
            sweep_price = high[pos] if pos >= 0 else np.nan
        sweep_times.append(bar_start[pos] + pd.Timedelta(minutes=1) if pos >= 0 else pd.NaT)
        sweep_prices.append(float(sweep_price) if np.isfinite(sweep_price) else np.nan)
    levels["active_minute_pos"] = active_positions
    levels["sweep_available_time"] = pd.to_datetime(sweep_times)
    levels["sweep_price"] = sweep_prices
    levels["unswept_at_dataset_end"] = levels["sweep_available_time"].isna()
    levels["lifetime_minutes"] = (
        pd.to_datetime(levels["sweep_available_time"]).fillna(bars.index[-1] + pd.Timedelta(minutes=1))
        - pd.to_datetime(levels["initial_available_time"])
    ).dt.total_seconds() / 60.0
    return levels


def save_swing_lifecycle(levels: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(path, index=False, compression="gzip")


def load_swing_lifecycle(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip")
    for name in ("pivot_time", "initial_available_time", "sweep_available_time"):
        if name in frame:
            frame[name] = as_datetime_ns(frame[name], errors="coerce")
    return frame


@dataclass(frozen=True)
class _LevelRecord:
    level_id: int
    side: str
    timeframe: str
    timeframe_min: int
    price: float
    available_time: pd.Timestamp
    sweep_time: pd.Timestamp | pd.NaT
    pivot_range_bp: float
    pivot_notional_ratio: float


def _record(row: object) -> _LevelRecord:
    return _LevelRecord(
        level_id=int(getattr(row, "level_id")),
        side=str(getattr(row, "level_side")),
        timeframe=str(getattr(row, "source_timeframe")),
        timeframe_min=int(getattr(row, "source_timeframe_min")),
        price=float(getattr(row, "level_price")),
        available_time=pd.Timestamp(getattr(row, "initial_available_time")),
        sweep_time=pd.Timestamp(getattr(row, "sweep_available_time"))
        if pd.notna(getattr(row, "sweep_available_time"))
        else pd.NaT,
        pivot_range_bp=float(getattr(row, "pivot_range_bp", np.nan)),
        pivot_notional_ratio=float(getattr(row, "pivot_notional_vs_past20", np.nan)),
    )


def _empty_inventory_row(config: LatentLiquidityPathAtlasConfig) -> dict[str, object]:
    row: dict[str, object] = {
        "unswept_relevant_count": 0,
        "unswept_opposite_count": 0,
        "unswept_nearest_distance_bp": np.nan,
        "unswept_nearest_age_minutes": np.nan,
        "unswept_oldest_age_minutes": np.nan,
        "unswept_median_age_minutes": np.nan,
        "unswept_distance_q25_bp": np.nan,
        "unswept_distance_q50_bp": np.nan,
        "unswept_distance_q75_bp": np.nan,
        "unswept_timeframe_diversity": 0,
        "unswept_max_pivot_range_bp": np.nan,
        "unswept_max_pivot_notional_vs_past20": np.nan,
        "unswept_max_level_available_time": pd.NaT,
    }
    for timeframe, _ in config.swing_timeframes:
        row[f"unswept_{timeframe}_relevant_count"] = 0
        row[f"unswept_{timeframe}_nearest_distance_bp"] = np.nan
        row[f"unswept_{timeframe}_oldest_age_minutes"] = np.nan
    for band in config.swing_confluence_bp:
        key = int(band) if float(band).is_integer() else str(band).replace(".", "p")
        row[f"unswept_confluence_count_{key}bp"] = 0
        row[f"unswept_confluence_timeframes_{key}bp"] = 0
        row[f"unswept_confluence_oldest_age_minutes_{key}bp"] = np.nan
    return row



def _finite_max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if len(finite) else np.nan


def _inventory_features(
    active: dict[int, _LevelRecord],
    event_time: pd.Timestamp,
    event_side: str,
    anchor_price: float,
    config: LatentLiquidityPathAtlasConfig,
) -> dict[str, object]:
    row = _empty_inventory_row(config)
    if not active or not np.isfinite(anchor_price) or anchor_price <= 0:
        return row
    relevant_side = "LOW" if event_side == "DOWN" else "HIGH"
    relevant: list[tuple[_LevelRecord, float, float]] = []
    opposite = 0
    for level in active.values():
        if level.side != relevant_side:
            opposite += 1
            continue
        if relevant_side == "LOW":
            distance = (anchor_price - level.price) / anchor_price * 1e4
        else:
            distance = (level.price - anchor_price) / anchor_price * 1e4
        # A tiny negative distance can arise because the current minute has
        # already touched the level but the 1m bar has not closed.  Retain it as
        # near-zero pre-release inventory rather than silently dropping it.
        if distance < -5.0:
            continue
        age = max((event_time - level.available_time).total_seconds() / 60.0, 0.0)
        relevant.append((level, float(max(distance, 0.0)), float(age)))
    row["unswept_relevant_count"] = len(relevant)
    row["unswept_opposite_count"] = opposite
    if not relevant:
        return row
    distances = np.asarray([item[1] for item in relevant], dtype=float)
    ages = np.asarray([item[2] for item in relevant], dtype=float)
    nearest_pos = int(np.argmin(distances))
    row.update(
        {
            "unswept_nearest_distance_bp": float(distances[nearest_pos]),
            "unswept_nearest_age_minutes": float(ages[nearest_pos]),
            "unswept_oldest_age_minutes": float(np.max(ages)),
            "unswept_median_age_minutes": float(np.median(ages)),
            "unswept_distance_q25_bp": float(np.quantile(distances, 0.25)),
            "unswept_distance_q50_bp": float(np.quantile(distances, 0.50)),
            "unswept_distance_q75_bp": float(np.quantile(distances, 0.75)),
            "unswept_timeframe_diversity": len({item[0].timeframe for item in relevant}),
            "unswept_max_pivot_range_bp": _finite_max([item[0].pivot_range_bp for item in relevant]),
            "unswept_max_pivot_notional_vs_past20": _finite_max(
                [item[0].pivot_notional_ratio for item in relevant]
            ),
            "unswept_max_level_available_time": max(item[0].available_time for item in relevant),
        }
    )
    for timeframe, _ in config.swing_timeframes:
        group = [item for item in relevant if item[0].timeframe == timeframe]
        row[f"unswept_{timeframe}_relevant_count"] = len(group)
        if group:
            group_dist = np.asarray([item[1] for item in group], dtype=float)
            group_age = np.asarray([item[2] for item in group], dtype=float)
            row[f"unswept_{timeframe}_nearest_distance_bp"] = float(np.min(group_dist))
            row[f"unswept_{timeframe}_oldest_age_minutes"] = float(np.max(group_age))
    for band in config.swing_confluence_bp:
        key = int(band) if float(band).is_integer() else str(band).replace(".", "p")
        group = [item for item in relevant if item[1] <= float(band)]
        row[f"unswept_confluence_count_{key}bp"] = len(group)
        row[f"unswept_confluence_timeframes_{key}bp"] = len({item[0].timeframe for item in group})
        if group:
            row[f"unswept_confluence_oldest_age_minutes_{key}bp"] = float(max(item[2] for item in group))
    return row


def attach_unswept_swing_inventory(
    event_features: pd.DataFrame,
    levels: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> pd.DataFrame:
    """Attach all-active-level inventory to each event in causal time order."""
    if event_features.empty:
        return event_features.copy()
    out = event_features.copy()
    out["event_time"] = as_datetime_ns(out["event_time"])
    out = out.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    if levels.empty:
        additions = pd.DataFrame([_empty_inventory_row(config) for _ in range(len(out))])
        return pd.concat([out, additions], axis=1)

    required = [name for name in _REQUIRED_LEVEL_COLUMNS if name not in levels]
    if required:
        raise ValueError(f"swing lifecycle missing columns: {required}")
    work = levels.copy()
    work["initial_available_time"] = as_datetime_ns(work["initial_available_time"])
    work["sweep_available_time"] = as_datetime_ns(work["sweep_available_time"], errors="coerce")
    records = [_record(row) for row in work.sort_values("initial_available_time", kind="mergesort").itertuples(index=False)]
    removals: list[tuple[int, int]] = []
    active: dict[int, _LevelRecord] = {}
    add_pos = 0
    rows: list[dict[str, object]] = []
    for event in out.itertuples(index=False):
        event_time = pd.Timestamp(event.event_time)
        while add_pos < len(records) and records[add_pos].available_time <= event_time:
            level = records[add_pos]
            active[level.level_id] = level
            if pd.notna(level.sweep_time):
                heapq.heappush(removals, (int(pd.Timestamp(level.sweep_time).value), level.level_id))
            add_pos += 1
        while removals and removals[0][0] <= int(event_time.value):
            _, level_id = heapq.heappop(removals)
            active.pop(level_id, None)
        anchor = float(getattr(event, "macro_pre_event_close", np.nan))
        if not np.isfinite(anchor):
            anchor = float(getattr(event, "pre_event_close", np.nan))
        rows.append(_inventory_features(active, event_time, str(event.event_side), anchor, config))
    additions = pd.DataFrame(rows)
    return pd.concat([out.reset_index(drop=True), additions], axis=1)
