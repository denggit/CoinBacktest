#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast next-open path, MFE/MAE and structural-survival outcomes for R03."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .config import ZoneStudyConfig

EPS = 1e-12


class RangeMinMaxIndex:
    """Iterative O(log n) range min/max queries."""

    def __init__(self, values: np.ndarray):
        arr = np.asarray(values, dtype=float)
        self.n = int(len(arr))
        size = 1
        while size < max(1, self.n):
            size <<= 1
        self.size = size
        self.min_tree = np.full(2 * size, np.inf, dtype=float)
        self.max_tree = np.full(2 * size, -np.inf, dtype=float)
        finite = np.isfinite(arr)
        self.min_tree[size : size + self.n] = np.where(finite, arr, np.inf)
        self.max_tree[size : size + self.n] = np.where(finite, arr, -np.inf)
        for node in range(size - 1, 0, -1):
            self.min_tree[node] = min(self.min_tree[node * 2], self.min_tree[node * 2 + 1])
            self.max_tree[node] = max(self.max_tree[node * 2], self.max_tree[node * 2 + 1])

    def query(self, start: int, end_inclusive: int) -> tuple[float, float]:
        if self.n == 0 or start > end_inclusive or start >= self.n or end_inclusive < 0:
            return np.nan, np.nan
        left = max(0, int(start)) + self.size
        right = min(self.n - 1, int(end_inclusive)) + self.size
        min_value = np.inf
        max_value = -np.inf
        while left <= right:
            if left & 1:
                min_value = min(min_value, self.min_tree[left])
                max_value = max(max_value, self.max_tree[left])
                left += 1
            if not (right & 1):
                min_value = min(min_value, self.min_tree[right])
                max_value = max(max_value, self.max_tree[right])
                right -= 1
            left >>= 1
            right >>= 1
        return (min_value if np.isfinite(min_value) else np.nan, max_value if np.isfinite(max_value) else np.nan)


def _token(value: float) -> str:
    return f"{float(value) * 100:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def attach_structural_path_outcomes(events: pd.DataFrame, primary: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    cfg = config.validate()
    if events.empty:
        return events.copy()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    close_index = SegmentThresholdIndex(close)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    out = events.copy().reset_index(drop=True)
    event_pos = pd.to_numeric(out["event_pos"], errors="raise").astype(np.int64).to_numpy()
    entry_pos = event_pos + 1
    valid = (entry_pos >= 0) & (entry_pos < len(bars))
    entry_price = np.full(len(out), np.nan, dtype=float)
    entry_price[valid] = open_[entry_pos[valid]]
    out["entry_reference_pos"] = np.where(valid, entry_pos, -1)
    out["entry_reference_time"] = pd.NaT
    if valid.any():
        out.loc[valid, "entry_reference_time"] = index[entry_pos[valid]].to_numpy()
    out["entry_reference_price"] = entry_price
    max_horizon = int(max(cfg.path_horizons))
    structural_reference = pd.to_numeric(out["sweep_low"], errors="coerce").to_numpy(dtype=float)
    break_threshold = structural_reference * (1.0 - float(cfg.structural_break_epsilon_bp) / 10_000.0)
    first_break = np.full(len(out), -1, dtype=np.int64)
    first_reclaim_floor = np.full(len(out), -1, dtype=np.int64)
    first_reclaim_ceiling = np.full(len(out), -1, dtype=np.int64)
    floor = pd.to_numeric(out.get("zone_floor_price", out["sweep_low"]), errors="coerce").to_numpy(dtype=float)
    ceiling = pd.to_numeric(out.get("zone_ceiling_price", floor), errors="coerce").to_numpy(dtype=float)
    for i in np.flatnonzero(valid):
        start = int(entry_pos[i])
        end = min(len(bars) - 1, start + max_horizon - 1)
        first_break[i] = low_index.first_leq(start, end, float(break_threshold[i]))
        first_reclaim_floor[i] = close_index.first_geq(start, end, float(floor[i]))
        first_reclaim_ceiling[i] = close_index.first_geq(start, end, float(ceiling[i]))
    out["first_lower_low_pos"] = first_break
    out["bars_to_lower_low"] = np.where(first_break >= 0, first_break - entry_pos + 1, np.nan)
    out["first_zone_floor_reclaim_pos"] = first_reclaim_floor
    out["bars_to_zone_floor_reclaim"] = np.where(first_reclaim_floor >= 0, first_reclaim_floor - entry_pos + 1, np.nan)
    out["first_zone_ceiling_reclaim_pos"] = first_reclaim_ceiling
    out["bars_to_zone_ceiling_reclaim"] = np.where(first_reclaim_ceiling >= 0, first_reclaim_ceiling - entry_pos + 1, np.nan)

    for horizon in cfg.path_horizons:
        h = int(horizon)
        terminal = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)
        survive = np.zeros(len(out), dtype=bool)
        reclaim_floor_by = np.zeros(len(out), dtype=bool)
        reclaim_ceiling_by = np.zeros(len(out), dtype=bool)
        for i in np.flatnonzero(valid):
            start = int(entry_pos[i])
            end = min(len(bars) - 1, start + h - 1)
            price = float(entry_price[i])
            if not np.isfinite(price) or price <= 0:
                continue
            low_value, _ = low_range.query(start, end)
            _, high_value = high_range.query(start, end)
            terminal[i] = close[end] / price - 1.0
            mfe[i] = high_value / price - 1.0 if np.isfinite(high_value) else np.nan
            mae[i] = low_value / price - 1.0 if np.isfinite(low_value) else np.nan
            survive[i] = first_break[i] < 0 or first_break[i] > end
            reclaim_floor_by[i] = first_reclaim_floor[i] >= 0 and first_reclaim_floor[i] <= end
            reclaim_ceiling_by[i] = first_reclaim_ceiling[i] >= 0 and first_reclaim_ceiling[i] <= end
        out[f"close_return_{h}m"] = terminal
        out[f"mfe_high_{h}m"] = mfe
        out[f"mae_low_{h}m"] = mae
        out[f"structural_low_survival_{h}m"] = survive
        out[f"zone_floor_reclaim_by_{h}m"] = reclaim_floor_by
        out[f"zone_ceiling_reclaim_by_{h}m"] = reclaim_ceiling_by

    mfe_before_break = np.full(len(out), np.nan, dtype=float)
    return_before_break = np.full(len(out), np.nan, dtype=float)
    for i in np.flatnonzero(valid):
        start = int(entry_pos[i])
        end_max = min(len(bars) - 1, start + max_horizon - 1)
        path_end = end_max if first_break[i] < 0 else max(start, int(first_break[i]) - 1)
        _, high_value = high_range.query(start, path_end)
        price = float(entry_price[i])
        if np.isfinite(high_value) and price > 0:
            mfe_before_break[i] = high_value / price - 1.0
        return_before_break[i] = close[path_end] / price - 1.0 if price > 0 else np.nan
    out[f"mfe_before_lower_low_{max_horizon}m"] = mfe_before_break
    out[f"close_return_before_lower_low_or_{max_horizon}m"] = return_before_break

    for target in cfg.tp_returns:
        token = _token(float(target))
        first_tp = np.full(len(out), -1, dtype=np.int64)
        before_break = np.zeros(len(out), dtype=bool)
        mae_before_tp = np.full(len(out), np.nan, dtype=float)
        for i in np.flatnonzero(valid):
            start = int(entry_pos[i])
            end = min(len(bars) - 1, start + max_horizon - 1)
            price = float(entry_price[i])
            if not np.isfinite(price) or price <= 0:
                continue
            pos = high_index.first_geq(start, end, price * (1.0 + float(target)))
            first_tp[i] = pos
            before_break[i] = pos >= 0 and (first_break[i] < 0 or pos < first_break[i])
            if pos >= 0:
                low_value, _ = low_range.query(start, pos)
                mae_before_tp[i] = low_value / price - 1.0 if np.isfinite(low_value) else np.nan
        out[f"first_tp_{token}_pos"] = first_tp
        out[f"bars_to_tp_{token}"] = np.where(first_tp >= 0, first_tp - entry_pos + 1, np.nan)
        out[f"tp_{token}_before_lower_low_{max_horizon}m"] = before_break
        out[f"mae_before_tp_{token}"] = mae_before_tp
    return out
