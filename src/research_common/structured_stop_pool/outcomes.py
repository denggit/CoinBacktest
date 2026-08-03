#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative next-open first-touch payoff outcomes for R09."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .config import StructuredStopPoolConfig, first_touch_specs


def attach_first_touch_outcomes(events: pd.DataFrame, primary: pd.DataFrame, config: StructuredStopPoolConfig) -> pd.DataFrame:
    cfg = config.validate()
    if events.empty:
        return events.copy()
    bars = normalize_primary_bars(primary)
    out = events.copy().reset_index(drop=True)
    event_pos = pd.to_numeric(out["event_pos"], errors="raise").astype(np.int64).to_numpy()
    entry_pos = event_pos + 1
    n = len(bars)
    valid = (entry_pos >= 0) & (entry_pos < n)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars.index)
    entry_price = np.full(len(out), np.nan, dtype=float)
    entry_price[valid] = open_[entry_pos[valid]]
    out["r09_entry_pos"] = np.where(valid, entry_pos, -1)
    out["r09_entry_time"] = pd.NaT
    if valid.any():
        out.loc[valid, "r09_entry_time"] = index[entry_pos[valid]].to_numpy()
    out["r09_entry_price"] = entry_price
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    one_x_cost = 2.0 * (float(cfg.fee_rate_per_side) + float(cfg.slippage_rate_per_side))
    stressed_cost = one_x_cost * float(cfg.stressed_cost_multiplier)

    for spec in first_touch_specs():
        token = spec.name.lower()
        first_tp = np.full(len(out), -1, dtype=np.int64)
        first_sl = np.full(len(out), -1, dtype=np.int64)
        exit_pos = np.full(len(out), -1, dtype=np.int64)
        gross = np.full(len(out), np.nan, dtype=float)
        outcome = np.full(len(out), "INVALID", dtype=object)
        same_bar_both = np.zeros(len(out), dtype=bool)
        for i in np.flatnonzero(valid):
            start = int(entry_pos[i])
            end = min(n - 1, start + int(spec.horizon_minutes) - 1)
            price = float(entry_price[i])
            if not np.isfinite(price) or price <= 0:
                continue
            tp_pos = high_index.first_geq(start, end, price * (1.0 + float(spec.tp_bp) / 10_000.0))
            sl_pos = low_index.first_leq(start, end, price * (1.0 - float(spec.sl_bp) / 10_000.0))
            first_tp[i] = tp_pos
            first_sl[i] = sl_pos
            if tp_pos >= 0 and sl_pos >= 0 and tp_pos == sl_pos:
                same_bar_both[i] = True
                exit_pos[i] = sl_pos
                gross[i] = -float(spec.sl_bp) / 10_000.0
                outcome[i] = "SL_CONSERVATIVE_SAME_BAR"
            elif sl_pos >= 0 and (tp_pos < 0 or sl_pos < tp_pos):
                exit_pos[i] = sl_pos
                gross[i] = -float(spec.sl_bp) / 10_000.0
                outcome[i] = "SL"
            elif tp_pos >= 0:
                exit_pos[i] = tp_pos
                gross[i] = float(spec.tp_bp) / 10_000.0
                outcome[i] = "TP"
            else:
                exit_pos[i] = end
                gross[i] = close[end] / price - 1.0
                outcome[i] = "TIME"
        out[f"{token}_first_tp_pos"] = first_tp
        out[f"{token}_first_sl_pos"] = first_sl
        out[f"{token}_exit_pos"] = exit_pos
        out[f"{token}_outcome"] = outcome
        out[f"{token}_same_bar_both_flag"] = same_bar_both
        out[f"{token}_tp_before_sl"] = np.isin(outcome, ["TP"])
        out[f"{token}_gross_return"] = gross
        out[f"{token}_net_return_1x_cost"] = gross - one_x_cost
        out[f"{token}_net_return_2x_cost"] = gross - stressed_cost
        out[f"{token}_bars_to_exit"] = np.where(exit_pos >= 0, exit_pos - entry_pos + 1, np.nan)
        out[f"{token}_exit_time"] = pd.NaT
        mask = exit_pos >= 0
        if mask.any():
            out.loc[mask, f"{token}_exit_time"] = index[exit_pos[mask]].to_numpy()
    return out
