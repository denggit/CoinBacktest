#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal next-open forward close-path labels for atlas events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AtlasConfig
from .pivots import normalize_primary_bars


def attach_forward_paths(events: pd.DataFrame, primary: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    cfg = config.validate()
    if events.empty:
        return events.copy()
    bars = normalize_primary_bars(primary)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars.index)
    out = events.copy()
    positions = pd.to_numeric(out["event_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    entry_pos = positions + 1
    valid_entry = (entry_pos >= 0) & (entry_pos < len(bars))
    entry_price = np.full(len(out), np.nan, dtype=float)
    entry_price[valid_entry] = open_[entry_pos[valid_entry]]
    out["entry_reference_pos"] = np.where(valid_entry, entry_pos, -1)
    out["entry_reference_time"] = pd.NaT
    if valid_entry.any():
        out.loc[valid_entry, "entry_reference_time"] = index[entry_pos[valid_entry]].to_numpy()
    out["entry_reference_price"] = entry_price

    for horizon in cfg.forward_horizons:
        h = int(horizon)
        ret = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)
        terminal = np.full(len(out), np.nan, dtype=float)
        for i in np.flatnonzero(valid_entry):
            start = int(entry_pos[i])
            end = min(len(close), start + h)
            path = close[start:end]
            path = path[np.isfinite(path)]
            price = entry_price[i]
            if path.size == 0 or not np.isfinite(price) or price <= 0:
                continue
            returns = path / price - 1.0
            terminal[i] = returns[-1]
            ret[i] = returns[-1]
            mfe[i] = float(np.max(returns))
            mae[i] = float(np.min(returns))
        out[f"close_return_{h}m"] = ret
        out[f"mfe_close_{h}m"] = mfe
        out[f"mae_close_{h}m"] = mae
        out[f"hit_up_0p25_{h}m"] = mfe >= 0.0025
        out[f"hit_up_0p50_{h}m"] = mfe >= 0.0050
        out[f"hit_up_1p00_{h}m"] = mfe >= 0.0100
    return out
