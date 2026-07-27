#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal derived features for Binance futures metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .models import parse_timedelta, window_column_tag


def add_derived_ratio_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    taker_ratio = pd.to_numeric(out["sum_taker_long_short_vol_ratio"], errors="coerce")
    out["taker_volume_imbalance"] = (taker_ratio - 1.0) / (taker_ratio + 1.0)
    ratio_map = {
        "count_toptrader_long_short_ratio": "top_trader_account_long_share",
        "sum_toptrader_long_short_ratio": "top_trader_position_long_share",
        "count_long_short_ratio": "global_account_long_share",
    }
    for source, target in ratio_map.items():
        values = pd.to_numeric(out[source], errors="coerce")
        out[target] = values / (1.0 + values)
    return out


def build_relative_features(
    raw: pd.DataFrame,
    *,
    windows: Sequence[str | pd.Timedelta],
    baseline_tolerance: str | pd.Timedelta,
) -> pd.DataFrame:
    out = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True).copy()
    times = pd.to_datetime(out["timestamp"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    oi_base = pd.to_numeric(out["sum_open_interest"], errors="coerce").to_numpy(dtype="float64")
    oi_usd = pd.to_numeric(out["sum_open_interest_value"], errors="coerce").to_numpy(dtype="float64")
    tolerance_ns = int(parse_timedelta(baseline_tolerance).value)

    for value in windows:
        window = parse_timedelta(value)
        tag = window_column_tag(window)
        target = times - int(window.value)
        positions = np.searchsorted(times, target, side="right") - 1
        valid = positions >= 0
        safe_positions = np.where(valid, positions, 0)
        baseline_times = times[safe_positions]
        valid &= baseline_times >= target - tolerance_ns

        base_prior = np.full(len(out), np.nan, dtype="float64")
        usd_prior = np.full(len(out), np.nan, dtype="float64")
        base_prior[valid] = oi_base[safe_positions[valid]]
        usd_prior[valid] = oi_usd[safe_positions[valid]]
        out[f"oi_base_change_{tag}"] = _safe_relative_change(oi_base, base_prior)
        out[f"oi_usd_change_{tag}"] = _safe_relative_change(oi_usd, usd_prior)
        out[f"oi_baseline_age_seconds_{tag}"] = np.where(
            valid,
            (times - baseline_times) / 1_000_000_000.0,
            np.nan,
        )
    return out


def set_index_mode(frame: pd.DataFrame, *, index_mode: str) -> pd.DataFrame:
    mode = str(index_mode).strip().lower()
    if mode == "none":
        return frame.reset_index(drop=True)
    if mode not in {"timestamp", "available_time"}:
        raise ValueError("index_mode must be timestamp/available_time/none")
    out = frame.copy()
    out[mode] = pd.to_datetime(out[mode])
    return out.set_index(mode, drop=False).sort_index()


def _safe_relative_change(current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    out = np.full(len(current), np.nan, dtype="float64")
    valid = np.isfinite(current) & np.isfinite(baseline) & (baseline != 0.0)
    out[valid] = current[valid] / baseline[valid] - 1.0
    return out
