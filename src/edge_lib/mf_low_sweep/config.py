#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for ETH MF Low Sweep A0 Footprint time48."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

EDGE_ID = "ETH_EDGE_MF_LOW_SWEEP_A0_FOOTPRINT"
VARIANT_NAME = "MF_LOW_SWEEP_TIME48"
SLEEVE_ID = "ETH_SLEEVE_MF_LOW_SWEEP_V1"


@dataclass(frozen=True)
class StopSpec:
    name: str
    mode: str
    value: float | None = None


@dataclass(frozen=True)
class UpgradeVariant:
    variant_name: str
    candidate_layer: str
    support_mode: str
    entry_mode: str
    exit_mode: str
    stop_spec: StopSpec


@dataclass(frozen=True)
class MarketCache:
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    reclaim_vol_base: np.ndarray


def build_mf_args(args: Any) -> SimpleNamespace:
    fee = float(getattr(args, "fee_rate", 0.00055))
    slippage = float(getattr(args, "slippage", getattr(args, "slippage_pct", 0.00020)))
    return SimpleNamespace(
        symbol=args.symbol,
        timeframe="1m",
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        out_dir=getattr(args, "out_dir", None),
        pivot_left=6,
        pivot_right=3,
        min_swing_age=3,
        max_swing_ages="12,24,48",
        min_swing_prominence_pcts="0.0015,0.0030",
        spike_pcts="0.0060,0.0080,0.0100,0.0120",
        breakout_pcts="0.0000,0.0005",
        variants="fade_close_through",
        wick_min_frac=0.45,
        close_through_buffer_pct=0.0,
        volume_window=120,
        atr_window=42,
        cvd_window=60,
        cvd_windows="5,15,30,60,120",
        volume_spike_threshold=1.50,
        buy_ratio_thresholds="0.60,0.65,0.70",
        buy_pressure_thresholds="0.60,0.65,0.70",
        delta_pressure_thresholds="0.00,0.10,0.20",
        cvd_pressure_thresholds="0.00,0.05,0.10",
        rolling_quantile_days="30,90",
        rolling_quantiles="0.75,0.80",
        entry_fee_rate=fee,
        exit_fee_rate=fee,
        entry_slippage_pct=slippage,
        exit_slippage_pct=slippage,
        entry_delay_bars=1,
        starting_equity=1.0,
        progress_every=1000,
        no_progress=False,
        candidate_layers="A0_fp_abs_delta_high",
        support_modes="single_swing",
        entry_modes="next_open",
        exit_modes="time48",
        upgrade_stop_specs="no_stop",
        context_sources="trade_bar,footprint",
        micro_timeframes="",
        micro_load_mode="local",
        cluster_lookback_bars=1440,
        cluster_tolerance_pcts="0.0020,0.0030",
        aged_swing_min_age=60,
        soft_spike_pct=0.0060,
        soft_close_pos=0.35,
        limit_tick_size=0.01,
        reclaim_volume_window=20,
        reclaim_volume_mult=1.0,
        target_entry_same_bar_min_delay=1,
        footprint_range_pct=0.0020,
        footprint_price_step=1.0,
        fp_low_delta_vneg_threshold=-0.10,
        fp_abs_delta_high_threshold=0.60,
        range_delta_mild_neg_min=-0.35,
        range_delta_mild_neg_max=0.05,
        range_pcts="0.0015,0.0020,0.0025",
        micro_last_seconds=20,
        micro_buy_sell_ratio_min=1.20,
        micro_sell_exhaustion_delta_min=-0.15,
        micro_no_new_low_buffer_pct=0.0,
        micro_large_delta_min=0.0,
    )


def primary_variant() -> UpgradeVariant:
    return UpgradeVariant(
        variant_name=VARIANT_NAME,
        candidate_layer="A0_fp_abs_delta_high",
        support_mode="single_swing",
        entry_mode="next_open",
        exit_mode="time48",
        stop_spec=StopSpec(name="no_stop", mode="none", value=None),
    )

