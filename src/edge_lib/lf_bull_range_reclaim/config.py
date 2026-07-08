#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for ETH LF Bull Range Reclaim V2."""

from __future__ import annotations

from dataclasses import dataclass

EDGE_ID = "ETH_EDGE_LF_BULL_RANGE_RECLAIM_V2"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.012, "max_total_notional_mult": 5.0, "max_units": 2, "max_risk_mult": 1.6},
    "high": {"unit_risk_per_trade": 0.020, "max_total_notional_mult": 8.0, "max_units": 3, "max_risk_mult": 1.8},
    "turbo": {"unit_risk_per_trade": 0.026, "max_total_notional_mult": 9.0, "max_units": 3, "max_risk_mult": 1.9},
    "ultra": {"unit_risk_per_trade": 0.032, "max_total_notional_mult": 10.0, "max_units": 3, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class BullRangeConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.020
    max_total_notional_mult: float = 8.0
    max_units: int = 3
    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.8
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = False
    atr_period: int = 20
    adx_period: int = 14
    rsi_period: int = 14
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 100
    pullback_lookback: int = 8
    pb_dist50: float = 0.015
    pb_dist100: float = 0.030
    d1_close_mult: float = 0.980
    d1_fast_mult: float = 0.970
    d1_slope_min: float = 0.000
    d1_mid_vs_slow_min: float = 0.980
    d1_max_dist: float = 0.200
    d1_min_dist: float = -0.080
    reclaim_mult: float = 1.000
    rsi_min: float = 48.0
    adx_min: float = 6.0
    adx_max: float = 16.0
    atr_min: float = 0.003
    atr_max: float = 0.050
    vol_mult: float = 0.80
    h4_max_dist50: float = 0.080
    secondary_adx_max: float = 22.0
    secondary_rsi_min: float = 52.0
    secondary_quality_mult: float = 0.35
    exit_ema50_mult: float = 0.970
    initial_atr_mult: float = 2.2
    trailing_atr_mult: float = 3.5
    add_every_r: float = 1.2
    max_hold_bars: int = 90
    cooldown_bars: int = 4
    breakeven_after_r: float = 0.80
    breakeven_lock_r: float = 0.05
    lock_after_2r: float = 1.60
    lock_2r: float = 0.60
    lock_after_3r: float = 2.60
    lock_3r: float = 1.20
    no_progress_bars: int = 60
    no_progress_min_r: float = 0.50
    d1_ema_fast: int = 20
    d1_ema_mid: int = 50
    d1_ema_slow: int = 100
    d1_slope_lookback: int = 10

