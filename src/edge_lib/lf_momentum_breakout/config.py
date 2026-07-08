#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for ETH LF Momentum Breakout V3."""

from __future__ import annotations

from dataclasses import dataclass

EDGE_ID = "ETH_EDGE_LF_MOMENTUM_BREAKOUT_V3"

PRESETS: dict[str, dict[str, float | int]] = {
    "stable": {"unit_risk_per_trade": 0.020, "max_total_notional_mult": 10.0, "max_units": 4, "max_risk_mult": 2.0},
    "high": {"unit_risk_per_trade": 0.026, "max_total_notional_mult": 11.0, "max_units": 4, "max_risk_mult": 2.0},
    "turbo": {"unit_risk_per_trade": 0.032, "max_total_notional_mult": 12.0, "max_units": 4, "max_risk_mult": 2.0},
    "ultra": {"unit_risk_per_trade": 0.040, "max_total_notional_mult": 12.0, "max_units": 4, "max_risk_mult": 2.0},
}


@dataclass(frozen=True)
class MomentumConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.020
    max_total_notional_mult: float = 10.0
    max_units: int = 4
    min_risk_mult: float = 0.35
    max_risk_mult: float = 2.0
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    enable_short: bool = True
    entry_lookback: int = 12
    exit_lookback: int = 12
    atr_period: int = 20
    adx_period: int = 14
    min_adx_long: float = 10.0
    min_adx_short: float = 16.0
    max_adx_long: float = 38.0
    max_adx_short: float = 42.0
    min_atr_pct: float = 0.0030
    max_atr_pct: float = 0.0700
    volume_window: int = 60
    volume_mult: float = 1.05
    max_d1_distance_long: float = 0.120
    max_d1_distance_short: float = 0.140
    initial_atr_mult: float = 2.2
    trailing_atr_mult: float = 4.0
    add_every_r: float = 1.0
    max_hold_bars: int = 180
    cooldown_bars: int = 4
    breakeven_after_r: float = 1.0
    breakeven_lock_r: float = 0.10
    lock_after_2r: float = 1.7
    lock_2r: float = 0.70
    lock_after_3r: float = 2.8
    lock_3r: float = 1.50
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_slope_min: float = -0.0300
    bear_slope_max: float = -0.0030
    w_ema_fast: int = 10
    w_ema_mid: int = 20
    w_slope_lookback: int = 4
    weak_long_quality_mult: float = 0.25
    mature_long_adx_threshold: float = 16.0
    mature_long_quality_mult: float = 0.50

