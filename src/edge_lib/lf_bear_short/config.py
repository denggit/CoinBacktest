#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for ETH LF Bear Short V3."""

from __future__ import annotations

from dataclasses import dataclass

EDGE_ID = "ETH_EDGE_LF_BEAR_SHORT_V3"

PRESETS: dict[str, dict[str, float | int | str]] = {
    "scout": {"unit_risk_per_trade": 0.022, "max_total_notional_mult": 8.0, "max_units": 4, "max_risk_mult": 2.0, "style": "breakdown"},
    "stable": {"unit_risk_per_trade": 0.018, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "high": {"unit_risk_per_trade": 0.022, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "turbo": {"unit_risk_per_trade": 0.030, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "ultra": {"unit_risk_per_trade": 0.040, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
    "max": {"unit_risk_per_trade": 0.055, "max_total_notional_mult": 11.0, "max_units": 5, "max_risk_mult": 2.3, "style": "bear_permission_v3"},
}


@dataclass(frozen=True)
class BearConfig:
    symbol: str = "ETH-USDT-SWAP"
    initial_capital: float = 1000.0
    unit_risk_per_trade: float = 0.022
    max_total_notional_mult: float = 11.0
    max_units: int = 5
    min_risk_mult: float = 0.25
    max_risk_mult: float = 2.3
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    style: str = "crash_continuation"
    d1_ema_fast: int = 20
    d1_ema_mid: int = 50
    d1_ema_slow: int = 100
    d1_ema_major: int = 200
    d1_slope_lookback: int = 10
    w_ema_fast: int = 10
    w_ema_mid: int = 20
    w_ema_slow: int = 40
    w_slope_lookback: int = 4
    initial_atr_mult: float = 2.5
    trailing_atr_mult: float = 4.5
    add_every_r: float = 1.0
    max_hold_bars: int = 360
    cooldown_bars: int = 8

