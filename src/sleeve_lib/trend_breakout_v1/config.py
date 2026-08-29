#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for the first executable ETH Portfolio V2 sleeve."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendBreakoutConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "15m"
    warmup_start_date: str = "2022-01-01"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-30 23:59:59"

    breakout_lookback: int = 48       # 12 hours on 15m bars
    stop_lookback: int = 16           # 4 hours
    atr_period: int = 32              # 8 hours
    ema_fast: int = 32                # 8 hours
    ema_slow: int = 96                # 24 hours

    initial_capital: float = 10_000.0
    base_risk_per_trade: float = 0.005
    max_notional_mult: float = 3.0
    fee_rate_per_side: float = 0.00055  # 0.11% round trip baseline
    slippage_pct: float = 0.00020
    min_stop_pct: float = 0.0025
    max_stop_pct: float = 0.0300
    target_r: float = 2.0
    cooldown_bars: int = 2

    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.00

    def validate(self) -> None:
        if self.breakout_lookback < 4 or self.stop_lookback < 2:
            raise ValueError("lookbacks are too short")
        if self.ema_fast <= 1 or self.ema_slow <= self.ema_fast:
            raise ValueError("ema_slow must be greater than ema_fast")
        if not 0 < self.base_risk_per_trade <= 0.02:
            raise ValueError("base_risk_per_trade must be in (0, 0.02]")
        if not 0 <= self.fee_rate_per_side < 0.01 or not 0 <= self.slippage_pct < 0.01:
            raise ValueError("cost assumptions are invalid")
        if not 0 < self.min_risk_mult <= self.max_risk_mult:
            raise ValueError("risk multiplier bounds are invalid")
