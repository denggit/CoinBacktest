#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen, non-optimized configuration for R09 structured stop-pool research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FirstTouchSpec:
    name: str
    tp_bp: float
    sl_bp: float
    horizon_minutes: int = 180


@dataclass(frozen=True)
class StructuredStopPoolConfig:
    timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    )
    frozen_train_end: str = "2025-01-01"
    atr_window_htf: int = 20
    equal_low_tolerance_atr: float = 0.25
    h1_decline_quantile: float = 0.50
    h1_rebound_quantile: float = 0.50
    h4_displacement_quantile: float = 0.75
    zone_merge_tolerance_bp: float = 10.0
    impulse_gap_bars: int = 5
    impulse_price_tolerance_bp: float = 50.0
    release_baseline_minutes: int = 60
    release_long_baseline_minutes: int = 240
    release_windows_minutes: tuple[int, ...] = (1, 5, 15)
    release_score_quantile: float = 0.75
    path_horizons: tuple[int, ...] = (15, 60, 180)
    tp_returns: tuple[float, ...] = (0.0015, 0.0025, 0.0050, 0.0100)
    structural_break_epsilon_bp: float = 0.01
    control_exclusion_bars: int = 5
    control_min_downside_atr: float = 0.25
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00010
    stressed_cost_multiplier: float = 2.0
    report_sample_rows: int = 50_000
    minimum_family_events: int = 100
    minimum_period_events: int = 30
    release_lift_gate_pp: float = 5.0
    reversal_lift_gate_pp: float = 5.0

    def validate(self) -> "StructuredStopPoolConfig":
        if not self.timeframes:
            raise ValueError("at least one timeframe is required")
        minutes = [int(v) for _, v in self.timeframes]
        if any(v <= 0 for v in minutes) or len(set(minutes)) != len(minutes):
            raise ValueError("timeframes must contain unique positive minutes")
        if self.atr_window_htf < 5:
            raise ValueError("atr_window_htf must be >= 5")
        if self.equal_low_tolerance_atr <= 0:
            raise ValueError("equal_low_tolerance_atr must be positive")
        for name, q in (
            ("h1_decline_quantile", self.h1_decline_quantile),
            ("h1_rebound_quantile", self.h1_rebound_quantile),
            ("h4_displacement_quantile", self.h4_displacement_quantile),
            ("release_score_quantile", self.release_score_quantile),
        ):
            if not 0 < float(q) < 1:
                raise ValueError(f"{name} must be inside (0, 1)")
        if self.zone_merge_tolerance_bp <= 0 or self.impulse_gap_bars < 0:
            raise ValueError("zone/impulse settings are invalid")
        if self.release_baseline_minutes < 20 or self.release_long_baseline_minutes < self.release_baseline_minutes:
            raise ValueError("release baselines are invalid")
        if any(int(v) <= 0 for v in self.release_windows_minutes):
            raise ValueError("release windows must be positive")
        if any(int(v) <= 0 for v in self.path_horizons):
            raise ValueError("path horizons must be positive")
        if any(float(v) <= 0 for v in self.tp_returns):
            raise ValueError("tp_returns must be positive")
        if min(self.fee_rate_per_side, self.slippage_rate_per_side) < 0:
            raise ValueError("cost rates cannot be negative")
        if self.stressed_cost_multiplier < 1:
            raise ValueError("stressed_cost_multiplier must be >= 1")
        if self.minimum_family_events <= 0 or self.minimum_period_events <= 0:
            raise ValueError("minimum sample gates must be positive")
        return self


def first_touch_specs() -> tuple[FirstTouchSpec, ...]:
    """Natural predeclared payoff shapes; this is not a parameter grid."""
    return (
        FirstTouchSpec("TP15_SL15", tp_bp=15.0, sl_bp=15.0),
        FirstTouchSpec("TP25_SL15", tp_bp=25.0, sl_bp=15.0),
        FirstTouchSpec("TP50_SL25", tp_bp=50.0, sl_bp=25.0),
    )
