#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R11 liquidity-magnet and risk-frontier research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityMagnetConfig:
    """Predeclared R11 design.

    Distance checkpoints and stop models are deliberately sparse natural
    specifications, not an optimization grid.
    """

    distance_bands_bp: tuple[float, ...] = (150.0, 100.0, 50.0, 25.0)
    zone_merge_tolerance_bp: float = 10.0
    front_run_buffer_bp: float = 5.0
    local_high_buffer_bp: float = 5.0
    local_high_windows_minutes: tuple[int, ...] = (15, 60)
    horizon_minutes: int = 180
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00010
    stressed_cost_multiplier: float = 2.0
    minimum_spec_events: int = 300
    minimum_period_events: int = 75
    minimum_directional_target_rate: float = 0.52
    minimum_positive_periods: int = 3
    report_sample_rows: int = 50_000

    def validate(self) -> "LiquidityMagnetConfig":
        if not self.distance_bands_bp:
            raise ValueError("distance_bands_bp cannot be empty")
        bands = tuple(float(v) for v in self.distance_bands_bp)
        if any(v <= 0 for v in bands) or len(set(bands)) != len(bands):
            raise ValueError("distance_bands_bp must be unique positive values")
        if self.zone_merge_tolerance_bp <= 0:
            raise ValueError("zone_merge_tolerance_bp must be positive")
        if self.front_run_buffer_bp < 0 or self.local_high_buffer_bp < 0:
            raise ValueError("buffers cannot be negative")
        if not self.local_high_windows_minutes or any(int(v) <= 1 for v in self.local_high_windows_minutes):
            raise ValueError("local_high_windows_minutes must contain values > 1")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if min(self.fee_rate_per_side, self.slippage_rate_per_side) < 0:
            raise ValueError("cost rates cannot be negative")
        if self.stressed_cost_multiplier < 1:
            raise ValueError("stressed_cost_multiplier must be >= 1")
        if self.minimum_spec_events <= 0 or self.minimum_period_events <= 0:
            raise ValueError("sample gates must be positive")
        if not 0.5 <= self.minimum_directional_target_rate < 1:
            raise ValueError("minimum_directional_target_rate must be in [0.5, 1)")
        if not 1 <= self.minimum_positive_periods <= 3:
            raise ValueError("minimum_positive_periods must be 1..3")
        return self


def stop_model_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "stop_model": "EQUAL_DISTANCE",
            "description": "Stop above entry by the same distance as the front-run liquidity target below entry.",
            "causal_source": "entry price and pre-existing target only",
        },
        {
            "stop_model": "LOCAL_HIGH_15M",
            "description": "Stop above the highest completed 1m high in the preceding 15 minutes plus a fixed 5bp buffer.",
            "causal_source": "closed bars through the signal bar",
        },
        {
            "stop_model": "LOCAL_HIGH_60M",
            "description": "Stop above the highest completed 1m high in the preceding 60 minutes plus a fixed 5bp buffer.",
            "causal_source": "closed bars through the signal bar",
        },
    )
