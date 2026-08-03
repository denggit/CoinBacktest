#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R10 structured pullback-entry research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PullbackTargetSpec:
    """One predeclared exit target; this is not a parameter grid."""

    name: str
    mode: str
    r_multiple: float | None = None


@dataclass(frozen=True)
class StructuredPullbackConfig:
    """Causal, non-optimized research settings for R10."""

    timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    )
    stop_buffer_bp: float = 5.0
    confluence_tolerance_bp: float = 10.0
    minimum_holding_minutes: int = 180
    maximum_holding_minutes: int = 4_320
    holding_timeframe_multiplier: int = 6
    fee_round_trip: float = 0.0011
    slippage_round_trip: float = 0.0002
    stressed_cost_multiplier: float = 2.0
    minimum_family_candidates: int = 100
    minimum_family_fills: int = 100
    minimum_period_fills: int = 30
    report_sample_rows: int = 50_000

    def validate(self) -> "StructuredPullbackConfig":
        if not self.timeframes:
            raise ValueError("at least one timeframe is required")
        minutes = [int(value) for _, value in self.timeframes]
        if any(value <= 0 for value in minutes) or len(set(minutes)) != len(minutes):
            raise ValueError("timeframes must contain unique positive minutes")
        if self.stop_buffer_bp < 0:
            raise ValueError("stop_buffer_bp cannot be negative")
        if self.confluence_tolerance_bp <= 0:
            raise ValueError("confluence_tolerance_bp must be positive")
        if self.minimum_holding_minutes <= 0:
            raise ValueError("minimum_holding_minutes must be positive")
        if self.maximum_holding_minutes < self.minimum_holding_minutes:
            raise ValueError("maximum_holding_minutes must be >= minimum_holding_minutes")
        if self.holding_timeframe_multiplier <= 0:
            raise ValueError("holding_timeframe_multiplier must be positive")
        if min(self.fee_round_trip, self.slippage_round_trip) < 0:
            raise ValueError("cost assumptions cannot be negative")
        if self.stressed_cost_multiplier < 1:
            raise ValueError("stressed_cost_multiplier must be >= 1")
        if min(
            self.minimum_family_candidates,
            self.minimum_family_fills,
            self.minimum_period_fills,
            self.report_sample_rows,
        ) <= 0:
            raise ValueError("sample/report gates must be positive")
        return self

    def holding_minutes(self, source_timeframe_min: int) -> int:
        raw = int(source_timeframe_min) * int(self.holding_timeframe_multiplier)
        return max(
            int(self.minimum_holding_minutes),
            min(int(self.maximum_holding_minutes), raw),
        )

    @property
    def realistic_round_trip_cost(self) -> float:
        return float(self.fee_round_trip + self.slippage_round_trip)

    @property
    def stressed_round_trip_cost(self) -> float:
        return float(self.realistic_round_trip_cost * self.stressed_cost_multiplier)


def target_specs() -> tuple[PullbackTargetSpec, ...]:
    """Natural structural/R targets declared before seeing R10 outcomes."""

    return (
        PullbackTargetSpec("H0", mode="STRUCTURAL_H0"),
        PullbackTargetSpec("R1", mode="R_MULTIPLE", r_multiple=1.0),
        PullbackTargetSpec("R2", mode="R_MULTIPLE", r_multiple=2.0),
        PullbackTargetSpec("R3", mode="R_MULTIPLE", r_multiple=3.0),
    )
