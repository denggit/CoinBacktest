#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen design for R12 post-sweep rejection/acceptance research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepAcceptanceConfig:
    checkpoints_minutes: tuple[int, ...] = (1, 3, 5, 10)
    horizon_minutes: int = 180
    target_r_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)
    stop_buffer_bp: float = 5.0
    persistent_accept_share: float = 2.0 / 3.0
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00010
    stressed_cost_multiplier: float = 2.0
    minimum_spec_events: int = 300
    minimum_period_events: int = 75
    minimum_promote_events: int = 1_000
    research_pf_gate: float = 1.20
    promote_pf_gate: float = 1.40
    report_sample_rows: int = 50_000

    def validate(self) -> "PostSweepAcceptanceConfig":
        cps = tuple(int(v) for v in self.checkpoints_minutes)
        if not cps or any(v <= 0 for v in cps) or tuple(sorted(set(cps))) != cps:
            raise ValueError("checkpoints_minutes must be unique ascending positive integers")
        if self.horizon_minutes <= max(cps):
            raise ValueError("horizon_minutes must exceed the largest checkpoint")
        if not self.target_r_multiples or any(float(v) <= 0 for v in self.target_r_multiples):
            raise ValueError("target_r_multiples must be positive")
        if self.stop_buffer_bp <= 0:
            raise ValueError("stop_buffer_bp must be positive")
        if not 0.5 <= self.persistent_accept_share <= 1.0:
            raise ValueError("persistent_accept_share must be in [0.5, 1]")
        if min(self.fee_rate_per_side, self.slippage_rate_per_side) < 0:
            raise ValueError("cost rates cannot be negative")
        if self.stressed_cost_multiplier < 1.0:
            raise ValueError("stressed_cost_multiplier must be >= 1")
        if min(self.minimum_spec_events, self.minimum_period_events, self.minimum_promote_events) <= 0:
            raise ValueError("sample gates must be positive")
        if self.promote_pf_gate < self.research_pf_gate:
            raise ValueError("promote_pf_gate must be >= research_pf_gate")
        return self


STATE_ORDER = (
    "PRESSURE_TEST_REJECT",
    "STRONG_REJECT",
    "REJECT",
    "RECLAIM_FAILED",
    "PERSISTENT_ACCEPT",
    "MIXED_BELOW",
)


def state_direction(state: str) -> str:
    return "LONG" if state in {"PRESSURE_TEST_REJECT", "STRONG_REJECT", "REJECT"} else "SHORT" if state in {"RECLAIM_FAILED", "PERSISTENT_ACCEPT"} else "NONE"
