#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed, non-optimized configuration for R08 staged execution research."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StageSpec:
    name: str
    weight: float
    trigger: str
    max_signal_elapsed: int

@dataclass(frozen=True)
class SchemeSpec:
    name: str
    stages: tuple[StageSpec, ...]

@dataclass(frozen=True)
class PostSweepStagedExecutionConfig:
    max_deployment_minutes: int = 60
    horizons: tuple[int, ...] = (15, 30, 60, 180)
    fee_rate_per_side: float = 0.00055
    slippage_rate_per_side: float = 0.00010
    stressed_slippage_rate_per_side: float = 0.00020
    report_sample_rows: int = 50_000

    def validate(self) -> "PostSweepStagedExecutionConfig":
        if self.max_deployment_minutes <= 0:
            raise ValueError("max_deployment_minutes must be positive")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be positive")
        if self.fee_rate_per_side < 0 or self.slippage_rate_per_side < 0:
            raise ValueError("cost rates cannot be negative")
        for scheme in scheme_specs():
            total = sum(stage.weight for stage in scheme.stages)
            if abs(total - 1.0) > 1e-12:
                raise ValueError(f"scheme {scheme.name} weights sum to {total}, expected 1")
        return self


def scheme_specs() -> tuple[SchemeSpec, ...]:
    """Frozen natural execution schemes; no parameter grid is used."""
    return (
        SchemeSpec(
            "FULL_FIRST_CHECKPOINT",
            (StageSpec("initial", 1.00, "INITIAL", 1),),
        ),
        SchemeSpec(
            "PROBE30_EARLY70",
            (
                StageSpec("probe", 0.30, "INITIAL", 1),
                StageSpec("main", 0.70, "EARLY_REJECTION", 30),
            ),
        ),
        SchemeSpec(
            "PROBE25_EARLY50_CONFIRMED25",
            (
                StageSpec("probe", 0.25, "INITIAL", 1),
                StageSpec("main", 0.50, "EARLY_REJECTION", 30),
                StageSpec("runner", 0.25, "CONFIRMED_REJECTION", 45),
            ),
        ),
        SchemeSpec(
            "PROBE25_CONFIRMED75",
            (
                StageSpec("probe", 0.25, "INITIAL", 1),
                StageSpec("main", 0.75, "CONFIRMED_REJECTION", 45),
            ),
        ),
        SchemeSpec(
            "PROBE20_EARLY40_STRONG40",
            (
                StageSpec("probe", 0.20, "INITIAL", 1),
                StageSpec("main", 0.40, "EARLY_REJECTION", 30),
                StageSpec("runner", 0.40, "STRONG_RECLAIM", 60),
            ),
        ),
        SchemeSpec(
            "WAIT_CONFIRMED100_DIAGNOSTIC",
            (StageSpec("main", 1.00, "CONFIRMED_REJECTION", 45),),
        ),
    )
