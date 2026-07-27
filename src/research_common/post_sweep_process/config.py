#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for R04 post-sweep continuation/exhaustion research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepConfig:
    """Predeclared mechanism-study settings.

    The values below define observation cadence and descriptive labels. They are
    not fitted from returns and are not final strategy thresholds.
    """

    observation_horizon_bars: int = 180
    dense_checkpoint_bars: int = 30
    fixed_checkpoint_bars: tuple[int, ...] = (45, 60, 90, 120, 180)
    flow_windows: tuple[int, ...] = (1, 3, 5, 15, 30)
    future_horizons: tuple[int, ...] = (5, 15, 30, 60, 180)
    micro_break_windows: tuple[int, ...] = (3, 5, 10)
    no_new_low_windows: tuple[int, ...] = (3, 5, 10, 15)
    large_mfe_returns: tuple[float, ...] = (0.005, 0.010, 0.020)
    reversal_mfe_return: float = 0.005
    continuation_mae_return: float = 0.005
    dominance_ratio: float = 1.5
    new_low_epsilon_bp: float = 0.0
    max_rows_per_event: int = 96
    sample_rows: int = 50_000

    def validate(self) -> "PostSweepConfig":
        if self.observation_horizon_bars < 5:
            raise ValueError("observation_horizon_bars must be >=5")
        if self.dense_checkpoint_bars < 1:
            raise ValueError("dense_checkpoint_bars must be positive")
        if not self.fixed_checkpoint_bars or any(int(v) < 1 for v in self.fixed_checkpoint_bars):
            raise ValueError("fixed_checkpoint_bars must be positive")
        if not self.flow_windows or any(int(v) < 1 for v in self.flow_windows):
            raise ValueError("flow_windows must be positive")
        if not self.future_horizons or any(int(v) < 1 for v in self.future_horizons):
            raise ValueError("future_horizons must be positive")
        if tuple(sorted(set(self.future_horizons))) != tuple(self.future_horizons):
            raise ValueError("future_horizons must be sorted unique")
        if max(self.future_horizons) > self.observation_horizon_bars:
            raise ValueError("future_horizons cannot exceed observation_horizon_bars")
        if not self.large_mfe_returns or any(float(v) <= 0 for v in self.large_mfe_returns):
            raise ValueError("large_mfe_returns must be positive")
        if self.reversal_mfe_return <= 0 or self.continuation_mae_return <= 0:
            raise ValueError("dominant path return thresholds must be positive")
        if self.dominance_ratio <= 1.0:
            raise ValueError("dominance_ratio must be >1")
        if self.new_low_epsilon_bp < 0:
            raise ValueError("new_low_epsilon_bp cannot be negative")
        if self.max_rows_per_event < 8:
            raise ValueError("max_rows_per_event must be >=8")
        return self
