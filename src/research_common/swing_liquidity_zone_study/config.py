#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for R03 Swing Liquidity Zone Sweep mechanism study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneStudyConfig:
    """Predeclared descriptive settings; none are selected from returns."""

    zone_merge_tolerance_bp: float = 10.0
    zone_merge_sensitivity_bp: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0)
    impulse_gap_bars: int = 5
    impulse_price_tolerance_bp: float = 50.0
    pre_windows: tuple[int, ...] = (5, 15, 60, 240, 1440)
    atr_windows: tuple[int, ...] = (60, 240, 1440)
    path_horizons: tuple[int, ...] = (5, 15, 30, 60, 180, 360, 720, 1440, 2880, 4320)
    tp_returns: tuple[float, ...] = (0.0025, 0.0050, 0.0100, 0.0200, 0.0300)
    structural_break_epsilon_bp: float = 0.01
    control_exclusion_bars: int = 5
    control_min_downside_atr: float = 0.25
    control_max_per_zone: int = 1

    def validate(self) -> "ZoneStudyConfig":
        if self.zone_merge_tolerance_bp <= 0:
            raise ValueError("zone_merge_tolerance_bp must be positive")
        if not self.zone_merge_sensitivity_bp or any(float(v) <= 0 for v in self.zone_merge_sensitivity_bp):
            raise ValueError("zone_merge_sensitivity_bp must be positive")
        if self.impulse_gap_bars < 0 or self.control_exclusion_bars < 0:
            raise ValueError("gap/exclusion bars cannot be negative")
        if self.impulse_price_tolerance_bp <= 0:
            raise ValueError("impulse_price_tolerance_bp must be positive")
        if not self.pre_windows or any(int(v) < 1 for v in self.pre_windows):
            raise ValueError("pre_windows must be positive")
        if not self.atr_windows or any(int(v) < 2 for v in self.atr_windows):
            raise ValueError("atr_windows must be >=2")
        if not self.path_horizons or any(int(v) < 1 for v in self.path_horizons):
            raise ValueError("path_horizons must be positive")
        if tuple(sorted(set(int(v) for v in self.path_horizons))) != tuple(self.path_horizons):
            raise ValueError("path_horizons must be sorted and unique")
        if not self.tp_returns or any(float(v) <= 0 for v in self.tp_returns):
            raise ValueError("tp_returns must be positive")
        if self.structural_break_epsilon_bp < 0:
            raise ValueError("structural_break_epsilon_bp cannot be negative")
        if self.control_min_downside_atr < 0:
            raise ValueError("control_min_downside_atr cannot be negative")
        if self.control_max_per_zone < 0:
            raise ValueError("control_max_per_zone cannot be negative")
        return self
