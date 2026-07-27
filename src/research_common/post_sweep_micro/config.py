#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for R06 post-sweep micro turning-point research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepMicroConfig:
    micro_timeframe: str = "1s"
    pre_window_seconds: int = 60
    post_window_seconds: int = 660
    rolling_windows_seconds: tuple[int, ...] = (3, 5, 10, 15, 30)
    future_horizons_seconds: tuple[int, ...] = (30, 60, 180, 300)
    first_passage_barriers_bp: tuple[float, ...] = (10.0, 15.0, 25.0)
    range_pcts: tuple[float, ...] = (0.0015, 0.0020, 0.0025)
    range_lookback_minutes: int = 30
    range_lookforward_minutes: int = 15
    oracle_min_mfe_60m: float = 0.005
    control_multiplier: float = 1.0
    round_trip_cost: float = 0.0011
    sample_rows: int = 50_000

    def validate(self) -> "PostSweepMicroConfig":
        if self.micro_timeframe != "1s":
            raise ValueError("R06 currently requires micro_timeframe='1s' so trigger timing is comparable")
        if self.pre_window_seconds < 30:
            raise ValueError("pre_window_seconds must be >= 30")
        if self.post_window_seconds < max(self.future_horizons_seconds) + 60:
            raise ValueError("post_window_seconds must cover the attempt minute plus the largest future horizon")
        if any(v <= 0 for v in self.rolling_windows_seconds):
            raise ValueError("rolling windows must be positive")
        if any(v <= 0 for v in self.future_horizons_seconds):
            raise ValueError("future horizons must be positive")
        if any(v <= 0 for v in self.first_passage_barriers_bp):
            raise ValueError("barriers must be positive")
        if any(v <= 0 for v in self.range_pcts):
            raise ValueError("range_pcts must be positive")
        if self.oracle_min_mfe_60m <= 0:
            raise ValueError("oracle_min_mfe_60m must be positive")
        if self.control_multiplier < 0:
            raise ValueError("control_multiplier must be >= 0")
        if self.round_trip_cost < 0:
            raise ValueError("round_trip_cost must be >= 0")
        return self
