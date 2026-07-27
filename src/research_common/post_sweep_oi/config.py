#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for R05 Binance OI post-sweep mechanism research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepOIConfig:
    """Predeclared OI alignment and descriptive-study settings.

    None of these values are fitted from returns. Binance metrics are 5-minute
    interval-end observations; ``publication_lag`` makes their use causal.
    """

    oi_windows: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h", "1d")
    future_oi_horizons: tuple[int, ...] = (15, 30, 60, 180)
    publication_lag: str = "1min"
    baseline_tolerance: str = "1min"
    alignment_tolerance: str = "10min"
    large_mfe_returns: tuple[float, ...] = (0.005, 0.010, 0.020)
    sample_rows: int = 20_000

    def validate(self) -> "PostSweepOIConfig":
        import pandas as pd

        if not self.oi_windows:
            raise ValueError("oi_windows must not be empty")
        if not self.future_oi_horizons or any(int(v) <= 0 for v in self.future_oi_horizons):
            raise ValueError("future_oi_horizons must be positive")
        if tuple(sorted(set(self.future_oi_horizons))) != tuple(self.future_oi_horizons):
            raise ValueError("future_oi_horizons must be sorted unique")
        if any(float(v) <= 0 for v in self.large_mfe_returns):
            raise ValueError("large_mfe_returns must be positive")
        if pd.Timedelta(self.publication_lag) < pd.Timedelta(0):
            raise ValueError("publication_lag must be non-negative")
        if pd.Timedelta(self.baseline_tolerance) < pd.Timedelta(0):
            raise ValueError("baseline_tolerance must be non-negative")
        if pd.Timedelta(self.alignment_tolerance) < pd.Timedelta("5min"):
            raise ValueError("alignment_tolerance must be at least one metrics interval")
        if self.sample_rows < 100:
            raise ValueError("sample_rows must be >=100")
        return self
