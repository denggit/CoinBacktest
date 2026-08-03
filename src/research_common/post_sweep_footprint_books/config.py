#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for R07 footprint + order-book absorption research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepFootprintBooksConfig:
    """Predeclared R07 mechanism-study settings.

    These values define data windows and natural market-structure neighborhoods;
    they are not searched for the best historical PnL.
    """

    range_pct: float = 0.0020
    footprint_price_step: float = 1.0
    footprint_chunk_days: int = 120
    footprint_lag_bars: int = 3
    low_zone_bins: tuple[int, ...] = (1, 3, 5)
    stacked_sell_ratio: float = 3.0
    books_depth: int = 5000
    books_lookback_seconds: int = 60
    books_max_staleness_seconds: int = 15
    books_min_valid_fraction: float = 0.50
    frozen_reference_period: str = "EARLY_2023_2024"
    sample_rows: int = 50_000
    minimum_period_events: int = 100

    def validate(self) -> "PostSweepFootprintBooksConfig":
        if self.range_pct <= 0:
            raise ValueError("range_pct must be > 0")
        if self.footprint_price_step <= 0:
            raise ValueError("footprint_price_step must be > 0")
        if self.footprint_chunk_days <= 0:
            raise ValueError("footprint_chunk_days must be > 0")
        if self.footprint_lag_bars < 1:
            raise ValueError("footprint_lag_bars must be >= 1")
        if not self.low_zone_bins or any(int(v) <= 0 for v in self.low_zone_bins):
            raise ValueError("low_zone_bins must contain positive integers")
        if tuple(sorted(set(int(v) for v in self.low_zone_bins))) != tuple(self.low_zone_bins):
            raise ValueError("low_zone_bins must be unique and sorted")
        if 3 not in self.low_zone_bins:
            raise ValueError("low_zone_bins must include 3 because R07's frozen feature contract uses low3")
        if self.stacked_sell_ratio <= 1:
            raise ValueError("stacked_sell_ratio must be > 1")
        if self.books_depth <= 0:
            raise ValueError("books_depth must be > 0")
        if self.books_lookback_seconds <= 0:
            raise ValueError("books_lookback_seconds must be > 0")
        if self.books_max_staleness_seconds < 0:
            raise ValueError("books_max_staleness_seconds must be >= 0")
        if not 0 <= self.books_min_valid_fraction <= 1:
            raise ValueError("books_min_valid_fraction must be in [0, 1]")
        if self.sample_rows <= 0:
            raise ValueError("sample_rows must be > 0")
        if self.minimum_period_events <= 0:
            raise ValueError("minimum_period_events must be > 0")
        return self
