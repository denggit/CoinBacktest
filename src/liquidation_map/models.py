#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Models for the transparent estimated liquidation heatmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LiquidationMapConfig:
    leverage_buckets: tuple[int, ...] = (5, 10, 20, 50)
    leverage_weights: tuple[float, ...] = (0.16, 0.34, 0.34, 0.16)
    maintenance_margin_rate: float = 0.005
    liquidation_fee_buffer: float = 0.002
    price_bucket_pct: float = 0.0025
    max_distance_pct: float = 0.12
    snapshot_every_bars: int = 5
    cohort_half_life_hours: float = 72.0
    crossed_level_survival: float = 0.15
    oi_reduction_cap: float = 0.70
    minimum_oi_delta_usd: float = 100_000.0
    max_cells_per_snapshot: int = 60
    top_zone_count: int = 3

    def validate(self) -> None:
        if not self.leverage_buckets or len(self.leverage_buckets) != len(self.leverage_weights):
            raise ValueError("leverage buckets and weights must be non-empty and equal length")
        if any(item <= 1 for item in self.leverage_buckets):
            raise ValueError("leverage must be > 1")
        if any(item < 0 for item in self.leverage_weights) or sum(self.leverage_weights) <= 0:
            raise ValueError("invalid leverage weights")
        if self.price_bucket_pct <= 0 or self.max_distance_pct <= 0:
            raise ValueError("price bucket and distance must be positive")
        if self.snapshot_every_bars <= 0 or self.max_cells_per_snapshot <= 0:
            raise ValueError("snapshot settings must be positive")


@dataclass(frozen=True)
class HeatmapCell:
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    price_low: float
    price_high: float
    intensity: float
    raw_notional: float
    side: str
    confidence: float
    label: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiquidationZone:
    side: str
    price_low: float
    price_high: float
    center_price: float
    raw_notional: float
    intensity: float
    distance_pct: float
    confidence: float


@dataclass(frozen=True)
class LiquidationMapResult:
    cells: list[HeatmapCell]
    row_frame: pd.DataFrame
    current_zones: list[LiquidationZone]
    diagnostics: dict[str, Any]
