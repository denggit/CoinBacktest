#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Typed configurations for causal ICT MSS research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplacementSpec:
    """One predeclared displacement/FVG quality definition.

    Ratios are measured against *past-only* rolling medians.  No global or
    future-fitted quantile enters signal construction.
    """

    name: str
    min_body_vs_past_median: float
    min_range_vs_past_median: float
    min_body_fraction: float
    max_close_from_extreme_fraction: float
    min_fvg_size_bp: float = 0.0


@dataclass(frozen=True)
class MSSResearchSpec:
    """Mechanism-level filter used for the fixed R01 research atlas."""

    spec_id: str
    description: str
    side: int = 0
    micro_order: int = 2
    structure_mode: str = "pre_sweep"
    min_htf_confirmed_order: int = 2
    min_max_timeframe_min: int = 15
    min_swept_timeframe_count: int = 1
    max_sweep_to_displacement_bars: int = 60
    displacement_name: str = "core"
    max_fill_wait_bars: int = 60
    neighborhood_group: str = "standalone"

    def validate(self) -> "MSSResearchSpec":
        if self.side not in {-1, 0, 1}:
            raise ValueError("side must be -1 (short), 0 (both), or 1 (long)")
        if self.micro_order < 1:
            raise ValueError("micro_order must be >= 1")
        if self.structure_mode not in {"pre_sweep", "rolling"}:
            raise ValueError("structure_mode must be pre_sweep or rolling")
        if self.min_htf_confirmed_order < 1:
            raise ValueError("min_htf_confirmed_order must be >= 1")
        if self.min_max_timeframe_min not in {15, 30, 60, 240}:
            raise ValueError("min_max_timeframe_min must be one of 15/30/60/240")
        if self.min_swept_timeframe_count < 1:
            raise ValueError("min_swept_timeframe_count must be >= 1")
        if self.max_sweep_to_displacement_bars < 1:
            raise ValueError("max_sweep_to_displacement_bars must be positive")
        if self.max_fill_wait_bars < 1:
            raise ValueError("max_fill_wait_bars must be positive")
        return self
