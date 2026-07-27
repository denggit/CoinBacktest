#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core models for the causal Market State Map.

V0.2 keeps price structure, volatility, order flow, price impact/absorption and
rolling location as orthogonal dimensions.  Human-readable labels are
observable context, not forecasts or executable trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MarketStateConfig:
    """Configuration for Market State Map V0.2.

    Window values are expressed in bars.  Every historical normalization ends
    one closed bar before the value being classified.  Rolling support and
    resistance levels also exclude the current bar.
    """

    fast_trend_window: int = 16
    trend_window: int = 64
    slow_trend_window: int = 240
    volatility_window: int = 30
    activity_window: int = 12
    baseline_window: int = 720

    directional_threshold: float = 0.24
    trend_exit_threshold: float = 0.10
    trend_confirm_bars: int = 3
    min_state_bars: int = 15
    fast_pulse_threshold: float = 0.20

    orderliness_low_quantile: float = 0.20
    orderliness_high_quantile: float = 0.80
    ordered_threshold: float = 0.58  # deprecated V0 compatibility field

    volatility_quiet_z: float = -0.80
    volatility_quiet_exit_z: float = -0.40
    volatility_expand_z: float = 0.80
    volatility_expand_exit_z: float = 0.40
    volatility_shock_z: float = 2.00
    volatility_shock_exit_z: float = 1.30
    volatility_confirm_bars: int = 3
    volatility_min_state_bars: int = 6

    # Rich trade-bar order flow.
    flow_fast_window: int = 3
    flow_window: int = 12
    flow_slow_window: int = 30
    flow_scale: float = 0.10
    flow_threshold: float = 0.06
    flow_acceleration_threshold: float = 0.035
    impact_expected_response: float = 0.20
    impact_effective_threshold: float = 0.25
    absorption_threshold: float = 0.35

    # Prior-bar rolling structure levels.
    location_window: int = 60
    structure_window: int = 240
    near_level_atr: float = 0.60
    sweep_min_atr: float = 0.03
    breakout_accept_atr: float = 0.10

    def validate(self) -> None:
        windows = (
            "fast_trend_window",
            "trend_window",
            "slow_trend_window",
            "volatility_window",
            "activity_window",
            "baseline_window",
            "flow_fast_window",
            "flow_window",
            "flow_slow_window",
            "location_window",
            "structure_window",
        )
        for name in windows:
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be >= 2")
        if not self.fast_trend_window < self.trend_window < self.slow_trend_window:
            raise ValueError("trend windows must satisfy fast < medium < slow")
        if not self.flow_fast_window < self.flow_window < self.flow_slow_window:
            raise ValueError("flow windows must satisfy fast < medium < slow")
        if self.location_window > self.structure_window:
            raise ValueError("location_window must be <= structure_window")
        largest_context = max(
            self.trend_window,
            self.volatility_window,
            self.flow_slow_window,
        )
        if self.baseline_window < largest_context:
            raise ValueError("baseline_window must cover the largest normalized state window")
        if self.min_state_bars < 1 or self.trend_confirm_bars < 1:
            raise ValueError("trend duration/confirmation bars must be >= 1")
        if self.volatility_min_state_bars < 1 or self.volatility_confirm_bars < 1:
            raise ValueError("volatility duration/confirmation bars must be >= 1")
        if not 0.0 < self.trend_exit_threshold < self.directional_threshold < 1.0:
            raise ValueError("trend thresholds must satisfy 0 < exit < enter < 1")
        if not 0.0 < self.fast_pulse_threshold < 1.0:
            raise ValueError("fast_pulse_threshold must be between 0 and 1")
        if not 0.0 < self.orderliness_low_quantile < self.orderliness_high_quantile < 1.0:
            raise ValueError("orderliness quantiles must satisfy 0 < low < high < 1")
        if not self.volatility_quiet_z < self.volatility_quiet_exit_z < 0.0:
            raise ValueError("quiet volatility thresholds must satisfy enter < exit < 0")
        if not 0.0 < self.volatility_expand_exit_z < self.volatility_expand_z:
            raise ValueError("expansion volatility thresholds must satisfy 0 < exit < enter")
        if not self.volatility_expand_z < self.volatility_shock_exit_z < self.volatility_shock_z:
            raise ValueError("shock thresholds must satisfy expansion < shock_exit < shock_enter")
        if not 0.0 < self.flow_scale <= 1.0:
            raise ValueError("flow_scale must be in (0, 1]")
        if not 0.0 < self.flow_threshold < 1.0:
            raise ValueError("flow_threshold must be in (0, 1)")
        if not 0.0 < self.flow_acceleration_threshold < 1.0:
            raise ValueError("flow_acceleration_threshold must be in (0, 1)")
        if not -1.0 < self.impact_expected_response < 1.0:
            raise ValueError("impact_expected_response must be in (-1, 1)")
        if not 0.0 < self.impact_effective_threshold < 1.0:
            raise ValueError("impact_effective_threshold must be in (0, 1)")
        if not 0.0 < self.absorption_threshold < 1.0:
            raise ValueError("absorption_threshold must be in (0, 1)")
        if self.near_level_atr <= 0.0 or self.sweep_min_atr < 0.0 or self.breakout_accept_atr < 0.0:
            raise ValueError("location ATR thresholds must be non-negative and near_level_atr > 0")


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    usable_rows: int
    duplicate_timestamps: int
    missing_ohlcv_rows: int
    invalid_price_rows: int
    monotonic_increasing: bool
    median_interval_seconds: float | None
    irregular_interval_ratio: float | None
    score: float
    usable: bool
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "usable_rows": self.usable_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_ohlcv_rows": self.missing_ohlcv_rows,
            "invalid_price_rows": self.invalid_price_rows,
            "monotonic_increasing": self.monotonic_increasing,
            "median_interval_seconds": self.median_interval_seconds,
            "irregular_interval_ratio": self.irregular_interval_ratio,
            "score": self.score,
            "usable": self.usable,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MarketStateSnapshot:
    timestamp: pd.Timestamp
    available_time: pd.Timestamp
    trend_score: float | None
    fast_trend_score: float | None
    medium_trend_score: float | None
    slow_trend_score: float | None
    trend_alignment_score: float | None
    orderliness_score: float | None
    orderliness_percentile: float | None
    volatility_score: float | None
    volatility_z: float | None
    activity_score: float | None
    activity_z: float | None
    trend_state: str
    trend_quality_state: str
    fast_pulse_state: str
    trend_phase: str
    trend_candidate_state: str
    trend_candidate_progress: float
    volatility_state: str
    primary_state: str
    trend_state_age: int
    volatility_state_age: int
    data_ready: bool
    orderflow_available: bool = False
    flow_score: float | None = None
    flow_persistence: float | None = None
    flow_acceleration: float | None = None
    flow_state: str = "unavailable"
    flow_price_effectiveness: float | None = None
    sell_absorption_score: float | None = None
    buy_absorption_score: float | None = None
    impact_state: str = "unavailable"
    structural_location_score: float | None = None
    location_state: str = "warmup"
    trade_context_state: str = "wait"
    trade_context_score: float | None = None


@dataclass(frozen=True)
class MarketStateSegment:
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    start_available_time: pd.Timestamp
    end_available_time: pd.Timestamp
    primary_state: str
    trend_state: str
    volatility_state: str
    bars: int
    mean_trend_score: float | None
    mean_orderliness_score: float | None
    mean_volatility_score: float | None
    mean_activity_score: float | None
    max_volatility_z: float | None


@dataclass(frozen=True)
class MarketStateResult:
    frame: pd.DataFrame
    segments: tuple[MarketStateSegment, ...]
    data_quality: DataQualityReport
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready_rows(self) -> int:
        if "data_ready" not in self.frame:
            return 0
        return int(self.frame["data_ready"].fillna(False).sum())
