#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03 medium-horizon swing research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import DEFAULT_AI_RESEARCH_CONFIG, PROJECT_ROOT


@dataclass(frozen=True)
class SwingTargetSpec:
    target_id: str
    target_move: float
    max_adverse_move: float
    horizon_hours: int

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id is required")
        if not 0 < self.max_adverse_move < self.target_move:
            raise ValueError("max_adverse_move must be positive and smaller than target_move")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SWING_TARGETS = (
    SwingTargetSpec("move3_lowmae_72h", target_move=0.030, max_adverse_move=0.0125, horizon_hours=72),
    SwingTargetSpec("move5_lowmae_120h", target_move=0.050, max_adverse_move=0.0175, horizon_hours=120),
)


@dataclass(frozen=True)
class SwingBaselineConfig:
    symbol: str = DEFAULT_AI_RESEARCH_CONFIG.symbol
    source_timeframe: str = "1m"
    warmup_start: str = DEFAULT_AI_RESEARCH_CONFIG.warmup_start
    research_start: str = DEFAULT_AI_RESEARCH_CONFIG.research_start
    research_end: str = DEFAULT_AI_RESEARCH_CONFIG.research_end
    validation_start: str = DEFAULT_AI_RESEARCH_CONFIG.validation_start
    locked_holdout_start: str = DEFAULT_AI_RESEARCH_CONFIG.sealed_holdout_start
    decision_interval_minutes: int = 15
    execution_delay_minutes: int = 1
    delay_scenarios_minutes: tuple[int, ...] = (1, 2, 5)
    round_trip_fee_rate: float = DEFAULT_AI_RESEARCH_CONFIG.round_trip_fee_rate
    slippage_rate_per_side: float = 0.0001
    cost_stress_multipliers: tuple[float, ...] = DEFAULT_AI_RESEARCH_CONFIG.cost_stress_multipliers
    feature_lookback_days: int = 180
    target_specs: tuple[SwingTargetSpec, ...] = DEFAULT_SWING_TARGETS
    signal_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    probability_margin: float = 0.05
    train_sample_cap: int = 500_000
    random_seed: int = 20260729
    lightgbm_n_estimators: int = 450
    lightgbm_learning_rate: float = 0.035
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 250
    lightgbm_feature_fraction: float = 0.85
    structural_swing_bars_4h: int = 8
    structural_buffer_atr: float = 0.15
    min_initial_stop_pct: float = 0.006
    max_initial_stop_pct: float = 0.018
    breakeven_trigger_pct: float = 0.015
    trailing_trigger_pct: float = 0.025
    strong_trend_trigger_pct: float = 0.040
    min_trailing_distance_pct: float = 0.006
    max_trailing_distance_pct: float = 0.015
    max_hold_hours: int = 120
    cache_dir: str = "data/cache/eth_ai_trading/r03_swing"
    report_dir: str = "data/reports/research/eth_ai_trading/03_swing_baseline"

    def validate(self) -> None:
        start = pd.Timestamp(self.research_start)
        validation = pd.Timestamp(self.validation_start)
        holdout = pd.Timestamp(self.locked_holdout_start)
        end = pd.Timestamp(self.research_end)
        if not start < validation < holdout <= end:
            raise ValueError("dates must satisfy research < validation < locked holdout <= end")
        if self.source_timeframe != "1m":
            raise ValueError("R03 must use the public 1m trade-bar interface")
        if self.decision_interval_minutes <= 0 or 60 % self.decision_interval_minutes != 0:
            raise ValueError("decision interval must be a positive divisor of one hour")
        if self.execution_delay_minutes not in self.delay_scenarios_minutes:
            raise ValueError("base execution delay must be included in delay scenarios")
        if any(value < self.execution_delay_minutes for value in self.delay_scenarios_minutes):
            raise ValueError("delay scenarios may not be faster than the base conservative execution")
        if not 0 < self.round_trip_fee_rate < 0.02:
            raise ValueError("invalid round-trip fee")
        if self.slippage_rate_per_side < 0:
            raise ValueError("slippage must be non-negative")
        if any(value < 1.0 for value in self.cost_stress_multipliers):
            raise ValueError("cost multipliers must be >= 1")
        if self.feature_lookback_days < 90:
            raise ValueError("daily context requires at least 90 lookback days")
        if not self.target_specs:
            raise ValueError("at least one target spec is required")
        if not 0 < min(self.signal_quantiles) < max(self.signal_quantiles) < 1:
            raise ValueError("signal quantiles must be inside (0, 1)")
        if not 0 <= self.probability_margin < 1:
            raise ValueError("probability margin must be inside [0, 1)")
        if not 0 < self.min_initial_stop_pct < self.max_initial_stop_pct:
            raise ValueError("invalid initial stop bounds")
        if not 0 < self.breakeven_trigger_pct < self.trailing_trigger_pct < self.strong_trend_trigger_pct:
            raise ValueError("profit-protection triggers must be increasing")
        if self.max_hold_hours < max(spec.horizon_hours for spec in self.target_specs):
            raise ValueError("max hold must cover the longest research target horizon")

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / self.cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    @property
    def max_horizon_hours(self) -> int:
        return max(spec.horizon_hours for spec in self.target_specs)

    @property
    def base_round_trip_cost(self) -> float:
        return self.round_trip_fee_rate + 2.0 * self.slippage_rate_per_side

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["delay_scenarios_minutes"] = list(self.delay_scenarios_minutes)
        payload["cost_stress_multipliers"] = list(self.cost_stress_multipliers)
        payload["signal_quantiles"] = list(self.signal_quantiles)
        payload["target_specs"] = [spec.to_dict() for spec in self.target_specs]
        return payload


DEFAULT_SWING_BASELINE_CONFIG = SwingBaselineConfig()
