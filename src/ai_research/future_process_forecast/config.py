#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.3 future market-process forecasting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.swing_long_context.config import DEFAULT_SWING_LONG_CONTEXT_CONFIG


STAGE_ID = "R03.3"
STAGE_NAME = "Future market-process forecast MVP"
PROCESS_TYPES = ("up_expansion", "down_expansion", "volatile_range", "low_opportunity")


@dataclass(frozen=True)
class DirectionalEventSpec:
    target_floor: float = 0.050
    target_atr_multiple: float = 3.0
    max_adverse_floor: float = 0.020
    max_adverse_target_fraction: float = 0.45
    horizon_hours: int = 24
    initial_window_hours: int = 3
    initial_move_floor: float = 0.008
    initial_target_fraction: float = 0.15
    prior_window_hours: int = 6
    prior_progress_cap: float = 0.30
    terminal_capture_fraction: float = 0.30
    refractory_hours: int = 12

    def validate(self) -> None:
        if not 0 < self.target_floor < 0.25:
            raise ValueError("invalid directional target floor")
        if not 0 < self.max_adverse_floor < self.target_floor:
            raise ValueError("invalid directional adverse floor")
        if self.horizon_hours < self.initial_window_hours:
            raise ValueError("directional horizon must cover the initial window")
        if not 0 < self.prior_progress_cap < 1:
            raise ValueError("prior progress cap must be in (0, 1)")


@dataclass(frozen=True)
class RangeEventSpec:
    side_move_floor: float = 0.018
    side_atr_multiple: float = 1.20
    total_range_floor: float = 0.050
    total_atr_multiple: float = 3.50
    horizon_hours: int = 12
    initial_window_hours: int = 3
    initial_range_fraction: float = 0.22
    prior_window_hours: int = 6
    prior_range_fraction_cap: float = 0.65
    terminal_share_cap: float = 0.35
    min_reversal_count: int = 3
    refractory_hours: int = 8

    def validate(self) -> None:
        if not 0 < self.side_move_floor < self.total_range_floor:
            raise ValueError("invalid volatile-range thresholds")
        if self.horizon_hours < self.initial_window_hours:
            raise ValueError("range horizon must cover the initial window")
        if self.min_reversal_count < 2:
            raise ValueError("range events require at least two reversals")


@dataclass(frozen=True)
class FutureProcessForecastConfig:
    symbol: str = DEFAULT_SWING_LONG_CONTEXT_CONFIG.base.symbol
    research_start: str = "2023-01-01"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01"
    decision_interval_minutes: int = 15
    forecast_horizons_hours: tuple[int, ...] = (6, 12, 24)
    event_scan_timeframe_minutes: int = 15
    event_atr_hours: int = 4
    event_atr_slow_days: int = 30
    directional: DirectionalEventSpec = DirectionalEventSpec()
    volatile_range: RangeEventSpec = RangeEventSpec()
    low_opportunity_range_floor: float = 0.025
    low_opportunity_atr_multiple: float = 2.25
    sample_stride_decisions: int = 4
    train_sample_cap: int = 300_000
    random_seed: int = 20260730
    architectures: tuple[str, ...] = (
        "macro_lightgbm",
        "multiframe_lightgbm",
        "multiframe_micro_lightgbm",
        "multiframe_micro_logistic",
    )
    micro_timeframe: str = "5s"
    micro_required: bool = True
    micro_windows_minutes: tuple[int, ...] = (5, 15, 60)
    micro_load_chunk_days: int = 31
    minimum_micro_coverage: float = 0.95
    signal_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    lightgbm_n_estimators: int = 420
    lightgbm_learning_rate: float = 0.035
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 300
    lightgbm_feature_fraction: float = 0.80
    logistic_c: float = 0.25
    logistic_max_iter: int = 2500
    base_cache_dir: str = DEFAULT_SWING_LONG_CONTEXT_CONFIG.base.cache_dir
    event_cache_dir: str = "data/cache/eth_ai_trading/r03_3_process_events"
    micro_cache_dir: str = "data/cache/eth_ai_trading/r03_3_micro_5s"
    report_dir: str = "data/reports/research/eth_ai_trading/03_3_future_process_forecast"

    def validate(self) -> None:
        self.directional.validate()
        self.volatile_range.validate()
        start = pd.Timestamp(self.research_start)
        end = pd.Timestamp(self.research_end)
        holdout = pd.Timestamp(self.sealed_holdout_start)
        if not start < end < holdout:
            raise ValueError("R03.3 research must end before the sealed 2026 holdout")
        if self.decision_interval_minutes != 15:
            raise ValueError("R03.3 must reuse the R03.2 15m decision axis")
        if tuple(sorted(set(self.forecast_horizons_hours))) != self.forecast_horizons_hours:
            raise ValueError("forecast horizons must be unique and increasing")
        if self.micro_timeframe not in {"1s", "5s", "10s"}:
            raise ValueError("micro timeframe must be a public trade-bar timeframe")
        if not 0 < self.minimum_micro_coverage <= 1:
            raise ValueError("invalid micro coverage gate")
        if self.sample_stride_decisions < 1:
            raise ValueError("sample stride must be positive")
        if not 0 < min(self.signal_quantiles) < max(self.signal_quantiles) < 1:
            raise ValueError("invalid signal quantiles")
        if "r03_2_long_context" not in self.base_cache_dir:
            raise ValueError("R03.3 must reuse the isolated R03.2 long-context cache")
        if "r03_3" not in self.event_cache_dir or "r03_3" not in self.micro_cache_dir:
            raise ValueError("R03.3 caches must be isolated")

    @property
    def base_cache_path(self) -> Path:
        return PROJECT_ROOT / self.base_cache_dir

    @property
    def event_cache_path(self) -> Path:
        return PROJECT_ROOT / self.event_cache_dir

    @property
    def micro_cache_path(self) -> Path:
        return PROJECT_ROOT / self.micro_cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["forecast_horizons_hours"] = list(self.forecast_horizons_hours)
        payload["architectures"] = list(self.architectures)
        payload["micro_windows_minutes"] = list(self.micro_windows_minutes)
        payload["signal_quantiles"] = list(self.signal_quantiles)
        return payload


DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG = FutureProcessForecastConfig()
