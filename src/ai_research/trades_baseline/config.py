#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for the first trades-only supervised baseline.

R01 is intentionally a complete research gate rather than a standalone data
inspection stage. It performs only a light public-loader smoke check, then
builds causal samples, trains simple models, converts predictions into trades,
and applies realistic cost and latency stress in one run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import DEFAULT_AI_RESEARCH_CONFIG, PROJECT_ROOT


@dataclass(frozen=True)
class TradesBaselineConfig:
    symbol: str = DEFAULT_AI_RESEARCH_CONFIG.symbol
    timeframe: str = "1s"
    warmup_start: str = DEFAULT_AI_RESEARCH_CONFIG.warmup_start
    research_start: str = DEFAULT_AI_RESEARCH_CONFIG.research_start
    research_end: str = DEFAULT_AI_RESEARCH_CONFIG.research_end
    validation_start: str = DEFAULT_AI_RESEARCH_CONFIG.validation_start
    sealed_holdout_start: str = DEFAULT_AI_RESEARCH_CONFIG.sealed_holdout_start
    decision_interval_seconds: int = DEFAULT_AI_RESEARCH_CONFIG.decision_interval_seconds
    feature_windows_seconds: tuple[int, ...] = (10, 30, 60, 300)
    return_windows_seconds: tuple[int, ...] = (5, 10, 30, 60, 300)
    horizons_seconds: tuple[int, ...] = (60, 180, 300, 900)
    base_latency_seconds: float = 0.5
    latency_scenarios_seconds: tuple[float, ...] = DEFAULT_AI_RESEARCH_CONFIG.latency_scenarios_seconds
    round_trip_fee_rate: float = DEFAULT_AI_RESEARCH_CONFIG.round_trip_fee_rate
    slippage_rate_per_side: float = 0.0001
    cost_stress_multipliers: tuple[float, ...] = DEFAULT_AI_RESEARCH_CONFIG.cost_stress_multipliers
    train_sample_cap: int = 2_000_000
    linear_sample_cap: int = 750_000
    prediction_chunk_rows: int = 600_000
    random_seed: int = 20260729
    signal_quantiles: tuple[float, ...] = (0.990, 0.995, 0.999)
    initial_capital: float = 10_000.0
    model_num_leaves: int = 31
    model_learning_rate: float = 0.05
    model_n_estimators: int = 350
    model_min_child_samples: int = 500
    model_feature_fraction: float = 0.85
    cache_dir: str = "data/cache/eth_ai_trading/r01_trades_only"
    report_dir: str = "data/reports/research/eth_ai_trading/01_trades_only_baseline"

    def validate(self) -> None:
        if self.timeframe != "1s":
            raise ValueError("R01 baseline requires the existing 1s trade-bar interface")
        if self.decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive")
        if any(v <= 0 for v in self.feature_windows_seconds + self.return_windows_seconds):
            raise ValueError("feature windows must be positive")
        if any(v <= 0 for v in self.horizons_seconds):
            raise ValueError("horizons must be positive")
        if self.base_latency_seconds not in self.latency_scenarios_seconds:
            raise ValueError("base latency must be included in latency scenarios")
        if self.round_trip_fee_rate <= 0 or self.slippage_rate_per_side < 0:
            raise ValueError("invalid transaction-cost assumptions")
        if not 0 < min(self.signal_quantiles) < max(self.signal_quantiles) < 1:
            raise ValueError("signal quantiles must be inside (0, 1)")
        if self.train_sample_cap <= 0 or self.linear_sample_cap <= 0:
            raise ValueError("sample caps must be positive")

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / self.cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    @property
    def max_history_seconds(self) -> int:
        return max(max(self.feature_windows_seconds), max(self.return_windows_seconds))

    @property
    def max_future_seconds(self) -> int:
        return max(self.horizons_seconds) + int(max(self.latency_scenarios_seconds)) + 5

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


DEFAULT_TRADES_BASELINE_CONFIG = TradesBaselineConfig()
