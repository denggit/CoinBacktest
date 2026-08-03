#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4 state-context opening-value ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4"
STAGE_NAME = "State-context directional opening-value ablation"

ABLATION_VARIANTS = (
    "base_multiframe",
    "base_plus_activity",
    "base_plus_directional_state",
    "base_plus_all_state",
    "base_plus_all_state_and_activity_persist",
    "state_only",
)


@dataclass(frozen=True)
class StateContextAblationConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    decision_interval_minutes: int = 15
    horizons_hours: tuple[int, ...] = (3, 6)
    primary_horizon_hours: int = 6
    risk_penalty: float = 1.25
    base_round_trip_cost: float = 0.0013
    signal_quantiles: tuple[float, ...] = (0.90, 0.95)
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0)
    variants: tuple[str, ...] = ABLATION_VARIANTS
    train_sample_cap: int = 400_000
    lightgbm_n_estimators: int = 420
    lightgbm_learning_rate: float = 0.035
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 300
    lightgbm_feature_fraction: float = 0.80
    random_state: int = 20260801
    minimum_test_rows: int = 20_000
    minimum_signal_count: int = 300
    minimum_rank_ic_increment: float = 0.005
    minimum_net_expectancy_increment: float = 0.0
    outcome_cache_dir: str = "data/cache/eth_ai_trading/r03_4_state_context_ablation"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_state_context_ablation"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4 must keep 2026 sealed")
        if self.decision_interval_minutes != 15:
            raise ValueError("R03.4 must retain the 15-minute decision axis")
        if tuple(sorted(set(self.horizons_hours))) != self.horizons_hours:
            raise ValueError("horizons must be unique and increasing")
        if self.primary_horizon_hours not in self.horizons_hours:
            raise ValueError("primary horizon must be present in horizons")
        if not 0 < self.risk_penalty <= 5:
            raise ValueError("invalid risk penalty")
        if not 0 < self.base_round_trip_cost < 0.02:
            raise ValueError("invalid cost assumption")
        if any(value < 1.0 for value in self.cost_stress_multipliers):
            raise ValueError("cost stress multipliers must be >= 1")
        if tuple(self.variants) != ABLATION_VARIANTS:
            raise ValueError("R03.4 ablation variants are frozen")
        if not 0 < min(self.signal_quantiles) < max(self.signal_quantiles) < 1:
            raise ValueError("signal quantiles must lie inside (0, 1)")
        if "r03_4" not in self.outcome_cache_dir or "03_4" not in self.report_dir:
            raise ValueError("R03.4 outputs must be isolated")

    @property
    def outcome_cache_path(self) -> Path:
        return PROJECT_ROOT / self.outcome_cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def outcome_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for horizon in self.horizons_hours:
            columns.extend(
                [
                    f"long_mfe_h{horizon}",
                    f"long_mae_h{horizon}",
                    f"short_mfe_h{horizon}",
                    f"short_mae_h{horizon}",
                    f"future_close_return_h{horizon}",
                    f"long_utility_h{horizon}",
                    f"short_utility_h{horizon}",
                ]
            )
        return tuple(columns)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["horizons_hours"] = list(self.horizons_hours)
        payload["signal_quantiles"] = list(self.signal_quantiles)
        payload["variants"] = list(self.variants)
        payload["cost_stress_multipliers"] = list(self.cost_stress_multipliers)
        return payload


DEFAULT_STATE_CONTEXT_ABLATION_CONFIG = StateContextAblationConfig()
