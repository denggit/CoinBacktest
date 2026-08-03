#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.1 long-opportunity state meta calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.1"
STAGE_NAME = "Long-opportunity two-stage soft-state calibration"

META_VARIANTS = (
    "base_identity",
    "score_only_meta",
    "score_plus_activity_meta",
    "score_plus_strategic_meta",
    "score_plus_strategic_activity_meta",
    "soft_state_only_meta",
)
STATE_META_VARIANTS = (
    "score_plus_activity_meta",
    "score_plus_strategic_meta",
    "score_plus_strategic_activity_meta",
)


@dataclass(frozen=True)
class LongStateCalibrationConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    primary_horizon_hours: int = 6
    risk_penalty: float = 1.25
    base_round_trip_cost: float = 0.0013
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0)
    variants: tuple[str, ...] = META_VARIANTS
    signal_quantiles: tuple[float, ...] = (0.90, 0.95)
    common_candidate_quantile: float = 0.80
    common_rerank_quantile: float = 0.50
    independent_event_cooldown_hours: int = 6
    episode_merge_gap_minutes: int = 30
    oof_min_train_days: int = 120
    oof_blocks: int = 4
    oof_embargo_hours: int = 18
    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    meta_n_estimators: int = 180
    meta_learning_rate: float = 0.03
    meta_num_leaves: int = 7
    meta_min_child_samples: int = 500
    random_state: int = 20260801
    minimum_test_rows: int = 20_000
    minimum_independent_events: int = 80
    minimum_long_utility_ic_increment: float = 0.003
    maximum_mae_worsening: float = 0.0002
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_1_long_state_meta_calibration"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.1 must keep 2026 sealed")
        if self.primary_horizon_hours != 6:
            raise ValueError("R03.4.1 freezes the six-hour long-opportunity target")
        if tuple(self.variants) != META_VARIANTS:
            raise ValueError("R03.4.1 meta variants are frozen")
        if not 0 < self.common_candidate_quantile < 1:
            raise ValueError("invalid common candidate quantile")
        if not 0 < self.common_rerank_quantile < 1:
            raise ValueError("invalid common rerank quantile")
        if self.oof_blocks < 3 or self.oof_min_train_days < 60:
            raise ValueError("OOF stacking needs at least three blocks and sixty warmup days")
        if self.oof_embargo_hours < self.primary_horizon_hours:
            raise ValueError("OOF embargo must cover the target horizon")
        if self.independent_event_cooldown_hours < self.primary_horizon_hours:
            raise ValueError("independent event cooldown must cover the target horizon")
        if not 0 < self.base_round_trip_cost < 0.02:
            raise ValueError("invalid cost assumption")
        if "03_4_1" not in self.report_dir:
            raise ValueError("R03.4.1 report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in ("cost_stress_multipliers", "variants", "signal_quantiles"):
            payload[key] = list(payload[key])
        return payload


DEFAULT_LONG_STATE_CALIBRATION_CONFIG = LongStateCalibrationConfig()
