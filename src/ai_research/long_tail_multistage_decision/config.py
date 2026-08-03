#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.3 multi-stage holding research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.3"
STAGE_NAME = "Expanded causal path training, multi-stage holding decisions and q70 opportunity expansion"


@dataclass(frozen=True)
class LongTailMultistageConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    primary_horizon_hours: int = 6
    path_horizon_hours: int = 120
    entry_delay_minutes: int = 1
    checkpoints_minutes: tuple[int, ...] = (60, 180, 360, 1440)
    train_event_quantile: float = 0.50
    evaluation_quantiles: tuple[float, ...] = (0.70, 0.90)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    risk_penalty: float = 1.25

    # Strict OOF entry-score timeline used only to build the holding-model train set.
    entry_oof_min_train_days: int = 120
    entry_oof_calibration_days: int = 45
    entry_oof_blocks: int = 6
    entry_oof_embargo_hours: int = 18

    # Holding-model OOF thresholds must cover the complete five-day label horizon.
    holding_oof_splits: int = 5
    holding_oof_embargo_hours: int = 120
    minimum_train_rows: int = 80
    minimum_class_rows: int = 12
    minimum_test_rows: int = 30

    classifier_n_estimators: int = 260
    classifier_learning_rate: float = 0.03
    classifier_num_leaves: int = 9
    classifier_min_child_samples: int = 25

    base_train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801

    # OOF-probability quantiles, not absolute probability cutoffs.
    high_failure_quantile: float = 0.85
    safe_failure_quantile: float = 0.40
    low_recovery_quantile: float = 0.25
    high_recovery_quantile: float = 0.65
    high_continuation_quantile: float = 0.65
    high_longhold_quantile: float = 0.65

    persistent_failure_max_mfe_48h: float = 0.0125
    recoverable_min_mfe_48h: float = 0.0125
    continuation_increment_6h_to_24h: float = 0.010
    longhold_increment_24h_to_120h: float = 0.015

    independent_event_cooldown_hours: int = 6
    episode_merge_gap_minutes: int = 30
    minimum_trades_per_year: int = 60
    minimum_pf_2x: float = 1.20
    maximum_mdd: float = 0.20
    maximum_top10_profit_share: float = 0.60
    minimum_positive_quarters: int = 6
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_3_long_tail_multistage_decision"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.3 must keep 2026 sealed")
        if self.primary_horizon_hours != 6:
            raise ValueError("the frozen opening model remains a six-hour utility model")
        if self.path_horizon_hours != 120:
            raise ValueError("R03.4.2.3 requires a five-day path for long-hold validation")
        if self.entry_delay_minutes != 1:
            raise ValueError("next-minute-open entry remains frozen")
        if self.checkpoints_minutes != (60, 180, 360, 1440):
            raise ValueError("multi-stage checkpoints are frozen")
        if not 0.40 <= self.train_event_quantile < min(self.evaluation_quantiles):
            raise ValueError("the OOF training pool must be broader than q70")
        if self.evaluation_quantiles != (0.70, 0.90):
            raise ValueError("q70 and q90 are the frozen OOS scopes")
        if self.entry_oof_embargo_hours < self.primary_horizon_hours:
            raise ValueError("entry OOF embargo must cover the opening target")
        if self.holding_oof_embargo_hours < self.path_horizon_hours:
            raise ValueError("holding OOF embargo must cover the five-day label")
        if self.independent_event_cooldown_hours < self.primary_horizon_hours:
            raise ValueError("event cooldown must cover the original target")
        if tuple(sorted(set(self.checkpoints_minutes))) != self.checkpoints_minutes:
            raise ValueError("checkpoints must be unique and increasing")
        if any(value < 1.0 for value in self.cost_multipliers):
            raise ValueError("cost multipliers must be >= 1")
        if "03_4_2_3" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["checkpoints_minutes"] = list(self.checkpoints_minutes)
        payload["evaluation_quantiles"] = list(self.evaluation_quantiles)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        return payload


DEFAULT_LONG_TAIL_MULTISTAGE_CONFIG = LongTailMultistageConfig()
