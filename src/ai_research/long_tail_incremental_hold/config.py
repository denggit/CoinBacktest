#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.6 incremental holding-value research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.6"
STAGE_NAME = "q70 incremental holding value and non-time exit signal research"


@dataclass(frozen=True)
class IncrementalHoldConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    primary_horizon_hours: int = 6
    path_horizon_hours: int = 120
    entry_delay_minutes: int = 1
    train_event_quantile: float = 0.50
    evaluation_quantiles: tuple[float, ...] = (0.70, 0.90)
    base_round_trip_cost: float = 0.0013
    risk_penalty: float = 1.25

    # Checkpoints are decision observations, not mandatory exits.
    checkpoints_minutes: tuple[int, ...] = (180, 360, 720, 1440, 2880)
    future_endpoints_minutes: tuple[int, ...] = (360, 720, 1440, 2880, 7200)

    # Strict rolling OOF opening-score history used to build holding-model train rows.
    entry_oof_min_train_days: int = 120
    entry_oof_calibration_days: int = 45
    entry_oof_blocks: int = 6
    entry_oof_embargo_hours: int = 18

    # Holding-value OOF contract. The embargo covers the five-day label window.
    holding_oof_splits: int = 5
    holding_oof_embargo_hours: int = 120
    minimum_train_rows: int = 120
    minimum_test_rows: int = 40
    minimum_oof_folds: int = 2

    regression_n_estimators: int = 280
    regression_learning_rate: float = 0.03
    regression_num_leaves: int = 11
    regression_min_child_samples: int = 30

    base_train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801

    # Minimum economic buffer used only for diagnostics; it is not an exit threshold.
    # Shared path extractor label fields; retained only for compatibility and diagnostics.
    persistent_failure_max_mfe_48h: float = 0.0125
    recoverable_min_mfe_48h: float = 0.0125
    continuation_increment_6h_to_24h: float = 0.010
    longhold_increment_24h_to_120h: float = 0.015

    positive_utility_buffer: float = 0.0005
    rank_bucket_count: int = 10
    decision_quantiles: tuple[float, ...] = (0.10, 0.20, 0.80, 0.90)

    # Cross-year signal gates. Passing this stage means the signal is worth integrating
    # into a recurrent policy in the next stage, not that a final live exit exists.
    minimum_rank_ic: float = 0.08
    minimum_top_bottom_spread: float = 0.004
    minimum_decile_monotonicity: float = 0.45
    minimum_sign_accuracy: float = 0.55

    independent_event_cooldown_hours: int = 6
    episode_merge_gap_minutes: int = 30
    disaster_stop_return: float = -0.030
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_6_incremental_hold_value"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.6 must keep 2026 sealed")
        if self.primary_horizon_hours != 6:
            raise ValueError("the frozen opening model remains the six-hour utility model")
        if self.path_horizon_hours != 120:
            raise ValueError("the research label window remains five days")
        if self.entry_delay_minutes != 1:
            raise ValueError("next-minute-open entry is frozen")
        if self.train_event_quantile >= min(self.evaluation_quantiles):
            raise ValueError("holding train pool must be broader than q70")
        if self.evaluation_quantiles != (0.70, 0.90):
            raise ValueError("q70 and q90 OOS scopes are frozen")
        if self.checkpoints_minutes != tuple(sorted(set(self.checkpoints_minutes))):
            raise ValueError("checkpoints must be unique and increasing")
        if max(self.future_endpoints_minutes) != self.path_horizon_hours * 60:
            raise ValueError("the final endpoint must match the five-day path horizon")
        if any(checkpoint >= max(self.future_endpoints_minutes) for checkpoint in self.checkpoints_minutes):
            raise ValueError("all checkpoints must occur before the final label endpoint")
        if self.entry_oof_embargo_hours < self.primary_horizon_hours:
            raise ValueError("entry OOF embargo must cover the opening target")
        if self.holding_oof_embargo_hours < self.path_horizon_hours:
            raise ValueError("holding OOF embargo must cover the future utility label")
        if not -0.10 < self.disaster_stop_return < -0.01:
            raise ValueError("disaster stop must remain a wide safety floor")
        if "03_4_2_6" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["checkpoints_minutes"] = list(self.checkpoints_minutes)
        payload["future_endpoints_minutes"] = list(self.future_endpoints_minutes)
        payload["evaluation_quantiles"] = list(self.evaluation_quantiles)
        payload["decision_quantiles"] = list(self.decision_quantiles)
        return payload


DEFAULT_INCREMENTAL_HOLD_CONFIG = IncrementalHoldConfig()
