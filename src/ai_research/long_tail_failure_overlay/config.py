#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.5 high-confidence failure overlay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.5"
STAGE_NAME = "q70 score-tier high-confidence persistent-failure exit overlay"


@dataclass(frozen=True)
class FailureOverlayConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    primary_horizon_hours: int = 6
    path_horizon_hours: int = 120
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    train_event_quantile: float = 0.50
    evaluation_quantile: float = 0.70
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    risk_penalty: float = 1.25

    # Strict rolling OOF opening-score history for the holding-risk model.
    entry_oof_min_train_days: int = 120
    entry_oof_calibration_days: int = 45
    entry_oof_blocks: int = 6
    entry_oof_embargo_hours: int = 18

    # Persistent-failure classifier OOF contract.
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

    # Labels remain ex-post; checkpoint features are strictly causal.
    persistent_failure_max_mfe_48h: float = 0.0125
    recoverable_min_mfe_48h: float = 0.0125
    continuation_increment_6h_to_24h: float = 0.010
    longhold_increment_24h_to_120h: float = 0.015
    checkpoints_minutes: tuple[int, ...] = (60, 180, 360, 1440)

    # Pre-registered probability quantiles. Higher opening-score tiers receive
    # more tolerance before an early exit is allowed.
    global_warning_quantile: float = 0.85
    global_confirm_quantile: float = 0.90
    ultra_confirm_quantile: float = 0.97
    tier_warning_quantiles: tuple[tuple[str, float], ...] = (
        ("q70_to_q80", 0.85),
        ("q80_to_q90", 0.90),
        ("q90_plus", 0.94),
    )
    tier_confirm_quantiles: tuple[tuple[str, float], ...] = (
        ("q70_to_q80", 0.90),
        ("q80_to_q90", 0.94),
        ("q90_plus", 0.97),
    )
    minimum_tier_threshold_rows: int = 30

    # Structural confirmation. Probability alone is never sufficient.
    standard_gate_count: int = 3
    ultra_gate_count: int = 4
    lower_low_share_minimum: float = 0.50
    underwater_fraction_minimum: float = 0.60
    maximum_recovery_from_trough: float = 0.0030

    # Safety floor is deliberately not optimized. It is evaluated separately
    # from the model overlay and uses next-minute-open execution after breach.
    disaster_stop_return: float = -0.030

    independent_event_cooldown_hours: int = 6
    episode_merge_gap_minutes: int = 30
    minimum_trades_per_year: int = 150
    minimum_pf_2x: float = 1.35
    maximum_mdd: float = 0.20
    maximum_top10_profit_share: float = 0.60
    minimum_positive_quarters: int = 6
    minimum_overlay_exit_share: float = 0.01
    maximum_overlay_exit_share: float = 0.15
    minimum_baseline_profit_retention: float = 0.90
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_5_q70_failure_overlay"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.5 must keep 2026 sealed")
        if self.primary_horizon_hours != 6:
            raise ValueError("the opening model remains the frozen six-hour model")
        if self.path_horizon_hours != 120:
            raise ValueError("persistent-failure labels require the frozen five-day path extractor")
        if self.train_event_quantile >= self.evaluation_quantile:
            raise ValueError("the OOF holding-model train pool must be broader than q70")
        if self.entry_oof_embargo_hours < self.primary_horizon_hours:
            raise ValueError("entry OOF embargo must cover the opening target")
        if self.holding_oof_embargo_hours < self.path_horizon_hours:
            raise ValueError("holding OOF embargo must cover the future label")
        if self.checkpoints_minutes != (60, 180, 360, 1440):
            raise ValueError("shared causal path extractor checkpoints changed")
        if not -0.10 < self.disaster_stop_return < -0.01:
            raise ValueError("disaster stop must remain a wide safety floor")
        if "03_4_2_5" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    @property
    def warning_quantiles(self) -> dict[str, float]:
        return dict(self.tier_warning_quantiles)

    @property
    def confirm_quantiles(self) -> dict[str, float]:
        return dict(self.tier_confirm_quantiles)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["checkpoints_minutes"] = list(self.checkpoints_minutes)
        payload["tier_warning_quantiles"] = dict(self.tier_warning_quantiles)
        payload["tier_confirm_quantiles"] = dict(self.tier_confirm_quantiles)
        return payload


DEFAULT_FAILURE_OVERLAY_CONFIG = FailureOverlayConfig()
