#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.2 causal path-health recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.2"
STAGE_NAME = "Causal early path-health, recovery and long-hold recognition"


@dataclass(frozen=True)
class LongTailPathRecognitionConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    discovery_signal_quantile: float = 0.70
    primary_signal_quantile: float = 0.90
    quality_control_quantile: float = 0.95
    entry_delay_minutes: int = 1
    analysis_horizon_hours: int = 48
    fixed_diagnostic_horizon_hours: int = 6
    risk_penalty: float = 1.25
    base_round_trip_cost: float = 0.0013
    checkpoints_minutes: tuple[int, ...] = (60, 180, 360)
    minimum_train_rows: int = 80
    minimum_class_rows: int = 18
    minimum_test_rows: int = 50
    oof_splits: int = 4
    oof_embargo_hours: int = 48
    high_risk_quantile: float = 0.80
    high_hold_quantile: float = 0.70
    broad_safe_quantile: float = 0.30
    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    classifier_n_estimators: int = 220
    classifier_learning_rate: float = 0.035
    classifier_num_leaves: int = 7
    classifier_min_child_samples: int = 20
    random_state: int = 20260801
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_2_long_tail_path_recognition"

    # Fixed labels. These are research targets, not live exit rules.
    recovery_min_24h_mfe: float = 0.010
    recovery_positive_24h_net: float = 0.0
    persistent_failure_max_24h_mfe: float = 0.010
    continuation_post6_mfe_increment: float = 0.010
    giveback_activation_mfe: float = 0.010
    giveback_future_loss_from_checkpoint: float = 0.0075

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.2 must keep 2026 sealed")
        if self.entry_delay_minutes != 1:
            raise ValueError("R03.4.2.2 freezes next-minute-open entry")
        if self.analysis_horizon_hours != 48:
            raise ValueError("R03.4.2.2 requires the frozen 48h path")
        if self.fixed_diagnostic_horizon_hours != 6:
            raise ValueError("R03.4.2.2 freezes the original 6h diagnostic")
        if not 0.5 <= self.discovery_signal_quantile < self.primary_signal_quantile < 1.0:
            raise ValueError("discovery pool must be broader than the q90 primary pool")
        if tuple(sorted(set(self.checkpoints_minutes))) != self.checkpoints_minutes:
            raise ValueError("checkpoints must be unique and increasing")
        if self.checkpoints_minutes[-1] != 360:
            raise ValueError("the last causal checkpoint must be T+360m")
        if self.oof_embargo_hours < self.analysis_horizon_hours:
            raise ValueError("OOF embargo must cover the complete future path label")
        if "03_4_2_2" not in self.report_dir:
            raise ValueError("R03.4.2.2 report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["checkpoints_minutes"] = list(self.checkpoints_minutes)
        return payload


DEFAULT_LONG_TAIL_PATH_RECOGNITION_CONFIG = LongTailPathRecognitionConfig()
