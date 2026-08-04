#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Immutable configuration for R03.4.2.16 one-time 2026 sealed validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.16"
STAGE_NAME = "one-time 2026 sealed validation of frozen C2 MF Long sleeve"


@dataclass(frozen=True)
class SealedHoldoutConfig:
    symbol: str = "ETH-USDT-SWAP"

    fit_start: str = "2023-01-01 00:00:00"
    calibration_start: str = "2025-10-01 00:00:00"
    calibration_end: str = "2025-12-31 23:59:59"
    holdout_start: str = "2026-01-01 00:00:00"
    holdout_end: str = "2026-06-30 23:59:59"
    post_holdout_boundary: str = "2026-07-01 00:00:00"
    embargo_hours: int = 18

    source_2_15_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_15_final_account_live_readiness"
    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_16_2026_sealed_validation"
    isolated_outcome_cache_dir: str = "data/cache/eth_ai_trading/r03_4_2_16_sealed_outcomes"

    evaluation_quantile: float = 0.70
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    live_price_risk_fraction: float = 0.0084
    initial_equity: float = 1.0
    hard_stop_distance: float = 0.02
    soft_failure_distance: float = 0.015

    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801
    episode_merge_gap_minutes: int = 30
    independent_event_cooldown_hours: int = 6

    anchor_delay_minutes: int = 1
    anchor_cost_multiplier: float = 2.0
    minimum_executed_cycles: int = 60
    minimum_anchor_total_return: float = 0.0
    minimum_anchor_profit_factor: float = 1.20
    maximum_anchor_mdd: float = 0.15
    maximum_stress_mdd: float = 0.18
    maximum_worst_net_r: float = 1.25
    minimum_positive_months: int = 3
    minimum_positive_quarters: int = 1
    maximum_top10_profit_share: float = 0.75
    minimum_return_without_top10: float = -0.05
    maximum_censored_cycles: int = 1

    def validate(self) -> None:
        fit_start = pd.Timestamp(self.fit_start)
        calibration_start = pd.Timestamp(self.calibration_start)
        calibration_end = pd.Timestamp(self.calibration_end)
        holdout_start = pd.Timestamp(self.holdout_start)
        holdout_end = pd.Timestamp(self.holdout_end)
        boundary = pd.Timestamp(self.post_holdout_boundary)
        fit_end = calibration_start - pd.Timedelta(hours=self.embargo_hours)
        if not fit_start < fit_end < calibration_start <= calibration_end < holdout_start <= holdout_end < boundary:
            raise ValueError("invalid frozen fit/calibration/holdout chronology")
        if self.evaluation_quantile != 0.70:
            raise ValueError("q70 opening threshold is frozen")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost stress grid changed")
        if abs(self.hard_stop_distance - 0.02) > 1e-12:
            raise ValueError("real 2% hard stop changed")
        if abs(self.soft_failure_distance - 0.015) > 1e-12:
            raise ValueError("1.5% completed-close soft failure changed")
        if abs(self.account_risk_fraction_per_full_r - 0.01) > 1e-12:
            raise ValueError("research one-R definition changed")
        if not 0 < self.live_price_risk_fraction <= self.account_risk_fraction_per_full_r:
            raise ValueError("invalid deployment price-risk budget")
        if self.anchor_delay_minutes not in self.entry_delay_minutes:
            raise ValueError("anchor delay missing from grid")
        if self.anchor_cost_multiplier not in self.cost_multipliers:
            raise ValueError("anchor cost missing from grid")
        for token, value in (
            ("03_4_2_15", self.source_2_15_report_dir),
            ("03_4_2_8a", self.source_2_8a_report_dir),
            ("03_4_2_16", self.report_dir),
            ("r03_4_2_16", self.isolated_outcome_cache_dir),
        ):
            if token not in value:
                raise ValueError(f"path drift: expected {token}")

    @property
    def fit_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.calibration_start) - pd.Timedelta(hours=self.embargo_hours)

    @property
    def source_2_15_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_15_report_dir

    @property
    def source_2_8a_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8a_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    @property
    def isolated_outcome_cache_path(self) -> Path:
        return PROJECT_ROOT / self.isolated_outcome_cache_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["fit_end"] = str(self.fit_end)
        payload["frozen_strategy"] = (
            "q70 immediate next 1m open + equal 1R + real 2% hard stop + "
            "1.5% completed 15m-close soft failure + failed_reclaim + no add-on + no fixed TP"
        )
        payload["holdout_use"] = "2026-01-01 through 2026-06-30; labels may score results only"
        payload["post_holdout_tuning"] = "FORBIDDEN"
        return payload


DEFAULT_SEALED_HOLDOUT_CONFIG = SealedHoldoutConfig()
