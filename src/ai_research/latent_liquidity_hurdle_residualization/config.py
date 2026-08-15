#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R02.3.1 zero-inflated nuisance residualization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02.3.1"
STAGE_NAME = "Zero-inflated nuisance residualization + reversal residual ranking"


@dataclass(frozen=True)
class HurdleResidualizationConfig:
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_3_1_hurdle_nuisance_residualization"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r02_3_1_hurdle_nuisance_residualization"
    )
    primary_label_window_seconds: int = 180
    expected_zone_count: int = 25
    train_period: str = "TRAIN_2023_2024"
    calibration_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )
    nuisance_initial_train_months: int = 6
    nuisance_forward_block_months: int = 6
    nuisance_purge_hours: int = 13
    split_boundary_purge_hours: int = 13
    nuisance_model_n_estimators: int = 160
    nuisance_model_learning_rate: float = 0.035
    nuisance_model_num_leaves: int = 15
    nuisance_model_max_depth: int = 4
    nuisance_model_min_child_samples: int = 160
    nuisance_train_cap_rows_per_side: int = 220_000
    nuisance_min_rows: int = 2_000
    nuisance_min_class_rows: int = 200
    nuisance_min_positive_rows: int = 500
    rank_relevance_grades: int = 5
    minimum_rank_groups: int = 400
    model_train_cap_rows_per_side: int = 180_000
    model_n_estimators: int = 320
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 80
    minimum_regression_rows: int = 1_000
    minimum_top1_touched: int = 50
    promotion_min_excess_spearman: float = 0.08
    promotion_min_reversal_spearman: float = 0.08
    promotion_min_top1_actual_expected_ratio: float = 1.20
    promotion_min_sweep_depth_spearman: float = 0.20
    promotion_min_oracle_top3_rate: float = 0.30
    continue_min_stable_spearman: float = 0.04
    residualization_max_abs_distance_spearman: float = 0.12
    reversal_residualization_max_abs_distance_spearman: float = 0.15
    random_state: int = 20260807

    def validate(self) -> None:
        if self.primary_label_window_seconds != 180:
            raise ValueError("R02.3.1 primary first-touch window is frozen at 180 seconds")
        if self.expected_zone_count != 25:
            raise ValueError("R02.3.1 expects the frozen 25-zone lattice per side")
        if self.nuisance_initial_train_months < 3:
            raise ValueError("nuisance initial train window is too small")
        if self.nuisance_forward_block_months < 1:
            raise ValueError("nuisance forward block must be positive")
        if self.nuisance_purge_hours < 12:
            raise ValueError("nuisance purge must cover the 12h first-touch horizon")
        if self.split_boundary_purge_hours < 12:
            raise ValueError("period-boundary purge must cover the 12h first-touch horizon")
        if self.rank_relevance_grades < 3:
            raise ValueError("ranking relevance requires at least 3 grades")
        if self.train_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("R02.3.1 frozen periods are inconsistent")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("R02.3.1 report must remain in the model namespace")

    @property
    def report_path(self) -> Path:
        self.validate()
        p = Path(self.report_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def cache_path(self) -> Path:
        self.validate()
        p = Path(self.cache_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        return payload


DEFAULT_CONFIG = HurdleResidualizationConfig()
