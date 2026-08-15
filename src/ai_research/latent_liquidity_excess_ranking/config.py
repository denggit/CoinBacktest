#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R02.3 distance-normalized excess liquidity ranking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02.3"
STAGE_NAME = "Distance-normalized excess liquidity + reversal quality ranking"


@dataclass(frozen=True)
class ExcessLiquidityRankingConfig:
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_3_distance_normalized_excess_liquidity_ranking"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r02_3_distance_normalized_excess_liquidity_ranking"
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
    model_train_cap_rows_per_side: int = 180_000
    minimum_rank_groups: int = 500
    rank_relevance_grades: int = 5
    model_n_estimators: int = 320
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 80
    random_state: int = 20260807
    normalizer_min_rows_per_distance: int = 50
    normalizer_iqr_floor: float = 0.10
    minimum_regression_rows: int = 1_000
    minimum_top1_touched: int = 50
    promotion_min_excess_spearman: float = 0.10
    promotion_min_excess_top1_ratio: float = 1.20
    promotion_min_reversal_spearman: float = 0.08
    promotion_min_sweep_depth_spearman: float = 0.20
    promotion_min_oracle_top3_rate: float = 0.30

    def validate(self) -> None:
        if self.primary_label_window_seconds != 180:
            raise ValueError("R02.3 primary first-touch window is frozen at 180 seconds")
        if self.expected_zone_count != 25:
            raise ValueError("R02.3 expects the frozen 25-zone lattice per side")
        if self.rank_relevance_grades < 3:
            raise ValueError("R02.3 requires at least 3 relevance grades")
        if self.normalizer_min_rows_per_distance < 2:
            raise ValueError("distance normalizer support is too small")
        if self.normalizer_iqr_floor <= 0:
            raise ValueError("normalizer IQR floor must be positive")
        if self.minimum_regression_rows < 20:
            raise ValueError("minimum regression rows is too small")
        if self.train_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("R02.3 frozen periods are inconsistent")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("R02.3 report must remain in the model namespace")

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


DEFAULT_CONFIG = ExcessLiquidityRankingConfig()
