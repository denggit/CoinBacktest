#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R02.1 conditional liquidity-pool strength learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02.1"
STAGE_NAME = "Conditional pool-strength / release-density deconfounding"


@dataclass(frozen=True)
class LatentLiquidityPoolStrengthConfig:
    symbol: str = "ETH-USDT-SWAP"
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_1_pool_strength_density_model"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r02_1_pool_strength_density_model"
    )
    primary_horizon_minutes: int = 720
    strength_quantile: float = 0.80
    model_train_cap_rows: int = 650_000
    model_eval_cap_rows_per_period: int = 450_000
    aggregation_decision_chunk_size: int = 1024
    random_state: int = 20260807
    model_n_estimators: int = 360
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 100
    train_period: str = "TRAIN_2023_2024"
    calibration_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )
    minimum_train_rows: int = 10_000
    minimum_class_rows: int = 500
    minimum_top1_touched: int = 50

    def validate(self) -> None:
        if self.primary_horizon_minutes <= 0:
            raise ValueError("primary horizon must be positive")
        if not 0.5 < self.strength_quantile < 0.95:
            raise ValueError("strength quantile must be in (0.5,0.95)")
        if self.train_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("frozen periods are inconsistent")
        if self.aggregation_decision_chunk_size < 64:
            raise ValueError("aggregation_decision_chunk_size must be >= 64")
        if self.minimum_top1_touched < 1:
            raise ValueError("minimum top1 touched count must be positive")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("R02.1 report must remain in model namespace")

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


DEFAULT_CONFIG = LatentLiquidityPoolStrengthConfig()
