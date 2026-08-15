#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R02.2 first-touch relative liquidity ranking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02.2"
STAGE_NAME = "First-touch relative liquidity ranking"


@dataclass(frozen=True)
class FirstTouchLiquidityRankingConfig:
    symbol: str = "ETH-USDT-SWAP"
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_2_first_touch_relative_liquidity_ranking"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r02_2_first_touch_relative_liquidity_ranking"
    )
    primary_horizon_minutes: int = 720
    label_windows_seconds: tuple[int, ...] = (30, 60, 180, 300)
    primary_label_window_seconds: int = 180
    touch_replay_chunk_days: int = 7
    max_fill_gap_seconds: int = 5
    pre_touch_baseline_seconds: int = 60
    model_train_cap_rows_per_side: int = 180_000
    random_state: int = 20260807
    rank_relevance_grades: int = 5
    model_n_estimators: int = 320
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 80
    train_period: str = "TRAIN_2023_2024"
    calibration_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )
    minimum_rank_groups: int = 500
    minimum_top1_touched: int = 50
    promotion_min_group_spearman: float = 0.15
    promotion_min_top1_density_lift: float = 1.25
    promotion_min_oracle_top3_rate: float = 0.30

    def validate(self) -> None:
        if self.primary_label_window_seconds not in self.label_windows_seconds:
            raise ValueError("primary label window must be listed")
        if tuple(sorted(set(self.label_windows_seconds))) != self.label_windows_seconds:
            raise ValueError("label windows must be unique and increasing")
        if self.primary_horizon_minutes <= 0:
            raise ValueError("primary touch horizon must be positive")
        if self.touch_replay_chunk_days < 1:
            raise ValueError("touch replay chunk days must be positive")
        if self.rank_relevance_grades < 3:
            raise ValueError("ranking relevance requires at least 3 grades")
        if self.train_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("frozen periods are inconsistent")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("R02.2 report must remain in model namespace")

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
        payload["label_windows_seconds"] = list(self.label_windows_seconds)
        payload["periods"] = list(self.periods)
        return payload


DEFAULT_CONFIG = FirstTouchLiquidityRankingConfig()
