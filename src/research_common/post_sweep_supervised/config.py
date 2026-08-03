#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R13 post-sweep supervised meta-labeling."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSweepSupervisedConfig:
    """Predeclared research design; values are not tuned on the final holdout."""

    train_end_exclusive: str = "2025-01-01"
    validation_end_exclusive: str = "2025-10-01"
    checkpoints_minutes: tuple[int, ...] = (0, 3, 5, 10)
    primary_checkpoint_minutes: int = 5
    primary_r_multiple: float = 2.0
    profitable_net_r_threshold: float = 0.25
    score_quantiles: tuple[float, ...] = (0.90, 0.95, 0.98)
    primary_score_quantile: float = 0.95
    minimum_train_events: int = 3_000
    minimum_validation_events: int = 600
    minimum_holdout_events: int = 600
    minimum_holdout_trades: int = 150
    minimum_module_coverage: float = 0.80
    minimum_pf: float = 1.30
    minimum_positive_month_rate: float = 0.70
    logistic_c: float = 0.10
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 220
    hgb_max_leaf_nodes: int = 15
    hgb_max_depth: int = 3
    hgb_min_samples_leaf: int = 100
    hgb_l2_regularization: float = 0.10
    random_state: int = 20260729
    trade_chunk_days: int = 7
    range_chunk_days: int = 30
    footprint_chunk_days: int = 120
    sample_rows: int = 10_000

    def validate(self) -> "PostSweepSupervisedConfig":
        if self.train_end_exclusive >= self.validation_end_exclusive:
            raise ValueError("train_end_exclusive must be before validation_end_exclusive")
        if not self.checkpoints_minutes or any(int(v) < 0 for v in self.checkpoints_minutes):
            raise ValueError("checkpoints_minutes must contain non-negative integers")
        if self.primary_checkpoint_minutes not in self.checkpoints_minutes:
            raise ValueError("primary checkpoint must be in checkpoints_minutes")
        if self.primary_r_multiple != 2.0:
            raise ValueError("R13 v1 freezes the primary path label at 2R")
        if not 0 < self.profitable_net_r_threshold < 2:
            raise ValueError("profitable_net_r_threshold must be in (0, 2)")
        if any(not 0 < float(q) < 1 for q in self.score_quantiles):
            raise ValueError("score_quantiles must be in (0, 1)")
        if self.primary_score_quantile not in self.score_quantiles:
            raise ValueError("primary_score_quantile must be predeclared in score_quantiles")
        if not 0 < self.minimum_module_coverage <= 1:
            raise ValueError("minimum_module_coverage must be in (0, 1]")
        if min(self.minimum_train_events, self.minimum_validation_events, self.minimum_holdout_events) <= 0:
            raise ValueError("minimum split event counts must be positive")
        if self.minimum_holdout_trades <= 0:
            raise ValueError("minimum_holdout_trades must be positive")
        if self.hgb_max_depth <= 0 or self.hgb_max_leaf_nodes <= 1 or self.hgb_min_samples_leaf <= 1:
            raise ValueError("invalid HGB complexity controls")
        if min(self.trade_chunk_days, self.range_chunk_days, self.footprint_chunk_days) <= 0:
            raise ValueError("chunk sizes must be positive")
        return self
