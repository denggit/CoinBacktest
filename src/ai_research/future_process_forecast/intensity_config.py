#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.3.2 continuous future-opportunity intensity research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

from .config import DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG, FutureProcessForecastConfig


STAGE_ID = "R03.3.2"
STAGE_NAME = "Continuous future opportunity intensity forecast"

TARGET_FAMILIES = (
    "future_range_pct",
    "future_max_directional_pct",
    "future_two_sided_pct",
    "future_range_atr_multiple",
)

CURRENT_STATE_DEFINITION: dict[str, dict[str, object]] = {
    "strategic_structure": {
        "meaning": "长期方向、价格所处长期区间位置、趋势年龄与长期回撤/反弹阶段",
        "feature_prefixes": ["tf1d_", "tf4h_", "cross_d1_", "cross_long_"],
    },
    "medium_process": {
        "meaning": "1H/4H推动、回调、恢复、趋势效率与结构延续状态",
        "feature_prefixes": ["tf4h_", "tf1h_", "cross_4h_"],
    },
    "short_structure": {
        "meaning": "30m/15m/5m/1m当前突破距离、区间位置、短周期方向和回调状态",
        "feature_prefixes": ["tf30m_", "tf15m_", "tf5m_", "tf1m_"],
    },
    "volatility_phase": {
        "meaning": "当前波动水平、压缩/扩张、波动率加速或衰退",
        "feature_keywords": ["rv_", "atr_pct", "volatility", "range_"],
    },
    "flow_impact": {
        "meaning": "主动买卖压力、大单压力、价格冲击效率、吸收和压力持续性",
        "feature_keywords": ["flow_imb", "large_flow", "taker_buy", "absorption", "impact", "pressure"],
    },
    "activity_liquidity_proxy": {
        "meaning": "成交活跃度、成交量异常、距离高低点和突破边界的位置",
        "feature_keywords": ["volume_z", "trade", "notional", "breakout_", "bars_since", "range_pos"],
    },
}


@dataclass(frozen=True)
class FutureIntensityConfig:
    base: FutureProcessForecastConfig = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    horizons_hours: tuple[int, ...] = (6, 12)
    target_families: tuple[str, ...] = TARGET_FAMILIES
    architectures: tuple[str, ...] = (
        "macro_lightgbm",
        "multiframe_lightgbm",
        "multiframe_micro_lightgbm",
    )
    sample_stride_decisions: int = 4
    train_sample_cap: int = 300_000
    target_clip_quantile: float = 0.999
    rank_quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)
    lightgbm_n_estimators: int = 420
    lightgbm_learning_rate: float = 0.035
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 300
    lightgbm_feature_fraction: float = 0.80
    minimum_rank_ic: float = 0.10
    minimum_top_decile_lift: float = 1.20
    minimum_decile_monotonicity: float = 0.70
    minimum_test_rows: int = 10_000
    target_cache_dir: str = "data/cache/eth_ai_trading/r03_3_2_future_intensity"
    report_dir: str = "data/reports/research/eth_ai_trading/03_3_2_future_intensity"

    def validate(self) -> None:
        self.base.validate()
        if tuple(sorted(set(self.horizons_hours))) != self.horizons_hours:
            raise ValueError("R03.3.2 horizons must be unique and increasing")
        if not self.horizons_hours or max(self.horizons_hours) > 24:
            raise ValueError("R03.3.2 first pass supports horizons up to 24h")
        if tuple(self.target_families) != TARGET_FAMILIES:
            raise ValueError("R03.3.2 target contract is frozen")
        if not set(self.architectures).issubset(set(self.base.architectures)):
            raise ValueError("R03.3.2 must reuse R03.3 feature architectures")
        if self.sample_stride_decisions < 1:
            raise ValueError("sample stride must be positive")
        if not 0.95 <= self.target_clip_quantile < 1.0:
            raise ValueError("invalid target clipping quantile")
        if tuple(sorted(set(self.rank_quantiles))) != self.rank_quantiles:
            raise ValueError("rank quantiles must be unique and increasing")
        if not 0 < self.minimum_rank_ic < 1:
            raise ValueError("invalid rank IC gate")
        if "r03_3_2" not in self.target_cache_dir or "03_3_2" not in self.report_dir:
            raise ValueError("R03.3.2 caches and reports must be isolated")
        if pd.Timestamp(self.base.research_end) >= pd.Timestamp(self.base.sealed_holdout_start):
            raise ValueError("2026 holdout must remain sealed")

    @property
    def target_cache_path(self) -> Path:
        return PROJECT_ROOT / self.target_cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def target_names(self) -> tuple[str, ...]:
        return tuple(f"{family}_h{h}" for family in self.target_families for h in self.horizons_hours)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["horizons_hours"] = list(self.horizons_hours)
        payload["target_families"] = list(self.target_families)
        payload["architectures"] = list(self.architectures)
        payload["rank_quantiles"] = list(self.rank_quantiles)
        payload["current_state_definition"] = CURRENT_STATE_DEFINITION
        return payload


DEFAULT_FUTURE_INTENSITY_CONFIG = FutureIntensityConfig()
