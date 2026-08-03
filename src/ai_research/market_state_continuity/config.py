#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.3.3 multi-timescale market-state continuity research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


STAGE_ID = "R03.3.3.1"
STAGE_NAME = "Market-state continuity calibration and transition-warning audit"


@dataclass(frozen=True)
class StateTargetSpec:
    target_id: str
    state_column: str
    horizon_hours: int
    layer: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_STATE_TARGETS = (
    StateTargetSpec("strategic_persist_h24", "strategic_state", 24, "strategic"),
    StateTargetSpec("strategic_persist_h72", "strategic_state", 72, "strategic"),
    StateTargetSpec("tactical_persist_h3", "tactical_state", 3, "tactical"),
    StateTargetSpec("tactical_persist_h6", "tactical_state", 6, "tactical"),
    StateTargetSpec("entry_persist_h1", "entry_state", 1, "entry"),
    StateTargetSpec("activity_persist_h3", "activity_state", 3, "activity"),
)


@dataclass(frozen=True)
class MarketStateContinuityConfig:
    symbol: str = "ETH-USDT-SWAP"
    source_timeframe: str = "1m"
    ordinary_kline_end: str = "2021-12-31 23:59:59"
    trade_bar_start: str = "2022-01-01 00:00:00"
    warmup_start: str = "2020-01-01 00:00:00"
    research_start: str = "2021-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    decision_interval_minutes: int = 15
    feature_lookback_days: int = 420
    structural_swing_bars_4h: int = 30
    cache_dir: str = "data/cache/eth_ai_trading/r03_3_3_1_universal_state"
    report_dir: str = "data/reports/research/eth_ai_trading/03_3_3_1_market_state_continuity_audit"
    targets: tuple[StateTargetSpec, ...] = field(default_factory=lambda: DEFAULT_STATE_TARGETS)
    architectures: tuple[str, ...] = ("universal_ohlcv_lightgbm", "trade_enhanced_lightgbm")
    # Strategic state is calibrated causally from prior daily scores. It is intentionally
    # not forced through the faster tactical/entry fixed threshold.
    strategic_threshold_lookback_days: int = 365
    strategic_threshold_min_days: int = 180
    strategic_long_enter_quantile: float = 0.85
    strategic_short_enter_quantile: float = 0.15
    strategic_long_exit_quantile: float = 0.60
    strategic_short_exit_quantile: float = 0.40
    strategic_fallback_long_enter: float = 0.12
    strategic_fallback_short_enter: float = -0.06
    strategic_fallback_long_exit: float = 0.04
    strategic_fallback_short_exit: float = -0.01
    direction_enter_threshold: float = 0.30
    direction_exit_threshold: float = 0.10
    activity_enter_threshold: float = 0.25
    activity_exit_threshold: float = 0.05
    minimum_auc: float = 0.60
    minimum_brier_skill: float = 0.0
    minimum_transition_lift: float = 1.25
    minimum_auc_increment_vs_mechanical: float = 0.01
    transition_alert_train_quantile: float = 0.10
    transition_alert_merge_gap_minutes: int = 60
    maximum_rows_per_fit: int = 500_000
    mechanical_baseline_max_rows: int = 250_000
    random_state: int = 42

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def report_path(self) -> Path:
        return Path(self.report_dir)

    @property
    def maximum_target_horizon_hours(self) -> int:
        return max(spec.horizon_hours for spec in self.targets)

    def target_names(self) -> tuple[str, ...]:
        return tuple(spec.target_id for spec in self.targets)

    def validate(self) -> None:
        if pd.Timestamp(self.warmup_start) >= pd.Timestamp(self.research_start):
            raise ValueError("R03.3.3 warmup must precede research start")
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.3.3 must keep 2026 sealed")
        # The boundary is intentionally adjacent, never overlapping or leaving a gap.
        if pd.Timestamp(self.ordinary_kline_end) + pd.Timedelta(seconds=1) != pd.Timestamp(self.trade_bar_start):
            raise ValueError("ordinary K-line and Trade Bar boundaries must be adjacent")
        if self.decision_interval_minutes != 15:
            raise ValueError("R03.3.3 frozen decision interval is 15 minutes")
        if self.feature_lookback_days < 400:
            raise ValueError("R03.3.3 requires at least 400 days of causal warmup")
        quantiles = (
            self.strategic_short_enter_quantile,
            self.strategic_short_exit_quantile,
            self.strategic_long_exit_quantile,
            self.strategic_long_enter_quantile,
        )
        if not (0 < quantiles[0] < quantiles[1] < quantiles[2] < quantiles[3] < 1):
            raise ValueError("invalid strategic causal quantile thresholds")
        if self.strategic_threshold_min_days < 60:
            raise ValueError("strategic threshold calibration needs at least 60 prior days")
        if self.strategic_threshold_lookback_days < self.strategic_threshold_min_days:
            raise ValueError("strategic threshold lookback must cover minimum calibration days")
        if not (0 < self.direction_exit_threshold < self.direction_enter_threshold < 1):
            raise ValueError("invalid direction hysteresis thresholds")
        if not (0 <= self.activity_exit_threshold < self.activity_enter_threshold < 1):
            raise ValueError("invalid activity hysteresis thresholds")
        if not (0 < self.transition_alert_train_quantile < 0.5):
            raise ValueError("transition alert quantile must be in (0, 0.5)")
        if self.transition_alert_merge_gap_minutes < self.decision_interval_minutes:
            raise ValueError("transition alert merge gap must cover at least one decision interval")
        if self.minimum_auc_increment_vs_mechanical < 0:
            raise ValueError("mechanical AUC increment gate cannot be negative")
        if not self.targets:
            raise ValueError("at least one continuity target is required")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["targets"] = [spec.to_dict() for spec in self.targets]
        return payload


DEFAULT_MARKET_STATE_CONTINUITY_CONFIG = MarketStateContinuityConfig()
