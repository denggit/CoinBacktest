#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.1 q90 event path atlas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.1"
STAGE_NAME = "Frozen q90 long-event complete path atlas and ex-post path typology"


@dataclass(frozen=True)
class LongTailPathAtlasConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    primary_signal_quantile: float = 0.90
    quality_control_quantile: float = 0.95
    entry_delay_minutes: int = 1
    analysis_horizon_hours: int = 48
    fixed_diagnostic_horizon_hours: int = 6
    risk_penalty: float = 1.25
    base_round_trip_cost: float = 0.0013
    checkpoint_minutes: tuple[int, ...] = (5, 15, 30, 60, 120, 180, 360, 720, 1440, 2880)
    upside_levels: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020, 0.030)
    downside_levels: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020, 0.030)
    cluster_count: int = 6
    minimum_discovery_events: int = 36
    minimum_oos_events_per_year: int = 100
    minimum_type_samples: int = 10
    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801
    export_full_minute_paths: bool = True
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_1_long_tail_path_atlas"

    # Fixed semantic path definitions. These are descriptive, not trading rules.
    immediate_target_pct: float = 0.010
    immediate_target_minutes: int = 60
    immediate_max_mae_before_target: float = 0.005
    delayed_recovery_mae: float = 0.0075
    delayed_recovery_underwater_minutes: int = 120
    early_spike_mfe: float = 0.015
    early_spike_minutes: int = 180
    early_spike_giveback: float = 0.0075
    slow_grind_min_mfe: float = 0.010
    slow_grind_peak_after_minutes: int = 180
    post6_continuation_increment: float = 0.010
    persistent_failure_max_24h_mfe: float = 0.010

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.1 must keep 2026 sealed")
        if self.primary_signal_quantile != 0.90:
            raise ValueError("R03.4.2.1 freezes q90 as the primary signal")
        if self.entry_delay_minutes != 1:
            raise ValueError("R03.4.2.1 freezes next-minute-open entry")
        if self.analysis_horizon_hours != 48:
            raise ValueError("R03.4.2.1 freezes a 48-hour complete path atlas")
        if self.fixed_diagnostic_horizon_hours != 6:
            raise ValueError("R03.4.2.1 freezes the original six-hour diagnostic")
        if self.risk_penalty != 1.25:
            raise ValueError("R03.4.2.1 freezes the original utility penalty")
        if tuple(sorted(set(self.checkpoint_minutes))) != self.checkpoint_minutes:
            raise ValueError("checkpoint minutes must be unique and increasing")
        if self.checkpoint_minutes[-1] != self.analysis_horizon_hours * 60:
            raise ValueError("last checkpoint must equal the full path horizon")
        if self.cluster_count < 3 or self.cluster_count > 10:
            raise ValueError("cluster_count must remain interpretable")
        if self.minimum_discovery_events < self.cluster_count * 4:
            raise ValueError("minimum discovery sample is too small for clustering")
        if "03_4_2_1" not in self.report_dir:
            raise ValueError("R03.4.2.1 report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["checkpoint_minutes"] = list(self.checkpoint_minutes)
        payload["upside_levels"] = list(self.upside_levels)
        payload["downside_levels"] = list(self.downside_levels)
        return payload


DEFAULT_LONG_TAIL_PATH_ATLAS_CONFIG = LongTailPathAtlasConfig()
