#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen descriptive contract for R03.4.2.17.

This stage may diagnose 2026 failure and describe counterfactual state gates,
but it may not promote a new strategy or overwrite the failed V1 seal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.17"
STAGE_NAME = "2026 sealed-failure attribution and Long market-state gate diagnostic"


@dataclass(frozen=True)
class StateGateDiagnosticConfig:
    symbol: str = "ETH-USDT-SWAP"
    analysis_start: str = "2024-01-01 00:00:00"
    analysis_end: str = "2026-07-31 23:59:59"
    state_warmup_start: str = "2023-01-01 00:00:00"
    decision_interval_minutes: int = 15

    source_2_15_report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_15_final_account_live_readiness"
    )
    source_2_16_report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_16_2026_sealed_validation"
    )
    source_2_16_1_report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_16_1_2026_july_forward_extension"
    )
    source_2_16_base_cache_dir: str = (
        "data/cache/eth_ai_trading/r03_4_2_16_1_july_long_context"
    )
    source_2_16_outcome_cache_dir: str = (
        "data/cache/eth_ai_trading/r03_4_2_16_1_july_forward_outcomes"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_17_state_gate_diagnostic"
    )

    anchor_delay_minutes: int = 1
    anchor_cost_multiplier: float = 2.0
    base_round_trip_cost: float = 0.0013
    q70_quantile: float = 0.70

    # Economic/descriptive boundaries. These are not selected on 2026 returns.
    near_90d_high_drawdown: float = -0.10
    deep_90d_drawdown: float = -0.25
    high_vol_ratio: float = 1.25
    low_vol_ratio: float = 0.80

    # Frozen model recipe; must match R03.4.2.16/16.1.
    fit_start: str = "2023-01-01 00:00:00"
    calibration_start: str = "2025-10-01 00:00:00"
    calibration_end: str = "2025-12-31 23:59:59"
    embargo_hours: int = 18
    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801

    def validate(self) -> None:
        if pd.Timestamp(self.state_warmup_start) >= pd.Timestamp(self.analysis_start):
            raise ValueError("state warmup must precede analysis start")
        if pd.Timestamp(self.analysis_start) >= pd.Timestamp(self.analysis_end):
            raise ValueError("analysis chronology invalid")
        if self.decision_interval_minutes != 15:
            raise ValueError("frozen decision interval changed")
        if self.anchor_delay_minutes != 1 or abs(self.anchor_cost_multiplier - 2.0) > 1e-12:
            raise ValueError("anchor scenario changed")
        if abs(self.base_round_trip_cost - 0.0013) > 1e-12:
            raise ValueError("cost contract changed")
        if abs(self.q70_quantile - 0.70) > 1e-12:
            raise ValueError("q70 contract changed")
        if not self.deep_90d_drawdown < self.near_90d_high_drawdown < 0:
            raise ValueError("drawdown diagnostic bands invalid")
        if not 0 < self.low_vol_ratio < 1 < self.high_vol_ratio:
            raise ValueError("volatility diagnostic bands invalid")
        for token, value in (
            ("03_4_2_15", self.source_2_15_report_dir),
            ("03_4_2_16_2026", self.source_2_16_report_dir),
            ("03_4_2_16_1", self.source_2_16_1_report_dir),
            ("r03_4_2_16_1", self.source_2_16_base_cache_dir),
            ("r03_4_2_16_1", self.source_2_16_outcome_cache_dir),
            ("03_4_2_17", self.report_dir),
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
    def source_2_16_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_16_report_dir

    @property
    def source_2_16_1_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_16_1_report_dir

    @property
    def source_base_cache_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_16_base_cache_dir

    @property
    def source_outcome_cache_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_16_outcome_cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["fit_end"] = str(self.fit_end)
        payload["research_role"] = "diagnostic only; no strategy promotion and no repair of the opened 2026 seal"
        payload["frozen_v1_status"] = "FAIL_2026_SEALED_HOLDOUT; NOT LIVE APPROVED"
        payload["counterfactual_gate_status"] = "descriptive development evidence only; requires future untouched validation"
        return payload


DEFAULT_STATE_GATE_DIAGNOSTIC_CONFIG = StateGateDiagnosticConfig()
