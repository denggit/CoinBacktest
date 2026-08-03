#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.4 q70 cross-year audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.4"
STAGE_NAME = "Frozen q70 versus q90 cross-year opening-pool audit"


@dataclass(frozen=True)
class Q70CrossYearAuditConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    diagnostic_horizon_hours: int = 6
    risk_penalty: float = 1.25
    evaluation_quantiles: tuple[float, ...] = (0.70, 0.90)
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    episode_merge_gap_minutes: int = 30
    independent_event_cooldown_hours: int = 6

    base_train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801

    minimum_pf_2x: float = 1.40
    minimum_incremental_band_pf_2x: float = 1.15
    maximum_drawdown: float = 0.20
    maximum_top10_profit_share: float = 0.60
    minimum_positive_quarters: int = 6
    minimum_trades_per_year: int = 150
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_4_q70_cross_year_audit"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.4 must keep 2026 sealed")
        if self.diagnostic_horizon_hours != 6:
            raise ValueError("the frozen opening model remains a six-hour diagnostic target")
        if self.risk_penalty != 1.25:
            raise ValueError("the frozen opening target keeps risk_penalty=1.25")
        if self.evaluation_quantiles != (0.70, 0.90):
            raise ValueError("R03.4.2.4 audits only q70 and q90")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay grid is frozen")
        if self.cost_multipliers != (1.0, 2.0, 3.0):
            raise ValueError("cost-stress grid is frozen")
        if self.independent_event_cooldown_hours < self.diagnostic_horizon_hours:
            raise ValueError("event cooldown must cover the diagnostic horizon")
        if "03_4_2_4" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["evaluation_quantiles"] = list(self.evaluation_quantiles)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        return payload


DEFAULT_Q70_CROSS_YEAR_AUDIT_CONFIG = Q70CrossYearAuditConfig()
