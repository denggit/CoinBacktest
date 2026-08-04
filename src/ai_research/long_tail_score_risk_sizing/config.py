#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.13 score-tier risk sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.13"
STAGE_NAME = "q70/q80/q90 score-tier risk sizing and account scaling"
SCORE_TIERS = ("q70_to_q80", "q80_to_q90", "q90_plus")


@dataclass(frozen=True)
class ScoreRiskPolicy:
    name: str
    q70_to_q80_r: float
    q80_to_q90_r: float
    q90_plus_r: float
    qualifying_candidate: bool = True
    diagnostic_only: bool = False

    def validate(self) -> None:
        values = self.risk_map
        if any(not 0.25 <= value <= 1.25 for value in values.values()):
            raise ValueError(f"invalid risk multiplier for {self.name}: {values}")
        if self.qualifying_candidate and max(values.values()) > 1.0 + 1e-12:
            raise ValueError(f"qualifying policy exceeds one-R: {self.name}")
        if self.diagnostic_only and self.qualifying_candidate:
            raise ValueError(f"diagnostic policy cannot qualify: {self.name}")

    @property
    def risk_map(self) -> dict[str, float]:
        return {
            "q70_to_q80": float(self.q70_to_q80_r),
            "q80_to_q90": float(self.q80_to_q90_r),
            "q90_plus": float(self.q90_plus_r),
        }

    @property
    def max_tail_r(self) -> float:
        return max(self.risk_map.values())


@dataclass(frozen=True)
class ScoreRiskConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    source_2_12_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_13_score_risk_sizing"

    source_policy: str = "C2_real_2p_soft1p5"
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    initial_equity: float = 1.0

    policies: tuple[ScoreRiskPolicy, ...] = (
        ScoreRiskPolicy("E075_equal_0p75R", 0.75, 0.75, 0.75, qualifying_candidate=False, diagnostic_only=True),
        ScoreRiskPolicy("E100_equal_1R", 1.00, 1.00, 1.00, qualifying_candidate=False),
        ScoreRiskPolicy("T1_mild_0p75_0p90_1p00", 0.75, 0.90, 1.00),
        ScoreRiskPolicy("T2_low_only_0p75_1p00_1p00", 0.75, 1.00, 1.00),
        ScoreRiskPolicy("T3_strong_0p60_0p80_1p00", 0.60, 0.80, 1.00),
        ScoreRiskPolicy("E125_equal_1p25R_DIAG", 1.25, 1.25, 1.25, qualifying_candidate=False, diagnostic_only=True),
    )

    minimum_return_retention_each_year: float = 0.95
    minimum_combined_return_ratio: float = 0.95
    maximum_mdd_multiple: float = 1.00
    minimum_calmar_improvement: float = 1.05
    minimum_positive_quarters_per_year: int = 3
    maximum_top10_profit_share_increase: float = 0.08
    maximum_candidate_tail_r: float = 1.0
    maximum_absolute_mdd: float = 0.12

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.13 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost grid changed")
        if not 0 < self.account_risk_fraction_per_full_r <= 0.02:
            raise ValueError("invalid one-R account fraction")
        if "03_4_2_12" not in self.source_2_12_report_dir:
            raise ValueError("2.12 source path drift")
        if "03_4_2_13" not in self.report_dir:
            raise ValueError("report path must be isolated")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        if "E100_equal_1R" not in names:
            raise ValueError("equal one-R anchor is required")
        for policy in self.policies:
            policy.validate()

    @property
    def source_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_12_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [
            {**asdict(policy), "risk_map": policy.risk_map, "max_tail_r": policy.max_tail_r}
            for policy in self.policies
        ]
        payload["frozen_execution"] = "q70 next 1m open + C2 real 2% stop + 1.5% completed-close soft failure + failed_reclaim"
        payload["position_formula"] = "equity * 1% * tier multiplier / executable 2% hard-stop distance"
        payload["holdout"] = "2026 sealed"
        return payload


DEFAULT_SCORE_RISK_CONFIG = ScoreRiskConfig()
