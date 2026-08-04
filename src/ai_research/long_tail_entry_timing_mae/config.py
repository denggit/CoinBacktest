#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.14 entry timing and MAE attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.14"
STAGE_NAME = "frozen C2 entry timing and MAE attribution"


@dataclass(frozen=True)
class EntryTimingPolicy:
    name: str
    mode: str
    max_wait_minutes: int = 0
    minimum_score_increase: float = 0.0
    maximum_chase_atr: float = 0.25
    pullback_fraction: float = 0.005
    reclaim_tolerance_fraction: float = 0.001
    qualifying_candidate: bool = True

    def validate(self) -> None:
        allowed = {"immediate", "score_rise", "score_rise_no_chase", "pullback_reclaim"}
        if self.mode not in allowed:
            raise ValueError(f"unsupported entry mode for {self.name}: {self.mode}")
        if self.mode == "immediate" and self.max_wait_minutes != 0:
            raise ValueError("immediate anchor cannot wait")
        if self.mode != "immediate" and not 15 <= self.max_wait_minutes <= 60:
            raise ValueError(f"invalid bounded wait for {self.name}")
        if self.minimum_score_increase < 0:
            raise ValueError("score increase cannot be negative")
        if not 0 <= self.maximum_chase_atr <= 1.0:
            raise ValueError("invalid chase allowance")
        if not 0 < self.pullback_fraction < 0.02:
            raise ValueError("invalid pullback fraction")
        if not 0 <= self.reclaim_tolerance_fraction < self.pullback_fraction:
            raise ValueError("invalid reclaim tolerance")


@dataclass(frozen=True)
class EntryTimingConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    source_2_12_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression"
    source_2_13_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_13_score_risk_sizing"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_14_entry_timing_mae"

    source_policy: str = "E100_equal_1R"
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    initial_equity: float = 1.0
    hard_stop_distance: float = 0.02
    soft_failure_distance: float = 0.015

    policies: tuple[EntryTimingPolicy, ...] = (
        EntryTimingPolicy("E0_immediate_C2", "immediate", qualifying_candidate=False),
        EntryTimingPolicy("E1_score_rise_30m", "score_rise", max_wait_minutes=30),
        EntryTimingPolicy("E2_score_rise_no_chase_45m", "score_rise_no_chase", max_wait_minutes=45, maximum_chase_atr=0.25),
        EntryTimingPolicy("E3_pullback_reclaim_60m", "pullback_reclaim", max_wait_minutes=60, pullback_fraction=0.005, reclaim_tolerance_fraction=0.001),
    )

    minimum_coverage_ratio: float = 0.90
    minimum_return_retention_each_year: float = 0.95
    minimum_combined_return_ratio: float = 1.00
    maximum_mdd_multiple: float = 1.10
    maximum_absolute_mdd: float = 0.12
    minimum_positive_quarters_per_year: int = 3
    minimum_mae60_improvement: float = 0.10
    minimum_win_rate_improvement: float = 0.01
    minimum_stop_share_reduction: float = 0.10
    maximum_top10_profit_share_increase: float = 0.08

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.14 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost grid changed")
        if abs(self.hard_stop_distance - 0.02) > 1e-12:
            raise ValueError("C2 real 2% hard stop must remain frozen")
        if abs(self.soft_failure_distance - 0.015) > 1e-12:
            raise ValueError("C2 1.5% soft failure must remain frozen")
        if not 0 < self.account_risk_fraction_per_full_r <= 0.02:
            raise ValueError("invalid one-R account fraction")
        for token, path in (("03_4_2_8a", self.source_2_8a_report_dir), ("03_4_2_12", self.source_2_12_report_dir), ("03_4_2_13", self.source_2_13_report_dir), ("03_4_2_14", self.report_dir)):
            if token not in path:
                raise ValueError(f"source/report path drift: {token}")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)) or "E0_immediate_C2" not in names:
            raise ValueError("entry policies must be unique and retain E0")
        for policy in self.policies:
            policy.validate()

    @property
    def source_2_8a_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8a_report_dir

    @property
    def source_2_12_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_12_report_dir

    @property
    def source_2_13_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_13_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [asdict(policy) for policy in self.policies]
        payload["frozen_strategy"] = "q70 + equal 1R + real 2% hard stop + 1.5% completed-close soft failure + failed_reclaim"
        payload["coverage_contract"] = "entry timing may wait at most 60 minutes and must retain at least 90% of frozen C2 cycles"
        payload["holdout"] = "2026 sealed"
        return payload


DEFAULT_ENTRY_TIMING_CONFIG = EntryTimingConfig()
