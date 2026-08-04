#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.15 final account and live-readiness audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.15"
STAGE_NAME = "frozen C2 final account robustness and live-readiness audit"


@dataclass(frozen=True)
class FinalAccountAuditConfig:
    symbol: str = "ETH-USDT-SWAP"
    oos_start: str = "2024-01-01 00:00:00"
    oos_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_14_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_14_entry_timing_mae"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_15_final_account_live_readiness"

    source_policy: str = "E0_immediate_C2"
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    anchor_delay_minutes: int = 1
    anchor_cost_multiplier: float = 2.0

    account_risk_fraction_per_full_r: float = 0.01
    hard_stop_distance: float = 0.02
    soft_failure_distance: float = 0.015
    contract_value_base: float = 0.1
    minimum_contracts: int = 1
    initial_equity_tiers: tuple[float, ...] = (500.0, 1000.0, 3000.0, 10000.0, 30000.0)

    minimum_anchor_total_return: float = 1.50
    maximum_anchor_mdd: float = 0.12
    minimum_anchor_profit_factor: float = 1.50
    minimum_positive_months: int = 16
    minimum_positive_quarters: int = 6
    maximum_losing_streak: int = 12
    maximum_drawdown_duration_days: int = 120
    minimum_return_without_top10: float = 0.0
    maximum_top10_profit_share: float = 0.45
    maximum_anchor_worst_net_r: float = 1.20
    maximum_stress_mdd: float = 0.15

    def validate(self) -> None:
        if pd.Timestamp(self.oos_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.15 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("delay stress grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost stress grid changed")
        if abs(self.hard_stop_distance - 0.02) > 1e-12:
            raise ValueError("C2 real 2% hard stop must remain frozen")
        if abs(self.soft_failure_distance - 0.015) > 1e-12:
            raise ValueError("C2 1.5% soft failure must remain frozen")
        if not 0 < self.account_risk_fraction_per_full_r <= 0.02:
            raise ValueError("invalid full-R fraction")
        if self.contract_value_base <= 0 or self.minimum_contracts < 1:
            raise ValueError("invalid contract sizing settings")
        if "03_4_2_14" not in self.source_2_14_report_dir or "03_4_2_15" not in self.report_dir:
            raise ValueError("source/report path drift")

    @property
    def source_2_14_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_14_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["initial_equity_tiers"] = list(self.initial_equity_tiers)
        payload["frozen_strategy"] = (
            "q70 immediate next 1m open + equal 1R + real 2% hard stop + "
            "1.5% completed-close soft failure + failed_reclaim"
        )
        payload["oos_contract"] = "continuous WF_2024 + WF_2025; 2023 is development/training history, not OOS account return"
        payload["holdout"] = "2026 sealed"
        return payload


DEFAULT_FINAL_ACCOUNT_AUDIT_CONFIG = FinalAccountAuditConfig()
