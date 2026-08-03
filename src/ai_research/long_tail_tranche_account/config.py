#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.8B account-level dual-slot research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig

STAGE_ID = "R03.4.2.8B"
STAGE_NAME = "account-level dual risk-slot coverage and tranche audit"


@dataclass(frozen=True)
class TranchePolicy:
    """Pre-registered virtual risk-slot policy.

    Slot weights are fractions of one full account-risk unit, not notional
    multipliers. Every tranche still sizes from risk budget / 3% disaster
    distance. No policy may hold more than two virtual tranches.
    """

    name: str
    slot_a_r: float
    slot_b_r: float
    max_tranches: int
    protection_gate: bool = False

    def validate(self) -> None:
        if self.max_tranches not in (1, 2):
            raise ValueError(f"unsupported max_tranches for {self.name}: {self.max_tranches}")
        if self.slot_a_r <= 0 or self.slot_b_r < 0:
            raise ValueError(f"invalid slot weights for {self.name}")
        if self.slot_a_r + self.slot_b_r > 1.0 + 1e-12:
            raise ValueError(f"{self.name} exceeds one full risk unit")
        if self.max_tranches == 1 and self.slot_b_r != 0:
            raise ValueError(f"single-slot policy {self.name} cannot allocate slot B")


@dataclass(frozen=True)
class TrancheAccountConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8b_dual_risk_slot_account"

    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    disaster_stop_distance: float = 0.03
    initial_equity: float = 1.0

    minimum_coverage_ratio: float = 0.70
    minimum_tranches_per_year: int = 300
    minimum_monthly_tranches: float = 25.0
    minimum_positive_quarters_per_year: int = 3
    maximum_account_drawdown: float = 0.20
    maximum_allocated_r: float = 1.0
    maximum_dangerous_second_add_share: float = 0.10
    maximum_losing_second_add_share: float = 0.15

    policies: tuple[TranchePolicy, ...] = (
        TranchePolicy("P0_single_1R", 1.00, 0.00, 1, False),
        TranchePolicy("P1_equal_05_05", 0.50, 0.50, 2, False),
        TranchePolicy("P2_primary_065_secondary_035", 0.65, 0.35, 2, False),
        TranchePolicy("P3_protected_065_035", 0.65, 0.35, 2, True),
    )

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.8B must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost-stress grid changed")
        if not 0.0 < self.account_risk_fraction_per_full_r <= 0.05:
            raise ValueError("account risk per full R must be realistic")
        if not 0.0 < self.disaster_stop_distance < 0.20:
            raise ValueError("invalid disaster stop distance")
        if self.maximum_allocated_r > 1.0 + 1e-12:
            raise ValueError("maximum allocated R cannot exceed one full unit")
        if "03_4_2_8a" not in self.source_report_dir:
            raise ValueError("R03.4.2.8B must use the frozen R03.4.2.8A source report")
        if "03_4_2_8b" not in self.report_dir:
            raise ValueError("report path must be isolated")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        for policy in self.policies:
            policy.validate()

    @property
    def source_report_path(self) -> Path:
        return PROJECT_ROOT / self.source_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def eligibility_config(self) -> TrancheEligibilityConfig:
        return TrancheEligibilityConfig()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [asdict(policy) for policy in self.policies]
        payload["fixed_exit_contract"] = "failed_reclaim plus 3% disaster protection"
        payload["maximum_virtual_tranches"] = 2
        payload["fixed_6h_role"] = "diagnostic benchmark only"
        payload["2026_status"] = "SEALED"
        return payload


DEFAULT_TRANCHE_ACCOUNT_CONFIG = TrancheAccountConfig()
