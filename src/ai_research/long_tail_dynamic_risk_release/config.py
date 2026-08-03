#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.9 causal protection and risk release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.9"
STAGE_NAME = "causal structural protection stop and dynamic risk-release tranche audit"


@dataclass(frozen=True)
class ProtectionPolicy:
    """One pre-registered hard-protection policy.

    ``mode`` controls which already-confirmed structural floor may become a
    live stop. The stop is always monotone and the 3% disaster floor remains
    the outer safety bound.
    """

    name: str
    mode: str

    def validate(self) -> None:
        if self.mode not in {"disaster_only", "latest_confirmed", "lagged_confirmed"}:
            raise ValueError(f"unsupported protection mode: {self.mode}")


@dataclass(frozen=True)
class DynamicReleasePolicy:
    """One causal maximum-two-tranche risk-release policy.

    The primary tranche always starts at one full R. A secondary tranche may
    only use risk that a live, enforceable protection stop has already
    released. ``max_secondary_r`` caps the new tranche; it is not a static
    reservation and never reduces a standalone primary entry.
    """

    name: str
    max_secondary_r: float
    minimum_release_r: float
    require_healthy_state: bool = True
    require_non_losing_active: bool = False
    allow_secondary: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.max_secondary_r <= 1.0:
            raise ValueError(f"invalid secondary cap for {self.name}")
        if not 0.0 <= self.minimum_release_r <= self.max_secondary_r + 1e-12:
            raise ValueError(f"invalid release threshold for {self.name}")
        if not self.allow_secondary and self.max_secondary_r != 0.0:
            raise ValueError(f"single-tranche policy {self.name} must have zero secondary cap")


@dataclass(frozen=True)
class DynamicRiskReleaseConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    source_2_8b_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8b_dual_risk_slot_account"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_9_dynamic_risk_release"

    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    disaster_stop_distance: float = 0.03
    initial_equity: float = 1.0
    maximum_live_remaining_r: float = 1.0
    maximum_virtual_tranches: int = 2

    protection_policies: tuple[ProtectionPolicy, ...] = (
        ProtectionPolicy("S0_disaster_only", "disaster_only"),
        ProtectionPolicy("S1_latest_confirmed", "latest_confirmed"),
        ProtectionPolicy("S2_lagged_confirmed", "lagged_confirmed"),
    )
    dynamic_policies: tuple[DynamicReleasePolicy, ...] = (
        DynamicReleasePolicy("D0_single_1R", 0.0, 0.0, allow_secondary=False),
        DynamicReleasePolicy("D1_release_cap035", 0.35, 0.20),
        DynamicReleasePolicy("D2_release_cap050", 0.50, 0.20),
        DynamicReleasePolicy("D3_release_cap050_non_losing", 0.50, 0.20, require_non_losing_active=True),
    )

    # Protection stop qualification. The stop must preserve the opening edge;
    # reducing MDD alone is not enough.
    minimum_protection_return_retention: float = 0.90
    maximum_protection_mdd_multiple: float = 1.10
    maximum_hard_stop_share: float = 0.40

    # Dynamic tranche qualification.
    minimum_coverage_ratio: float = 0.70
    minimum_monthly_tranches: float = 25.0
    minimum_dynamic_return_retention: float = 0.95
    maximum_dynamic_mdd_multiple: float = 1.10
    maximum_losing_second_add_share: float = 0.15
    maximum_broken_second_add_share: float = 0.0
    minimum_positive_quarters_per_year: int = 3

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.9 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost-stress grid changed")
        if self.maximum_virtual_tranches != 2:
            raise ValueError("R03.4.2.9 permits at most two virtual tranches")
        if not 0.0 < self.account_risk_fraction_per_full_r <= 0.05:
            raise ValueError("invalid account risk per full R")
        if abs(self.disaster_stop_distance - 0.03) > 1e-12:
            raise ValueError("3% disaster protection is frozen")
        if self.maximum_live_remaining_r > 1.0 + 1e-12:
            raise ValueError("live remaining risk cannot exceed one full R")
        if "03_4_2_8a" not in self.source_2_8a_report_dir:
            raise ValueError("2.8A source path drift")
        if "03_4_2_8b" not in self.source_2_8b_report_dir:
            raise ValueError("2.8B source path drift")
        if "03_4_2_9" not in self.report_dir:
            raise ValueError("report path must be isolated")
        protection_names = [policy.name for policy in self.protection_policies]
        dynamic_names = [policy.name for policy in self.dynamic_policies]
        if len(protection_names) != len(set(protection_names)):
            raise ValueError("protection policy names must be unique")
        if len(dynamic_names) != len(set(dynamic_names)):
            raise ValueError("dynamic policy names must be unique")
        for policy in self.protection_policies:
            policy.validate()
        for policy in self.dynamic_policies:
            policy.validate()

    @property
    def source_2_8a_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8a_report_dir

    @property
    def source_2_8b_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8b_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["protection_policies"] = [asdict(policy) for policy in self.protection_policies]
        payload["dynamic_policies"] = [asdict(policy) for policy in self.dynamic_policies]
        payload["frozen_entry"] = "q70 ML opening pool"
        payload["frozen_exit"] = "failed_reclaim deterministic structural exit"
        payload["fixed_6h_role"] = "diagnostic benchmark only"
        payload["2026_status"] = "SEALED"
        return payload


DEFAULT_DYNAMIC_RISK_RELEASE_CONFIG = DynamicRiskReleaseConfig()
