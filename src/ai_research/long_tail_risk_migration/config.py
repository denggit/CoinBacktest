#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.10 partial de-risking and risk migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.10"
STAGE_NAME = "soft-structure partial de-risking and q70 risk-migration account audit"


@dataclass(frozen=True)
class MigrationPolicy:
    """One pre-registered account policy.

    ``partial_reduce_fraction`` is applied once to the active tranche when a
    causally confirmed, already-proven structure first enters the soft BROKEN
    state while the tranche is non-losing. It is a real partial close, not a
    synthetic stop tightening.

    ``migration_target_r`` is the maximum account-R budget transferred to a
    later q70 event. The simulator first uses capacity physically released by
    prior partial closes/exits. If needed, it reduces the active tranche at the
    new event's execution open before opening the new tranche. Simultaneous
    cycle risk never exceeds the original one-R dollar budget.
    """

    name: str
    partial_reduce_fraction: float = 0.0
    migration_target_r: float = 0.0
    allow_migration: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.partial_reduce_fraction < 1.0:
            raise ValueError(f"invalid partial reduction for {self.name}")
        if not 0.0 <= self.migration_target_r <= 0.50:
            raise ValueError(f"invalid migration cap for {self.name}")
        if not self.allow_migration and self.migration_target_r != 0.0:
            raise ValueError(f"non-migration policy {self.name} has migration risk")


@dataclass(frozen=True)
class RiskMigrationConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    source_2_8b_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8b_dual_risk_slot_account"
    source_2_9_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_9_dynamic_risk_release"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_10_risk_migration"

    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    disaster_stop_distance: float = 0.03
    initial_equity: float = 1.0
    maximum_virtual_tranches: int = 2
    maximum_cycle_r: float = 1.0
    minimum_migration_r: float = 0.20
    minimum_root_remaining_fraction: float = 0.35

    policies: tuple[MigrationPolicy, ...] = (
        MigrationPolicy("P0_single_1R"),
        MigrationPolicy("R1_soft_break_reduce025", partial_reduce_fraction=0.25),
        MigrationPolicy("R2_soft_break_reduce050", partial_reduce_fraction=0.50),
        MigrationPolicy("M1_signal_migrate035", migration_target_r=0.35, allow_migration=True),
        MigrationPolicy("M2_signal_migrate050", migration_target_r=0.50, allow_migration=True),
        MigrationPolicy(
            "H1_reduce025_then_migrate035",
            partial_reduce_fraction=0.25,
            migration_target_r=0.35,
            allow_migration=True,
        ),
    )

    # Guardrails. Frequency is a first-class target; a policy cannot pass by
    # deleting most q70 opportunities.
    minimum_return_retention_each_year: float = 0.95
    minimum_combined_return_ratio: float = 1.00
    maximum_mdd_multiple: float = 1.10
    minimum_coverage_ratio_for_migration: float = 0.70
    minimum_monthly_tranches_for_migration: float = 25.0
    minimum_positive_quarters_per_year: int = 3
    maximum_losing_migration_share: float = 0.10
    maximum_broken_migration_share: float = 0.0
    maximum_cycle_r_tolerance: float = 0.02

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.10 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost grid changed")
        if abs(self.disaster_stop_distance - 0.03) > 1e-12:
            raise ValueError("3% disaster protection is frozen")
        if self.maximum_virtual_tranches != 2:
            raise ValueError("maximum virtual tranches must remain two")
        if self.maximum_cycle_r > 1.0 + 1e-12:
            raise ValueError("cycle risk cannot exceed one R")
        if not 0.0 < self.minimum_migration_r <= 0.50:
            raise ValueError("invalid minimum migration R")
        if not 0.0 < self.minimum_root_remaining_fraction < 1.0:
            raise ValueError("invalid minimum root remaining fraction")
        if not 0.0 < self.account_risk_fraction_per_full_r <= 0.05:
            raise ValueError("invalid full-R account risk")
        if "03_4_2_8a" not in self.source_2_8a_report_dir:
            raise ValueError("2.8A source path drift")
        if "03_4_2_8b" not in self.source_2_8b_report_dir:
            raise ValueError("2.8B source path drift")
        if "03_4_2_9" not in self.source_2_9_report_dir:
            raise ValueError("2.9 source path drift")
        if "03_4_2_10" not in self.report_dir:
            raise ValueError("report path must be isolated")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        for policy in self.policies:
            policy.validate()

    @property
    def source_2_8a_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8a_report_dir

    @property
    def source_2_8b_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_8b_report_dir

    @property
    def source_2_9_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_9_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [asdict(policy) for policy in self.policies]
        payload["frozen_entry"] = "q70 ML opening pool"
        payload["frozen_exit"] = "3% disaster protection plus deterministic failed_reclaim"
        payload["hard_pivot_stop"] = "ABANDONED"
        payload["fixed_6h_role"] = "diagnostic benchmark only"
        payload["2026_status"] = "SEALED"
        return payload


DEFAULT_RISK_MIGRATION_CONFIG = RiskMigrationConfig()
