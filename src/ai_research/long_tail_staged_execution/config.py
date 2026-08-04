#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.11 staged entry and pyramiding audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.11"
STAGE_NAME = "staged entry, soft-failure sizing and asymmetric pyramiding audit"


@dataclass(frozen=True)
class StagedExecutionPolicy:
    """One pre-registered execution policy.

    ``base_r`` and every value in ``add_r`` are fractions of the account's
    one-R dollar budget.  ``base_sizing_stop_distance`` controls position
    quantity only.  The exchange-style base disaster floor remains 3% unless
    explicitly documented as a tail-risk sizing experiment.

    Add-ons never reduce or reset the base tranche.  They use independent
    stops and may therefore be swept without closing the original winner.
    """

    name: str
    mode: str
    base_r: float = 1.0
    base_sizing_stop_distance: float = 0.03
    soft_failure_distance: float = 0.0
    add_r: tuple[float, ...] = ()
    trigger_n: tuple[float, ...] = ()
    add_stop_n: float = 1.0
    max_cycle_hard_r: float = 1.0
    max_notional_to_equity: float = 1.50
    require_profit_cover: bool = True

    def validate(self) -> None:
        allowed = {"baseline", "soft_failure", "staged_dual_path", "turtle", "pyramid"}
        if self.mode not in allowed:
            raise ValueError(f"unsupported mode for {self.name}: {self.mode}")
        if not 0.0 < self.base_r <= 1.0:
            raise ValueError(f"invalid base_r for {self.name}")
        if not 0.005 <= self.base_sizing_stop_distance <= 0.03:
            raise ValueError(f"invalid base sizing distance for {self.name}")
        if not 0.0 <= self.soft_failure_distance <= 0.03:
            raise ValueError(f"invalid soft-failure distance for {self.name}")
        if len(self.add_r) != len(self.trigger_n):
            raise ValueError(f"add/trigger length mismatch for {self.name}")
        if len(self.add_r) > 2:
            raise ValueError(f"maximum two add-ons for {self.name}")
        if any(value <= 0.0 or value > 0.60 for value in self.add_r):
            raise ValueError(f"invalid add risk for {self.name}")
        if any(value <= 0.0 for value in self.trigger_n):
            raise ValueError(f"invalid trigger N for {self.name}")
        if not 0.50 <= self.add_stop_n <= 1.50:
            raise ValueError(f"invalid add stop N for {self.name}")
        if not 1.0 <= self.max_cycle_hard_r <= 2.0:
            raise ValueError(f"invalid max cycle R for {self.name}")
        if not 0.30 <= self.max_notional_to_equity <= 1.50:
            raise ValueError(f"invalid notional cap for {self.name}")
        if self.mode == "baseline" and (self.add_r or self.soft_failure_distance):
            raise ValueError("baseline cannot contain add-ons or soft failure")
        if self.mode == "soft_failure" and self.soft_failure_distance <= 0:
            raise ValueError("soft-failure policy requires a threshold")
        if self.mode in {"staged_dual_path", "turtle", "pyramid"} and not self.add_r:
            raise ValueError(f"{self.mode} requires add-on risk")
        hard_tail_r = self.base_r * 0.03 / self.base_sizing_stop_distance + sum(self.add_r)
        if hard_tail_r > self.max_cycle_hard_r + 1e-12:
            raise ValueError(
                f"declared hard-tail risk exceeds cap for {self.name}: {hard_tail_r:.3f}R"
            )

    @property
    def declared_hard_tail_r(self) -> float:
        return float(self.base_r * 0.03 / self.base_sizing_stop_distance + sum(self.add_r))


@dataclass(frozen=True)
class StagedExecutionConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    source_2_8b_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8b_dual_risk_slot_account"
    source_2_9_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_9_dynamic_risk_release"
    source_2_10_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_10_risk_migration"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_11_staged_entry_pyramiding"

    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    disaster_stop_distance: float = 0.03
    initial_equity: float = 1.0

    # Causal volatility unit from prior 60 completed one-minute bars.
    minimum_n_pct: float = 0.0050
    maximum_n_pct: float = 0.0150
    staged_pullback_arm_n: float = 0.50
    minimum_add_stop_pct: float = 0.0050
    maximum_add_stop_pct: float = 0.0125
    minimum_executed_add_r: float = 0.10
    maximum_virtual_tranches: int = 3
    maximum_account_tail_r: float = 2.0
    maximum_notional_to_equity: float = 1.50

    policies: tuple[StagedExecutionPolicy, ...] = (
        StagedExecutionPolicy(
            name="P0_single_1R",
            mode="baseline",
        ),
        StagedExecutionPolicy(
            name="F1_soft_failure_1p5",
            mode="soft_failure",
            base_r=1.0,
            base_sizing_stop_distance=0.015,
            soft_failure_distance=0.015,
            max_cycle_hard_r=2.0,
        ),
        StagedExecutionPolicy(
            name="S1_stage060_add040",
            mode="staged_dual_path",
            base_r=0.60,
            add_r=(0.40,),
            trigger_n=(1.0,),
            add_stop_n=1.0,
            max_cycle_hard_r=1.0,
            require_profit_cover=False,
        ),
        StagedExecutionPolicy(
            name="T1_turtle_add035",
            mode="turtle",
            base_r=1.0,
            add_r=(0.35,),
            trigger_n=(1.0,),
            add_stop_n=1.0,
            max_cycle_hard_r=1.35,
        ),
        StagedExecutionPolicy(
            name="P1_pyramid_add035x2",
            mode="pyramid",
            base_r=1.0,
            add_r=(0.35, 0.35),
            trigger_n=(1.0, 2.0),
            add_stop_n=0.75,
            max_cycle_hard_r=1.70,
        ),
    )

    # Unified gates. A policy cannot pass merely by raising nominal exposure.
    minimum_return_retention_each_year: float = 0.95
    minimum_combined_return_ratio: float = 1.00
    maximum_mdd_multiple: float = 1.20
    minimum_positive_quarters_per_year: int = 3
    maximum_winner_to_loser_share: float = 0.05
    maximum_addon_loss_share_of_base_profit: float = 0.35
    maximum_tail_r_tolerance: float = 0.02
    maximum_notional_tolerance: float = 0.02

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.11 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost grid changed")
        if abs(self.disaster_stop_distance - 0.03) > 1e-12:
            raise ValueError("3% disaster protection is frozen")
        if self.maximum_virtual_tranches != 3:
            raise ValueError("base plus at most two add-ons")
        if self.maximum_account_tail_r > 2.0 + 1e-12:
            raise ValueError("account tail risk cannot exceed two R")
        if self.maximum_notional_to_equity > 1.50 + 1e-12:
            raise ValueError("research notional cap cannot exceed 1.5x")
        if not 0 < self.account_risk_fraction_per_full_r <= 0.02:
            raise ValueError("invalid account one-R fraction")
        if not 0 < self.minimum_n_pct <= self.maximum_n_pct:
            raise ValueError("invalid causal N clamp")
        if "03_4_2_10" not in self.source_2_10_report_dir:
            raise ValueError("2.10 source path drift")
        if "03_4_2_11" not in self.report_dir:
            raise ValueError("report path must be isolated")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        for policy in self.policies:
            policy.validate()
            if policy.declared_hard_tail_r > self.maximum_account_tail_r + 1e-12:
                raise ValueError(f"{policy.name} exceeds account tail-risk cap")
            if policy.max_notional_to_equity > self.maximum_notional_to_equity + 1e-12:
                raise ValueError(f"{policy.name} exceeds account notional cap")

    @property
    def source_2_10_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_10_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [
            {
                **asdict(policy),
                "add_r": list(policy.add_r),
                "trigger_n": list(policy.trigger_n),
                "declared_hard_tail_r": policy.declared_hard_tail_r,
            }
            for policy in self.policies
        ]
        payload["dynamic_position_formula"] = "risk dollars / executable stop distance"
        payload["frozen_entry"] = "q70 ML opening pool"
        payload["frozen_base_exit"] = "3% disaster protection plus deterministic failed_reclaim"
        payload["add_on_exit"] = "independent add-on stop or base failed_reclaim, whichever occurs first"
        payload["hard_pivot_stop"] = "ABANDONED_FOR_BASE"
        payload["fixed_6h_role"] = "diagnostic benchmark only"
        payload["2026_status"] = "SEALED"
        return payload


DEFAULT_STAGED_EXECUTION_CONFIG = StagedExecutionConfig()
