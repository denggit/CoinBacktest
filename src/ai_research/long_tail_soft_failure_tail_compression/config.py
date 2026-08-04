#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.12 soft-failure tail compression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.12"
STAGE_NAME = "soft-failure attribution and real one-R tail compression"


@dataclass(frozen=True)
class TailCompressionPolicy:
    """One pre-registered initial-stop policy.

    ``sizing_stop_distance`` and ``hard_stop_distance`` are fractions of entry
    price for fixed policies.  Adaptive policies derive one frozen distance
    from prior completed one-minute ATR at entry.  A qualifying policy must
    size from the same distance as its executable hard stop, so its declared
    worst-case price risk is one account-R before fees/slippage.
    """

    name: str
    mode: str
    sizing_stop_distance: float = 0.03
    hard_stop_distance: float = 0.03
    soft_failure_distance: float = 0.0
    adaptive_atr_multiple: float = 0.0
    adaptive_min_distance: float = 0.015
    adaptive_max_distance: float = 0.03
    adaptive_soft_fraction: float = 0.0
    qualifying_candidate: bool = True

    def validate(self) -> None:
        allowed = {"baseline", "reference", "fixed", "adaptive"}
        if self.mode not in allowed:
            raise ValueError(f"unsupported mode for {self.name}: {self.mode}")
        if self.mode == "adaptive":
            if self.adaptive_atr_multiple <= 0:
                raise ValueError(f"adaptive ATR multiple missing for {self.name}")
            if not 0.005 <= self.adaptive_min_distance <= self.adaptive_max_distance <= 0.03:
                raise ValueError(f"invalid adaptive clamp for {self.name}")
            if not 0.0 <= self.adaptive_soft_fraction < 1.0:
                raise ValueError(f"invalid adaptive soft fraction for {self.name}")
        else:
            if not 0.005 <= self.sizing_stop_distance <= 0.03:
                raise ValueError(f"invalid sizing distance for {self.name}")
            if not 0.005 <= self.hard_stop_distance <= 0.03:
                raise ValueError(f"invalid hard stop for {self.name}")
            if not 0.0 <= self.soft_failure_distance < self.hard_stop_distance + 1e-12:
                raise ValueError(f"invalid soft failure distance for {self.name}")
        if self.mode == "baseline" and self.soft_failure_distance:
            raise ValueError("baseline cannot contain soft failure")
        if self.qualifying_candidate and self.mode != "adaptive":
            if abs(self.sizing_stop_distance - self.hard_stop_distance) > 1e-12:
                raise ValueError(f"qualifying policy {self.name} must size from its real hard stop")

    @property
    def declared_hard_tail_r(self) -> float:
        if self.mode == "adaptive":
            return 1.0
        return float(self.hard_stop_distance / self.sizing_stop_distance)


@dataclass(frozen=True)
class TailCompressionConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2024-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    source_2_8a_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"
    source_2_8b_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8b_dual_risk_slot_account"
    source_2_9_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_9_dynamic_risk_release"
    source_2_10_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_10_risk_migration"
    source_2_11_report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_11_staged_entry_pyramiding"
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression"

    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    account_risk_fraction_per_full_r: float = 0.01
    initial_equity: float = 1.0

    policies: tuple[TailCompressionPolicy, ...] = (
        TailCompressionPolicy(
            name="P0_single_1R",
            mode="baseline",
            sizing_stop_distance=0.03,
            hard_stop_distance=0.03,
            qualifying_candidate=False,
        ),
        TailCompressionPolicy(
            name="F1_reference_1p5size_3ptail",
            mode="reference",
            sizing_stop_distance=0.015,
            hard_stop_distance=0.03,
            soft_failure_distance=0.015,
            qualifying_candidate=False,
        ),
        TailCompressionPolicy(
            name="C2_real_2p_soft1p5",
            mode="fixed",
            sizing_stop_distance=0.02,
            hard_stop_distance=0.02,
            soft_failure_distance=0.015,
        ),
        TailCompressionPolicy(
            name="C15_real_1p5_hard",
            mode="fixed",
            sizing_stop_distance=0.015,
            hard_stop_distance=0.015,
            soft_failure_distance=0.0,
        ),
        TailCompressionPolicy(
            name="C15_real_1p5_soft1p0",
            mode="fixed",
            sizing_stop_distance=0.015,
            hard_stop_distance=0.015,
            soft_failure_distance=0.01,
        ),
        TailCompressionPolicy(
            name="V1_causal_volatility_1R",
            mode="adaptive",
            adaptive_atr_multiple=2.0,
            adaptive_min_distance=0.015,
            adaptive_max_distance=0.03,
            adaptive_soft_fraction=0.75,
        ),
    )

    # Candidate gates. F1 is a reference only because its real tail is two-R.
    minimum_return_retention_each_year: float = 0.95
    minimum_combined_return_ratio: float = 1.10
    maximum_mdd_multiple: float = 1.40
    maximum_absolute_mdd: float = 0.12
    minimum_positive_quarters_per_year: int = 3
    maximum_winner_to_loser_share: float = 0.05
    maximum_tail_r: float = 1.02
    minimum_mean_notional_to_equity: float = 0.45
    maximum_worst_cycle_loss_r: float = 1.25
    attribution_materiality: float = 0.001

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.12 must keep 2026 sealed")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay grid changed")
        if self.cost_multipliers != (2.0, 3.0):
            raise ValueError("cost grid changed")
        if not 0 < self.account_risk_fraction_per_full_r <= 0.02:
            raise ValueError("invalid one-R account fraction")
        if "03_4_2_11" not in self.source_2_11_report_dir:
            raise ValueError("2.11 source path drift")
        if "03_4_2_12" not in self.report_dir:
            raise ValueError("report path must be isolated")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        for policy in self.policies:
            policy.validate()
            if policy.qualifying_candidate and policy.declared_hard_tail_r > self.maximum_tail_r:
                raise ValueError(f"qualifying policy exceeds one-R tail: {policy.name}")

    @property
    def source_2_11_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_11_report_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [
            {**asdict(policy), "declared_hard_tail_r": policy.declared_hard_tail_r}
            for policy in self.policies
        ]
        payload["position_formula"] = "one-R account dollars / frozen executable hard-stop distance"
        payload["frozen_entry"] = "q70 ML opening pool at next observable 1m open"
        payload["frozen_profit_exit"] = "deterministic failed_reclaim; no fixed-time profit exit"
        payload["frozen_holdout"] = "2026 sealed"
        return payload


DEFAULT_TAIL_COMPRESSION_CONFIG = TailCompressionConfig()
