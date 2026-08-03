#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.7 causal non-time structural exits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2.7"
STAGE_NAME = "q70 causal structural state machine and non-time exit audit"


@dataclass(frozen=True)
class StructuralPolicy:
    """One pre-registered structural state-machine variant.

    None of these fields is a holding-time limit. Bars are only the causal
    observation cadence used to confirm market structure.
    """

    name: str
    exit_on_failed_reclaim: bool = False
    enable_profit_guard: bool = False


@dataclass(frozen=True)
class StructuralExitConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"

    evaluation_quantile: float = 0.70
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    diagnostic_horizon_hours: int = 6
    base_round_trip_cost: float = 0.0013
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    risk_penalty: float = 1.25

    base_train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801

    episode_merge_gap_minutes: int = 30
    independent_event_cooldown_hours: int = 6

    # Causal structure contract. Fifteen-minute bars are completed before a
    # state transition can execute at the next one-minute open.
    structure_bar_minutes: int = 15
    pre_entry_history_minutes: int = 360
    pivot_left_bars: int = 2
    pivot_right_bars: int = 2
    atr_window_bars: int = 12
    structure_buffer_atr_multiple: float = 0.20
    minimum_structure_buffer_bps: float = 5.0

    # Wide safety floor, deliberately not optimized.
    disaster_stop_return: float = -0.030

    # Profit protection is structure-confirmed; these values only activate the
    # guard and never force a time-based exit.
    profit_activation_return: float = 0.015
    profit_activation_atr_multiple: float = 4.0
    minimum_peak_giveback_return: float = 0.0075
    peak_giveback_fraction: float = 0.45
    profit_guard_declining_bars: int = 3

    policies: tuple[StructuralPolicy, ...] = (
        StructuralPolicy(name="confirmed_structure"),
        StructuralPolicy(name="failed_reclaim", exit_on_failed_reclaim=True),
        StructuralPolicy(name="profit_guard", enable_profit_guard=True),
    )

    minimum_pf_2x: float = 1.35
    minimum_trades_per_year: int = 100
    minimum_positive_quarters: int = 6
    maximum_mdd: float = 0.20
    maximum_top10_profit_share: float = 0.60
    maximum_censored_share: float = 0.05
    minimum_profit_retention_vs_fixed: float = 0.90
    minimum_relative_mdd_improvement: float = 0.10

    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_7_non_time_structural_exit"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.7 must keep 2026 sealed")
        if self.evaluation_quantile != 0.70:
            raise ValueError("q70 is the frozen main research pool")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (1.0, 2.0, 3.0):
            raise ValueError("cost-stress grid changed")
        if self.structure_bar_minutes <= 0 or 60 % self.structure_bar_minutes != 0:
            raise ValueError("structure bars must divide one hour")
        if self.pre_entry_history_minutes % self.structure_bar_minutes:
            raise ValueError("pre-entry history must align to structure bars")
        if self.pivot_left_bars < 1 or self.pivot_right_bars < 1:
            raise ValueError("causal pivots require left and right confirmation bars")
        if self.atr_window_bars < 4:
            raise ValueError("ATR window is too short")
        if not -0.10 < self.disaster_stop_return < -0.01:
            raise ValueError("disaster stop must remain a wide safety floor")
        if any("time" in policy.name.lower() for policy in self.policies):
            raise ValueError("candidate policy names must not imply a time exit")
        if len({policy.name for policy in self.policies}) != len(self.policies):
            raise ValueError("structural policy names must be unique")
        if "03_4_2_7" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["policies"] = [asdict(policy) for policy in self.policies]
        return payload


DEFAULT_STRUCTURAL_EXIT_CONFIG = StructuralExitConfig()
