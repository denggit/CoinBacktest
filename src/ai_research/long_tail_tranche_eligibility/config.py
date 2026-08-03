#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2.8A occupied-signal diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.long_tail_structural_exit.config import StructuralExitConfig, StructuralPolicy

STAGE_ID = "R03.4.2.8A"
STAGE_NAME = "occupied q70 signal atlas and risk-controlled tranche eligibility gate"


@dataclass(frozen=True)
class TrancheEligibilityConfig:
    """Pre-registered diagnostic contract before any second tranche is traded.

    This stage does not add size. It determines whether occupied q70 signals
    contain a causal, cross-year subset that is worth taking into the separate
    R03.4.2.8B account-risk simulation.
    """

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

    structure_bar_minutes: int = 15
    pre_entry_history_minutes: int = 360
    pivot_left_bars: int = 2
    pivot_right_bars: int = 2
    atr_window_bars: int = 12
    structure_buffer_atr_multiple: float = 0.20
    minimum_structure_buffer_bps: float = 5.0
    disaster_stop_return: float = -0.030

    # This is a gate, not a sizing multiplier. A signal is not eligible for
    # later tranche simulation unless a causal candidate protection level has
    # removed at least this fraction of the original disaster-risk distance.
    minimum_released_risk_fraction: float = 0.25
    maximum_structure_age_minutes: int = 360

    minimum_eligible_events_per_year: int = 30
    minimum_pf_2x: float = 1.25
    minimum_positive_quarters_per_year: int = 3
    maximum_top10_profit_share: float = 0.60
    maximum_eligible_losing_position_share: float = 0.10

    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2.8A must keep 2026 sealed")
        if self.evaluation_quantile != 0.70:
            raise ValueError("q70 is the frozen opening pool")
        if self.entry_delay_minutes != (1, 3, 5):
            raise ValueError("entry-delay stress grid changed")
        if self.cost_multipliers != (1.0, 2.0, 3.0):
            raise ValueError("cost-stress grid changed")
        if not 0.0 < self.minimum_released_risk_fraction < 1.0:
            raise ValueError("released-risk gate must be between zero and one")
        if self.maximum_structure_age_minutes < self.structure_bar_minutes:
            raise ValueError("independent structure age must cover at least one structure bar")
        if self.minimum_eligible_events_per_year < 1:
            raise ValueError("eligible-event gate must be positive")
        if "03_4_2_8a" not in self.report_dir:
            raise ValueError("report path must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def structural_config(self) -> StructuralExitConfig:
        """Return the exact frozen failed-reclaim structural contract."""

        return StructuralExitConfig(
            symbol=self.symbol,
            research_start=self.research_start,
            research_end=self.research_end,
            sealed_holdout_start=self.sealed_holdout_start,
            evaluation_quantile=self.evaluation_quantile,
            entry_delay_minutes=self.entry_delay_minutes,
            diagnostic_horizon_hours=self.diagnostic_horizon_hours,
            base_round_trip_cost=self.base_round_trip_cost,
            cost_multipliers=self.cost_multipliers,
            risk_penalty=self.risk_penalty,
            base_train_sample_cap=self.base_train_sample_cap,
            base_n_estimators=self.base_n_estimators,
            base_learning_rate=self.base_learning_rate,
            base_num_leaves=self.base_num_leaves,
            base_min_child_samples=self.base_min_child_samples,
            random_state=self.random_state,
            episode_merge_gap_minutes=self.episode_merge_gap_minutes,
            independent_event_cooldown_hours=self.independent_event_cooldown_hours,
            structure_bar_minutes=self.structure_bar_minutes,
            pre_entry_history_minutes=self.pre_entry_history_minutes,
            pivot_left_bars=self.pivot_left_bars,
            pivot_right_bars=self.pivot_right_bars,
            atr_window_bars=self.atr_window_bars,
            structure_buffer_atr_multiple=self.structure_buffer_atr_multiple,
            minimum_structure_buffer_bps=self.minimum_structure_buffer_bps,
            disaster_stop_return=self.disaster_stop_return,
            policies=(StructuralPolicy(name="failed_reclaim", exit_on_failed_reclaim=True),),
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["frozen_structural_policy"] = "failed_reclaim"
        payload["tranche_execution_in_this_stage"] = False
        return payload


DEFAULT_TRANCHE_ELIGIBILITY_CONFIG = TrancheEligibilityConfig()
