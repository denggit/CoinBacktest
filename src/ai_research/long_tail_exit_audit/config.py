#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.4.2 q90 long-tail path exit audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R03.4.2"
STAGE_NAME = "Frozen q90 long-opportunity path exit and rolling-renewal audit"


@dataclass(frozen=True)
class ExitRecipe:
    """One preregistered exit recipe.

    ``stop_lookback_minutes=0`` is reserved for the fixed-six-hour diagnostic
    baseline. Every tradeable recipe uses a causal pre-entry structural stop.
    """

    name: str
    stop_lookback_minutes: int
    take_profit_r: float | None = None
    trail_activation_r: float | None = None
    trail_giveback_r: float | None = None
    renewal_quantile: float | None = None
    invalidation_quantile: float | None = None
    invalidation_confirmations: int = 0
    minimum_invalidation_hold_minutes: int = 60
    checkpoint_hours: int = 6
    safety_cap_hours: int = 24
    is_time_baseline: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("exit recipe name is required")
        if self.is_time_baseline:
            if self.stop_lookback_minutes != 0:
                raise ValueError("time baseline cannot use a structural stop")
            return
        if self.stop_lookback_minutes not in {60, 180}:
            raise ValueError(f"unsupported stop lookback: {self.stop_lookback_minutes}")
        if self.take_profit_r is not None and self.take_profit_r <= 0:
            raise ValueError("take-profit R must be positive")
        if (self.trail_activation_r is None) != (self.trail_giveback_r is None):
            raise ValueError("trail activation and giveback must be specified together")
        if self.trail_activation_r is not None:
            if self.trail_activation_r <= 0 or self.trail_giveback_r is None or self.trail_giveback_r <= 0:
                raise ValueError("invalid trailing parameters")
            if self.trail_giveback_r >= self.trail_activation_r:
                raise ValueError("giveback must be smaller than activation")
        if self.renewal_quantile is not None and not 0 < self.renewal_quantile < 1:
            raise ValueError("renewal quantile must be inside (0, 1)")
        if self.invalidation_quantile is not None:
            if not 0 < self.invalidation_quantile < 1:
                raise ValueError("invalidation quantile must be inside (0, 1)")
            if self.invalidation_confirmations < 1:
                raise ValueError("invalidation confirmations are required")
        if self.checkpoint_hours < 1 or self.safety_cap_hours < self.checkpoint_hours:
            raise ValueError("invalid checkpoint/safety horizon")


EXIT_RECIPES: tuple[ExitRecipe, ...] = (
    ExitRecipe(
        name="fixed_6h_diagnostic",
        stop_lookback_minutes=0,
        safety_cap_hours=6,
        is_time_baseline=True,
    ),
    ExitRecipe(
        name="s60_tp_1p5r",
        stop_lookback_minutes=60,
        take_profit_r=1.5,
        safety_cap_hours=24,
    ),
    ExitRecipe(
        name="s60_tp_2p0r",
        stop_lookback_minutes=60,
        take_profit_r=2.0,
        safety_cap_hours=24,
    ),
    ExitRecipe(
        name="s180_tp_2p0r",
        stop_lookback_minutes=180,
        take_profit_r=2.0,
        safety_cap_hours=36,
    ),
    ExitRecipe(
        name="s60_trail_a1p0_g0p5",
        stop_lookback_minutes=60,
        trail_activation_r=1.0,
        trail_giveback_r=0.5,
        safety_cap_hours=24,
    ),
    ExitRecipe(
        name="s60_trail_a1p5_g0p75",
        stop_lookback_minutes=60,
        trail_activation_r=1.5,
        trail_giveback_r=0.75,
        safety_cap_hours=36,
    ),
    ExitRecipe(
        name="s60_renew_q70_trail",
        stop_lookback_minutes=60,
        trail_activation_r=1.0,
        trail_giveback_r=0.5,
        renewal_quantile=0.70,
        safety_cap_hours=48,
    ),
    ExitRecipe(
        name="s60_renew_q60_trail",
        stop_lookback_minutes=60,
        trail_activation_r=1.5,
        trail_giveback_r=0.75,
        renewal_quantile=0.60,
        safety_cap_hours=48,
    ),
    ExitRecipe(
        name="s180_renew_q70_trail",
        stop_lookback_minutes=180,
        trail_activation_r=1.0,
        trail_giveback_r=0.5,
        renewal_quantile=0.70,
        safety_cap_hours=48,
    ),
    ExitRecipe(
        name="s60_renew_q70_invalidate_q50_trail",
        stop_lookback_minutes=60,
        trail_activation_r=1.0,
        trail_giveback_r=0.5,
        renewal_quantile=0.70,
        invalidation_quantile=0.50,
        invalidation_confirmations=2,
        minimum_invalidation_hold_minutes=60,
        safety_cap_hours=48,
    ),
)


@dataclass(frozen=True)
class LongTailExitAuditConfig:
    symbol: str = "ETH-USDT-SWAP"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    primary_signal_quantile: float = 0.90
    quality_control_quantile: float = 0.95
    primary_horizon_hours: int = 6
    risk_penalty: float = 1.25
    base_round_trip_cost: float = 0.0013
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    entry_delay_minutes: tuple[int, ...] = (1, 3, 5)
    episode_merge_gap_minutes: int = 30
    independent_event_cooldown_hours: int = 6
    structural_buffer_bps: float = 4.0
    atr_buffer_multiple: float = 0.50
    s60_min_stop_pct: float = 0.0035
    s60_max_stop_pct: float = 0.0180
    s180_min_stop_pct: float = 0.0045
    s180_max_stop_pct: float = 0.0220
    risk_budget_fraction: float = 0.01
    maximum_notional_multiple: float = 1.50
    train_sample_cap: int = 400_000
    base_n_estimators: int = 420
    base_learning_rate: float = 0.035
    base_num_leaves: int = 31
    base_min_child_samples: int = 300
    random_state: int = 20260801
    recipes: tuple[ExitRecipe, ...] = EXIT_RECIPES
    minimum_trades_per_year: int = 80
    minimum_2x_profit_factor: float = 1.20
    minimum_positive_quarters: int = 6
    minimum_2x_expectancy_retention_vs_fixed6h: float = 0.60
    maximum_risk_sized_drawdown: float = 0.20
    maximum_safety_cap_share: float = 0.20
    maximum_top10_profit_share: float = 0.60
    report_dir: str = "data/reports/research/eth_ai_trading/03_4_2_long_tail_exit_audit"

    def validate(self) -> None:
        if pd.Timestamp(self.research_end) >= pd.Timestamp(self.sealed_holdout_start):
            raise ValueError("R03.4.2 must keep 2026 sealed")
        if self.primary_signal_quantile != 0.90:
            raise ValueError("R03.4.2 freezes q90 as the primary signal")
        if self.primary_horizon_hours != 6:
            raise ValueError("R03.4.2 freezes the original six-hour target")
        if self.risk_penalty != 1.25:
            raise ValueError("R03.4.2 freezes the original risk penalty")
        if tuple(self.cost_stress_multipliers) != (1.0, 2.0, 3.0):
            raise ValueError("R03.4.2 cost stress grid is frozen")
        if tuple(self.entry_delay_minutes) != (1, 3, 5):
            raise ValueError("R03.4.2 delay audit is frozen")
        if self.independent_event_cooldown_hours < self.primary_horizon_hours:
            raise ValueError("event cooldown must cover the original target horizon")
        if not 0 < self.risk_budget_fraction <= 0.02:
            raise ValueError("invalid per-trade risk budget")
        if self.maximum_notional_multiple <= 0:
            raise ValueError("invalid notional cap")
        if "03_4_2" not in self.report_dir:
            raise ValueError("R03.4.2 report path must be isolated")
        names = [recipe.name for recipe in self.recipes]
        if len(names) != len(set(names)):
            raise ValueError("exit recipe names must be unique")
        for recipe in self.recipes:
            recipe.validate()

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    @property
    def signal_quantiles(self) -> tuple[float, ...]:
        return (self.primary_signal_quantile, self.quality_control_quantile)

    def stop_bounds(self, lookback_minutes: int) -> tuple[float, float]:
        if lookback_minutes == 60:
            return self.s60_min_stop_pct, self.s60_max_stop_pct
        if lookback_minutes == 180:
            return self.s180_min_stop_pct, self.s180_max_stop_pct
        raise ValueError(f"unsupported stop lookback {lookback_minutes}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["cost_stress_multipliers"] = list(self.cost_stress_multipliers)
        payload["entry_delay_minutes"] = list(self.entry_delay_minutes)
        payload["recipes"] = [asdict(recipe) for recipe in self.recipes]
        return payload


DEFAULT_LONG_TAIL_EXIT_AUDIT_CONFIG = LongTailExitAuditConfig()
