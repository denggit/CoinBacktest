#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen R01 opportunity-model + executable strategy baseline configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CACHE_DIR, PROJECT_ROOT
from .opportunity import DEFAULT_TRADE_TEMPLATES, TradeTemplate


DEFAULT_R01_REPORT_DIR = (
    PROJECT_ROOT / "data" / "reports" / "research" / "eth_ai_trading" /
    "rl_market_agent_v1" / "r01_opportunity_strategy"
)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: str
    train_end_exclusive: str
    calibration_start: str
    calibration_end_exclusive: str
    oos_start: str
    oos_end_exclusive: str


DEFAULT_FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        "WF_2024",
        "2023-01-01 00:00:00", "2023-10-01 00:00:00",
        "2023-10-01 00:00:00", "2024-01-01 00:00:00",
        "2024-01-01 00:00:00", "2025-01-01 00:00:00",
    ),
    WalkForwardFold(
        "WF_2025",
        "2023-01-01 00:00:00", "2024-10-01 00:00:00",
        "2024-10-01 00:00:00", "2025-01-01 00:00:00",
        "2025-01-01 00:00:00", "2026-01-01 00:00:00",
    ),
)


@dataclass(frozen=True)
class R01Config:
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    r00_cache_dir: str = str(DEFAULT_CACHE_DIR)
    report_dir: str = str(DEFAULT_R01_REPORT_DIR)
    symbol: str = "ETH-USDT-SWAP"
    round_trip_cost: float = 0.0011
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    delay_stress_minutes: tuple[int, ...] = (0, 1, 2)
    top_trade_removal_counts: tuple[int, ...] = (1, 5, 10)
    risk_per_trade: float = 0.01
    max_notional_multiple: float = 2.0
    threshold_quantiles: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95)
    min_calibration_trades: int = 20
    calibration_max_drawdown_pct: float = 20.0
    random_state: int = 20260817
    folds: tuple[WalkForwardFold, ...] = DEFAULT_FOLDS
    trade_templates: tuple[TradeTemplate, ...] = DEFAULT_TRADE_TEMPLATES

    def validate(self) -> None:
        if self.round_trip_cost <= 0:
            raise ValueError("round_trip_cost must be positive")
        if not (0 < self.risk_per_trade <= 0.02):
            raise ValueError("risk_per_trade must be in (0, 2%]")
        if self.max_notional_multiple <= 0:
            raise ValueError("max_notional_multiple must be positive")
        if not self.delay_stress_minutes or any(int(x) < 0 for x in self.delay_stress_minutes):
            raise ValueError("delay_stress_minutes must be non-negative")
        if 0 not in self.delay_stress_minutes:
            raise ValueError("delay_stress_minutes must include base delay 0")
        if any(int(x) <= 0 for x in self.top_trade_removal_counts):
            raise ValueError("top_trade_removal_counts must be positive")
        if any(not (0 < q < 1) for q in self.threshold_quantiles):
            raise ValueError("threshold_quantiles must be in (0,1)")
        if not self.trade_templates:
            raise ValueError("at least one trade template is required")
        for template in self.trade_templates:
            template.validate()
        seal = pd.Timestamp(self.sealed_holdout_start)
        for fold in self.folds:
            train_start = pd.Timestamp(fold.train_start)
            train_end = pd.Timestamp(fold.train_end_exclusive)
            calibration_start = pd.Timestamp(fold.calibration_start)
            calibration_end = pd.Timestamp(fold.calibration_end_exclusive)
            oos_start = pd.Timestamp(fold.oos_start)
            oos_end = pd.Timestamp(fold.oos_end_exclusive)
            if not (
                train_start < train_end <= calibration_start
                < calibration_end <= oos_start < oos_end <= seal
            ):
                raise ValueError(f"invalid or sealed-overlapping fold: {fold}")

    @property
    def cache_path(self) -> Path:
        return Path(self.r00_cache_dir)

    @property
    def report_path(self) -> Path:
        return Path(self.report_dir)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        return payload


DEFAULT_R01_CONFIG = R01Config()
