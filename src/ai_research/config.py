#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen governance defaults for the ETH AI trading research programme.

These values describe the initial research contract. They are not a model
hyper-parameter grid and must not be silently changed to improve a backtest.
Any future change requires an explicit research-plan revision and report note.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "reports" / "research" / "eth_ai_trading"
DEFAULT_PLAN_DOC = PROJECT_ROOT / "docs" / "ETH_AI_TRADING_RESEARCH_PLAN.md"


@dataclass(frozen=True)
class AIResearchConfig:
    """Project-level research and execution assumptions.

    The input data resolution and decision cadence are intentionally separate:
    one-second trade bars preserve microstructure information, while the first
    deployable baseline makes a decision every five seconds.
    """

    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    validation_start: str = "2025-01-01 00:00:00"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    research_end: str = "2026-06-30 23:59:59"
    input_bar_seconds: int = 1
    decision_interval_seconds: int = 5
    round_trip_fee_rate: float = 0.0011
    latency_scenarios_seconds: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0)
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    execution_style: str = "market_order_first"
    primary_timezone_policy: str = "project_local_timezone_plus_8"

    def validate(self) -> None:
        warmup = pd.Timestamp(self.warmup_start)
        research = pd.Timestamp(self.research_start)
        validation = pd.Timestamp(self.validation_start)
        holdout = pd.Timestamp(self.sealed_holdout_start)
        end = pd.Timestamp(self.research_end)
        if not warmup <= research < validation < holdout <= end:
            raise ValueError(
                "research dates must satisfy warmup <= research < validation "
                "< sealed_holdout <= research_end"
            )
        if self.input_bar_seconds <= 0:
            raise ValueError("input_bar_seconds must be positive")
        if self.decision_interval_seconds < self.input_bar_seconds:
            raise ValueError("decision cadence cannot be faster than the input bar")
        if self.decision_interval_seconds % self.input_bar_seconds != 0:
            raise ValueError("decision interval must be an integer multiple of input bar seconds")
        if self.round_trip_fee_rate <= 0:
            raise ValueError("round_trip_fee_rate must be positive")
        if not self.latency_scenarios_seconds or any(value < 0 for value in self.latency_scenarios_seconds):
            raise ValueError("latency scenarios must be non-empty and non-negative")
        if not self.cost_stress_multipliers or any(value < 1 for value in self.cost_stress_multipliers):
            raise ValueError("cost stress multipliers must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["latency_scenarios_seconds"] = list(self.latency_scenarios_seconds)
        payload["cost_stress_multipliers"] = list(self.cost_stress_multipliers)
        return payload


DEFAULT_AI_RESEARCH_CONFIG = AIResearchConfig()
