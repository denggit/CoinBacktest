#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for the clean-sheet ETH RL market-agent programme.

R00 does not train a trading policy.  It creates a causally aligned state and
forward-path dataset that later supervised and offline-RL stages share.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "research"
    / "eth_ai_trading"
    / "rl_market_agent_v1"
    / "r00_causal_state_dataset"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "eth_ai_trading" / "rl_market_agent_v1" / "r00_4"


@dataclass(frozen=True)
class RLMarketAgentConfig:
    """Project-level data contract for the new ETH AI/RL research line."""

    dataset_revision: str = "R00.4"
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    sealed_holdout_start: str = "2026-01-01 00:00:00"
    research_end: str = "2026-08-15 23:59:59"

    decision_interval: str = "5min"
    label_horizons_minutes: tuple[int, ...] = (15, 30, 60, 180, 360)

    # Core sources.  Fixed-timeframe K-lines are intentionally higher timeframe;
    # the 1m execution/path axis comes from the tick-derived 1m trade-bar cache.
    kline_timeframes: tuple[str, ...] = ("5m", "15m", "1H", "4H", "1D")
    trade_bar_timeframe: str = "1m"
    micro_trade_bar_timeframe: str = "5s"

    # Existing range/footprint data products.  All are read through src.data_feed.
    range_pcts: tuple[float, ...] = (0.0015, 0.0020, 0.0025)
    footprint_range_pct: float = 0.0020
    footprint_price_step: float = 1.0

    # R00 keeps enough left context to compute all micro rolling features while
    # still reading one month at a time.  Long context lives on higher timeframes.
    micro_context_minutes: int = 1_440
    trade_windows_minutes: tuple[int, ...] = (5, 15, 30, 60, 180, 360)
    micro_windows_seconds: tuple[int, ...] = (30, 60, 300, 900)
    range_windows_minutes: tuple[int, ...] = (5, 15, 30, 60, 180)

    round_trip_fee_rate: float = 0.0011
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)

    # Data policy.  Core 1m trade bars are mandatory.  Enrichment sources are
    # measured, not silently required, so incomplete historical coverage cannot
    # make the build unusable or leak calendar information without disclosure.
    require_core_trade_bars: bool = True
    require_all_kline_context: bool = True
    require_micro_trade_bars: bool = False
    require_range_bars: bool = False
    require_footprint: bool = False

    cache_dir: str = str(DEFAULT_CACHE_DIR)
    report_dir: str = str(DEFAULT_REPORT_DIR)

    def validate(self) -> None:
        if not self.dataset_revision.strip():
            raise ValueError("dataset_revision must be non-empty")
        warmup = pd.Timestamp(self.warmup_start)
        start = pd.Timestamp(self.research_start)
        holdout = pd.Timestamp(self.sealed_holdout_start)
        end = pd.Timestamp(self.research_end)
        if not warmup < start < holdout <= end:
            raise ValueError("dates must satisfy warmup < research_start < sealed_holdout_start <= research_end")
        if self.decision_end < start:
            raise ValueError("research_end must leave room for the largest forward-label horizon")
        step = pd.Timedelta(self.decision_interval)
        if step <= pd.Timedelta(0):
            raise ValueError("decision_interval must be positive")
        if step % pd.Timedelta(minutes=1) != pd.Timedelta(0):
            raise ValueError("R00 decision_interval must be an integer number of minutes")
        horizons = tuple(int(x) for x in self.label_horizons_minutes)
        if not horizons or any(x <= 0 for x in horizons) or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("label_horizons_minutes must be unique, positive, and sorted")
        if any(x <= 0 for x in self.trade_windows_minutes):
            raise ValueError("trade_windows_minutes must be positive")
        if any(x <= 0 or x % 5 != 0 for x in self.micro_windows_seconds):
            raise ValueError("micro_windows_seconds must be positive multiples of the 5s micro bar")
        if any(x <= 0 for x in self.range_windows_minutes):
            raise ValueError("range_windows_minutes must be positive")
        if self.round_trip_fee_rate <= 0:
            raise ValueError("round_trip_fee_rate must be positive")
        if not self.cost_stress_multipliers or any(x < 1 for x in self.cost_stress_multipliers):
            raise ValueError("cost_stress_multipliers must all be >= 1")
        if self.micro_context_minutes < max(self.trade_windows_minutes):
            raise ValueError("micro_context_minutes must cover the largest trade window")
        if self.footprint_range_pct not in self.range_pcts:
            raise ValueError("footprint_range_pct must be one of range_pcts")

    @property
    def max_label_horizon(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=max(self.label_horizons_minutes))

    @property
    def decision_end(self) -> pd.Timestamp:
        """Last decision timestamp with a fully observable largest-horizon label."""

        return pd.Timestamp(self.research_end) - self.max_label_horizon

    @property
    def micro_context(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=int(self.micro_context_minutes))

    @property
    def report_path(self) -> Path:
        return Path(self.report_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in (
            "label_horizons_minutes",
            "kline_timeframes",
            "range_pcts",
            "trade_windows_minutes",
            "micro_windows_seconds",
            "range_windows_minutes",
            "cost_stress_multipliers",
        ):
            payload[key] = list(payload[key])
        return payload


DEFAULT_CONFIG = RLMarketAgentConfig()
