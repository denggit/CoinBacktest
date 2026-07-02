#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Typed models for reusable event-study research."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

EntryAssumption = Literal["next_open"]


@dataclass(frozen=True)
class CostConfig:
    """Simple round-trip cost model used by event-study labels.

    The labels remain trade-direction signed returns. Costs are subtracted from
    the signed gross return so a positive edge must be large enough to survive
    fee/slippage pressure before it enters strategy development.
    """

    entry_fee_rate: float = 0.00055
    exit_fee_rate: float = 0.00055
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0

    @property
    def round_trip_cost_pct(self) -> float:
        return float(self.entry_fee_rate + self.exit_fee_rate + self.entry_slippage_pct + self.exit_slippage_pct)


@dataclass(frozen=True)
class EventStudyConfig:
    """Configuration for a causal event study on an OHLCV execution axis."""

    horizons: tuple[int, ...] = (1, 3, 6, 12)
    mfe_mae_horizon: int = 12
    entry_assumption: EntryAssumption = "next_open"
    entry_delay_bars: int = 1
    cost: CostConfig = field(default_factory=CostConfig)
    signal_time_col: str = "signal_time"
    side_col: str = "side"
    event_name_col: str | None = "event_name"
    event_id_col: str | None = None
    context_available_time_cols: tuple[str, ...] = ()
    min_count: int = 30
    progress_every: int = 0

    def __post_init__(self) -> None:
        if self.entry_assumption != "next_open":
            raise ValueError("Only entry_assumption='next_open' is supported in EventStudyConfig v1.")
        if self.entry_delay_bars < 1:
            raise ValueError("entry_delay_bars must be >= 1 so signals are executed after the closed signal bar.")
        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if any(int(h) < self.entry_delay_bars for h in self.horizons):
            raise ValueError("Each horizon must be >= entry_delay_bars for next-open labels.")
        if int(self.mfe_mae_horizon) < self.entry_delay_bars:
            raise ValueError("mfe_mae_horizon must be >= entry_delay_bars.")


@dataclass(frozen=True)
class EventStudyResult:
    """Container returned by run_event_study."""

    events: pd.DataFrame
    overview: pd.DataFrame
    yearly: pd.DataFrame
    side_stats: pd.DataFrame
    horizon_stats: pd.DataFrame
    causal_audit: pd.DataFrame
    meta: dict[str, object]

    def write(self, out_dir: str | Path) -> None:
        """Write the standard event-study report files to disk."""
        from .reports import write_event_study_report

        write_event_study_report(self, out_dir)
