#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Executable strategy contract used by ETH portfolio development."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


StrategyClass = Literal["core", "rare_event"]
StrategyStage = Literal[
    "planned",
    "in_development",
    "backtest_failed",
    "backtest_candidate",
    "portfolio_candidate",
    "shadow",
    "live",
    "archived",
]


@dataclass(frozen=True)
class StrategyContract:
    """Minimum information required before an idea can be called a strategy.

    The fields are intentionally execution-oriented.  Feature studies may feed
    these rules, but a strategy cannot graduate with only an event label or a
    forward-return statistic.
    """

    strategy_id: str
    name: str
    sleeve_id: str
    family: str
    strategy_class: StrategyClass
    stage: StrategyStage
    symbol: str
    signal_timeframe: str
    execution_timeframe: str
    setup: str
    trigger: str
    entry: str
    initial_stop: str
    exit_logic: str
    sizing: str
    invalidation: str
    causal_timing: str = "closed_bar_signal_next_bar_open_execution"
    target_min_trades: int = 300
    notes: str = ""

    def validate(self) -> None:
        required_text = {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "sleeve_id": self.sleeve_id,
            "family": self.family,
            "symbol": self.symbol,
            "signal_timeframe": self.signal_timeframe,
            "execution_timeframe": self.execution_timeframe,
            "setup": self.setup,
            "trigger": self.trigger,
            "entry": self.entry,
            "initial_stop": self.initial_stop,
            "exit_logic": self.exit_logic,
            "sizing": self.sizing,
            "invalidation": self.invalidation,
            "causal_timing": self.causal_timing,
        }
        empty = [name for name, value in required_text.items() if not str(value).strip()]
        if empty:
            raise ValueError(f"strategy contract missing required fields: {empty}")
        if self.strategy_class == "core" and self.target_min_trades < 100:
            raise ValueError("core strategy target_min_trades must be >= 100")
        if self.target_min_trades <= 0:
            raise ValueError("target_min_trades must be > 0")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)
