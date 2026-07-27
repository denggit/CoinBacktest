#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen martingale variants and engine-level risk/cost configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MartingaleVariant:
    """Frozen parameters for one user-requested martingale ladder."""

    key: str
    display_name: str
    initial_add_ratio: float
    entry_drop_pct: float
    spacing_multiplier: float
    amount_multiplier: float
    take_profit_pct: float
    max_additions: int
    leverage: float

    @property
    def total_orders(self) -> int:
        return 1 + int(self.max_additions)

    def validate(self) -> None:
        if not self.key:
            raise ValueError("variant key cannot be empty")
        if self.initial_add_ratio <= 0:
            raise ValueError("initial_add_ratio must be > 0")
        if not 0 < self.entry_drop_pct < 1:
            raise ValueError("entry_drop_pct must be in (0, 1)")
        if self.spacing_multiplier <= 0:
            raise ValueError("spacing_multiplier must be > 0")
        if self.amount_multiplier <= 0:
            raise ValueError("amount_multiplier must be > 0")
        if not 0 < self.take_profit_pct < 1:
            raise ValueError("take_profit_pct must be in (0, 1)")
        if self.max_additions < 0:
            raise ValueError("max_additions must be >= 0")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")


VARIANTS: dict[str, MartingaleVariant] = {
    "midterm": MartingaleVariant(
        key="midterm",
        display_name="中期趋势加仓",
        initial_add_ratio=0.94,
        entry_drop_pct=0.0100,
        spacing_multiplier=1.00,
        amount_multiplier=1.05,
        take_profit_pct=0.0410,
        max_additions=8,
        leverage=10.0,
    ),
    "aggressive": MartingaleVariant(
        key="aggressive",
        display_name="短期进取型投资",
        initial_add_ratio=0.54,
        entry_drop_pct=0.0053,
        spacing_multiplier=1.00,
        amount_multiplier=1.10,
        take_profit_pct=0.0410,
        max_additions=12,
        leverage=13.0,
    ),
    "longterm": MartingaleVariant(
        key="longterm",
        display_name="长期稳健投资",
        initial_add_ratio=1.11,
        entry_drop_pct=0.0137,
        spacing_multiplier=1.00,
        amount_multiplier=1.05,
        take_profit_pct=0.0500,
        max_additions=7,
        leverage=9.0,
    ),
}


@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 1000.0
    capital_utilization: float = 1.0
    fee_rate: float = 0.00055
    maintenance_margin_rate: float = 0.005
    force_close_end: bool = True

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0 < self.capital_utilization <= 1:
            raise ValueError("capital_utilization must be in (0, 1]")
        if self.fee_rate < 0 or self.fee_rate >= 0.1:
            raise ValueError("fee_rate must be in [0, 0.1)")
        if self.maintenance_margin_rate < 0 or self.maintenance_margin_rate >= 0.5:
            raise ValueError("maintenance_margin_rate must be in [0, 0.5)")
        if self.fee_rate + self.maintenance_margin_rate >= 1:
            raise ValueError("fee_rate + maintenance_margin_rate must be < 1")
