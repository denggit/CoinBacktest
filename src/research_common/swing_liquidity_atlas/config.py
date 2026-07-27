#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration for the causal unconsumed swing-liquidity atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


_TIMEFRAME_ALIASES = {
    "15m": ("15m", 15),
    "30m": ("30m", 30),
    "1h": ("1H", 60),
    "1H": ("1H", 60),
    "60m": ("1H", 60),
    "4h": ("4H", 240),
    "4H": ("4H", 240),
    "240m": ("4H", 240),
    "1d": ("1D", 1440),
    "1D": ("1D", 1440),
    "1440m": ("1D", 1440),
}


def normalize_timeframe(value: str) -> tuple[str, int]:
    raw = str(value).strip()
    if raw not in _TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported swing timeframe: {value!r}")
    return _TIMEFRAME_ALIASES[raw]


def normalize_timeframes(values: Sequence[str]) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_timeframe(value)
        if item[0] in seen:
            continue
        seen.add(item[0])
        out.append(item)
    if not out:
        raise ValueError("At least one swing timeframe is required")
    return tuple(out)


@dataclass(frozen=True)
class AtlasConfig:
    """Predeclared event-atlas settings.

    These settings define a broad observation universe.  They are not tuned
    strategy thresholds and are never used to choose a profitable candidate.
    """

    timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    )
    confirmation_orders: tuple[int, ...] = (1, 2, 3, 5)
    approach_distance_bp: float = 200.0
    touch_distance_bp: float = 5.0
    sweep_epsilon_bp: float = 0.01
    acceptance_depth_bp: float = 50.0
    acceptance_consecutive_closes: int = 3
    resolution_horizon_bars: int = 180
    forward_horizons: tuple[int, ...] = (5, 15, 30, 60, 180)
    confluence_tolerances_bp: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0)

    def validate(self) -> "AtlasConfig":
        if not self.timeframes:
            raise ValueError("timeframes cannot be empty")
        orders = tuple(sorted(set(int(v) for v in self.confirmation_orders)))
        if not orders or orders[0] != 1 or any(v < 1 for v in orders):
            raise ValueError("confirmation_orders must contain 1 and only positive integers")
        if self.approach_distance_bp <= self.touch_distance_bp:
            raise ValueError("approach_distance_bp must exceed touch_distance_bp")
        if self.touch_distance_bp < 0 or self.sweep_epsilon_bp < 0:
            raise ValueError("touch/sweep distances cannot be negative")
        if self.acceptance_depth_bp <= 0:
            raise ValueError("acceptance_depth_bp must be positive")
        if self.acceptance_consecutive_closes < 1:
            raise ValueError("acceptance_consecutive_closes must be positive")
        if self.resolution_horizon_bars < 1:
            raise ValueError("resolution_horizon_bars must be positive")
        if not self.forward_horizons or any(int(v) < 1 for v in self.forward_horizons):
            raise ValueError("forward_horizons must be positive")
        return self
