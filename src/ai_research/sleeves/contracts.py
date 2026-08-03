#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared contracts for the three ETH AI trading sleeves.

The contracts deliberately stop at a target-position intent.  They contain no
exchange adapter, database access, or live-order code, so CoinBacktest research
and the later AetherEdge strategy plugin can implement the same schema without
importing each other's runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd


SleeveId = Literal["short_horizon", "intraday_trend", "swing"]
Direction = Literal["long", "short", "flat"]
ExitStyle = Literal[
    "micro_dynamic_with_time_cap",
    "structure_state_trailing_with_time_cap",
]


@dataclass(frozen=True)
class SleeveSpec:
    sleeve_id: SleeveId
    display_name: str
    decision_cadence: str
    intended_hold_min_minutes: int
    intended_hold_max_minutes: int
    target_move_min: float
    target_move_max: float
    direction_timeframes: tuple[str, ...]
    entry_timeframes: tuple[str, ...]
    exit_style: ExitStyle
    max_hold_is_safety_only: bool

    def __post_init__(self) -> None:
        if self.intended_hold_min_minutes < 0:
            raise ValueError("minimum hold must be non-negative")
        if self.intended_hold_max_minutes <= self.intended_hold_min_minutes:
            raise ValueError("maximum hold must exceed minimum hold")
        if not 0 < self.target_move_min <= self.target_move_max:
            raise ValueError("target move range must be positive and ordered")
        if not self.direction_timeframes or not self.entry_timeframes:
            raise ValueError("direction and entry timeframes must be declared")
        if self.sleeve_id != "short_horizon" and not self.max_hold_is_safety_only:
            raise ValueError("intraday and swing sleeves may not use time as the primary exit")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction_timeframes"] = list(self.direction_timeframes)
        payload["entry_timeframes"] = list(self.entry_timeframes)
        return payload


@dataclass(frozen=True)
class ModelEvidence:
    """One model/state source's causal evidence at a decision timestamp."""

    source_id: str
    sleeve_id: SleeveId
    asof: pd.Timestamp
    direction: Direction
    success_probability: float
    expected_move: float
    predicted_mfe: float
    predicted_mae: float
    horizon_minutes: int
    feature_version: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.feature_version.strip():
            raise ValueError("source_id and feature_version are required")
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("success_probability must be in [0, 1]")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        object.__setattr__(self, "asof", pd.Timestamp(self.asof))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asof"] = str(self.asof)
        return payload


@dataclass(frozen=True)
class TradeCandidate:
    """A sleeve-level candidate, not a direct order."""

    candidate_id: str
    sleeve_id: SleeveId
    decision_time: pd.Timestamp
    entry_not_before: pd.Timestamp
    direction: Direction
    score: float
    expected_move: float
    predicted_mfe: float
    predicted_mae: float
    invalidation_price: float | None
    max_hold_minutes: int
    evidence: tuple[ModelEvidence, ...]

    def __post_init__(self) -> None:
        decision = pd.Timestamp(self.decision_time)
        entry = pd.Timestamp(self.entry_not_before)
        if entry <= decision:
            raise ValueError("entry_not_before must be after decision_time")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id is required")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be in [0, 1]")
        if self.max_hold_minutes <= 0:
            raise ValueError("max_hold_minutes must be positive")
        if self.direction == "flat" and self.invalidation_price is not None:
            raise ValueError("flat candidates may not define an invalidation price")
        if any(item.asof > decision for item in self.evidence):
            raise ValueError("candidate evidence may not be newer than decision_time")
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "entry_not_before", entry)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision_time"] = str(self.decision_time)
        payload["entry_not_before"] = str(self.entry_not_before)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


@dataclass(frozen=True)
class SleeveContribution:
    sleeve_id: SleeveId
    direction: Direction
    raw_score: float
    risk_weight: float
    target_fraction: float
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_score <= 1.0:
            raise ValueError("raw_score must be in [0, 1]")
        if not 0.0 <= self.risk_weight <= 1.0:
            raise ValueError("risk_weight must be in [0, 1]")
        if abs(self.target_fraction) > 1.0:
            raise ValueError("target_fraction must be inside [-1, 1]")


@dataclass(frozen=True)
class TargetPositionDecision:
    """Single ETH target-position contract consumed by a later strategy plugin."""

    decision_time: pd.Timestamp
    direction: Direction
    target_fraction: float
    contributions: tuple[SleeveContribution, ...]
    risk_vetoes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if abs(self.target_fraction) > 1.0:
            raise ValueError("target_fraction must be inside [-1, 1]")
        if self.risk_vetoes and abs(self.target_fraction) > 1e-12:
            raise ValueError("risk vetoes require a flat target")
        if self.direction == "flat" and abs(self.target_fraction) > 1e-12:
            raise ValueError("flat decision must have zero target_fraction")
        if self.direction == "long" and self.target_fraction <= 0:
            raise ValueError("long decision requires positive target_fraction")
        if self.direction == "short" and self.target_fraction >= 0:
            raise ValueError("short decision requires negative target_fraction")
        object.__setattr__(self, "decision_time", pd.Timestamp(self.decision_time))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision_time"] = str(self.decision_time)
        payload["contributions"] = [asdict(item) for item in self.contributions]
        payload["risk_vetoes"] = list(self.risk_vetoes)
        return payload
