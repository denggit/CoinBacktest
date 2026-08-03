#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen R03.1 configuration for target-centric swing entry research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.swing_baseline.config import DEFAULT_SWING_BASELINE_CONFIG, SwingBaselineConfig


@dataclass(frozen=True)
class ExitPolicySpec:
    policy_id: str
    use_structural_stop: bool
    enable_profit_protection: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_EXIT_POLICIES = (
    ExitPolicySpec("fixed_adverse_target", use_structural_stop=False, enable_profit_protection=False),
    ExitPolicySpec("structural_protected_target", use_structural_stop=True, enable_profit_protection=True),
)


@dataclass(frozen=True)
class SwingEntryMvpConfig:
    """R03.1 keeps the original model/features and replaces labels + trade management."""

    base: SwingBaselineConfig = field(
        default_factory=lambda: replace(
            DEFAULT_SWING_BASELINE_CONFIG,
            report_dir="data/reports/research/eth_ai_trading/03_1_swing_entry_mvp",
        )
    )
    exact_label_cache_dir: str = "data/cache/eth_ai_trading/r03_1_exact_outcomes"
    report_dir: str = "data/reports/research/eth_ai_trading/03_1_swing_entry_mvp"
    architectures: tuple[str, ...] = (
        "high_logistic",
        "high_lightgbm",
        "hierarchical_lightgbm",
    )
    direction_modes: tuple[str, ...] = ("long", "short")
    exit_policies: tuple[ExitPolicySpec, ...] = DEFAULT_EXIT_POLICIES
    score_margin: float = 0.03
    protection_trigger_fraction: float = 0.50
    locked_profit_fraction: float = 0.30
    cooldown_minutes: int = 60
    same_bar_policy: str = "adverse_first"

    def validate(self) -> None:
        self.base.validate()
        if not self.architectures:
            raise ValueError("R03.1 requires at least one architecture")
        if set(self.direction_modes) - {"long", "short"}:
            raise ValueError("R03.1 direction modes must be long and/or short")
        if not self.exit_policies:
            raise ValueError("R03.1 requires at least one exit policy")
        if not 0 <= self.score_margin < 1:
            raise ValueError("score_margin must be inside [0, 1)")
        if not 0 < self.protection_trigger_fraction < 1:
            raise ValueError("protection trigger must be inside (0, 1)")
        if not 0 <= self.locked_profit_fraction < self.protection_trigger_fraction:
            raise ValueError("locked profit fraction must be below protection trigger")
        if self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes must be non-negative")
        if self.same_bar_policy != "adverse_first":
            raise ValueError("R03.1 only supports the conservative adverse_first policy")

    @property
    def exact_label_cache_path(self) -> Path:
        return PROJECT_ROOT / self.exact_label_cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "base": self.base.to_dict(),
            "exact_label_cache_dir": self.exact_label_cache_dir,
            "report_dir": self.report_dir,
            "architectures": list(self.architectures),
            "direction_modes": list(self.direction_modes),
            "exit_policies": [policy.to_dict() for policy in self.exit_policies],
            "score_margin": self.score_margin,
            "protection_trigger_fraction": self.protection_trigger_fraction,
            "locked_profit_fraction": self.locked_profit_fraction,
            "cooldown_minutes": self.cooldown_minutes,
            "same_bar_policy": self.same_bar_policy,
        }


DEFAULT_SWING_ENTRY_MVP_CONFIG = SwingEntryMvpConfig()
