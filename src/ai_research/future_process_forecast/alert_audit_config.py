#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.3.1 actionable early-warning audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

from .config import DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG, FutureProcessForecastConfig


STAGE_ID = "R03.3.1"
STAGE_NAME = "Actionable process early-warning audit"


@dataclass(frozen=True)
class ProcessAlertAuditConfig:
    base: FutureProcessForecastConfig = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    processes: tuple[str, ...] = ("up_expansion", "down_expansion", "volatile_range")
    horizons_hours: tuple[int, ...] = (6, 12)
    architectures: tuple[str, ...] = (
        "macro_lightgbm",
        "multiframe_lightgbm",
        "multiframe_micro_lightgbm",
    )
    signal_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    alert_merge_gap_hours: float = 1.0
    early_start_grace_hours: float = 2.0
    max_actionable_progress: float = 0.25
    min_remaining_directional_move: float = 0.025
    min_remaining_range_move: float = 0.030
    minimum_events_per_fold: int = 20
    minimum_actionable_precision: float = 0.20
    minimum_event_coverage: float = 0.30
    maximum_late_ongoing_rate: float = 0.25
    maximum_alerts_per_month: float = 15.0
    report_dir: str = "data/reports/research/eth_ai_trading/03_3_1_process_alert_value_audit"

    def validate(self) -> None:
        self.base.validate()
        valid_processes = {"up_expansion", "down_expansion", "volatile_range"}
        if not self.processes or not set(self.processes).issubset(valid_processes):
            raise ValueError("R03.3.1 supports only tradable R03.3 process heads")
        if not self.horizons_hours or not set(self.horizons_hours).issubset(set(self.base.forecast_horizons_hours)):
            raise ValueError("R03.3.1 horizons must be available in the frozen R03.3 labels")
        if not self.architectures or not set(self.architectures).issubset(set(self.base.architectures)):
            raise ValueError("R03.3.1 architectures must reuse R03.3 model families")
        if not 0 < self.alert_merge_gap_hours <= 6:
            raise ValueError("invalid alert merge gap")
        if not 0 <= self.early_start_grace_hours <= 6:
            raise ValueError("invalid early-start grace")
        if not 0 < self.max_actionable_progress < 1:
            raise ValueError("invalid actionable progress cap")
        if not 0 < self.min_remaining_directional_move < self.base.directional.target_floor:
            raise ValueError("invalid remaining directional move gate")
        if not 0 < self.min_remaining_range_move < 0.20:
            raise ValueError("invalid remaining range gate")
        if tuple(sorted(set(self.signal_quantiles))) != self.signal_quantiles:
            raise ValueError("signal quantiles must be unique and increasing")
        if "03_3_1" not in self.report_dir:
            raise ValueError("R03.3.1 report directory must be isolated")

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["base"] = self.base.to_dict()
        for key in ("processes", "horizons_hours", "architectures", "signal_quantiles"):
            payload[key] = list(payload[key])
        return payload


DEFAULT_PROCESS_ALERT_AUDIT_CONFIG = ProcessAlertAuditConfig()
