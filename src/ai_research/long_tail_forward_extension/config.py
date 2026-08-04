#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Immutable contract for the July-2026 forward extension of frozen C2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.long_tail_sealed_holdout.config import SealedHoldoutConfig

STAGE_ID = "R03.4.2.16.1"
STAGE_NAME = "July-2026 forward extension of the unchanged frozen C2 MF Long sleeve"
FOLD_ID = "WF_2026_07_FORWARD"


@dataclass(frozen=True)
class ForwardExtensionConfig(SealedHoldoutConfig):
    """Use the original pre-2026 fit/calibration and score July only."""

    holdout_start: str = "2026-07-01 00:00:00"
    holdout_end: str = "2026-07-31 23:59:59"
    post_holdout_boundary: str = "2026-08-01 00:00:00"

    source_2_16_report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_16_2026_sealed_validation"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/03_4_2_16_1_2026_july_forward_extension"
    )
    isolated_outcome_cache_dir: str = (
        "data/cache/eth_ai_trading/r03_4_2_16_1_july_forward_outcomes"
    )
    isolated_base_cache_dir: str = (
        "data/cache/eth_ai_trading/r03_4_2_16_1_july_long_context"
    )

    # One month is a forward diagnostic, not a second full qualification gate.
    minimum_executed_cycles: int = 8
    minimum_anchor_profit_factor: float = 1.0
    maximum_anchor_mdd: float = 0.15
    maximum_stress_mdd: float = 0.18
    minimum_positive_months: int = 1
    minimum_positive_quarters: int = 0
    maximum_top10_profit_share: float = 1.0
    minimum_return_without_top10: float = -0.20

    def validate(self) -> None:
        super().validate()
        if pd.Timestamp(self.holdout_start) != pd.Timestamp("2026-07-01 00:00:00"):
            raise ValueError("July forward start is frozen")
        if pd.Timestamp(self.holdout_end) != pd.Timestamp("2026-07-31 23:59:59"):
            raise ValueError("July forward end is frozen")
        if pd.Timestamp(self.post_holdout_boundary) != pd.Timestamp("2026-08-01 00:00:00"):
            raise ValueError("post-July boundary is frozen")
        if "03_4_2_16_2026_sealed_validation" not in self.source_2_16_report_dir:
            raise ValueError("R03.4.2.16 source path drift")
        if "03_4_2_16_1" not in self.report_dir:
            raise ValueError("R03.4.2.16.1 report path drift")
        if "r03_4_2_16_1" not in self.isolated_outcome_cache_dir:
            raise ValueError("R03.4.2.16.1 outcome cache path drift")
        if "r03_4_2_16_1" not in self.isolated_base_cache_dir:
            raise ValueError("R03.4.2.16.1 base cache path drift")

    @property
    def source_2_16_path(self) -> Path:
        return PROJECT_ROOT / self.source_2_16_report_dir

    @property
    def isolated_base_cache_path(self) -> Path:
        return PROJECT_ROOT / self.isolated_base_cache_dir

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload.update(
            {
                "stage_id": STAGE_ID,
                "stage_name": STAGE_NAME,
                "source_2_16_report_dir": self.source_2_16_report_dir,
                "isolated_base_cache_dir": self.isolated_base_cache_dir,
                "window_role": "new forward extension; not a repair or re-opening of the failed H1 seal",
                "january_june_use": "comparison only; forbidden for fit, calibration or rule selection",
                "july_use": "inference and diagnostic scoring only",
            }
        )
        return payload


DEFAULT_FORWARD_EXTENSION_CONFIG = ForwardExtensionConfig()
