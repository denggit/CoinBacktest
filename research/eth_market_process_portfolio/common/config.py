"""Frozen project-level defaults for ETH market-process portfolio research.

The values in this module are research governance defaults, not tunable strategy
parameters. Individual studies may narrow their data window when a source has
shorter coverage, but they must report that limitation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYMBOL = "ETH-USDT-SWAP"
DEFAULT_WARMUP_START = pd.Timestamp("2022-01-01 00:00:00")
DEFAULT_RESEARCH_START = pd.Timestamp("2023-01-01 00:00:00")
DEFAULT_RESEARCH_END = pd.Timestamp("2026-06-30 23:59:59")
DEFAULT_ROUND_TRIP_COST = 0.0011
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "reports" / "research" / "eth_market_process_portfolio"


@dataclass(frozen=True)
class ResearchWindow:
    warmup_start: pd.Timestamp = DEFAULT_WARMUP_START
    research_start: pd.Timestamp = DEFAULT_RESEARCH_START
    research_end: pd.Timestamp = DEFAULT_RESEARCH_END

    def validate(self) -> None:
        if self.warmup_start > self.research_start:
            raise ValueError("warmup_start must be <= research_start")
        if self.research_start > self.research_end:
            raise ValueError("research_start must be <= research_end")
