from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceLockedConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_start: str = "2026-01-01 00:00:00"
    round_trip_cost: float = 0.0011
    initial_capital: float = 100_000.0
    timezone_offset_hours: int = 8
    report_root: Path = Path("data/reports/research/eth_strategy_factory/r03_source_locked")
    flat_exposure_threshold: float = 0.05
    low_exposure_threshold: float = 0.25

    @property
    def one_way_cost(self) -> float:
        return self.round_trip_cost / 2.0


ZARATTINI_LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
