from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TurtlePathConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    discovery_end: str = "2024-12-31 23:59:59"
    sealed_start: str = "2026-01-01 00:00:00"
    round_trip_cost: float = 0.0011
    initial_capital: float = 100_000.0
    timezone_offset_hours: int = 8
    report_root: Path = Path("data/reports/research/eth_strategy_factory/r04_turtle_path_atlas")
    checkpoints_minutes: tuple[int, ...] = (5, 15, 30, 60, 240, 720, 1440, 4320, 10080)
