from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TournamentConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_start: str = "2026-01-01 00:00:00"
    round_trip_cost: float = 0.0011
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_notional_leverage: float = 2.0
    timezone_offset_hours: int = 8
    base_execution_delay_minutes: int = 0
    report_root: Path = Path("data/reports/research/eth_strategy_factory/v1")
    footprint_range_pct: float = 0.0020
    footprint_price_step: float = 1.0
    survivor_max_mdd_pct: float = 20.0
    survivor_min_pf: float = 1.0
    survivor_min_positive_years: int = 2
    survivor_min_trades: int = 20
    portfolio_max_strategies: int = 5

    @property
    def one_way_cost(self) -> float:
        return self.round_trip_cost / 2.0
