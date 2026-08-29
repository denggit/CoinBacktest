from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContinuousPortfolioConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2025-12-31 23:59:59"
    sealed_start: str = "2026-01-01 00:00:00"
    round_trip_cost: float = 0.0011
    initial_capital: float = 100_000.0
    timezone_offset_hours: int = 8
    report_root: Path = Path("data/reports/research/eth_strategy_factory/r02_continuous_portfolio")
    flat_exposure_threshold: float = 0.05
    low_exposure_threshold: float = 0.25

    @property
    def one_way_cost(self) -> float:
        return self.round_trip_cost / 2.0


@dataclass(frozen=True)
class PortfolioSpec:
    spec_id: str
    name: str
    volatility_target: float = 0.25
    volatility_window_days: int = 90
    max_abs_exposure: float = 1.50
    deadband: float = 0.10
    use_drawdown_governor: bool = False
    max_rebalance_step: float | None = None


def frozen_specs() -> tuple[PortfolioSpec, ...]:
    """Small pre-registered component ablation, not a parameter search."""
    return (
        PortfolioSpec(
            spec_id="CP01_CORE_VOL",
            name="Core ensemble + volatility targeting",
            use_drawdown_governor=False,
        ),
        PortfolioSpec(
            spec_id="CP02_DD_GOV",
            name="Core ensemble + volatility targeting + drawdown governor",
            use_drawdown_governor=True,
        ),
        PortfolioSpec(
            spec_id="CP03_DD_GOV_SMOOTH",
            name="Core ensemble + drawdown governor + rebalance step cap",
            use_drawdown_governor=True,
            max_rebalance_step=0.50,
        ),
    )
