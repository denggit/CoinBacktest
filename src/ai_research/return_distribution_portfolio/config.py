from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT


@dataclass(frozen=True)
class ReturnDistributionConfig:
    """Frozen first-pass contract for the continuous-position research mainline.

    This is deliberately a small baseline contract, not a hyper-parameter grid.
    Any later changes should be versioned in the research log instead of being
    silently tuned after looking at OOS results.
    """

    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2026-06-30 23:59:59"
    decision_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (30, 120, 360, 1440, 4320)
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    round_trip_fee_rate: float = 0.0011
    shard_context_days: int = 10
    train_stride: int = 3
    train_sample_cap: int = 300_000
    lightgbm_n_estimators: int = 320
    lightgbm_learning_rate: float = 0.035
    lightgbm_num_leaves: int = 31
    lightgbm_min_child_samples: int = 300
    feature_fraction: float = 0.85
    cache_dir: str = "data/cache/eth_return_distribution_portfolio/v1"
    report_dir: str = "data/reports/research/eth_return_distribution_portfolio/01_price_flow_distribution_baseline"

    def validate(self) -> None:
        warmup = pd.Timestamp(self.warmup_start)
        start = pd.Timestamp(self.research_start)
        end = pd.Timestamp(self.research_end)
        if not warmup < start <= end:
            raise ValueError("dates must satisfy warmup_start < research_start <= research_end")
        if self.decision_minutes <= 0:
            raise ValueError("decision_minutes must be positive")
        if not self.horizons_minutes or tuple(sorted(set(self.horizons_minutes))) != self.horizons_minutes:
            raise ValueError("horizons_minutes must be unique and increasing")
        if any(h <= 0 or h % self.decision_minutes != 0 for h in self.horizons_minutes):
            raise ValueError("each horizon must be a positive multiple of decision_minutes")
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and increasing")
        if self.quantiles[0] <= 0 or self.quantiles[-1] >= 1 or 0.5 not in self.quantiles:
            raise ValueError("quantiles must lie inside (0,1) and include 0.50")
        if self.round_trip_fee_rate <= 0:
            raise ValueError("round_trip_fee_rate must be positive")
        if self.shard_context_days < 4:
            raise ValueError("shard_context_days must cover multi-day features and 72h target path")
        if self.train_stride < 1 or self.train_sample_cap < 10_000:
            raise ValueError("invalid training sampling contract")
        if "eth_return_distribution_portfolio" not in self.cache_dir:
            raise ValueError("cache_dir must be isolated to this mainline")

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / self.cache_dir

    @property
    def report_path(self) -> Path:
        return PROJECT_ROOT / self.report_dir

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = asdict(self)
        out["horizons_minutes"] = list(self.horizons_minutes)
        out["quantiles"] = list(self.quantiles)
        return out


DEFAULT_CONFIG = ReturnDistributionConfig()
