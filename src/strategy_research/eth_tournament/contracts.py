from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd


class SourceClass(str, Enum):
    SOURCE_FAITHFUL = "SOURCE_FAITHFUL"
    SOURCE_VARIANT = "SOURCE_VARIANT"
    SOURCE_INSPIRED_ENGINEERING = "SOURCE_INSPIRED_ENGINEERING"


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    family_id: str
    family_name: str
    variant_name: str
    source_class: SourceClass
    source_title: str
    source_url: str
    rules_summary: str
    data_requirements: tuple[str, ...]
    engine: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["source_class"] = self.source_class.value
        out["data_requirements"] = ",".join(self.data_requirements)
        return out


@dataclass(frozen=True)
class EntryEvent:
    signal_time: pd.Timestamp
    side: int
    stop_distance: float | None = None
    target_distance: float | None = None
    max_hold_minutes: int | None = None
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitEvent:
    signal_time: pd.Timestamp
    side: int
    tag: str = "RULE_EXIT"


@dataclass
class StrategySignals:
    entries: list[EntryEvent] = field(default_factory=list)
    exits: list[ExitEvent] = field(default_factory=list)
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class BacktestResult:
    strategy_id: str
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    metrics: dict[str, Any]
    causal_audit: dict[str, Any]
    stress_rows: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class PortfolioSelectionKey:
    """Smaller tuple is better; ranking order frozen by the user."""

    max_flat_days: float
    max_consecutive_losing_days: int
    max_drawdown_pct: float
    neg_cagr_pct: float
    neg_total_return_pct: float

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any]) -> "PortfolioSelectionKey":
        return cls(
            max_flat_days=float(metrics.get("max_flat_days", float("inf"))),
            max_consecutive_losing_days=int(metrics.get("max_consecutive_losing_days", 10**9)),
            max_drawdown_pct=float(metrics.get("max_drawdown_pct", float("inf"))),
            neg_cagr_pct=-float(metrics.get("cagr_pct", -float("inf"))),
            neg_total_return_pct=-float(metrics.get("total_return_pct", -float("inf"))),
        )


SignalBuilder = Callable[[Any, StrategySpec], StrategySignals]
