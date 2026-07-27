#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure models for the offline order-book liquidity map.

The models intentionally contain no filesystem or UI code.  Historical OKX
files are normalized in :mod:`src.data_feed.okx_books_loader`, while replay,
aggregation and feature construction live in :mod:`src.liquidity_map`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    size_contracts: float
    order_count: int = 0

    @classmethod
    def from_sequence(cls, values: Iterable[Any]) -> "BookLevel":
        row = list(values)
        if len(row) < 2:
            raise ValueError(f"book level needs at least price and size: {row!r}")
        order_index = 3 if len(row) >= 4 else 2 if len(row) >= 3 else None
        count = 0
        if order_index is not None:
            try:
                count = int(float(row[order_index] or 0))
            except (TypeError, ValueError):
                count = 0
        return cls(price=float(row[0]), size_contracts=float(row[1]), order_count=count)


@dataclass(frozen=True, slots=True)
class BookEvent:
    ts_ms: int
    action: str
    bids: tuple[BookLevel, ...] = ()
    asks: tuple[BookLevel, ...] = ()
    seq_id: int | None = None
    prev_seq_id: int | None = None
    source_file: str = ""
    source_line: int = 0

    @property
    def is_snapshot(self) -> bool:
        return self.action == "snapshot"


@dataclass(frozen=True)
class LiquidityMapConfig:
    """Build configuration shared by the prebuilder and feature store."""

    symbol: str = "ETH-USDT-SWAP"
    books_depth: int = 400
    price_step: float = 1.0
    feature_seconds: int = 1
    heatmap_seconds: int = 15
    contract_value_base: float = 0.1
    max_distance_pct: float = 0.08
    max_levels_per_side: int = 60
    min_store_depth_base: float = 0.05
    min_store_ratio: float = 0.01
    large_depth_ratio: float = 0.50
    decision_delay_ms: int = 1000
    max_book_staleness_seconds: int = 30
    strict_sequence: bool = True

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.books_depth <= 0:
            raise ValueError("books_depth must be > 0")
        if self.price_step <= 0:
            raise ValueError("price_step must be > 0")
        if self.feature_seconds <= 0:
            raise ValueError("feature_seconds must be > 0")
        if self.heatmap_seconds < self.feature_seconds:
            raise ValueError("heatmap_seconds must be >= feature_seconds")
        if self.heatmap_seconds % self.feature_seconds:
            raise ValueError("heatmap_seconds must be a multiple of feature_seconds")
        if self.contract_value_base <= 0:
            raise ValueError("contract_value_base must be > 0")
        if not 0 < self.max_distance_pct <= 1:
            raise ValueError("max_distance_pct must be in (0, 1]")
        if self.max_levels_per_side < 0:
            raise ValueError("max_levels_per_side must be >= 0 (0 means no cap)")
        if self.min_store_depth_base < 0:
            raise ValueError("min_store_depth_base must be >= 0")
        if not 0 <= self.min_store_ratio <= 1:
            raise ValueError("min_store_ratio must be in [0, 1]")
        if not 0 < self.large_depth_ratio <= 1:
            raise ValueError("large_depth_ratio must be in (0, 1]")
        if self.decision_delay_ms < 0:
            raise ValueError("decision_delay_ms must be >= 0")
        if self.max_book_staleness_seconds <= 0:
            raise ValueError("max_book_staleness_seconds must be > 0")

    @property
    def feature_ms(self) -> int:
        return self.feature_seconds * 1000

    @property
    def heatmap_ms(self) -> int:
        return self.heatmap_seconds * 1000

    @property
    def max_book_staleness_ms(self) -> int:
        return self.max_book_staleness_seconds * 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityBuildStats:
    day: str
    book_events: int = 0
    snapshots: int = 0
    updates: int = 0
    sequence_gaps: int = 0
    invalid_events: int = 0
    book_feature_rows: int = 0
    heatmap_cells: int = 0
    raw_trade_rows: int = 0
    trade_buckets: int = 0
    first_event_ms: int | None = None
    last_event_ms: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
