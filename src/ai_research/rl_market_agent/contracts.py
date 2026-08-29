#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Typed contracts shared by R00 and later model/policy stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    causal_available_at: str
    description: str
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon_minutes: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardRecord:
    shard_id: str
    start_time: str
    end_time: str
    rows: int
    feature_count: int
    label_count: int
    features_path: str
    labels_path: str
    timestamps_path: str
    flags_path: str
    sealed_holdout: bool
    sha256_features: str
    sha256_labels: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceCoverageRecord:
    shard_id: str
    source: str
    expected_rows: int
    available_rows: int
    coverage_ratio: float
    first_available_time: str | None
    last_available_time: str | None
    required: bool
    status: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalAuditRecord:
    shard_id: str
    source: str
    rows_checked: int
    future_visibility_violations: int
    max_available_minus_decision_seconds: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class PortfolioSelectionKey:
    """Lexicographic champion-selection contract frozen by the user.

    Smaller tuple values are better.  Returns are negated so larger CAGR/total
    return rank ahead only after the three risk/continuity metrics tie.
    """

    max_flat_days: float
    max_consecutive_losing_days: int
    max_drawdown_pct: float
    neg_cagr_pct: float
    neg_total_return_pct: float

    @classmethod
    def from_metrics(
        cls,
        *,
        max_flat_days: float,
        max_consecutive_losing_days: int,
        max_drawdown_pct: float,
        cagr_pct: float,
        total_return_pct: float,
    ) -> "PortfolioSelectionKey":
        return cls(
            max_flat_days=float(max_flat_days),
            max_consecutive_losing_days=int(max_consecutive_losing_days),
            max_drawdown_pct=float(max_drawdown_pct),
            neg_cagr_pct=-float(cagr_pct),
            neg_total_return_pct=-float(total_return_pct),
        )


def rows_to_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "to_dict"):
            out.append(row.to_dict())
        else:
            out.append(dict(row))
    return out


def relative_or_absolute(path: str | Path, root: str | Path) -> str:
    p = Path(path).resolve()
    r = Path(root).resolve()
    try:
        return p.relative_to(r).as_posix()
    except ValueError:
        return str(p)
