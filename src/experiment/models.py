#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Typed metadata models for research/backtest experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping


ExperimentStage = Literal[
    "idea",
    "research",
    "edge",
    "backtest",
    "portfolio",
    "promotion",
    "live",
]

ExperimentStatus = Literal[
    "idea",
    "researching",
    "rejected",
    "edge_found",
    "candidate",
    "backtest_passed",
    "backtest_failed",
    "portfolio_pending",
    "portfolio_passed",
    "portfolio_rejected",
    "promoted",
    "live",
    "archived",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_id(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("experiment id must not be empty")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(ch not in allowed for ch in normalized):
        raise ValueError(
            "experiment id may contain only A-Z, 0-9 and underscore: "
            f"{value!r}"
        )
    return normalized


@dataclass(frozen=True)
class ExperimentRecord:
    """Lifecycle metadata for one ETH edge/research/backtest line."""

    id: str
    title: str
    stage: ExperimentStage = "idea"
    status: ExperimentStatus = "idea"
    symbol: str = "ETH-USDT-SWAP"
    family: str = "unclassified"
    hypothesis: str = ""
    owner: str = "research"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    data_required: tuple[str, ...] = ()
    reports: Mapping[str, str] = field(default_factory=dict)
    decision: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id))
        object.__setattr__(
            self,
            "data_required",
            tuple(str(item) for item in self.data_required),
        )
        object.__setattr__(self, "reports", dict(self.reports))
        object.__setattr__(self, "decision", dict(self.decision))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", data["id"])),
            stage=data.get("stage", "idea"),
            status=data.get("status", "idea"),
            symbol=str(data.get("symbol", "ETH-USDT-SWAP")),
            family=str(data.get("family", "unclassified")),
            hypothesis=str(data.get("hypothesis", "")),
            owner=str(data.get("owner", "research")),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            data_required=tuple(data.get("data_required", ())),
            reports=dict(data.get("reports", {})),
            decision=dict(data.get("decision", {})),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "stage": self.stage,
            "status": self.status,
            "symbol": self.symbol,
            "family": self.family,
            "hypothesis": self.hypothesis,
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data_required": list(self.data_required),
            "reports": dict(self.reports),
            "decision": dict(self.decision),
            "notes": self.notes,
        }

    def with_update(self, **changes: Any) -> "ExperimentRecord":
        payload = self.to_dict()
        payload.update(changes)
        payload["updated_at"] = utc_now_iso()
        return ExperimentRecord.from_dict(payload)
