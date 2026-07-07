#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metadata model for single-market ETH edge records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from src.experiment.models import normalize_id, utc_now_iso


EdgeStatus = Literal[
    "idea",
    "researching",
    "rejected",
    "edge_found",
    "candidate",
    "backtest_passed",
    "portfolio_pending",
    "portfolio_passed",
    "promoted",
    "live",
    "archived",
]


@dataclass(frozen=True)
class EdgeRecord:
    """One reusable ETH market edge and its lifecycle state."""

    id: str
    name: str
    family: str
    status: EdgeStatus = "idea"
    symbol: str = "ETH-USDT-SWAP"
    horizon: str = ""
    causal_timing: str = "closed_bar_or_later"
    data_required: tuple[str, ...] = ()
    research_report: str = ""
    backtest_report: str = ""
    portfolio_report: str = ""
    edge_summary: Mapping[str, Any] = field(default_factory=dict)
    decision: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id))
        object.__setattr__(
            self,
            "data_required",
            tuple(str(item) for item in self.data_required),
        )
        object.__setattr__(self, "edge_summary", dict(self.edge_summary))
        object.__setattr__(self, "decision", dict(self.decision))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EdgeRecord":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            family=str(data.get("family", "unclassified")),
            status=data.get("status", "idea"),
            symbol=str(data.get("symbol", "ETH-USDT-SWAP")),
            horizon=str(data.get("horizon", "")),
            causal_timing=str(data.get("causal_timing", "closed_bar_or_later")),
            data_required=tuple(data.get("data_required", ())),
            research_report=str(data.get("research_report", "")),
            backtest_report=str(data.get("backtest_report", "")),
            portfolio_report=str(data.get("portfolio_report", "")),
            edge_summary=dict(data.get("edge_summary", {})),
            decision=dict(data.get("decision", {})),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "symbol": self.symbol,
            "horizon": self.horizon,
            "causal_timing": self.causal_timing,
            "data_required": list(self.data_required),
            "research_report": self.research_report,
            "backtest_report": self.backtest_report,
            "portfolio_report": self.portfolio_report,
            "edge_summary": dict(self.edge_summary),
            "decision": dict(self.decision),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    def with_update(self, **changes: Any) -> "EdgeRecord":
        payload = self.to_dict()
        payload.update(changes)
        payload["updated_at"] = utc_now_iso()
        return EdgeRecord.from_dict(payload)
