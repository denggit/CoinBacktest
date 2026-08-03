#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Typed stage-plan contracts for ETH AI research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .config import AIResearchConfig


StageOwner = Literal["coinbacktest", "aetheredge", "cross_project"]


@dataclass(frozen=True)
class StageDefinition:
    """One gated stage in the research-to-live programme."""

    stage_id: str
    name: str
    owner: StageOwner
    goal: str
    depends_on: tuple[str, ...] = ()
    ai_methods: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ()
    acceptance_gates: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = str(self.stage_id).strip().upper()
        if not normalized or not normalized.startswith("R") or not normalized[1:].isdigit():
            raise ValueError(f"invalid stage id: {self.stage_id!r}")
        object.__setattr__(self, "stage_id", normalized)
        for field_name in (
            "depends_on",
            "ai_methods",
            "deliverables",
            "acceptance_gates",
            "stop_conditions",
        ):
            values = tuple(str(item).strip() for item in getattr(self, field_name) if str(item).strip())
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "depends_on",
            "ai_methods",
            "deliverables",
            "acceptance_gates",
            "stop_conditions",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class ResearchPlan:
    """Canonical ordered stage plan and frozen research assumptions."""

    plan_id: str
    title: str
    version: int
    config: AIResearchConfig
    stages: tuple[StageDefinition, ...]
    plan_doc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "title": self.title,
            "version": self.version,
            "config": self.config.to_dict(),
            "plan_doc": self.plan_doc,
            "stages": [stage.to_dict() for stage in self.stages],
        }
