#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Funnel audit that prevents research from filtering thousands of events into noise.

The audit separates *hard-filter* loss from normal execution loss.  A single-
position strategy can legitimately execute fewer trades than it signals because
it is already in a position; that is different from stacking filters until only
rare historical examples survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd


StageKind = Literal["source", "hard_filter", "score", "execution"]
StrategyClass = Literal["core", "rare_event"]
Severity = Literal["info", "warning", "fail"]


@dataclass(frozen=True)
class FunnelStage:
    name: str
    count: int
    kind: StageKind
    note: str = ""

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("funnel stage name must not be empty")
        if int(self.count) < 0:
            raise ValueError(f"funnel stage count must be >= 0: {self.name}={self.count}")


@dataclass(frozen=True)
class FunnelPolicy:
    """Fixed anti-overfit/frequency policy.

    The defaults are deliberately broad rather than tuned to a particular
    strategy.  Rare-event sleeves can use the same audit with their own lower
    trade-count requirement.
    """

    strategy_class: StrategyClass = "core"
    min_executed_trades_core: int = 300
    min_executed_trades_rare: int = 40
    warn_hard_filter_retention: float = 0.50
    fail_hard_filter_retention: float = 0.20
    min_total_hard_filter_retention_core: float = 0.10
    min_total_hard_filter_retention_rare: float = 0.02
    min_execution_rate: float = 0.05

    @property
    def min_executed_trades(self) -> int:
        return self.min_executed_trades_core if self.strategy_class == "core" else self.min_executed_trades_rare

    @property
    def min_total_hard_filter_retention(self) -> float:
        if self.strategy_class == "core":
            return self.min_total_hard_filter_retention_core
        return self.min_total_hard_filter_retention_rare


@dataclass(frozen=True)
class FunnelIssue:
    code: str
    severity: Severity
    message: str
    stage: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "stage": self.stage,
            "message": self.message,
        }


@dataclass(frozen=True)
class FunnelAudit:
    stages: tuple[FunnelStage, ...]
    policy: FunnelPolicy
    issues: tuple[FunnelIssue, ...]
    hard_filter_retention: float
    execution_rate: float
    executed_trades: int

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "fail" for issue in self.issues)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        previous: int | None = None
        for stage in self.stages:
            retention = None if previous is None or previous <= 0 else stage.count / previous
            rows.append(
                {
                    "stage": stage.name,
                    "kind": stage.kind,
                    "count": int(stage.count),
                    "previous_count": previous,
                    "retention_vs_previous": retention,
                    "dropped_vs_previous": None if previous is None else previous - stage.count,
                    "note": stage.note,
                }
            )
            previous = stage.count
        return pd.DataFrame(rows)

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame([issue.to_dict() for issue in self.issues])

    def summary(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "strategy_class": self.policy.strategy_class,
            "source_events": int(self.stages[0].count) if self.stages else 0,
            "executed_trades": int(self.executed_trades),
            "hard_filter_retention": float(self.hard_filter_retention),
            "execution_rate": float(self.execution_rate),
            "min_executed_trades": int(self.policy.min_executed_trades),
            "fail_count": sum(issue.severity == "fail" for issue in self.issues),
            "warning_count": sum(issue.severity == "warning" for issue in self.issues),
        }


def _validate_stages(stages: Sequence[FunnelStage]) -> None:
    if len(stages) < 2:
        raise ValueError("funnel audit requires at least two stages")
    if stages[0].kind != "source":
        raise ValueError("first funnel stage must have kind='source'")
    names: set[str] = set()
    previous = None
    for stage in stages:
        stage.validate()
        if stage.name in names:
            raise ValueError(f"duplicate funnel stage name: {stage.name}")
        names.add(stage.name)
        if previous is not None and stage.count > previous:
            raise ValueError(
                f"funnel counts must be non-increasing: {stage.name}={stage.count} > previous={previous}"
            )
        previous = stage.count
    if not any(stage.kind == "execution" for stage in stages):
        raise ValueError("funnel requires at least one execution stage")


def audit_funnel(stages: Sequence[FunnelStage], policy: FunnelPolicy | None = None) -> FunnelAudit:
    policy = policy or FunnelPolicy()
    stages = tuple(stages)
    _validate_stages(stages)
    issues: list[FunnelIssue] = []

    source_count = stages[0].count
    last_hard_filter_count = source_count
    previous_count = source_count

    for stage in stages[1:]:
        retention = stage.count / previous_count if previous_count > 0 else 0.0
        if stage.kind == "hard_filter":
            last_hard_filter_count = stage.count
            if retention < policy.fail_hard_filter_retention:
                issues.append(
                    FunnelIssue(
                        code="SEVERE_HARD_FILTER_COLLAPSE",
                        severity="fail",
                        stage=stage.name,
                        message=(
                            f"hard filter retained only {retention:.2%} of the previous stage; "
                            "convert quality information to scoring/sizing or justify it as a rare-event sleeve"
                        ),
                    )
                )
            elif retention < policy.warn_hard_filter_retention:
                issues.append(
                    FunnelIssue(
                        code="HARD_FILTER_STEEP_DROP",
                        severity="warning",
                        stage=stage.name,
                        message=f"hard filter retained {retention:.2%} of the previous stage",
                    )
                )
        previous_count = stage.count

    hard_retention = last_hard_filter_count / source_count if source_count > 0 else 0.0
    if hard_retention < policy.min_total_hard_filter_retention:
        issues.append(
            FunnelIssue(
                code="TOTAL_HARD_FILTER_COLLAPSE",
                severity="fail",
                stage="hard_filters",
                message=(
                    f"all hard filters retain only {hard_retention:.2%} of source events; "
                    f"policy floor is {policy.min_total_hard_filter_retention:.2%}"
                ),
            )
        )

    execution_stages = [stage for stage in stages if stage.kind == "execution"]
    executed = execution_stages[-1].count
    execution_denominator = last_hard_filter_count
    execution_rate = executed / execution_denominator if execution_denominator > 0 else 0.0
    if executed < policy.min_executed_trades:
        issues.append(
            FunnelIssue(
                code="TOO_FEW_EXECUTED_TRADES",
                severity="fail",
                stage=execution_stages[-1].name,
                message=(
                    f"executed trades={executed} is below {policy.strategy_class} strategy floor "
                    f"{policy.min_executed_trades}"
                ),
            )
        )
    if execution_rate < policy.min_execution_rate:
        issues.append(
            FunnelIssue(
                code="LOW_EXECUTION_RATE",
                severity="warning",
                stage=execution_stages[-1].name,
                message=(
                    f"only {execution_rate:.2%} of post-filter opportunities became trades; "
                    "inspect occupancy, cooldown, stop validity and execution constraints"
                ),
            )
        )

    if not issues:
        issues.append(
            FunnelIssue(
                code="FUNNEL_HEALTHY",
                severity="info",
                message="no hard-filter collapse or frequency failure detected",
            )
        )

    return FunnelAudit(
        stages=stages,
        policy=policy,
        issues=tuple(issues),
        hard_filter_retention=float(hard_retention),
        execution_rate=float(execution_rate),
        executed_trades=int(executed),
    )
