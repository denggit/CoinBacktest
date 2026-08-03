from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.ai_research.models import ResearchPlan, StageDefinition
from src.ai_research.plan import DEFAULT_RESEARCH_PLAN, validate_research_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_DOC = REPO_ROOT / "docs" / "ETH_AI_TRADING_RESEARCH_PLAN.md"


def test_default_plan_is_valid_and_sequential() -> None:
    validate_research_plan(DEFAULT_RESEARCH_PLAN)
    ids = [stage.stage_id for stage in DEFAULT_RESEARCH_PLAN.stages]
    assert ids == [f"R{i:02d}" for i in range(15)]
    assert DEFAULT_RESEARCH_PLAN.version == 4
    assert DEFAULT_RESEARCH_PLAN.config.input_bar_seconds == 1
    assert DEFAULT_RESEARCH_PLAN.config.decision_interval_seconds == 5
    assert DEFAULT_RESEARCH_PLAN.config.round_trip_fee_rate == pytest.approx(0.0011)

    covered_methods = {method for stage in DEFAULT_RESEARCH_PLAN.stages for method in stage.ai_methods}
    assert {
        "ai_assisted_research",
        "supervised_learning",
        "market_state_recognition",
        "signal_scoring",
        "deep_sequence_learning",
        "reinforcement_learning",
        "execution_optimisation",
        "risk_and_portfolio_management",
    } <= covered_methods


def test_each_stage_is_present_in_authoritative_doc() -> None:
    text = PLAN_DOC.read_text(encoding="utf-8")
    for stage in DEFAULT_RESEARCH_PLAN.stages:
        assert f"## {stage.stage_id} —" in text, stage.stage_id


def test_plan_rejects_missing_or_forward_dependency() -> None:
    invalid_stage = StageDefinition(
        stage_id="R15",
        name="invalid",
        owner="coinbacktest",
        goal="invalid dependency",
        depends_on=("R99",),
        acceptance_gates=("never",),
    )
    invalid_plan = replace(
        DEFAULT_RESEARCH_PLAN,
        stages=DEFAULT_RESEARCH_PLAN.stages + (invalid_stage,),
    )
    with pytest.raises(ValueError, match="missing stage"):
        validate_research_plan(invalid_plan)

    first = replace(DEFAULT_RESEARCH_PLAN.stages[0], depends_on=("R01",))
    forward_plan = ResearchPlan(
        plan_id="FORWARD",
        title="forward",
        version=1,
        config=DEFAULT_RESEARCH_PLAN.config,
        stages=(first, DEFAULT_RESEARCH_PLAN.stages[1]),
        plan_doc="docs/test.md",
    )
    with pytest.raises(ValueError, match="must appear earlier"):
        validate_research_plan(forward_plan)
