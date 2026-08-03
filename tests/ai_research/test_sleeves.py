from __future__ import annotations

import pandas as pd
import pytest

from src.ai_research.sleeves import (
    ModelEvidence,
    SLEEVE_SPECS,
    SleeveContribution,
    TargetPositionDecision,
    TradeCandidate,
)


def test_three_sleeves_are_independent_and_complete() -> None:
    assert set(SLEEVE_SPECS) == {"short_horizon", "intraday_trend", "swing"}
    short = SLEEVE_SPECS["short_horizon"]
    intraday = SLEEVE_SPECS["intraday_trend"]
    swing = SLEEVE_SPECS["swing"]
    assert short.target_move_max < intraday.target_move_min
    assert intraday.target_move_max < swing.target_move_min
    assert intraday.max_hold_is_safety_only
    assert swing.max_hold_is_safety_only
    assert swing.intended_hold_min_minutes == 0
    assert swing.direction_timeframes == ("1D", "4H", "1H")
    assert swing.entry_timeframes == ("30m", "15m", "5m", "1m")


def test_trade_candidate_rejects_future_evidence() -> None:
    decision = pd.Timestamp("2025-01-01 12:00:00")
    evidence = ModelEvidence(
        source_id="swing_direction",
        sleeve_id="swing",
        asof=decision + pd.Timedelta(minutes=1),
        direction="long",
        success_probability=0.7,
        expected_move=0.04,
        predicted_mfe=0.05,
        predicted_mae=0.01,
        horizon_minutes=4_320,
        feature_version="r03-v1",
    )
    with pytest.raises(ValueError, match="newer than decision"):
        TradeCandidate(
            candidate_id="bad",
            sleeve_id="swing",
            decision_time=decision,
            entry_not_before=decision + pd.Timedelta(minutes=1),
            direction="long",
            score=0.7,
            expected_move=0.04,
            predicted_mfe=0.05,
            predicted_mae=0.01,
            invalidation_price=2_900.0,
            max_hold_minutes=7_200,
            evidence=(evidence,),
        )


def test_target_position_is_single_net_contract_and_risk_veto_flattens() -> None:
    contribution = SleeveContribution(
        sleeve_id="swing",
        direction="long",
        raw_score=0.8,
        risk_weight=0.5,
        target_fraction=0.4,
        candidate_id="swing-1",
    )
    decision = TargetPositionDecision(
        decision_time=pd.Timestamp("2025-01-01"),
        direction="long",
        target_fraction=0.4,
        contributions=(contribution,),
    )
    assert decision.to_dict()["target_fraction"] == 0.4
    with pytest.raises(ValueError, match="risk vetoes require a flat target"):
        TargetPositionDecision(
            decision_time=pd.Timestamp("2025-01-01"),
            direction="long",
            target_fraction=0.4,
            contributions=(contribution,),
            risk_vetoes=("kill_switch",),
        )
