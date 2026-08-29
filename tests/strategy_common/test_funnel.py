from __future__ import annotations

import pytest

from src.strategy_common import FunnelPolicy, FunnelStage, StrategyContract, audit_funnel


def test_core_funnel_passes_when_scoring_preserves_events_and_frequency_is_healthy() -> None:
    audit = audit_funnel(
        [
            FunnelStage("events", 10_000, "source"),
            FunnelStage("scored", 10_000, "score"),
            FunnelStage("eligible", 8_000, "execution"),
            FunnelStage("trades", 1_200, "execution"),
        ],
        FunnelPolicy(strategy_class="core"),
    )
    assert audit.passed
    assert audit.hard_filter_retention == 1.0
    assert audit.executed_trades == 1_200


def test_core_funnel_fails_when_hard_filters_collapse_events() -> None:
    audit = audit_funnel(
        [
            FunnelStage("events", 20_000, "source"),
            FunnelStage("mss", 6_000, "hard_filter"),
            FunnelStage("fvg", 800, "hard_filter"),
            FunnelStage("session", 140, "hard_filter"),
            FunnelStage("trades", 60, "execution"),
        ],
        FunnelPolicy(strategy_class="core"),
    )
    codes = {issue.code for issue in audit.issues}
    assert not audit.passed
    assert "SEVERE_HARD_FILTER_COLLAPSE" in codes
    assert "TOTAL_HARD_FILTER_COLLAPSE" in codes
    assert "TOO_FEW_EXECUTED_TRADES" in codes


def test_funnel_rejects_increasing_counts() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        audit_funnel(
            [
                FunnelStage("events", 100, "source"),
                FunnelStage("trades", 101, "execution"),
            ]
        )


def test_strategy_contract_requires_real_exit_and_stop_rules() -> None:
    contract = StrategyContract(
        strategy_id="TEST",
        name="Test",
        sleeve_id="TEST_SLEEVE",
        family="test",
        strategy_class="core",
        stage="in_development",
        symbol="ETH-USDT-SWAP",
        signal_timeframe="15m",
        execution_timeframe="15m",
        setup="setup",
        trigger="trigger",
        entry="entry",
        initial_stop="",
        exit_logic="exit",
        sizing="sizing",
        invalidation="invalid",
        target_min_trades=300,
    )
    with pytest.raises(ValueError, match="initial_stop"):
        contract.validate()
