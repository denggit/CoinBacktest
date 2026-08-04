from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_score_risk_sizing.analysis import (
    build_cross_year_order_audit,
    build_tier_attribution,
    policy_gate,
)
from src.ai_research.long_tail_score_risk_sizing.config import ScoreRiskConfig, ScoreRiskPolicy
from src.ai_research.long_tail_score_risk_sizing.inputs import load_score_risk_inputs
from src.ai_research.long_tail_score_risk_sizing.simulator import simulate_score_risk_account


def _path() -> object:
    index = pd.date_range("2024-01-01", periods=20, freq="min")
    close = np.linspace(100.0, 102.0, len(index))
    frame = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close}, index=index)
    return prepare_minute_path_frame(frame)


def _source() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = pd.DataFrame([
        {"event_id": "a", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "decision_time": pd.Timestamp("2024-01-01 00:00"), "score": 0.1, "score_percentile": 0.75, "score_tier": "q70_to_q80", "hard_stop_distance": 0.02, "hard_stop_exit": False, "soft_failure_exit": False},
        {"event_id": "b", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "decision_time": pd.Timestamp("2024-01-01 00:05"), "score": 0.2, "score_percentile": 0.85, "score_tier": "q80_to_q90", "hard_stop_distance": 0.02, "hard_stop_exit": False, "soft_failure_exit": True},
        {"event_id": "c", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "decision_time": pd.Timestamp("2024-01-01 00:10"), "score": 0.3, "score_percentile": 0.95, "score_tier": "q90_plus", "hard_stop_distance": 0.02, "hard_stop_exit": False, "soft_failure_exit": False},
    ])
    legs = pd.DataFrame([
        {"event_id": "a", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "entry_time": pd.Timestamp("2024-01-01 00:01"), "exit_time": pd.Timestamp("2024-01-01 00:04"), "entry_price": 100.1, "exit_price": 100.4, "exit_reason": "failed_reclaim"},
        {"event_id": "b", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "entry_time": pd.Timestamp("2024-01-01 00:06"), "exit_time": pd.Timestamp("2024-01-01 00:09"), "entry_price": 100.6, "exit_price": 100.9, "exit_reason": "soft_failure"},
        {"event_id": "c", "fold_id": "WF_2024", "delay_minutes": 1, "cost_multiplier": 2.0, "entry_time": pd.Timestamp("2024-01-01 00:11"), "exit_time": pd.Timestamp("2024-01-01 00:14"), "entry_price": 101.1, "exit_price": 101.4, "exit_reason": "failed_reclaim"},
    ])
    return cycles, legs


def test_config_rejects_qualifying_policy_above_one_r() -> None:
    policy = ScoreRiskPolicy("bad", 1.0, 1.0, 1.1)
    with pytest.raises(ValueError):
        policy.validate()


def test_simulator_preserves_all_cycles_and_applies_tier_risk() -> None:
    cycles, legs = _source()
    policy = ScoreRiskPolicy("tiered", 0.75, 0.9, 1.0)
    config = ScoreRiskConfig(policies=(ScoreRiskPolicy("E100_equal_1R", 1, 1, 1, qualifying_candidate=False), policy))
    result = simulate_score_risk_account(cycles, legs, path=_path(), fold_id="WF_2024", policy=policy, delay_minutes=1, cost_multiplier=2.0, config=config, progress=False)
    assert len(result.cycles) == 3
    assert result.cycles["risk_multiplier"].tolist() == [0.75, 0.9, 1.0]
    assert result.cycles["base_notional_to_equity"].iloc[2] > result.cycles["base_notional_to_equity"].iloc[0]
    assert result.summary["max_risk_multiplier"] == 1.0


def test_equal_risk_uses_half_notional_for_two_percent_stop() -> None:
    cycles, legs = _source()
    policy = ScoreRiskPolicy("E100_equal_1R", 1, 1, 1, qualifying_candidate=False)
    config = ScoreRiskConfig(policies=(policy,))
    result = simulate_score_risk_account(cycles.iloc[:1], legs.iloc[:1], path=_path(), fold_id="WF_2024", policy=policy, delay_minutes=1, cost_multiplier=2.0, config=config, progress=False)
    assert result.cycles["base_notional_to_equity"].iloc[0] == pytest.approx(0.5)


def test_tier_attribution_and_order_audit_detect_non_monotonic_year() -> None:
    rows = []
    for fold, means in {"WF_2024": [0.03, 0.02, 0.01], "WF_2025": [0.01, 0.02, 0.03]}.items():
        for tier, value in zip(("q70_to_q80", "q80_to_q90", "q90_plus"), means):
            rows.append({"fold_id": fold, "delay_minutes": 1, "cost_multiplier": 2.0, "score_tier": tier, "cycle_return": value, "hard_stop_exit": False, "soft_failure_exit": False})
    attribution = build_tier_attribution(pd.DataFrame(rows))
    audit = build_cross_year_order_audit(attribution)
    assert audit.set_index("fold_id").loc["WF_2024", "monotonic_score_order"] == False  # noqa: E712
    assert audit.set_index("fold_id").loc["WF_2025", "monotonic_score_order"] == True  # noqa: E712


def test_policy_gate_requires_return_retention_and_calmar_improvement() -> None:
    config = ScoreRiskConfig(policies=(
        ScoreRiskPolicy("E100_equal_1R", 1, 1, 1, qualifying_candidate=False),
        ScoreRiskPolicy("candidate", 0.75, 0.9, 1.0),
    ))
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                rows.append({"fold_id": fold, "policy": "E100_equal_1R", "delay_minutes": delay, "cost_multiplier": cost, "total_net_return": 1.0, "max_drawdown": -0.1, "top10_profit_share": 0.3, "total_return_without_top10": 0.3, "positive_quarters": 4, "max_risk_multiplier": 1.0})
                rows.append({"fold_id": fold, "policy": "candidate", "delay_minutes": delay, "cost_multiplier": cost, "total_net_return": 0.96, "max_drawdown": -0.08, "top10_profit_share": 0.31, "total_return_without_top10": 0.25, "positive_quarters": 4, "max_risk_multiplier": 1.0})
    gate = policy_gate(pd.DataFrame(rows), config)
    assert bool(gate.set_index("policy").loc["candidate", "pass_to_next_stage"])


def test_loader_requires_passed_2_12_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "00_run_manifest.json").write_text(json.dumps({"stage": "R03.4.2.12"}), encoding="utf-8")
    (root / "99_decision.md").write_text("FAIL", encoding="utf-8")
    for name in ("04_selected_p0_cycles.csv", "07_account_cycles.csv", "08_account_legs.csv", "11_policy_summary.csv", "12_policy_gate.csv", "13_causal_audit.csv", "15_failures.csv"):
        pd.DataFrame().to_csv(root / name, index=False)
    config = ScoreRiskConfig(source_2_12_report_dir="data/reports/research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression")
    monkeypatch.setattr(type(config), "source_path", property(lambda self: root))
    with pytest.raises(RuntimeError, match="passed C2"):
        load_score_risk_inputs(config)
