from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ai_research.long_tail_final_account_audit.analysis import (
    build_continuous_scenarios,
    build_gate,
    build_lot_size_audit,
    build_model_governance,
    build_risk_reserve_audit,
)
from src.ai_research.long_tail_final_account_audit.config import FinalAccountAuditConfig
from src.ai_research.long_tail_final_account_audit.inputs import load_final_audit_inputs


def _cycles() -> pd.DataFrame:
    rows = []
    for fold, year in (("WF_2024", 2024), ("WF_2025", 2025)):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                for index, value in enumerate((0.02, -0.005, 0.01, -0.004), start=1):
                    entry = pd.Timestamp(year=year, month=index, day=1, hour=1)
                    rows.append(
                        {
                            "event_id": f"{fold}-{delay}-{cost}-{index}",
                            "fold_id": fold,
                            "policy": "E0_immediate_C2",
                            "delay_minutes": delay,
                            "cost_multiplier": cost,
                            "entry_time": entry,
                            "exit_time": entry + pd.Timedelta(hours=10),
                            "cycle_return": value - 0.0005 * (cost - 2.0),
                            "hard_stop_exit": value < 0,
                            "soft_failure_exit": False,
                            "exit_reason": "real_hard_stop_intrabar" if value < 0 else "failed_reclaim_below_structure",
                        }
                    )
    return pd.DataFrame(rows)


def _daily() -> pd.DataFrame:
    rows = []
    for fold, year in (("WF_2024", 2024), ("WF_2025", 2025)):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                equity = 1.0
                for month, value in enumerate((1.02, 0.995, 1.01, 0.996), start=1):
                    equity *= value
                    rows.append(
                        {
                            "date": pd.Timestamp(year=year, month=month, day=28),
                            "equity": equity,
                            "fold_id": fold,
                            "policy": "E0_immediate_C2",
                            "delay_minutes": delay,
                            "cost_multiplier": cost,
                        }
                    )
    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                rows.append(
                    {
                        "fold_id": fold,
                        "policy": "E0_immediate_C2",
                        "delay_minutes": delay,
                        "cost_multiplier": cost,
                        "final_equity": 1.0207,
                        "max_drawdown": -0.02,
                    }
                )
    return pd.DataFrame(rows)


def test_config_freezes_strategy_and_holdout() -> None:
    config = FinalAccountAuditConfig()
    config.validate()
    assert config.hard_stop_distance == 0.02
    assert config.soft_failure_distance == 0.015
    with pytest.raises(ValueError):
        FinalAccountAuditConfig(oos_end="2026-01-01").validate()


def test_continuous_account_does_not_reset_between_years() -> None:
    config = FinalAccountAuditConfig()
    cycles, daily, scenarios, months, quarters = build_continuous_scenarios(_cycles(), _daily(), _summary(), config)
    anchor = scenarios.loc[(scenarios.delay_minutes == 1) & (scenarios.cost_multiplier == 2.0)].iloc[0]
    one_year = (1.02 * 0.995 * 1.01 * 0.996)
    assert anchor.final_equity == pytest.approx(one_year * one_year, rel=1e-6)
    assert len(cycles) == 48
    assert not daily.empty and not months.empty and not quarters.empty


def test_lot_sizing_floors_contracts_and_never_exceeds_target_risk() -> None:
    config = FinalAccountAuditConfig(initial_equity_tiers=(1000.0,))
    legs = pd.DataFrame(
        {
            "policy": ["E0_immediate_C2", "E0_immediate_C2"],
            "delay_minutes": [1, 1],
            "cost_multiplier": [2.0, 2.0],
            "entry_price": [2000.0, 4000.0],
        }
    )
    audit = build_lot_size_audit(legs, config, live_price_risk_fraction=0.0084).iloc[0]
    assert audit.maximum_actual_price_risk_fraction <= 0.01 + 1e-12
    assert audit.untradable_share == 0.0


def test_risk_reserve_caps_price_budget_below_one_percent() -> None:
    config = FinalAccountAuditConfig()
    scenarios = pd.DataFrame([{"delay_minutes": 1, "cost_multiplier": 2.0, "worst_net_r": 1.125}])
    audit = build_risk_reserve_audit(scenarios, config).iloc[0]
    assert audit.maximum_price_risk_budget_for_1pct_net_tail == pytest.approx(0.01 / 1.125)
    assert audit.recommended_live_price_risk_budget <= 0.009


def test_model_governance_separates_retrain_from_promotion() -> None:
    governance = build_model_governance()
    monthly = governance.loc[governance.cadence.astype(str).str.startswith("monthly")]
    assert not monthly.empty
    assert monthly.promotion.astype(str).str.contains("forbidden|none").any()


def test_gate_requires_all_final_readiness_checks() -> None:
    config = FinalAccountAuditConfig(
        minimum_anchor_total_return=0.5,
        minimum_positive_months=1,
        minimum_positive_quarters=1,
    )
    scenarios = pd.DataFrame(
        [
            {
                "delay_minutes": delay,
                "cost_multiplier": cost,
                "total_return": 1.0,
                "max_drawdown": -0.08,
                "profit_factor": 1.7,
                "positive_months": 18,
                "positive_quarters": 8,
                "longest_losing_streak": 7,
                "max_drawdown_duration_days": 70,
                "total_return_without_top10": 0.2,
                "top10_profit_share": 0.3,
                "worst_net_r": 1.1,
            }
            for delay in (1, 3, 5)
            for cost in (2.0, 3.0)
        ]
    )
    risk = build_risk_reserve_audit(scenarios, config)
    gate = build_gate(scenarios, risk, config)
    assert bool(gate["pass"].all())


def test_loader_requires_passed_r214(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "r214"
    root.mkdir()
    (root / "99_decision.md").write_text("BLOCKED_DATA", encoding="utf-8")
    config = FinalAccountAuditConfig()
    monkeypatch.setattr(type(config), "source_2_14_path", property(lambda self: root))
    with pytest.raises(RuntimeError, match="must pass"):
        load_final_audit_inputs(config)
