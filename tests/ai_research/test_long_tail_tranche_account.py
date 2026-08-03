from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_tranche_account.analysis import policy_gate
from src.ai_research.long_tail_tranche_account.config import TrancheAccountConfig, TranchePolicy
from src.ai_research.long_tail_tranche_account.simulator import select_policy_trades, simulate_account


def _structural(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    records = []
    for event_id, entry, exit_, gross in rows:
        entry_ts = pd.Timestamp(entry)
        exit_ts = pd.Timestamp(exit_)
        entry_price = 100.0
        records.append(
            {
                "event_id": event_id,
                "fold_id": "WF_2025",
                "decision_time": entry_ts - pd.Timedelta(minutes=1),
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "delay_minutes": 1,
                "signal_quantile": 0.70,
                "score": 1.0,
                "score_percentile": 0.80,
                "score_tier": "q80_to_q90",
                "entry_price": entry_price,
                "exit_price": entry_price * (1.0 + gross),
                "gross_return": gross,
                "mfe": max(gross, 0.0),
                "mae": min(gross, 0.0),
                "holding_minutes": int((exit_ts - entry_ts) / pd.Timedelta(minutes=1)),
                "exit_reason": "failed_reclaim_below_structure",
                "is_censored": False,
                "standalone_outcome": "failed_reclaim",
            }
        )
    return pd.DataFrame(records)


def _path(start: str = "2025-01-01", periods: int = 24 * 60) -> object:
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.linspace(100.0, 110.0, periods)
    frame = pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        },
        index=index,
    )
    return prepare_minute_path_frame(frame)


def test_config_has_only_pre_registered_max_two_slot_policies() -> None:
    config = TrancheAccountConfig()
    config.validate()
    assert [policy.name for policy in config.policies] == [
        "P0_single_1R",
        "P1_equal_05_05",
        "P2_primary_065_secondary_035",
        "P3_protected_065_035",
    ]
    assert all(policy.max_tranches <= 2 for policy in config.policies)
    assert all(policy.slot_a_r + policy.slot_b_r <= 1.0 for policy in config.policies)


def test_equal_timestamp_remains_occupied_for_p0() -> None:
    structural = _structural(
        [
            ("a", "2025-01-01 00:01", "2025-01-01 06:01", 0.02),
            ("b", "2025-01-01 06:01", "2025-01-01 12:01", 0.02),
            ("c", "2025-01-01 06:02", "2025-01-01 12:02", 0.02),
        ]
    )
    selection = select_policy_trades(
        structural,
        policy=TranchePolicy("P0_single_1R", 1.0, 0.0, 1),
        pair_diagnostics=pd.DataFrame(),
    )
    assert selection.accepted["event_id"].tolist() == ["a", "c"]
    assert selection.decisions.loc[selection.decisions["event_id"] == "b", "reason"].iloc[0] == "risk_slots_full"


def test_equal_slots_accept_two_and_reject_third_overlap() -> None:
    structural = _structural(
        [
            ("a", "2025-01-01 00:01", "2025-01-01 10:01", 0.02),
            ("b", "2025-01-01 02:01", "2025-01-01 12:01", 0.03),
            ("c", "2025-01-01 04:01", "2025-01-01 14:01", 0.04),
        ]
    )
    selection = select_policy_trades(
        structural,
        policy=TranchePolicy("P1_equal_05_05", 0.5, 0.5, 2),
        pair_diagnostics=pd.DataFrame(),
    )
    assert selection.accepted[["event_id", "slot", "risk_weight_r"]].to_dict("records") == [
        {"event_id": "a", "slot": "A", "risk_weight_r": 0.5},
        {"event_id": "b", "slot": "B", "risk_weight_r": 0.5},
    ]
    assert selection.decisions.loc[selection.decisions["event_id"] == "c", "reason"].iloc[0] == "risk_slots_full"


def test_protected_policy_blocks_dangerous_but_not_ambiguous_second_signal() -> None:
    structural = _structural(
        [
            ("a", "2025-01-01 00:01", "2025-01-01 10:01", 0.02),
            ("b", "2025-01-01 02:01", "2025-01-01 12:01", 0.03),
            ("c", "2025-01-01 04:01", "2025-01-01 14:01", 0.04),
        ]
    )
    dangerous = pd.DataFrame(
        [
            {
                "root_event_id": "a",
                "event_id": "b",
                "signal_class": "dangerous_average_down",
                "current_return_vs_root": -0.01,
                "score_up_price_down": True,
                "protected_policy_block": True,
            },
            {
                "root_event_id": "a",
                "event_id": "c",
                "signal_class": "ambiguous_no_add",
                "current_return_vs_root": 0.005,
                "score_up_price_down": False,
                "protected_policy_block": False,
            },
        ]
    )
    selection = select_policy_trades(
        structural,
        policy=TranchePolicy("P3_protected_065_035", 0.65, 0.35, 2, True),
        pair_diagnostics=dangerous,
    )
    assert selection.accepted["event_id"].tolist() == ["a", "c"]
    b = selection.decisions.loc[selection.decisions["event_id"] == "b"].iloc[0]
    assert b["reason"] == "dangerous_or_broken_active_structure"


def test_account_simulation_uses_risk_budget_not_notional_multiplier() -> None:
    path = _path(periods=12 * 60 + 2)
    structural = _structural(
        [
            ("a", "2025-01-01 00:01", "2025-01-01 06:01", 0.02),
            ("b", "2025-01-01 02:01", "2025-01-01 08:01", 0.03),
        ]
    )
    policy = TranchePolicy("P1_equal_05_05", 0.5, 0.5, 2)
    selection = select_policy_trades(structural, policy=policy, pair_diagnostics=pd.DataFrame())
    config = TrancheAccountConfig(
        research_start="2025-01-01",
        research_end="2025-01-01 12:01",
        minimum_tranches_per_year=1,
        minimum_monthly_tranches=0.0,
    )
    result = simulate_account(
        selection.accepted,
        path=path,
        fold_id="WF_2025",
        policy=policy,
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=pd.Timestamp("2025-01-01 00:00"),
        test_end=pd.Timestamp("2025-01-01 12:01"),
        config=config,
    )
    assert len(result.trades) == 2
    assert result.summary["max_slot_r"] <= 1.0
    assert np.allclose(result.trades["risk_weight_r"], 0.5)
    assert (result.trades["notional"] < 1.0).all()


def test_policy_gate_requires_both_years_and_delay_stress() -> None:
    config = TrancheAccountConfig(
        minimum_tranches_per_year=1,
        minimum_monthly_tranches=0.0,
        maximum_dangerous_second_add_share=1.0,
        maximum_losing_second_add_share=1.0,
    )
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                rows.append(
                    {
                        "fold_id": fold,
                        "policy": "P0_single_1R",
                        "delay_minutes": delay,
                        "cost_multiplier": cost,
                        "total_net_return": 0.10,
                    }
                )
                rows.append(
                    {
                        "fold_id": fold,
                        "policy": "P1_equal_05_05",
                        "delay_minutes": delay,
                        "cost_multiplier": cost,
                        "total_net_return": 0.20,
                        "coverage_ratio": 0.80,
                        "executed_events": 350,
                        "monthly_tranches": 29.0,
                        "max_drawdown": -0.10,
                        "positive_quarters": 4,
                        "total_return_without_top10": 0.05,
                        "max_slot_r": 1.0,
                        "dangerous_second_add_share": 0.0,
                        "losing_active_second_add_share": 0.0,
                    }
                )
    gate = policy_gate(pd.DataFrame(rows), config)
    p1 = gate.loc[gate["policy"] == "P1_equal_05_05"].iloc[0]
    assert bool(p1["pass_to_entry_stop_research"])

    broken = pd.DataFrame(rows)
    broken.loc[
        (broken["policy"] == "P1_equal_05_05")
        & (broken["fold_id"] == "WF_2024")
        & (broken["delay_minutes"] == 5)
        & (broken["cost_multiplier"] == 3.0),
        "total_net_return",
    ] = -0.01
    failed = policy_gate(broken, config)
    assert not bool(failed.loc[failed["policy"] == "P1_equal_05_05", "pass_to_entry_stop_research"].iloc[0])
