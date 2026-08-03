from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_dynamic_risk_release.analysis import policy_gate
from src.ai_research.long_tail_dynamic_risk_release.config import (
    DynamicReleasePolicy,
    DynamicRiskReleaseConfig,
    ProtectionPolicy,
)
from src.ai_research.long_tail_dynamic_risk_release.protection import (
    candidate_stop_from_levels,
    stop_fill_price,
)
from src.ai_research.long_tail_dynamic_risk_release.simulator import (
    select_dynamic_trades,
    simulate_dynamic_account,
)
from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame


def _trades() -> pd.DataFrame:
    rows = []
    for event_id, entry, exit_, gross in (
        ("a", "2025-01-01 00:01", "2025-01-01 10:01", 0.03),
        ("b", "2025-01-01 02:01", "2025-01-01 12:01", 0.02),
        ("c", "2025-01-01 04:01", "2025-01-01 14:01", 0.01),
    ):
        entry_ts = pd.Timestamp(entry)
        rows.append(
            {
                "event_id": event_id,
                "fold_id": "WF_2025",
                "decision_time": entry_ts - pd.Timedelta(minutes=1),
                "entry_time": entry_ts,
                "exit_time": pd.Timestamp(exit_),
                "delay_minutes": 1,
                "score": 1.0,
                "signal_quantile": 0.70,
                "score_percentile": 0.80,
                "score_tier": "q80_to_q90",
                "entry_price": 100.0,
                "exit_price": 100.0 * (1.0 + gross),
                "gross_return": gross,
                "mfe": max(gross, 0.0),
                "mae": min(gross, 0.0),
                "holding_minutes": int((pd.Timestamp(exit_) - entry_ts) / pd.Timedelta(minutes=1)),
                "exit_reason": "failed_reclaim_below_structure",
                "is_censored": False,
                "protection_policy": "S2_lagged_confirmed",
                "initial_stop_price": 97.0,
            }
        )
    return pd.DataFrame(rows)


def _path(periods: int = 16 * 60) -> object:
    index = pd.date_range("2025-01-01", periods=periods, freq="1min")
    close = np.linspace(100.0, 104.0, periods)
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


def test_config_freezes_full_primary_and_maximum_two_tranches() -> None:
    config = DynamicRiskReleaseConfig()
    config.validate()
    assert config.maximum_virtual_tranches == 2
    assert config.dynamic_policies[0].name == "D0_single_1R"
    assert config.dynamic_policies[0].max_secondary_r == 0.0
    assert all(policy.max_secondary_r <= 1.0 for policy in config.dynamic_policies)


def test_latest_and_lagged_structure_use_different_confirmed_levels() -> None:
    levels = [95.0, 98.0, 101.0]
    latest = candidate_stop_from_levels(
        policy=ProtectionPolicy("latest", "latest_confirmed"),
        disaster_stop=97.0,
        structural_levels=levels,
        buffer_value=0.5,
    )
    lagged = candidate_stop_from_levels(
        policy=ProtectionPolicy("lagged", "lagged_confirmed"),
        disaster_stop=97.0,
        structural_levels=levels,
        buffer_value=0.5,
    )
    assert latest == 100.5
    assert lagged == 97.5


def test_lagged_stop_does_not_activate_without_a_new_confirmed_floor() -> None:
    stop = candidate_stop_from_levels(
        policy=ProtectionPolicy("lagged", "lagged_confirmed"),
        disaster_stop=97.0,
        structural_levels=[96.0],
        buffer_value=0.5,
    )
    assert stop == 97.0


def test_stop_fill_is_conservative_on_gap() -> None:
    assert stop_fill_price(open_price=96.0, low_price=95.0, stop_price=97.0) == 96.0
    assert stop_fill_price(open_price=98.0, low_price=96.5, stop_price=97.0) == 97.0
    assert stop_fill_price(open_price=98.0, low_price=97.5, stop_price=97.0) is None


def test_primary_keeps_full_r_and_secondary_uses_only_released_risk() -> None:
    trades = _trades()
    pairs = pd.DataFrame(
        [
            {
                "root_event_id": "a",
                "event_id": "b",
                "state": "HEALTHY",
                "pending_failed_reclaim_exit": False,
                "current_return": 0.01,
                "stop_price": 98.5,
                "released_risk_fraction": 0.50,
                "score_up_price_down": False,
            }
        ]
    )
    policy = DynamicReleasePolicy("release", 0.35, 0.20)
    selection = select_dynamic_trades(
        trades,
        protection_policy="S2_lagged_confirmed",
        dynamic_policy=policy,
        pair_diagnostics=pairs,
    )
    accepted = selection.accepted[["event_id", "risk_weight_r", "entry_role"]].to_dict("records")
    assert accepted == [
        {"event_id": "a", "risk_weight_r": 1.0, "entry_role": "primary"},
        {"event_id": "b", "risk_weight_r": 0.35, "entry_role": "secondary"},
    ]
    assert selection.decisions.loc[selection.decisions["event_id"] == "c", "reason"].iloc[0] == "maximum_two_tranches"


def test_no_enforceable_release_means_no_secondary() -> None:
    trades = _trades().iloc[:2].copy()
    pairs = pd.DataFrame(
        [
            {
                "root_event_id": "a",
                "event_id": "b",
                "state": "HEALTHY",
                "pending_failed_reclaim_exit": False,
                "current_return": 0.01,
                "stop_price": 97.3,
                "released_risk_fraction": 0.10,
                "score_up_price_down": False,
            }
        ]
    )
    selection = select_dynamic_trades(
        trades,
        protection_policy="S2_lagged_confirmed",
        dynamic_policy=DynamicReleasePolicy("release", 0.35, 0.20),
        pair_diagnostics=pairs,
    )
    assert selection.accepted["event_id"].tolist() == ["a"]
    assert selection.decisions.loc[selection.decisions["event_id"] == "b", "reason"].iloc[0] == "insufficient_enforceable_risk_release"


def test_non_losing_policy_blocks_underwater_add() -> None:
    trades = _trades().iloc[:2].copy()
    pairs = pd.DataFrame(
        [
            {
                "root_event_id": "a",
                "event_id": "b",
                "state": "HEALTHY",
                "pending_failed_reclaim_exit": False,
                "current_return": -0.005,
                "stop_price": 98.5,
                "released_risk_fraction": 0.50,
                "score_up_price_down": True,
            }
        ]
    )
    selection = select_dynamic_trades(
        trades,
        protection_policy="S2_lagged_confirmed",
        dynamic_policy=DynamicReleasePolicy("protected", 0.50, 0.20, require_non_losing_active=True),
        pair_diagnostics=pairs,
    )
    assert selection.accepted["event_id"].tolist() == ["a"]
    assert selection.decisions.loc[selection.decisions["event_id"] == "b", "reason"].iloc[0] == "active_position_still_losing"


def test_account_caps_live_remaining_r_while_initial_r_sum_can_exceed_one() -> None:
    path = _path()
    trades = _trades().iloc[:2].copy()
    trades["dynamic_policy"] = "release"
    trades["risk_weight_r"] = [1.0, 0.35]
    trades["entry_role"] = ["primary", "secondary"]
    trades["active_root_event_id"] = ["", "a"]
    trades["released_account_r_at_entry"] = [0.0, 0.5]
    trades["active_remaining_r_at_entry"] = [0.0, 0.5]
    trades["active_current_return"] = [np.nan, 0.01]
    trades["active_state"] = ["", "HEALTHY"]
    trades["active_pending_failed_reclaim"] = [False, False]
    updates = pd.DataFrame(
        [
            {
                "event_id": "a",
                "delay_minutes": 1,
                "protection_policy": "S2_lagged_confirmed",
                "effective_time": pd.Timestamp("2025-01-01 01:01"),
                "stop_price": 98.5,
            },
            {
                "event_id": "b",
                "delay_minutes": 1,
                "protection_policy": "S2_lagged_confirmed",
                "effective_time": pd.Timestamp("2025-01-01 02:01"),
                "stop_price": 97.0,
            },
        ]
    )
    config = DynamicRiskReleaseConfig(
        research_start="2025-01-01",
        research_end="2025-01-01 15:59",
        minimum_monthly_tranches=0.0,
    )
    result = simulate_dynamic_account(
        trades,
        stop_updates=updates,
        path=path,
        fold_id="WF_2025",
        protection_policy="S2_lagged_confirmed",
        dynamic_policy=DynamicReleasePolicy("release", 0.35, 0.20),
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=pd.Timestamp("2025-01-01 00:00"),
        test_end=pd.Timestamp("2025-01-01 15:59"),
        config=config,
        progress=False,
    )
    assert result.summary["max_live_remaining_r"] <= 1.02
    assert result.summary["max_initial_r_sum"] > 1.0


def test_gate_requires_cross_year_return_not_just_lower_drawdown() -> None:
    config = DynamicRiskReleaseConfig(minimum_monthly_tranches=0.0, minimum_coverage_ratio=0.0)
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                rows.append(
                    {
                        "fold_id": fold,
                        "protection_policy": "S0_disaster_only",
                        "dynamic_policy": "D0_single_1R",
                        "delay_minutes": delay,
                        "cost_multiplier": cost,
                        "total_net_return": 0.50,
                        "max_drawdown": -0.08,
                        "coverage_ratio": 0.55,
                        "monthly_tranches": 20.0,
                        "positive_quarters": 4,
                        "total_return_without_top10": 0.10,
                        "max_live_remaining_r": 1.0,
                        "losing_second_add_share": 0.0,
                        "broken_second_add_share": 0.0,
                    }
                )
                rows.append(
                    {
                        "fold_id": fold,
                        "protection_policy": "S2_lagged_confirmed",
                        "dynamic_policy": "D1_release_cap035",
                        "delay_minutes": delay,
                        "cost_multiplier": cost,
                        "total_net_return": 0.40,
                        "max_drawdown": -0.04,
                        "coverage_ratio": 0.80,
                        "monthly_tranches": 30.0,
                        "positive_quarters": 4,
                        "total_return_without_top10": 0.10,
                        "max_live_remaining_r": 1.0,
                        "losing_second_add_share": 0.0,
                        "broken_second_add_share": 0.0,
                    }
                )
    protection = pd.DataFrame(
        [
            {"fold_id": fold, "protection_policy": "S0_disaster_only", "delay_minutes": delay, "hard_stop_share": 0.0}
            for fold in ("WF_2024", "WF_2025") for delay in (1, 3, 5)
        ]
        + [
            {"fold_id": fold, "protection_policy": "S2_lagged_confirmed", "delay_minutes": delay, "hard_stop_share": 0.2}
            for fold in ("WF_2024", "WF_2025") for delay in (1, 3, 5)
        ]
    )
    gate = policy_gate(pd.DataFrame(rows), protection, config)
    candidate = gate.loc[gate["dynamic_policy"] == "D1_release_cap035"].iloc[0]
    assert not bool(candidate["cross_year_total_improvement"])
    assert not bool(candidate["pass_to_next_stage"])
