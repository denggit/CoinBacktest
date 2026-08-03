from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_risk_migration.analysis import policy_gate
from src.ai_research.long_tail_risk_migration.config import (
    MigrationPolicy,
    RiskMigrationConfig,
)
from src.ai_research.long_tail_risk_migration.simulator import simulate_risk_migration_account
from src.ai_research.long_tail_risk_migration.structure import build_candidate_pair_snapshots


def _path(periods: int = 240, *, drift: float = 0.01):
    index = pd.date_range("2024-01-01", periods=periods, freq="min")
    close = 100.0 + np.arange(periods) * drift
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
        },
        index=index,
    )
    return prepare_minute_path_frame(frame)


def _event(event_id: str, entry_minute: int, exit_minute: int, *, score: float = 0.8) -> dict[str, object]:
    entry_time = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=entry_minute)
    exit_time = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=exit_minute)
    entry_price = 100.0 + entry_minute * 0.01
    exit_price = 100.0 + exit_minute * 0.01
    return {
        "event_id": event_id,
        "fold_id": "WF_2024",
        "decision_time": entry_time - pd.Timedelta(minutes=1),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "delay_minutes": 1,
        "signal_quantile": 0.70,
        "score": score,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": "failed_reclaim",
        "is_censored": False,
    }


def _timeline(event_id: str, effective_minute: int, *, state: str, broken: bool = False, proven: bool = True, current_return: float = 0.01) -> dict[str, object]:
    effective = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=effective_minute)
    return {
        "event_id": event_id,
        "fold_id": "WF_2024",
        "delay_minutes": 1,
        "structure_close_time": effective - pd.Timedelta(minutes=1),
        "effective_time": effective,
        "state": state,
        "entered_broken_this_bar": broken,
        "pending_failed_reclaim_exit": False,
        "current_return": current_return,
        "proven_structure": proven,
        "floor_raised": proven,
        "higher_low_confirmed": proven,
        "recoveries": 0,
    }


def _run(policy: MigrationPolicy, events: list[dict[str, object]], timelines: list[dict[str, object]]):
    config = RiskMigrationConfig(
        policies=(policy,),
        minimum_return_retention_each_year=0.0,
        minimum_combined_return_ratio=0.0,
        minimum_positive_quarters_per_year=0,
        minimum_coverage_ratio_for_migration=0.0,
        minimum_monthly_tranches_for_migration=0.0,
    )
    return simulate_risk_migration_account(
        pd.DataFrame(events),
        pd.DataFrame(timelines),
        path=_path(),
        fold_id="WF_2024",
        policy=policy,
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-01-01 03:59"),
        config=config,
        progress=False,
    )


def test_config_keeps_frozen_boundaries() -> None:
    config = RiskMigrationConfig()
    config.validate()
    assert config.disaster_stop_distance == 0.03
    assert config.maximum_virtual_tranches == 2
    assert config.maximum_cycle_r == 1.0


def test_p0_skips_overlapping_event() -> None:
    result = _run(
        MigrationPolicy("P0_single_1R"),
        [_event("a", 10, 120), _event("b", 40, 80)],
        [_timeline("a", 30, state="HEALTHY")],
    )
    assert result.trades["event_id"].tolist() == ["a"]
    assert result.summary["coverage_ratio"] == 0.5


def test_soft_break_reduction_is_real_partial_close() -> None:
    result = _run(
        MigrationPolicy("R1", partial_reduce_fraction=0.25),
        [_event("a", 10, 120)],
        [_timeline("a", 30, state="BROKEN", broken=True, proven=True, current_return=0.01)],
    )
    assert result.summary["partial_reduce_actions"] == 1
    assert set(result.legs["leg_type"]) == {"partial_reduce", "final_exit"}
    trade = result.trades.iloc[0]
    assert trade["partial_leg_count"] == 1


def test_soft_break_reduction_does_not_fire_while_losing() -> None:
    path = _path(drift=-0.01)
    policy = MigrationPolicy("R1", partial_reduce_fraction=0.25)
    config = RiskMigrationConfig(policies=(policy,))
    result = simulate_risk_migration_account(
        pd.DataFrame([_event("a", 10, 120)]),
        pd.DataFrame([_timeline("a", 30, state="BROKEN", broken=True, proven=True, current_return=-0.01)]),
        path=path,
        fold_id="WF_2024",
        policy=policy,
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-01-01 03:59"),
        config=config,
        progress=False,
    )
    assert result.summary["partial_reduce_actions"] == 0


def test_signal_migration_reduces_old_before_opening_new() -> None:
    result = _run(
        MigrationPolicy("M1", migration_target_r=0.35, allow_migration=True),
        [_event("a", 10, 150), _event("b", 40, 90)],
        [_timeline("a", 30, state="HEALTHY", current_return=0.01)],
    )
    assert result.trades["event_id"].tolist() == ["a", "b"]
    assert result.summary["secondary_tranches"] == 1
    assert result.summary["migration_release_actions"] == 1
    assert result.summary["max_cycle_allocated_r"] <= 1.0 + 1e-9
    assert "migration_release" in set(result.legs["leg_type"])


def test_signal_migration_blocked_for_broken_root() -> None:
    result = _run(
        MigrationPolicy("M1", migration_target_r=0.35, allow_migration=True),
        [_event("a", 10, 150), _event("b", 40, 90)],
        [_timeline("a", 30, state="BROKEN", broken=True, current_return=0.01)],
    )
    assert result.trades["event_id"].tolist() == ["a"]
    assert (result.decisions["reason"] == "root_structure_not_healthy").any()


def test_pair_snapshot_never_uses_future_structure() -> None:
    structural = pd.DataFrame([_event("a", 10, 150), _event("b", 40, 90)])
    timelines = pd.DataFrame(
        [
            {
                **_timeline("a", 30, state="HEALTHY"),
                "structure_close_time": pd.Timestamp("2024-01-01 00:29"),
                "effective_time": pd.Timestamp("2024-01-01 00:30"),
            },
            {
                **_timeline("a", 50, state="BROKEN"),
                "structure_close_time": pd.Timestamp("2024-01-01 00:49"),
                "effective_time": pd.Timestamp("2024-01-01 00:50"),
            },
        ]
    )
    pairs = build_candidate_pair_snapshots(structural, timelines, delay_minutes=1)
    row = pairs.loc[pairs["event_id"] == "b"].iloc[0]
    assert row["state"] == "HEALTHY"
    assert row["snapshot_structure_close_time"] <= pd.Timestamp("2024-01-01 00:39")


def test_policy_gate_requires_cross_year_consistency() -> None:
    rows = []
    for fold, p0_ret, candidate_ret in [("WF_2024", 0.50, 0.49), ("WF_2025", 0.60, 0.40)]:
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                for policy, ret in [("P0_single_1R", p0_ret), ("M1_signal_migrate035", candidate_ret)]:
                    rows.append(
                        {
                            "fold_id": fold,
                            "policy": policy,
                            "delay_minutes": delay,
                            "cost_multiplier": cost,
                            "total_net_return": ret,
                            "max_drawdown": -0.05,
                            "total_return_without_top10": 0.10,
                            "positive_quarters": 4,
                            "max_cycle_allocated_r": 1.0,
                            "coverage_ratio": 0.80,
                            "monthly_tranches": 30.0,
                            "losing_migration_share": 0.0,
                            "broken_migration_share": 0.0,
                        }
                    )
    config = RiskMigrationConfig(
        policies=(
            MigrationPolicy("P0_single_1R"),
            MigrationPolicy("M1_signal_migrate035", migration_target_r=0.35, allow_migration=True),
        )
    )
    gate = policy_gate(pd.DataFrame(rows), config)
    candidate = gate.loc[gate["policy"] == "M1_signal_migrate035"].iloc[0]
    assert not bool(candidate["pass_to_next_stage"])


def test_p0_matches_frozen_dual_slot_simulator() -> None:
    from src.ai_research.long_tail_tranche_account.config import TrancheAccountConfig, TranchePolicy
    from src.ai_research.long_tail_tranche_account.simulator import select_policy_trades, simulate_account

    events = pd.DataFrame([_event("a", 10, 120), _event("b", 40, 80), _event("c", 130, 180)])
    old_config = TrancheAccountConfig(
        policies=(TranchePolicy("P0_single_1R", 1.0, 0.0, 1),),
        minimum_tranches_per_year=0,
        minimum_monthly_tranches=0.0,
    )
    old_policy = old_config.policies[0]
    selection = select_policy_trades(events, policy=old_policy, pair_diagnostics=pd.DataFrame())
    old = simulate_account(
        selection.accepted,
        path=_path(),
        fold_id="WF_2024",
        policy=old_policy,
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-01-01 03:59"),
        config=old_config,
        progress=False,
    )
    new = _run(MigrationPolicy("P0_single_1R"), events.to_dict("records"), [])
    assert new.trades["event_id"].tolist() == old.trades["event_id"].tolist()
    assert np.isclose(new.summary["total_net_return"], old.summary["total_net_return"], atol=1e-12)
    assert np.isclose(new.summary["max_drawdown"], old.summary["max_drawdown"], atol=1e-12)
