from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_staged_execution.analysis import enrich_summaries, policy_gate
from src.ai_research.long_tail_staged_execution.config import (
    StagedExecutionConfig,
    StagedExecutionPolicy,
)
from src.ai_research.long_tail_staged_execution.simulator import simulate_staged_execution_account


def _path(close: np.ndarray):
    index = pd.date_range("2024-01-01", periods=len(close), freq="min")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
        },
        index=index,
    )
    return prepare_minute_path_frame(frame)


def _event(path, *, entry_position: int = 80, exit_position: int = 220) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "e1",
                "fold_id": "WF_2024",
                "decision_time": path.index[entry_position - 1],
                "entry_time": path.index[entry_position],
                "exit_time": path.index[exit_position],
                "delay_minutes": 1,
                "signal_quantile": 0.70,
                "score": 0.8,
                "entry_price": float(path.open[entry_position]),
                "exit_price": float(path.open[exit_position]),
                "exit_reason": "failed_reclaim_below_structure",
                "is_censored": False,
            }
        ]
    )


def _timeline(path, states: list[tuple[int, str, float]]) -> pd.DataFrame:
    rows = []
    for position, state, current_close in states:
        rows.append(
            {
                "event_id": "e1",
                "fold_id": "WF_2024",
                "delay_minutes": 1,
                "effective_time": path.index[position],
                "structure_close_time": path.index[position - 1],
                "state": state,
                "pending_failed_reclaim_exit": False,
                "current_close": current_close,
            }
        )
    return pd.DataFrame(rows)


def _run(policy: StagedExecutionPolicy, path, timeline: pd.DataFrame | None = None):
    config = StagedExecutionConfig(
        policies=(policy,),
        minimum_return_retention_each_year=0.0,
        minimum_combined_return_ratio=0.0,
        minimum_positive_quarters_per_year=0,
        maximum_winner_to_loser_share=1.0,
        maximum_addon_loss_share_of_base_profit=10.0,
    )
    return simulate_staged_execution_account(
        _event(path),
        timeline if timeline is not None else pd.DataFrame(),
        path=path,
        fold_id="WF_2024",
        policy=policy,
        delay_minutes=1,
        cost_multiplier=2.0,
        test_start=path.index[0],
        test_end=path.index[-1],
        config=config,
        progress=False,
    )


def test_config_declares_one_point_five_notional_and_two_r_tail_caps() -> None:
    config = StagedExecutionConfig()
    config.validate()
    assert config.maximum_notional_to_equity == 1.5
    assert config.maximum_account_tail_r == 2.0
    assert config.maximum_virtual_tranches == 3
    assert max(policy.declared_hard_tail_r for policy in config.policies) <= 2.0


def test_p0_uses_one_r_with_three_percent_base_stop() -> None:
    close = 100 + np.linspace(0, 3, 300)
    result = _run(StagedExecutionPolicy("P0", "baseline"), _path(close))
    cycle = result.cycles.iloc[0]
    assert cycle["add_count"] == 0
    assert np.isclose(cycle["max_hard_tail_r"], 1.0, atol=1e-9)
    assert 0.30 < cycle["base_notional_to_equity"] < 0.35


def test_soft_failure_sizes_larger_but_keeps_three_percent_tail() -> None:
    close = np.full(300, 100.0)
    close[80:] = np.linspace(100.0, 97.0, 220)
    path = _path(close)
    policy = StagedExecutionPolicy(
        "F1",
        "soft_failure",
        base_sizing_stop_distance=0.015,
        soft_failure_distance=0.015,
        max_cycle_hard_r=2.0,
    )
    timeline = _timeline(path, [(150, "HEALTHY", 98.4)])
    result = _run(policy, path, timeline)
    cycle = result.cycles.iloc[0]
    assert cycle["soft_failure_exit"]
    assert 0.63 < cycle["base_notional_to_equity"] < 0.70
    assert np.isclose(cycle["max_hard_tail_r"], 2.0, atol=1e-9)


def test_staged_dual_path_adds_without_exceeding_one_r() -> None:
    close = 100 + np.linspace(0, 4, 300)
    path = _path(close)
    policy = StagedExecutionPolicy(
        "S1",
        "staged_dual_path",
        base_r=0.60,
        add_r=(0.40,),
        trigger_n=(1.0,),
        max_cycle_hard_r=1.0,
        require_profit_cover=False,
    )
    timeline = _timeline(path, [(100, "HEALTHY", float(path.close[99]))])
    result = _run(policy, path, timeline)
    cycle = result.cycles.iloc[0]
    assert cycle["add_count"] == 1
    assert cycle["max_hard_tail_r"] <= 1.0 + 1e-9
    assert cycle["max_notional_to_equity"] > cycle["base_notional_to_equity"]


def test_add_stop_does_not_close_base() -> None:
    close = 100 + np.linspace(0, 2.5, 300)
    close[180:190] -= np.linspace(0, 2.0, 10)
    path = _path(close)
    policy = StagedExecutionPolicy(
        "T1",
        "turtle",
        add_r=(0.35,),
        trigger_n=(1.0,),
        max_cycle_hard_r=1.35,
        require_profit_cover=False,
    )
    timeline = _timeline(path, [(100, "HEALTHY", float(path.close[99]))])
    result = _run(policy, path, timeline)
    assert (result.legs["exit_reason"] == "independent_add_stop").any()
    base = result.legs.loc[result.legs["tranche_role"].eq("base")].iloc[0]
    assert base["exit_reason"] == "failed_reclaim_below_structure"


def test_pyramid_never_exceeds_three_layers_or_declared_tail_r() -> None:
    close = 100 + np.linspace(0, 6, 300)
    path = _path(close)
    policy = StagedExecutionPolicy(
        "P1",
        "pyramid",
        add_r=(0.35, 0.35),
        trigger_n=(1.0, 2.0),
        add_stop_n=0.75,
        max_cycle_hard_r=1.70,
    )
    timeline = _timeline(path, [(100, "HEALTHY", float(path.close[99]))])
    result = _run(policy, path, timeline)
    cycle = result.cycles.iloc[0]
    assert cycle["add_count"] <= 2
    assert cycle["max_hard_tail_r"] <= 1.70 + 1e-9
    assert result.daily_equity["active_tranches"].max() <= 3


def test_broken_structure_blocks_add() -> None:
    close = 100 + np.linspace(0, 4, 300)
    path = _path(close)
    policy = StagedExecutionPolicy(
        "T1",
        "turtle",
        add_r=(0.35,),
        trigger_n=(1.0,),
        max_cycle_hard_r=1.35,
    )
    timeline = _timeline(path, [(90, "BROKEN", float(path.close[89]))])
    result = _run(policy, path, timeline)
    assert result.cycles.iloc[0]["add_count"] == 0


def test_counterfactual_marks_baseline_winner_turned_loser() -> None:
    cycles = pd.DataFrame(
        [
            {"fold_id": "WF_2024", "policy": "P0_single_1R", "delay_minutes": 1, "cost_multiplier": 2.0, "event_id": "a", "cycle_return": 0.02},
            {"fold_id": "WF_2024", "policy": "T1", "delay_minutes": 1, "cost_multiplier": 2.0, "event_id": "a", "cycle_return": -0.01},
        ]
    )
    summary = pd.DataFrame(
        [
            {"fold_id": "WF_2024", "policy": "P0_single_1R", "delay_minutes": 1, "cost_multiplier": 2.0},
            {"fold_id": "WF_2024", "policy": "T1", "delay_minutes": 1, "cost_multiplier": 2.0},
        ]
    )
    enriched = enrich_summaries(summary, cycles)
    row = enriched.loc[enriched["policy"].eq("T1")].iloc[0]
    assert row["winner_to_loser_share"] == 1.0


def test_gate_rejects_cross_year_return_loss() -> None:
    rows = []
    for fold, candidate_return in [("WF_2024", 0.55), ("WF_2025", 0.40)]:
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                for policy, ret in [("P0_single_1R", 0.60), ("T1", candidate_return)]:
                    rows.append(
                        {
                            "fold_id": fold,
                            "policy": policy,
                            "delay_minutes": delay,
                            "cost_multiplier": cost,
                            "total_net_return": ret,
                            "max_drawdown": -0.08,
                            "total_return_without_top10": 0.10,
                            "positive_quarters": 4,
                            "winner_to_loser_share": 0.0,
                            "max_hard_tail_r": 1.35,
                            "max_notional_to_equity": 1.0,
                            "addon_loss_share_of_base_profit": 0.0,
                        }
                    )
    config = StagedExecutionConfig(
        policies=(
            StagedExecutionPolicy("P0_single_1R", "baseline"),
            StagedExecutionPolicy("T1", "turtle", add_r=(0.35,), trigger_n=(1.0,), max_cycle_hard_r=1.35),
        )
    )
    gate = policy_gate(pd.DataFrame(rows), config)
    candidate = gate.loc[gate["policy"].eq("T1")].iloc[0]
    assert not bool(candidate["pass_to_next_stage"])
