from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_soft_failure_tail_compression.analysis import (
    build_f1_attribution,
    policy_gate,
)
from src.ai_research.long_tail_soft_failure_tail_compression.config import (
    TailCompressionConfig,
    TailCompressionPolicy,
)
from src.ai_research.long_tail_soft_failure_tail_compression.simulator import (
    simulate_tail_compression_account,
)


def _path(close: np.ndarray, *, low: np.ndarray | None = None):
    index = pd.date_range("2024-01-01", periods=len(close), freq="min")
    low_values = close - 0.08 if low is None else low
    frame = pd.DataFrame(
        {
            "open": close,
            "high": np.maximum(close + 0.08, low_values),
            "low": low_values,
            "close": close,
        },
        index=index,
    )
    return prepare_minute_path_frame(frame)


def _events(path, *, entry_position: int = 80, exit_position: int = 220) -> pd.DataFrame:
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


def _timeline(path, *, effective_position: int, current_close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "e1",
                "fold_id": "WF_2024",
                "delay_minutes": 1,
                "effective_time": path.index[effective_position],
                "structure_close_time": path.index[effective_position - 1],
                "state": "HEALTHY",
                "pending_failed_reclaim_exit": False,
                "current_close": current_close,
            }
        ]
    )


def _run(policy: TailCompressionPolicy, path, timeline: pd.DataFrame | None = None):
    config = TailCompressionConfig(
        policies=(policy,),
        minimum_return_retention_each_year=0.0,
        minimum_combined_return_ratio=0.0,
        minimum_positive_quarters_per_year=0,
        maximum_winner_to_loser_share=1.0,
        minimum_mean_notional_to_equity=0.0,
        maximum_worst_cycle_loss_r=10.0,
    )
    return simulate_tail_compression_account(
        _events(path),
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


def test_config_marks_only_real_one_r_policies_as_candidates() -> None:
    config = TailCompressionConfig()
    config.validate()
    policies = {policy.name: policy for policy in config.policies}
    assert np.isclose(policies["P0_single_1R"].declared_hard_tail_r, 1.0)
    assert np.isclose(policies["F1_reference_1p5size_3ptail"].declared_hard_tail_r, 2.0)
    assert not policies["F1_reference_1p5size_3ptail"].qualifying_candidate
    assert all(
        policy.declared_hard_tail_r <= 1.02
        for policy in config.policies
        if policy.qualifying_candidate
    )


def test_p0_preserves_one_r_and_one_third_notional() -> None:
    path = _path(100 + np.linspace(0, 3, 300))
    result = _run(TailCompressionPolicy("P0_single_1R", "baseline", qualifying_candidate=False), path)
    cycle = result.cycles.iloc[0]
    assert np.isclose(cycle["max_hard_tail_r"], 1.0)
    assert 0.30 < cycle["base_notional_to_equity"] < 0.35
    assert cycle["exit_reason"] == "failed_reclaim_below_structure"


def test_f1_reference_is_two_r_tail_and_two_thirds_notional() -> None:
    close = np.full(300, 100.0)
    close[80:] = np.linspace(100, 98.0, 220)
    path = _path(close)
    policy = TailCompressionPolicy(
        "F1_reference_1p5size_3ptail",
        "reference",
        sizing_stop_distance=0.015,
        hard_stop_distance=0.03,
        soft_failure_distance=0.015,
        qualifying_candidate=False,
    )
    result = _run(policy, path, _timeline(path, effective_position=150, current_close=98.4))
    cycle = result.cycles.iloc[0]
    assert np.isclose(cycle["max_hard_tail_r"], 2.0)
    assert 0.63 < cycle["base_notional_to_equity"] < 0.70
    assert cycle["soft_failure_exit"]


def test_real_two_percent_policy_is_half_notional_and_one_r_tail() -> None:
    path = _path(100 + np.linspace(0, 2, 300))
    policy = TailCompressionPolicy(
        "C2",
        "fixed",
        sizing_stop_distance=0.02,
        hard_stop_distance=0.02,
        soft_failure_distance=0.015,
    )
    result = _run(policy, path)
    cycle = result.cycles.iloc[0]
    assert np.isclose(cycle["max_hard_tail_r"], 1.0)
    assert 0.47 < cycle["base_notional_to_equity"] < 0.53


def test_real_hard_stop_is_gap_aware_and_precedes_soft_exit() -> None:
    close = np.full(300, 100.0)
    close[120] = 98.0
    path = _path(close)
    policy = TailCompressionPolicy(
        "C15",
        "fixed",
        sizing_stop_distance=0.015,
        hard_stop_distance=0.015,
        soft_failure_distance=0.01,
    )
    result = _run(policy, path, _timeline(path, effective_position=120, current_close=98.9))
    cycle = result.cycles.iloc[0]
    assert cycle["hard_stop_exit"]
    assert not cycle["soft_failure_exit"]
    assert cycle["exit_reason"] == "real_hard_stop_gap"


def test_soft_failure_executes_at_effective_next_open_before_intrabar_stop() -> None:
    close = np.full(300, 100.0)
    low = close - 0.05
    close[120] = 98.7
    low[120] = 98.2
    path = _path(close, low=low)
    policy = TailCompressionPolicy(
        "C2",
        "fixed",
        sizing_stop_distance=0.02,
        hard_stop_distance=0.02,
        soft_failure_distance=0.015,
    )
    result = _run(policy, path, _timeline(path, effective_position=120, current_close=98.4))
    cycle = result.cycles.iloc[0]
    assert cycle["soft_failure_exit"]
    assert not cycle["hard_stop_exit"]
    assert cycle["exit_time"] == path.index[120]


def test_adaptive_stop_uses_prior_completed_atr_and_stays_one_r() -> None:
    close = 100 + np.sin(np.arange(300) / 5.0) * 0.3
    path = _path(close)
    policy = TailCompressionPolicy(
        "V1",
        "adaptive",
        adaptive_atr_multiple=2.0,
        adaptive_min_distance=0.015,
        adaptive_max_distance=0.03,
        adaptive_soft_fraction=0.75,
    )
    result = _run(policy, path)
    cycle = result.cycles.iloc[0]
    assert 0.015 <= cycle["hard_stop_distance"] <= 0.03
    assert np.isclose(cycle["max_hard_tail_r"], 1.0)
    assert np.isclose(cycle["soft_failure_distance"], cycle["hard_stop_distance"] * 0.75)


def test_f1_attribution_normalizes_two_r_before_classification() -> None:
    path = _path(100 + np.linspace(0, 2, 300))
    cycles = pd.DataFrame(
        [
            {
                "fold_id": "WF_2024",
                "delay_minutes": 1,
                "cost_multiplier": 2.0,
                "event_id": "e1",
                "policy": "P0_single_1R",
                "entry_time": path.index[80],
                "source_exit_time": path.index[220],
                "cycle_return": -0.01,
                "source_exit_reason": "failed_reclaim_below_structure",
                "max_hard_tail_r": 1.0,
                "soft_failure_exit": False,
            },
            {
                "fold_id": "WF_2024",
                "delay_minutes": 1,
                "cost_multiplier": 2.0,
                "event_id": "e1",
                "policy": "F1_soft_failure_1p5",
                "entry_time": path.index[80],
                "source_exit_time": path.index[220],
                "cycle_return": -0.01,
                "source_exit_reason": "failed_reclaim_below_structure",
                "max_hard_tail_r": 2.0,
                "soft_failure_exit": True,
            },
        ]
    )
    legs = pd.DataFrame(
        [
            {
                "fold_id": "WF_2024",
                "delay_minutes": 1,
                "cost_multiplier": 2.0,
                "event_id": "e1",
                "policy": "F1_soft_failure_1p5",
                "tranche_role": "base",
                "entry_price": 100.0,
                "exit_time": path.index[150],
                "exit_price": 99.0,
                "exit_reason": "soft_failure_confirmed_close",
            }
        ]
    )
    result = build_f1_attribution(cycles, legs, fold_id="WF_2024", path=path, materiality=0.001)
    row = result.iloc[0]
    assert np.isclose(row["f1_1r_equivalent_return"], -0.005)
    assert row["attribution_class"] == "EFFECTIVE_LOSS_REDUCTION"


def test_gate_excludes_two_r_reference_even_with_high_return() -> None:
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                for policy, ret, tail, notional in [
                    ("P0_single_1R", 0.60, 1.0, 0.33),
                    ("F1_reference_1p5size_3ptail", 1.20, 2.0, 0.67),
                    ("C2_real_2p_soft1p5", 0.70, 1.0, 0.50),
                ]:
                    rows.append(
                        {
                            "fold_id": fold,
                            "policy": policy,
                            "delay_minutes": delay,
                            "cost_multiplier": cost,
                            "total_net_return": ret,
                            "max_drawdown": -0.09,
                            "total_return_without_top10": 0.10,
                            "positive_quarters": 4,
                            "winner_to_loser_share": 0.0,
                            "max_hard_tail_r": tail,
                            "mean_base_notional_to_equity": notional,
                            "worst_cycle_loss_r": 1.1,
                        }
                    )
    config = TailCompressionConfig(
        policies=(
            TailCompressionPolicy("P0_single_1R", "baseline", qualifying_candidate=False),
            TailCompressionPolicy(
                "F1_reference_1p5size_3ptail",
                "reference",
                sizing_stop_distance=0.015,
                hard_stop_distance=0.03,
                soft_failure_distance=0.015,
                qualifying_candidate=False,
            ),
            TailCompressionPolicy(
                "C2_real_2p_soft1p5",
                "fixed",
                sizing_stop_distance=0.02,
                hard_stop_distance=0.02,
                soft_failure_distance=0.015,
            ),
        ),
    )
    gate = policy_gate(pd.DataFrame(rows), config)
    f1 = gate.loc[gate["policy"].eq("F1_reference_1p5size_3ptail")].iloc[0]
    c2 = gate.loc[gate["policy"].eq("C2_real_2p_soft1p5")].iloc[0]
    assert not bool(f1["pass_to_next_stage"])
    assert bool(c2["pass_to_next_stage"])
