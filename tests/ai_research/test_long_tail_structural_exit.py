from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_structural_exit.analysis import stable_candidates
from src.ai_research.long_tail_structural_exit.config import StructuralExitConfig, StructuralPolicy
from src.ai_research.long_tail_structural_exit.simulator import (
    score_tier,
    simulate_fixed_diagnostic,
    simulate_structural_event,
)
from src.ai_research.long_tail_structural_exit.structure import (
    build_event_bars,
    confirmed_pivots,
)


def _bars(post: list[tuple[float, float, float]]) -> tuple[list[float], list[float], list[float]]:
    closes = [100.0] * 24
    highs = [100.3] * 24
    lows = [99.7] * 24
    lows[20] = 99.0
    for close, high, low in post:
        closes.append(close)
        highs.append(high)
        lows.append(low)
    return closes, highs, lows


def _path(post: list[tuple[float, float, float]]):
    closes, highs, lows = _bars(post)
    rows = len(closes) * 15
    index = pd.date_range("2025-01-01", periods=rows, freq="1min")
    frame_rows: list[dict[str, float]] = []
    previous = closes[0]
    for close, high, low in zip(closes, highs, lows, strict=True):
        values = np.linspace(previous, close, 15)
        for minute, value in enumerate(values):
            open_value = previous if minute == 0 else float(values[minute - 1])
            frame_rows.append(
                {
                    "open": float(open_value),
                    "high": float(max(high, open_value, value)),
                    "low": float(min(low, open_value, value)),
                    "close": float(value),
                }
            )
        previous = close
    frame = pd.DataFrame(frame_rows, index=index)
    return prepare_minute_path_frame(frame)


def _event(path, entry_position: int = 360) -> EventCandidate:
    decision_ns = int(path.timestamps_ns[entry_position] - pd.Timedelta(minutes=1).value)
    return EventCandidate("event", decision_ns, 1.0, 0.70)


def test_score_tiers_remain_visible() -> None:
    assert score_tier(0.70) == "q70_to_q80"
    assert score_tier(0.85) == "q80_to_q90"
    assert score_tier(0.95) == "q90_plus"


def test_candidate_config_has_no_holding_time_exit() -> None:
    config = StructuralExitConfig()
    payload = str(config.to_dict()).lower()
    assert "maximum_holding" not in payload
    assert "safety_cap" not in payload
    assert all("time" not in policy.name for policy in config.policies)


def test_pivots_are_only_available_after_right_confirmation() -> None:
    path = _path([(100.2, 100.4, 99.9)] * 8)
    config = StructuralExitConfig()
    bars = build_event_bars(path, entry_position=360, end_position=len(path.timestamps_ns) - 1, config=config)
    assert bars is not None
    pivots = confirmed_pivots(bars, left_bars=2, right_bars=2)
    floor = [pivot for pivot in pivots if pivot.kind == "low" and pivot.pivot_index == 20]
    assert len(floor) == 1
    assert floor[0].confirmation_index == 22


def test_future_after_pivot_confirmation_does_not_change_that_pivot() -> None:
    path = _path([(100.2, 100.4, 99.9)] * 8)
    config = StructuralExitConfig()
    bars = build_event_bars(path, entry_position=360, end_position=len(path.timestamps_ns) - 1, config=config)
    assert bars is not None
    before = [p for p in confirmed_pivots(bars, left_bars=2, right_bars=2) if p.pivot_index == 20]
    changed_post = [(120.0, 121.0, 119.0)] * 8
    changed = _path(changed_post)
    changed_bars = build_event_bars(changed, entry_position=360, end_position=len(changed.timestamps_ns) - 1, config=config)
    assert changed_bars is not None
    after = [p for p in confirmed_pivots(changed_bars, left_bars=2, right_bars=2) if p.pivot_index == 20]
    assert [(p.kind, p.price, p.confirmation_index) for p in before] == [
        (p.kind, p.price, p.confirmation_index) for p in after
    ]


def test_broken_floor_can_reclaim_without_forced_exit() -> None:
    post = [
        (98.8, 98.9, 98.5),
        (99.4, 99.5, 98.7),
        (99.8, 100.0, 99.2),
        (100.2, 100.4, 99.7),
        (100.5, 100.7, 100.0),
        (100.8, 101.0, 100.3),
        (101.0, 101.2, 100.6),
        (101.2, 101.4, 100.8),
    ] + [(101.2, 101.4, 101.0)] * 20
    path = _path(post)
    config = StructuralExitConfig(policies=(StructuralPolicy("confirmed_structure"),))
    trade = simulate_structural_event(
        _event(path),
        fold_id="WF_2025",
        policy=config.policies[0],
        delay_minutes=1,
        percentile=0.75,
        path=path,
        oos_end_ns=int(path.timestamps_ns[-1]),
        config=config,
    )
    assert trade is not None
    values = trade.to_dict()
    assert values["recoveries"] >= 1
    assert values["is_censored"]
    assert values["exit_reason"].startswith("censored_")


def test_lower_high_then_lower_low_confirms_exit() -> None:
    post = [
        (98.6, 98.70, 98.50),
        (98.4, 98.60, 98.40),
        (98.3, 98.50, 98.30),
        (98.5, 98.65, 98.35),
        (98.7, 98.85, 98.50),
        (98.5, 98.65, 98.40),
        (98.4, 98.55, 98.30),
        (98.1, 98.30, 97.90),
    ] + [(98.0, 98.2, 97.8)] * 20
    path = _path(post)
    config = StructuralExitConfig(policies=(StructuralPolicy("confirmed_structure"),))
    trade = simulate_structural_event(
        _event(path),
        fold_id="WF_2025",
        policy=config.policies[0],
        delay_minutes=1,
        percentile=0.95,
        path=path,
        oos_end_ns=int(path.timestamps_ns[-1]),
        config=config,
    )
    assert trade is not None
    values = trade.to_dict()
    assert not values["is_censored"]
    assert values["exit_reason"] == "confirmed_lower_high_lower_low"


def test_disaster_breach_executes_at_next_minute_open() -> None:
    post = [(96.5, 100.1, 96.0)] + [(96.8, 97.0, 96.4)] * 27
    path = _path(post)
    config = StructuralExitConfig(policies=(StructuralPolicy("confirmed_structure"),))
    trade = simulate_structural_event(
        _event(path),
        fold_id="WF_2025",
        policy=config.policies[0],
        delay_minutes=1,
        percentile=0.75,
        path=path,
        oos_end_ns=int(path.timestamps_ns[-1]),
        config=config,
    )
    assert trade is not None
    values = trade.to_dict()
    assert values["exit_reason"] == "disaster_stop"
    entry_pos = 360
    breach_pos = entry_pos
    expected_exit_time = pd.Timestamp(int(path.timestamps_ns[breach_pos + 1]), unit="ns")
    assert pd.Timestamp(values["exit_time"]) == expected_exit_time



def test_fixed_six_hour_diagnostic_uses_bar_close_not_next_open() -> None:
    post = [(100.0 + 0.1 * i, 100.3 + 0.1 * i, 99.7 + 0.1 * i) for i in range(28)]
    path = _path(post)
    config = StructuralExitConfig()
    trade = simulate_fixed_diagnostic(
        _event(path),
        fold_id="WF_2025",
        policy="fixed_6h",
        delay_minutes=1,
        percentile=0.75,
        path=path,
        config=config,
        disaster_protected=False,
    )
    assert trade is not None
    values = trade.to_dict()
    entry_pos = 360
    exit_pos = entry_pos + 360 - 1
    assert np.isclose(values["exit_price"], path.close[exit_pos])
    assert np.isclose(values["gross_return"], path.close[exit_pos] / path.open[entry_pos] - 1.0)

def test_stable_candidate_requires_same_policy_in_both_years() -> None:
    config = StructuralExitConfig(minimum_trades_per_year=10, minimum_positive_quarters=2)
    rows = []
    comparisons = []
    periods = []
    for fold in ("WF_2024", "WF_2025"):
        for delay, cost in ((1, 2.0), (1, 3.0), (5, 2.0)):
            rows.append(
                {
                    "fold_id": fold,
                    "policy": "confirmed_structure",
                    "policy_kind": "non_time_structural_candidate",
                    "delay_minutes": delay,
                    "metric_scope": "all_positions_mark_to_market",
                    "cost_multiplier": cost,
                    "trades": 100,
                    "mean_net_return": 0.003,
                    "profit_factor": 1.6,
                    "max_drawdown": -0.12,
                    "top10_profit_share": 0.4,
                    "mean_net_without_top10": 0.001,
                    "censored_share": 0.01,
                }
            )
        comparisons.append(
            {
                "fold_id": fold,
                "policy": "confirmed_structure",
                "total_return_delta": 0.10,
                "profit_retention_ratio": 1.05,
                "relative_mdd_improvement": 0.20,
            }
        )
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            periods.append(
                {
                    "fold_id": fold,
                    "policy": "confirmed_structure",
                    "delay_minutes": 1,
                    "cost_multiplier": 2.0,
                    "quarter": f"{fold[-4:]}{quarter}",
                    "mean_net_return": 0.001,
                }
            )
    stable = stable_candidates(pd.DataFrame(rows), pd.DataFrame(periods), pd.DataFrame(comparisons), config)
    assert bool(stable.iloc[0]["passes_profit_upgrade"])
    incomplete = stable_candidates(pd.DataFrame(rows).loc[lambda x: x.fold_id == "WF_2025"], pd.DataFrame(periods), pd.DataFrame(comparisons), config)
    assert not bool(incomplete.iloc[0]["complete_2024_2025"])
