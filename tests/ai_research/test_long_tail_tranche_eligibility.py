from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_tranche_eligibility.analysis import tranche_gate
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.long_tail_tranche_eligibility.simulator import (
    build_occupancy_map,
    classify_occupied_signal,
    failed_reclaim_snapshots,
)


def _path(post: list[tuple[float, float, float]]):
    closes = [100.0] * 24 + [row[0] for row in post]
    highs = [100.3] * 24 + [row[1] for row in post]
    lows = [99.7] * 24 + [row[2] for row in post]
    lows[20] = 99.0
    rows: list[dict[str, float]] = []
    previous = closes[0]
    for close, high, low in zip(closes, highs, lows, strict=True):
        values = np.linspace(previous, close, 15)
        for minute, value in enumerate(values):
            open_value = previous if minute == 0 else float(values[minute - 1])
            rows.append(
                {
                    "open": float(open_value),
                    "high": float(max(high, open_value, value)),
                    "low": float(min(low, open_value, value)),
                    "close": float(value),
                }
            )
        previous = close
    frame = pd.DataFrame(rows, index=pd.date_range("2025-01-01", periods=len(rows), freq="1min"))
    return prepare_minute_path_frame(frame)


def _event(path, entry_position: int = 360) -> EventCandidate:
    return EventCandidate(
        "root",
        int(path.timestamps_ns[entry_position] - pd.Timedelta(minutes=1).value),
        1.0,
        0.70,
    )


def test_config_is_diagnostic_and_keeps_2026_sealed() -> None:
    config = TrancheEligibilityConfig()
    payload = config.to_dict()
    assert payload["tranche_execution_in_this_stage"] is False
    assert payload["frozen_structural_policy"] == "failed_reclaim"
    assert pd.Timestamp(config.research_end) < pd.Timestamp(config.sealed_holdout_start)


def test_occupancy_map_keeps_root_and_marks_later_events() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_id": "a",
                "entry_time": "2025-01-01 00:01:00",
                "decision_time": "2025-01-01 00:00:00",
                "exit_time": "2025-01-01 12:00:00",
                "score": 1.0,
                "exit_reason": "failed_reclaim_below_structure",
            },
            {
                "event_id": "b",
                "entry_time": "2025-01-01 06:01:00",
                "decision_time": "2025-01-01 06:00:00",
                "exit_time": "2025-01-01 18:00:00",
                "score": 2.0,
                "exit_reason": "failed_reclaim_below_structure",
            },
            {
                "event_id": "c",
                "entry_time": "2025-01-01 13:01:00",
                "decision_time": "2025-01-01 13:00:00",
                "exit_time": "2025-01-01 20:00:00",
                "score": 1.5,
                "exit_reason": "failed_reclaim_below_structure",
            },
        ]
    )
    result = build_occupancy_map(frame)
    assert result.executed["event_id"].tolist() == ["a", "c"]
    assert result.occupied[["event_id", "root_event_id"]].to_dict("records") == [
        {"event_id": "b", "root_event_id": "a"}
    ]


def test_broken_or_pending_failed_reclaim_signal_is_never_eligible() -> None:
    config = TrancheEligibilityConfig()
    signal_class, reason, eligible = classify_occupied_signal(
        {
            "state": "BROKEN",
            "current_return_vs_root": -0.01,
            "score_delta_vs_root": 0.5,
            "released_risk_fraction": 0.8,
            "pending_failed_reclaim_exit": False,
        },
        config=config,
    )
    assert signal_class == "dangerous_average_down"
    assert reason == "failed_reclaim_process_active"
    assert not eligible


def test_score_up_price_down_without_protection_is_dangerous() -> None:
    config = TrancheEligibilityConfig()
    signal_class, reason, eligible = classify_occupied_signal(
        {
            "state": "HEALTHY",
            "current_return_vs_root": -0.005,
            "score_delta_vs_root": 0.2,
            "released_risk_fraction": 0.05,
            "pending_failed_reclaim_exit": False,
            "floor_raised": False,
        },
        config=config,
    )
    assert signal_class == "dangerous_average_down"
    assert reason == "score_up_price_down_without_protection"
    assert not eligible


def test_healthy_trend_requires_structure_and_meaningful_candidate_risk_release() -> None:
    config = TrancheEligibilityConfig(minimum_released_risk_fraction=0.25)
    signal_class, _, eligible = classify_occupied_signal(
        {
            "state": "HEALTHY",
            "current_return_vs_root": 0.02,
            "score_delta_vs_root": -0.1,
            "released_risk_fraction": 0.50,
            "pending_failed_reclaim_exit": False,
            "independent_structure_confirmed": True,
            "floor_raised": True,
            "higher_low_confirmed": True,
            "recoveries": 0,
            "latest_low_after_recovery": False,
            "structure_age_minutes": 60.0,
        },
        config=config,
    )
    assert signal_class == "healthy_trend"
    assert eligible


def test_snapshot_does_not_use_bars_after_observation() -> None:
    base_post = [
        (100.5, 100.8, 100.0),
        (101.0, 101.2, 100.4),
        (101.4, 101.6, 100.8),
        (101.1, 101.5, 100.9),
        (101.7, 101.9, 101.0),
        (102.0, 102.2, 101.5),
        (102.3, 102.5, 101.9),
        (102.6, 102.8, 102.1),
    ]
    path = _path(base_post + [(102.6, 102.8, 102.2)] * 12)
    event = _event(path)
    observation = int(path.timestamps_ns[360 + 15 * 6 - 1])
    config = TrancheEligibilityConfig().structural_config()
    before = failed_reclaim_snapshots(
        event,
        delay_minutes=1,
        observation_times_ns=(observation,),
        path=path,
        end_time_ns=int(path.timestamps_ns[-1]),
        config=config,
    )[observation]

    changed = _path(base_post + [(80.0, 81.0, 79.0)] * 12)
    after = failed_reclaim_snapshots(
        _event(changed),
        delay_minutes=1,
        observation_times_ns=(observation,),
        path=changed,
        end_time_ns=int(changed.timestamps_ns[-1]),
        config=config,
    )[observation]
    keys = [
        "state",
        "active_floor",
        "current_return_vs_root",
        "structure_breaks",
        "recoveries",
        "candidate_hard_stop_price",
    ]
    assert {key: before[key] for key in keys} == {key: after[key] for key in keys}



def test_off_boundary_delay_uses_latest_completed_structure_bar() -> None:
    post = [(101.0 + 0.1 * i, 101.3 + 0.1 * i, 100.7 + 0.1 * i) for i in range(20)]
    path = _path(post)
    entry_position = 360
    event = EventCandidate(
        "root_delay3",
        int(path.timestamps_ns[entry_position] - pd.Timedelta(minutes=3).value),
        1.0,
        0.70,
    )
    observation_position = entry_position + 20
    observation = int(path.timestamps_ns[observation_position])
    config = TrancheEligibilityConfig().structural_config()
    before = failed_reclaim_snapshots(
        event,
        delay_minutes=3,
        observation_times_ns=(observation,),
        path=path,
        end_time_ns=int(path.timestamps_ns[-1]),
        config=config,
    )[observation]

    frame = pd.DataFrame(
        {
            "open": path.open.copy(),
            "high": path.high.copy(),
            "low": path.low.copy(),
            "close": path.close.copy(),
        },
        index=path.index,
    )
    future = slice(observation_position + 1, observation_position + 10)
    frame.iloc[future, frame.columns.get_loc("open")] = 80.0
    frame.iloc[future, frame.columns.get_loc("high")] = 81.0
    frame.iloc[future, frame.columns.get_loc("low")] = 79.0
    frame.iloc[future, frame.columns.get_loc("close")] = 80.0
    changed = prepare_minute_path_frame(frame)
    after = failed_reclaim_snapshots(
        event,
        delay_minutes=3,
        observation_times_ns=(observation,),
        path=changed,
        end_time_ns=int(changed.timestamps_ns[-1]),
        config=config,
    )[observation]
    keys = ["state", "active_floor", "current_close", "structure_breaks", "recoveries"]
    assert {key: before[key] for key in keys} == {key: after[key] for key in keys}

def test_gate_requires_same_delay_to_pass_both_years() -> None:
    config = TrancheEligibilityConfig(
        minimum_eligible_events_per_year=12,
        minimum_positive_quarters_per_year=1,
        maximum_top10_profit_share=2.0,
    )
    atlas_rows = []
    quarter_rows = []
    # More than ten events are required because the production gate must remain
    # positive after removing the ten largest winners. Include a small losing
    # tail so PF remains finite rather than relying on an all-winner sample.
    gross_path = [0.015] * 12 + [0.008] * 6 + [-0.002] * 2
    structural_path = [0.018] * 12 + [0.009] * 6 + [-0.003] * 2
    for fold in ("WF_2024", "WF_2025"):
        for gross, structural_gross in zip(gross_path, structural_path, strict=True):
            atlas_rows.append(
                {
                    "fold_id": fold,
                    "delay_minutes": 1,
                    "eligible_for_tranche_simulation": True,
                    "fixed6h_gross_return": gross,
                    "standalone_failed_reclaim_gross_return": structural_gross,
                    "current_return_vs_root": 0.01,
                }
            )
        quarter_rows.append(
            {
                "fold_id": fold,
                "delay_minutes": 1,
                "quarter": f"{fold[-4:]}Q1",
                "mean_net_return": 0.01,
            }
        )
    atlas = pd.DataFrame(atlas_rows)
    gate = tranche_gate(atlas, pd.DataFrame(quarter_rows), config)
    main = gate.loc[gate["delay_minutes"] == 1].iloc[0]
    assert bool(main["pass_to_tranche_simulation"])

    one_year = atlas.loc[atlas["fold_id"] == "WF_2025"]
    failed = tranche_gate(one_year, pd.DataFrame(quarter_rows), config)
    assert not bool(failed.loc[failed["delay_minutes"] == 1, "pass_to_tranche_simulation"].iloc[0])
