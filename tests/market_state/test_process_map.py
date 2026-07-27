#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.process_map import ProcessMapConfig, ProcessMapEngine, stage_event_mask


def _frame(rows: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="1min")
    close = 2000.0 + np.linspace(0.0, 20.0, rows)
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "available_time": idx + pd.Timedelta(minutes=1),
            "data_ready": True,
            "volatility_state": "normal",
            "flow_state": "balanced",
            "flow_score": 0.0,
            "flow_strength": 0.5,
            "impact_state": "neutral",
            "sell_absorption_score": 0.0,
            "buy_absorption_score": 0.0,
            "location_state": "middle_zone",
            "structural_location_score": 0.0,
        },
        index=idx,
    )
    return frame


def test_long_reversal_requires_ordered_later_bars() -> None:
    frame = _frame()
    frame.loc[frame.index[20], ["flow_state", "flow_score", "impact_state"]] = ["sell_persistent", -0.4, "sell_effective"]
    frame.loc[frame.index[25], ["impact_state", "sell_absorption_score"]] = ["sell_absorbed", 0.8]
    frame.loc[frame.index[30], "location_state"] = "downside_sweep_reclaim"
    frame.loc[frame.index[35], ["flow_state", "flow_score"]] = ["buy_building", 0.3]

    result = ProcessMapEngine(ProcessMapConfig(semantic_version="v3", minimum_probability_samples=2)).compute(frame)
    completed = result.episodes.loc[
        result.episodes["family"].eq("long_reversal") & result.episodes["completed"].eq(True)
    ]
    assert len(completed) == 1
    row = completed.iloc[0]
    assert int(row["stage_1_pos"]) == 20
    assert int(row["stage_2_pos"]) == 25
    assert int(row["stage_3_pos"]) == 30
    assert int(row["stage_4_pos"]) == 35
    assert bool(stage_event_mask(result.frame, "long_reversal", 4).iloc[35])
    assert result.frame.iloc[35]["process_family"] == "long_reversal"
    assert result.frame.iloc[35]["process_status"] == "complete"


def test_same_bar_conditions_do_not_cascade() -> None:
    frame = _frame(80)
    pos = 20
    frame.loc[frame.index[pos], ["flow_state", "flow_score", "impact_state", "sell_absorption_score", "location_state"]] = [
        "sell_persistent", -0.5, "sell_absorbed", 0.8, "downside_sweep_reclaim"
    ]
    result = ProcessMapEngine(ProcessMapConfig(semantic_version="v3", minimum_probability_samples=2)).compute(frame)
    assert int(result.frame.iloc[pos]["long_reversal_stage"]) == 1
    assert not bool(result.episodes.get("completed", pd.Series(dtype=bool)).fillna(False).any())


def test_expiry_stops_stale_process() -> None:
    frame = _frame(100)
    frame.loc[frame.index[10], ["flow_state", "flow_score", "impact_state"]] = ["sell_persistent", -0.5, "sell_effective"]
    cfg = ProcessMapConfig(semantic_version="v3", reversal_pressure_to_absorption_bars=5, minimum_probability_samples=2)
    result = ProcessMapEngine(cfg).compute(frame)
    episode = result.episodes.loc[result.episodes["family"].eq("long_reversal")].iloc[0]
    assert episode["status"] == "expired"
    assert episode["expiry_reason"] == "stage_1_timeout"
    assert int(result.frame.iloc[17]["long_reversal_stage"]) == 0


def test_append_invariance_of_process_state() -> None:
    frame = _frame(180)
    for base in (20, 70, 120):
        frame.loc[frame.index[base], ["flow_state", "flow_score", "impact_state"]] = ["sell_persistent", -0.4, "sell_effective"]
        frame.loc[frame.index[base + 4], ["impact_state", "sell_absorption_score"]] = ["sell_absorbed", 0.8]
        frame.loc[frame.index[base + 8], "location_state"] = "downside_sweep_reclaim"
        frame.loc[frame.index[base + 12], ["flow_state", "flow_score"]] = ["buy_building", 0.3]
    cfg = ProcessMapConfig(semantic_version="v3", minimum_probability_samples=2)
    engine = ProcessMapEngine(cfg)
    prefix = engine.compute(frame.iloc[:140]).frame
    full = engine.compute(frame).frame.iloc[:140]
    columns = [
        "long_reversal_stage", "process_family", "process_stage", "process_status",
        "process_completion_probability", "process_direction_probability",
    ]
    pd.testing.assert_frame_equal(prefix[columns], full[columns], check_dtype=False)


def test_probability_does_not_use_current_event_future() -> None:
    frame = _frame(260)
    for base in (20, 70, 120, 170, 220):
        frame.loc[frame.index[base], ["flow_state", "flow_score", "impact_state"]] = ["sell_persistent", -0.4, "sell_effective"]
        frame.loc[frame.index[base + 3], ["impact_state", "sell_absorption_score"]] = ["sell_absorbed", 0.8]
        frame.loc[frame.index[base + 6], "location_state"] = "downside_sweep_reclaim"
        frame.loc[frame.index[base + 9], ["flow_state", "flow_score"]] = ["buy_building", 0.3]
        frame.loc[frame.index[base + 10: base + 20], "close"] += 10.0
    cfg = ProcessMapConfig(
        semantic_version="v3",
        probability_horizons_bars=(5, 15, 60),
        default_reversal_horizon_bars=5,
        default_breakout_horizon_bars=15,
        minimum_probability_samples=2,
    )
    result = ProcessMapEngine(cfg).compute(frame)
    completion_positions = np.flatnonzero(stage_event_mask(result.frame, "long_reversal", 4).to_numpy())
    assert len(completion_positions) >= 4
    first, second, third = completion_positions[:3]
    assert pd.isna(result.frame.iloc[first]["process_direction_probability"])
    # At the second event, the first event's 5-bar outcome is known; the second
    # event cannot include its own future result.
    assert int(result.frame.iloc[second]["process_direction_samples"]) <= 1
    assert int(result.frame.iloc[third]["process_direction_samples"]) <= 2
