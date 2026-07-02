#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import pandas as pd

from src.research_common.event_study import CostConfig, EventStudyConfig, run_event_study


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00:00", periods=6, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0, 102.0, 99.0, 98.0],
            "high": [101.0, 104.0, 104.0, 103.0, 100.0, 99.0],
            "low": [99.0, 100.0, 101.0, 98.0, 97.0, 96.0],
            "close": [101.0, 103.0, 102.0, 99.0, 98.0, 97.0],
        },
        index=idx,
    )


def test_run_event_study_next_open_labels_and_costs() -> None:
    bars = _bars()
    events = pd.DataFrame(
        {
            "signal_time": [bars.index[0], bars.index[2]],
            "side": [1, -1],
            "event_name": ["breakout_up", "breakout_down"],
        }
    )
    cfg = EventStudyConfig(horizons=(1, 2), mfe_mae_horizon=2, cost=CostConfig(entry_fee_rate=0.001, exit_fee_rate=0.001), min_count=1)

    result = run_event_study(bars, events, cfg)

    assert len(result.events) == 2
    first = result.events.iloc[0]
    assert first["entry_time"] == bars.index[1]
    assert math.isclose(float(first["entry_price"]), 101.0)
    assert math.isclose(float(first["next_open_ret_h1_gross"]), 103.0 / 101.0 - 1.0)
    assert math.isclose(float(first["next_open_ret_h1_net"]), 103.0 / 101.0 - 1.0 - 0.002)
    assert math.isclose(float(first["mfe_h2"]), 104.0 / 101.0 - 1.0)
    assert math.isclose(float(first["mae_h2"]), 100.0 / 101.0 - 1.0)

    second = result.events.iloc[1]
    assert second["entry_time"] == bars.index[3]
    assert math.isclose(float(second["next_open_ret_h2_gross"]), 1.0 - 98.0 / 102.0)
    assert int(result.meta["causal_fail_count"]) == 0
    assert not result.causal_audit["causal_fail_flag"].any()
    assert set(result.overview["metric"]) == {"next_open_ret_h1_net", "next_open_ret_h2_net"}


def test_run_event_study_flags_missing_signal_bar_and_context_leak() -> None:
    bars = _bars()
    events = pd.DataFrame(
        {
            "signal_time": [pd.Timestamp("2024-01-01 00:00:30"), bars.index[1]],
            "side": [1, 1],
            "tf5m_available_time": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:03:00")],
        }
    )
    cfg = EventStudyConfig(horizons=(1,), mfe_mae_horizon=1, context_available_time_cols=("tf5m_available_time",), min_count=1)

    result = run_event_study(bars, events, cfg)

    assert result.causal_audit["causal_fail_flag"].tolist() == [True, True]
    assert result.causal_audit["signal_on_bar_index_flag"].tolist() == [False, True]
    assert result.causal_audit["context_available_time_flag"].tolist() == [False, True]
