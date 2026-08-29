#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r23 import R23Config, regularize_trade_bars, simulate_frozen_panic_long


def _features() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=15, freq="1min")
    return pd.DataFrame(index=index, data={
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        "source_bar_observed_flag": 1,
    })


def _event(features: pd.DataFrame) -> pd.DataFrame:
    event_time = features.index[0]
    return pd.DataFrame([{
        "event_id": "R23_EVENT_000001", "source_event": "strict_flow",
        "event_bar_time": event_time, "signal_available_time": event_time + pd.Timedelta(minutes=1),
        "entry_time": event_time + pd.Timedelta(minutes=3), "event_low": 99.0, "event_high": 101.0,
    }])


def test_r23_multi_sweep_failure_exits_next_open() -> None:
    features = _features()
    features.loc[features.index[3], ["low", "close"]] = [98.9, 99.5]
    features.loc[features.index[5], ["low", "close"]] = [98.7, 98.8]
    features.loc[features.index[6], "open"] = 98.6
    trades = simulate_frozen_panic_long(features, _event(features), split="discovery", split_start=features.index[0], split_end=features.index[-1] + pd.Timedelta(minutes=1))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "MULTI_SWEEP_DEEPER_FAIL"
    assert trade["exit_decision_bar_time"] == features.index[5]
    assert trade["exit_time"] == features.index[6]
    assert np.isclose(trade["exit_price"], 98.6)


def test_r23_missing_source_bar_censors_open_path() -> None:
    features = _features()
    features.loc[features.index[5], "source_bar_observed_flag"] = 0
    trades = simulate_frozen_panic_long(features, _event(features), split="discovery", split_start=features.index[0], split_end=features.index[-1] + pd.Timedelta(minutes=1))
    trade = trades.iloc[0]
    assert trade["path_status"] == "data_gap_censored"
    assert pd.isna(trade["gross_return"])


def test_regularize_trade_bars_preserves_clock_with_flat_placeholder() -> None:
    index = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:02", "2023-01-01 00:03"])
    raw = pd.DataFrame(index=index, data={"open": [100, 102, 103], "high": [101, 103, 104], "low": [99, 101, 102], "close": [100, 102, 103], "volume": [1, 2, 3]})
    out = regularize_trade_bars(raw)
    missing = pd.Timestamp("2023-01-01 00:01")
    assert len(out) == 4
    assert out.loc[missing, "source_bar_observed_flag"] == 0
    assert np.isclose(out.loc[missing, "open"], 100.0)
    assert np.isclose(out.loc[missing, "volume"], 0.0)

