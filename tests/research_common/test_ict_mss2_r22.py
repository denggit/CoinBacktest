#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r22 import R22Config, simulate_catchup


def _event(timestamp: pd.Timestamp, direction: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": "R22_EVENT_000001", "trade_direction": direction,
        "signal_bar_time": timestamp - pd.Timedelta(hours=1), "signal_available_time": timestamp,
        "eth_atr20": 2.0, "beta_prior": 1.1, "btc_return_1h": 0.02 * direction,
        "eth_return_1h": 0.005 * direction, "btc_impulse_z": 2.5 * direction, "lag_z": 1.0,
    }])


def _bars(start: pd.Timestamp, periods: int = 180) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    return pd.DataFrame(index=index, data={"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0})


def test_r22_same_bar_is_stop_first() -> None:
    start = pd.Timestamp("2023-01-02 01:00")
    bars = _bars(start)
    bars.loc[start, ["high", "low"]] = [104.0, 96.0]
    trades = simulate_catchup(bars, _event(start), target_r=1.0, direction=1, split="discovery", split_start=pd.Timestamp("2023-01-01"), split_end=pd.Timestamp("2025-01-01"))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "STOP"
    assert trade["exit_time"] == start
    assert np.isclose(trade["exit_price"], 97.0)


def test_r22_later_gap_through_stop_uses_worse_open() -> None:
    start = pd.Timestamp("2023-01-02 01:00")
    bars = _bars(start)
    gap = start + pd.Timedelta(minutes=60)
    bars.loc[gap, ["open", "high", "low", "close"]] = [96.0, 97.0, 95.0, 96.0]
    trades = simulate_catchup(bars, _event(start), target_r=1.0, direction=1, split="discovery", split_start=pd.Timestamp("2023-01-01"), split_end=pd.Timestamp("2025-01-01"))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "STOP"
    assert np.isclose(trade["exit_price"], 96.0)


def test_r22_timeout_uses_boundary_open() -> None:
    start = pd.Timestamp("2023-01-02 01:00")
    bars = _bars(start, periods=24 * 60 + 2)
    deadline = start + pd.Timedelta(hours=24)
    bars.loc[deadline, "open"] = 101.0
    trades = simulate_catchup(bars, _event(start), target_r=2.0, direction=1, split="discovery", split_start=pd.Timestamp("2023-01-01"), split_end=pd.Timestamp("2025-01-01"))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "TIME_EXIT"
    assert trade["exit_time"] == deadline
    assert np.isclose(trade["exit_price"], 101.0)

