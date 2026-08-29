#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r21 import (
    R21Config,
    R21Model,
    r21_causal_audit,
    simulate_daily_channel,
)


def test_daily_channel_enters_next_day_and_stops_first() -> None:
    index = pd.date_range("2023-01-01", periods=30, freq="1D")
    daily = pd.DataFrame(index=index, data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    daily["atr20"] = 2.0
    daily["entry_high_20"] = 101.0
    daily["entry_low_20"] = 99.0
    daily["exit_high_10"] = 101.0
    daily["exit_low_10"] = 99.0
    daily.loc[index[20], "close"] = 102.0
    daily.loc[index[21], ["open", "high", "low", "close"]] = [102.0, 103.0, 97.0, 100.0]
    bars = daily[["open", "high", "low", "close"]].copy()
    bars["volume"] = 1.0
    trades = simulate_daily_channel(
        bars,
        daily,
        model=R21Model("D20_X10", 20, 10),
        direction=1,
        split="discovery",
        split_start=index[0],
        split_end=index[-1] + pd.Timedelta(days=1),
    )
    closed = trades.loc[trades["path_status"].eq("included")].iloc[0]
    assert closed["entry_time"] == index[21]
    assert closed["exit_time"] == index[21]
    assert closed["exit_reason"] == "INITIAL_ATR_STOP"
    assert np.isclose(closed["entry_price"], 102.0)
    assert np.isclose(closed["exit_price"], 98.0)
    audit = r21_causal_audit(trades, config=R21Config())
    assert int(audit["violations"].sum()) == 0


def test_daily_channel_gap_through_stop_uses_worse_open() -> None:
    index = pd.date_range("2023-01-01", periods=31, freq="1D")
    daily = pd.DataFrame(index=index, data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    daily["atr20"] = 2.0
    daily["entry_high_20"] = 101.0
    daily["entry_low_20"] = 99.0
    daily["exit_high_10"] = 101.0
    daily["exit_low_10"] = 99.0
    daily.loc[index[20], "close"] = 102.0
    daily.loc[index[21], ["open", "high", "low", "close"]] = [102.0, 103.0, 100.0, 100.0]
    daily.loc[index[22], ["open", "high", "low", "close"]] = [97.0, 100.0, 96.0, 98.0]
    bars = daily[["open", "high", "low", "close"]].copy()
    bars["volume"] = 1.0
    trades = simulate_daily_channel(
        bars,
        daily,
        model=R21Model("D20_X10", 20, 10),
        direction=1,
        split="discovery",
        split_start=index[0],
        split_end=index[-1] + pd.Timedelta(days=1),
    )
    closed = trades.loc[trades["path_status"].eq("included")].iloc[0]
    assert closed["exit_reason"] == "INITIAL_ATR_STOP"
    assert np.isclose(closed["initial_stop_price"], 98.0)
    assert np.isclose(closed["exit_price"], 97.0)


def test_channel_exit_executes_before_next_day_stop_check() -> None:
    index = pd.date_range("2023-01-01", periods=31, freq="1D")
    daily = pd.DataFrame(index=index, data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    daily["atr20"] = 2.0
    daily["entry_high_20"] = 101.0
    daily["entry_low_20"] = 99.0
    daily["exit_high_10"] = 101.0
    daily["exit_low_10"] = 99.0
    daily.loc[index[20], "close"] = 102.0
    daily.loc[index[21], ["open", "high", "low", "close"]] = [102.0, 103.0, 100.0, 98.0]
    daily.loc[index[22], ["open", "high", "low", "close"]] = [97.0, 100.0, 90.0, 95.0]
    bars = daily[["open", "high", "low", "close"]].copy()
    bars["volume"] = 1.0
    trades = simulate_daily_channel(
        bars,
        daily,
        model=R21Model("D20_X10", 20, 10),
        direction=1,
        split="discovery",
        split_start=index[0],
        split_end=index[-1] + pd.Timedelta(days=1),
    )
    closed = trades.loc[trades["path_status"].eq("included")].iloc[0]
    assert closed["exit_time"] == index[22]
    assert closed["exit_reason"] == "CHANNEL_EXIT_NEXT_OPEN"
    assert np.isclose(closed["exit_price"], 97.0)
