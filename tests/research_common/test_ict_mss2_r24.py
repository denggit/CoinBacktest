#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r24 import simulate_funding_unwind


def _bars(start: pd.Timestamp, periods: int = 500) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    return pd.DataFrame(index=index, data={"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0})


def _event(entry: pd.Timestamp, direction: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id":"R24_EVENT_000001", "trade_direction":direction,
        "event_bar_time":entry-pd.Timedelta(hours=1), "signal_available_time":entry, "entry_time":entry,
        "return_1h":-0.02*direction, "pre_settlement_z":-2.0*direction, "atr20":2.0,
    }])


def test_r24_stop_first_on_ambiguous_entry_bar() -> None:
    entry=pd.Timestamp("2023-01-02 08:00"); bars=_bars(entry)
    bars.loc[entry,["high","low"]]=[104.0,96.0]
    trade=simulate_funding_unwind(bars,_event(entry,1),target_r=1.0,direction=1,split="discovery",split_start=pd.Timestamp("2023-01-01"),split_end=pd.Timestamp("2025-01-01")).iloc[0]
    assert trade["exit_reason"]=="STOP"
    assert np.isclose(trade["exit_price"],97.0)


def test_r24_time_exit_at_next_schedule_open() -> None:
    entry=pd.Timestamp("2023-01-02 08:00"); bars=_bars(entry,periods=8*60+2)
    deadline=entry+pd.Timedelta(hours=8); bars.loc[deadline,"open"]=101.0
    trade=simulate_funding_unwind(bars,_event(entry,1),target_r=2.0,direction=1,split="discovery",split_start=pd.Timestamp("2023-01-01"),split_end=pd.Timestamp("2025-01-01")).iloc[0]
    assert trade["exit_reason"]=="TIME_EXIT"
    assert trade["exit_time"]==deadline
    assert np.isclose(trade["exit_price"],101.0)

