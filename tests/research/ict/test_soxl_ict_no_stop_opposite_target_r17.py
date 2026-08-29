from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.no_stop_opposite_target import (
    NoStopReplayConfig,
    replay_no_stop_to_opposite_or_close,
    summarize_no_stop,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ


def _bars(values):
    n = len(next(iter(values.values())))
    idx = pd.date_range("2026-08-05 08:30", periods=n, freq="1min", tz=NY_TZ)
    return idx, pd.DataFrame(values, index=idx)


def test_old_stop_can_be_washout_then_opposite_tp():
    idx, bars = _bars({
        "open":  [100,100,99,98,99,101,104,105],
        "high":  [101,101,100,99,102,104,106,106],
        "low":   [99,99,97,96,98,100,103,104],
        "close": [100,99,98,98,101,103,105,105],
    })
    fills = pd.DataFrame([{
        "ny_date":"2026-08-05","filled":True,"fill_time":idx[1],"trade_side":"LONG","entry_order_type":"limit",
        "entry_price":100.0,"entry_price_replay":np.nan,"target_price":105.0,"stop_price":98.0,
        "stop_hit":True,"stop_time":idx[2],"milestone_100_before_stop":False,"net_return_exit_100":-0.0211,
    }])
    out = replay_no_stop_to_opposite_or_close(bars, fills, config=NoStopReplayConfig(round_trip_cost=0.0011))
    r = out.iloc[0]
    assert r["no_stop_exit_reason"] == "opposite_liquidity_tp"
    assert bool(r["no_stop_tp_hit"])
    assert bool(r["rescued_after_old_terminal_stop"])
    assert r["no_stop_net_return"] > 0


def test_no_tp_exits_at_session_close_and_keeps_tail_risk():
    idx, bars = _bars({
        "open":  [100,100,98,96,95],
        "high":  [101,101,99,97,96],
        "low":   [99,98,96,94,93],
        "close": [100,99,97,95,94],
    })
    fills = pd.DataFrame([{
        "ny_date":"2026-08-05","filled":True,"fill_time":idx[1],"trade_side":"LONG","entry_order_type":"limit",
        "entry_price":100.0,"entry_price_replay":np.nan,"target_price":110.0,"stop_price":98.0,
        "stop_hit":True,"stop_time":idx[1],"milestone_100_before_stop":False,"net_return_exit_100":-0.0211,
    }])
    out = replay_no_stop_to_opposite_or_close(bars, fills)
    r = out.iloc[0]
    assert r["no_stop_exit_reason"] == "session_close"
    assert not bool(r["no_stop_tp_hit"])
    assert np.isclose(r["no_stop_exit_price"], 94.0)
    assert r["no_stop_net_return"] < -0.05
    assert r["no_stop_mae_pct"] <= -0.07


def test_limit_same_bar_target_is_conservative():
    idx, bars = _bars({
        "open":  [100,100,100],
        "high":  [101,110,102],
        "low":   [99,99,99],
        "close": [100,101,101],
    })
    fills = pd.DataFrame([{
        "ny_date":"2026-08-05","filled":True,"fill_time":idx[1],"trade_side":"LONG","entry_order_type":"limit",
        "entry_price":100.0,"target_price":105.0,"stop_price":98.0,"stop_hit":False,
        "milestone_100_before_stop":True,"net_return_exit_100":0.0489,
    }])
    out = replay_no_stop_to_opposite_or_close(bars, fills)
    r = out.iloc[0]
    assert bool(r["same_bar_fill_target_ambiguous"])
    assert not bool(r["no_stop_tp_hit"])
    assert r["no_stop_exit_reason"] == "session_close"


def test_summary_reports_old_stop_rescues_and_pf():
    q = pd.DataFrame({
        "no_stop_valid":[True,True],
        "no_stop_net_return":[0.05,-0.02],
        "no_stop_tp_hit":[True,False],
        "no_stop_is_profitable":[True,False],
        "no_stop_exit_reason":["opposite_liquidity_tp","session_close"],
        "no_stop_mae_pct":[-0.01,-0.08],
        "no_stop_mae_old_r":[-0.5,-4.0],
        "old_terminal_stop_hit":[True,True],
        "rescued_after_old_terminal_stop":[True,False],
        "same_bar_fill_target_ambiguous":[False,False],
        "net_return_exit_100":[-0.02,-0.02],
        "milestone_100_before_stop":[False,False],
    })
    s = summarize_no_stop(q).iloc[0]
    assert np.isclose(s["tp_rate"], 0.5)
    assert np.isclose(s["profit_factor"], 2.5)
    assert np.isclose(s["rescued_share_of_old_stop_hits"], 0.5)
