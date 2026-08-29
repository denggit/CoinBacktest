from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.rl_market_agent.opportunity import TradeTemplate
from src.ai_research.rl_market_agent.strategy import evaluate_trades, replay_strategy


def test_same_minute_tp_and_sl_is_conservative_stop():
    idx = pd.date_range("2025-01-01 00:00", periods=10, freq="1min")
    path = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [102.0] + [100.2] * 9,
        "low": [98.0] + [99.8] * 9,
        "close": [100.0] * 10,
    }, index=idx)
    trades = replay_strategy(
        decision_times_ns=np.array([idx[0].value], dtype=np.int64),
        long_scores=np.array([0.02]), short_scores=np.array([-0.01]),
        path_1m=path, template=TradeTemplate("T", 5, 0.01, 0.005),
        long_threshold=0.001, short_threshold=0.001,
        round_trip_cost=0.0011,
    )
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "SL"
    assert bool(trades.iloc[0]["same_bar_both_hit"]) is True
    assert np.isclose(trades.iloc[0]["gross_price_return"], -0.005)


def test_metrics_include_priority_fields():
    trades = pd.DataFrame([
        {
            "entry_time": "2025-01-01 00:00", "exit_time": "2025-01-01 01:00",
            "gross_price_return": 0.01, "notional_multiple": 1.0,
            "same_bar_both_hit": False,
        },
        {
            "entry_time": "2025-01-03 00:00", "exit_time": "2025-01-03 01:00",
            "gross_price_return": -0.005, "notional_multiple": 1.0,
            "same_bar_both_hit": False,
        },
    ])
    m = evaluate_trades(trades, start="2025-01-01", end_exclusive="2025-01-05", round_trip_cost=0.001)
    for key in ("max_flat_days", "max_consecutive_losing_days", "max_drawdown_pct", "cagr_pct", "total_return_pct"):
        assert key in m
    assert m["max_flat_days"] >= 1.0


def test_entry_delay_uses_later_open_without_changing_signal_time():
    idx = pd.date_range("2025-01-01 00:00", periods=10, freq="1min")
    path = pd.DataFrame({
        "open": np.arange(100.0, 110.0),
        "high": np.arange(100.0, 110.0) + 0.1,
        "low": np.arange(100.0, 110.0) - 0.1,
        "close": np.arange(100.0, 110.0),
    }, index=idx)
    trades = replay_strategy(
        decision_times_ns=np.array([idx[0].value], dtype=np.int64),
        long_scores=np.array([0.02]), short_scores=np.array([-0.01]),
        path_1m=path, template=TradeTemplate("T", 3, 0.50, 0.50),
        long_threshold=0.001, short_threshold=0.001,
        round_trip_cost=0.0011, entry_delay_minutes=2,
    )
    assert trades.iloc[0]["signal_time"] == str(idx[0])
    assert trades.iloc[0]["entry_time"] == str(idx[2])
    assert np.isclose(trades.iloc[0]["entry_price"], 102.0)


def test_mdd_includes_intratrade_mae_not_only_exit_equity():
    trades = pd.DataFrame([{
        "entry_time": "2025-01-01 00:00", "exit_time": "2025-01-01 01:00",
        "gross_price_return": 0.01, "mfe_price_return": 0.02, "mae_price_return": -0.01,
        "notional_multiple": 1.0, "same_bar_both_hit": False,
    }])
    m = evaluate_trades(trades, start="2025-01-01", end_exclusive="2025-01-02", round_trip_cost=0.001)
    assert m["total_return_pct"] > 0
    assert m["max_drawdown_pct"] >= 1.0
