from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.ict.broad_position_management import (
    BroadPositionManagementConfig,
    SCENARIOS,
    replay_position_scenario,
    replay_position_scenarios,
    select_discovery_policy,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ


def _bars(values: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-02 09:30", periods=len(values), freq="1min", tz=NY_TZ)
    return pd.DataFrame(values, columns=["open", "high", "low", "close"], index=idx).assign(volume=1000.0)


def _trade(fill_time: pd.Timestamp) -> dict[str, object]:
    return {
        "ny_date": "2026-06-02", "path_event_id": "p1", "event_id": "p1",
        "range_model": "prominent_15m_pair_0830", "entry_archetype": "mss_first_visible_any_break_next_open_market",
        "execution_tf": "1m", "execution_tf_minutes": 1, "trade_side": "LONG", "entry_order_type": "market_next_open",
        "entry_available_time": fill_time, "entry_price": 100.0, "entry_price_replay": 100.0,
        "fill_time": fill_time, "filled": True, "stop_price": 99.0, "target_price": 104.0,
    }


def test_same_bar_stop_beats_partial_and_target():
    bars = _bars([(100.0, 104.5, 98.8, 103.0), (103.0, 104.5, 102.5, 104.0)])
    t = _trade(bars.index[0])
    out = replay_position_scenario(bars, t, pd.DataFrame(), scenario="partial25_1r_be", round_trip_cost=0.0011)
    assert out["management_exit_reason"] == "initial_stop"
    assert not out["partial_taken"]


def test_becomes_protective_only_on_next_bar():
    # First bar reaches +1R but closes safely; second bar can then hit BE.
    bars = _bars([(100.0, 101.2, 99.2, 100.8), (100.8, 100.9, 99.9, 100.1), (100.1, 104.2, 100.0, 104.0)])
    t = _trade(bars.index[0])
    out = replay_position_scenario(bars, t, pd.DataFrame(), scenario="be_after_1r", round_trip_cost=0.0011)
    assert out["management_exit_reason"] == "protective_stop"
    assert abs(float(out["management_exit_price"]) - 100.0) < 1e-12


def test_all_management_scenarios_preserve_base_trade_count():
    bars = _bars([(100.0, 100.5, 99.5, 100.2), (100.2, 104.2, 100.0, 104.0)])
    t = pd.DataFrame([_trade(bars.index[0])])
    out = replay_position_scenarios(bars, t, round_trip_cost=0.0011, cost_multipliers=(1.0,), delays=(0,), scenarios=SCENARIOS)
    assert len(out) == len(SCENARIOS)
    assert out["managed"].all()


def test_discovery_selector_ignores_oos_rows_and_prefers_stability():
    rows = [
        {"management_scenario": "a", "period": "discovery_2023H2_2024", "cost_multiple": 1.0, "entry_delay_minutes": 0, "trades_per_session": 0.8, "mean_net_return": 0.001, "profit_factor": 1.2, "longest_no_trade_sessions": 2, "max_consecutive_losses": 5, "max_drawdown": -0.10, "cagr": 0.2, "total_return": 0.4},
        {"management_scenario": "b", "period": "discovery_2023H2_2024", "cost_multiple": 1.0, "entry_delay_minutes": 0, "trades_per_session": 0.8, "mean_net_return": 0.001, "profit_factor": 1.2, "longest_no_trade_sessions": 2, "max_consecutive_losses": 3, "max_drawdown": -0.20, "cagr": 0.5, "total_return": 1.0},
        {"management_scenario": "a", "period": "validation_2025", "cost_multiple": 1.0, "entry_delay_minutes": 0, "trades_per_session": 0.8, "mean_net_return": -0.5, "profit_factor": 0.1, "longest_no_trade_sessions": 99, "max_consecutive_losses": 99, "max_drawdown": -0.9, "cagr": -0.9, "total_return": -0.9},
    ]
    selected = select_discovery_policy(pd.DataFrame(rows), minimum_trades_per_session=0.5)
    assert selected["selected_policy"] == "b"


def test_r20_script_self_test():
    root = Path(__file__).resolve().parents[3]
    script = root / "research" / "ict" / "soxl_premarket_mss_fvg" / "20_broad_position_management_backtest.py"
    spec = importlib.util.spec_from_file_location("soxl_r20", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--self-test"]) == 0


def test_broad_signal_loader_keeps_wick_only_and_earliest_cross_tf(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "research" / "ict" / "soxl_premarket_mss_fvg" / "20_broad_position_management_backtest.py"
    spec = importlib.util.spec_from_file_location("soxl_r20_loader", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cache = tmp_path / "r15"; cache.mkdir()
    pd.DataFrame([
        {"ny_date":"2026-06-02","range_model":mod.DEFAULT_RANGE_MODEL,"path_event_id":"p1","first_raid_side":"low","first_raid_time":"2026-06-02T13:31:00Z","target_price":104.0,"source_level_price":99.0},
        {"ny_date":"2026-06-03","range_model":mod.DEFAULT_RANGE_MODEL,"path_event_id":"p2","first_raid_side":"high","first_raid_time":"2026-06-03T13:31:00Z","target_price":96.0,"source_level_price":101.0},
    ]).to_csv(cache / "03_daily_path_outcomes.csv", index=False)
    pd.DataFrame([
        # p1: 2m appears first and is wick-only. It must be retained instead of filtered.
        {"ny_date":"2026-06-02","range_model":mod.DEFAULT_RANGE_MODEL,"path_event_id":"p1","event_id":"p1","trade_side":"LONG","execution_tf":"2m","execution_tf_minutes":2,"break_available_time":"2026-06-02T13:34:00Z","break_wick_cross":True,"break_close_cross":False,"terminal_extreme_time":"2026-06-02T13:32:00Z","terminal_extreme_price":99.0,"mss_reference_time":"2026-06-02T13:30:00Z","mss_reference_available_time":"2026-06-02T13:32:00Z","causal_visibility_percentile":0.7},
        {"ny_date":"2026-06-02","range_model":mod.DEFAULT_RANGE_MODEL,"path_event_id":"p1","event_id":"p1","trade_side":"LONG","execution_tf":"1m","execution_tf_minutes":1,"break_available_time":"2026-06-02T13:35:00Z","break_wick_cross":True,"break_close_cross":True,"terminal_extreme_time":"2026-06-02T13:32:00Z","terminal_extreme_price":99.0,"mss_reference_time":"2026-06-02T13:30:00Z","mss_reference_available_time":"2026-06-02T13:32:00Z","causal_visibility_percentile":0.8},
        {"ny_date":"2026-06-03","range_model":mod.DEFAULT_RANGE_MODEL,"path_event_id":"p2","event_id":"p2","trade_side":"SHORT","execution_tf":"1m","execution_tf_minutes":1,"break_available_time":"2026-06-03T13:34:00Z","break_wick_cross":True,"break_close_cross":True,"terminal_extreme_time":"2026-06-03T13:32:00Z","terminal_extreme_price":101.0,"mss_reference_time":"2026-06-03T13:30:00Z","mss_reference_available_time":"2026-06-03T13:32:00Z","causal_visibility_percentile":0.6},
    ]).to_csv(cache / "06_causal_mss_narratives.csv", index=False)
    args = mod.parse_args(["--visible-swing-percentile","0.5"])
    signals, paths = mod._load_broad_signals(args, cache)
    assert len(paths) == 2
    assert len(signals) == 2
    p1 = signals.loc[signals["path_event_id"].eq("p1")].iloc[0]
    assert str(p1["execution_tf"]) == "2m"
    assert bool(p1["break_close_cross"]) is False


def test_materialize_broad_entry_uses_next_available_open():
    root = Path(__file__).resolve().parents[3]
    script = root / "research" / "ict" / "soxl_premarket_mss_fvg" / "20_broad_position_management_backtest.py"
    spec = importlib.util.spec_from_file_location("soxl_r20_materialize", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    idx = pd.date_range("2026-06-02 09:30", periods=10, freq="1min", tz=NY_TZ)
    bars = pd.DataFrame({"open":np.arange(100.0,110.0),"high":np.arange(100.5,110.5),"low":np.arange(99.5,109.5),"close":np.arange(100.2,110.2),"volume":1000.0}, index=idx)
    signal = pd.DataFrame([{"ny_date":"2026-06-02","path_event_id":"p1","event_id":"p1","range_model":mod.DEFAULT_RANGE_MODEL,"trade_side":"LONG","execution_tf":"2m","execution_tf_minutes":2,"break_available_time":pd.Timestamp("2026-06-02 09:34",tz=NY_TZ),"break_close_cross":False,"terminal_extreme_price":99.0,"target_price":108.0}])
    out = mod._materialize_broad_market_entries(signal,bars)
    assert bool(out.iloc[0]["filled"])
    assert out.iloc[0]["fill_time"] == idx[4]
    assert float(out.iloc[0]["entry_price"]) == float(bars.iloc[4]["open"])
    assert not bool(out.iloc[0]["close_confirmed_mss"])
