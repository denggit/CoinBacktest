from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "eth_market_process_portfolio" / "integration" / "01_environment_conditioned_strategy_lab.py"
spec = importlib.util.spec_from_file_location("environment_strategy_r02", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_range_context_uses_only_completed_range_bars() -> None:
    minute_index = pd.date_range("2024-01-01 00:00:00", periods=6, freq="1min")
    range_bars = pd.DataFrame(
        {
            "bar_id": [1, 2, 3],
            "end_ts": pd.to_datetime(
                ["2024-01-01 00:00:30", "2024-01-01 00:02:10", "2024-01-01 00:04:00"]
            ),
            "duration_seconds": [30.0, 60.0, 45.0],
            "delta_notional": [10.0, -20.0, 30.0],
            "notional": [100.0, 100.0, 100.0],
            "direction": [1.0, -1.0, 1.0],
        }
    )

    out = module.build_range_context(minute_index, range_bars)

    assert pd.isna(out.loc[minute_index[0], "range_available_time"])
    assert out.loc[minute_index[1], "range_available_time"] == pd.Timestamp("2024-01-01 00:00:30")
    assert out.loc[minute_index[2], "range_available_time"] == pd.Timestamp("2024-01-01 00:00:30")
    assert out.loc[minute_index[3], "range_available_time"] == pd.Timestamp("2024-01-01 00:02:10")
    assert bool(out.loc[minute_index[3], "range_context_causal"])
    assert (pd.to_datetime(out["range_available_time"].dropna()) <= out.index[out["range_available_time"].notna()]).all()


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [100.2] * 10,
            "low": [99.8] * 10,
            "close": [100.0] * 10,
        },
        index=idx,
    )


def _candidate(signal_time: pd.Timestamp) -> dict[str, object]:
    return {
        "signal_time": signal_time,
        "definition": "base",
        "family": "compression_breakout",
        "environment": "compression",
        "side": 1,
        "structural_stop": 99.0,
        "target_reference": 100.0,
        "range_context_causal": True,
    }


def test_next_open_and_pessimistic_same_bar_stop_first() -> None:
    bars = _bars()
    # Signal at bar 0, entry at bar 1 open. Both stop (99) and 1.8R target
    # (101.8) are touched in the entry bar; replay must assume stop first.
    bars.iloc[1, bars.columns.get_loc("low")] = 98.5
    bars.iloc[1, bars.columns.get_loc("high")] = 102.0
    market = module.build_bar_arrays(bars)
    trade = module.simulate_candidate(
        _candidate(bars.index[0]),
        market,
        module.Scenario("base"),
        module.LabConfig(),
    )

    assert trade is not None
    assert trade["entry_time"] == bars.index[1]
    assert trade["exit_time"] == bars.index[1]
    assert trade["exit_reason"] == "same_bar_stop_first"
    assert bool(trade["same_bar_ambiguous"])
    assert bool(trade["causal_entry"])
    assert abs(float(trade["gross_return"]) + 0.01) < 1e-12
    assert abs(float(trade["net_return"]) + 0.0111) < 1e-12


def test_delay_scenario_moves_entry_forward() -> None:
    bars = _bars()
    bars.iloc[4, bars.columns.get_loc("high")] = 102.0
    market = module.build_bar_arrays(bars)
    trade = module.simulate_candidate(
        _candidate(bars.index[0]),
        market,
        module.Scenario("delay_1m", delay_bars=1),
        module.LabConfig(),
    )

    assert trade is not None
    assert trade["entry_time"] == bars.index[2]


def test_cooldown_removes_side_conflicts_and_near_duplicates() -> None:
    ts = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:20:00",
        ]
    )
    candidates = pd.DataFrame(
        {
            "signal_time": ts,
            "definition": ["base"] * 4,
            "family": ["compression_breakout"] * 4,
            "side": [1, -1, 1, 1],
        }
    )

    out = module.apply_cooldown(candidates, 15)

    # The simultaneous long/short pair is removed.  00:05 is then the first
    # valid event and 00:20 is exactly one cooldown later.
    assert list(out["signal_time"]) == [pd.Timestamp("2024-01-01 00:05:00"), pd.Timestamp("2024-01-01 00:20:00")]


def test_nonoverlap_is_enforced_across_chunks() -> None:
    trades = pd.DataFrame(
        {
            "scenario": ["base", "base", "base"],
            "definition": ["base", "base", "base"],
            "family": ["compression_breakout"] * 3,
            "signal_time": pd.to_datetime(["2024-12-31 23:58", "2025-01-01 00:01", "2025-01-01 00:11"]),
            "entry_time": pd.to_datetime(["2024-12-31 23:59", "2025-01-01 00:02", "2025-01-01 00:12"]),
            "exit_time": pd.to_datetime(["2025-01-01 00:10", "2025-01-01 00:05", "2025-01-01 00:20"]),
        }
    )

    out = module.enforce_nonoverlap(trades)

    assert list(out["entry_time"]) == [pd.Timestamp("2024-12-31 23:59"), pd.Timestamp("2025-01-01 00:12")]


def test_compression_environment_can_emit_breakout_candidate() -> None:
    idx = pd.date_range("2024-01-01", periods=220, freq="1min")
    n = len(idx)
    frame = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.1),
            "low": np.full(n, 99.9),
            "close": np.full(n, 100.0),
            "vol_ratio_30": np.full(n, 1.0),
            "range_count_ratio_15m": np.full(n, 1.0),
            "range_available_time": idx,
            "range_context_causal": np.full(n, True),
            "range_delta_ratio": np.zeros(n),
            "delta_ratio_2": np.zeros(n),
            "delta_ratio_3": np.zeros(n),
            "large_delta_ratio_3": np.zeros(n),
            "notional_ratio_base": np.ones(n),
            "buy_notional_ratio_base": np.ones(n),
            "sell_notional_ratio_base": np.ones(n),
            "price_return_3": np.zeros(n),
            "efficiency_15": np.full(n, 0.5),
            "efficiency_30": np.full(n, 0.5),
            "efficiency_60": np.full(n, 0.5),
            "width_30": np.full(n, 0.006),
            "width_60": np.full(n, 0.012),
            "return_15": np.zeros(n),
            "prior_high_15": np.full(n, 100.1),
            "prior_low_15": np.full(n, 99.9),
            "prior_high_30": np.full(n, 100.0),
            "prior_low_30": np.full(n, 99.5),
            "prior_high_60": np.full(n, 100.5),
            "prior_low_60": np.full(n, 99.5),
            "mid_60": np.full(n, 100.0),
            "ema_60": np.full(n, 100.0),
            "close_pos": np.full(n, 0.5),
            "lower_wick_frac": np.zeros(n),
            "upper_wick_frac": np.zeros(n),
            "down_move_norm": np.zeros(n),
            "up_move_norm": np.zeros(n),
            "delta_reversal_short": np.zeros(n),
        },
        index=idx,
    )
    signal_pos = 190
    # The prior ten bars are a completed compression state.
    frame.iloc[signal_pos - 10 : signal_pos, frame.columns.get_loc("vol_ratio_30")] = 0.60
    frame.iloc[signal_pos - 10 : signal_pos, frame.columns.get_loc("range_count_ratio_15m")] = 0.60
    frame.iloc[signal_pos - 10 : signal_pos, frame.columns.get_loc("efficiency_30")] = 0.20
    frame.iloc[signal_pos, frame.columns.get_loc("close")] = 100.10
    frame.iloc[signal_pos, frame.columns.get_loc("high")] = 100.12
    frame.iloc[signal_pos, frame.columns.get_loc("delta_ratio_3")] = 0.25
    frame.iloc[signal_pos, frame.columns.get_loc("large_delta_ratio_3")] = 0.08
    frame.iloc[signal_pos, frame.columns.get_loc("notional_ratio_base")] = 2.0
    frame.iloc[signal_pos, frame.columns.get_loc("price_return_3")] = 0.002
    frame.iloc[signal_pos, frame.columns.get_loc("range_delta_ratio")] = 0.10

    candidates = module.build_strategy_candidates(frame, next(x for x in module.DEFINITIONS if x.name == "base"))

    found = candidates[
        (candidates["signal_time"] == idx[signal_pos])
        & (candidates["family"] == "compression_breakout")
        & (candidates["side"] == 1)
    ]
    assert len(found) == 1


def test_range_context_accepts_loader_style_end_ts_index_and_column() -> None:
    minute_index = pd.date_range("2024-01-01", periods=3, freq="1min")
    range_bars = pd.DataFrame(
        {
            "bar_id": [1, 2],
            "end_ts": pd.to_datetime(["2024-01-01 00:00:30", "2024-01-01 00:01:30"]),
            "duration_seconds": [30.0, 60.0],
            "delta_notional": [10.0, -5.0],
            "notional": [100.0, 100.0],
            "direction": [1.0, -1.0],
        }
    ).set_index("end_ts", drop=False)
    range_bars.index.name = "end_ts"

    out = module.build_range_context(minute_index, range_bars)

    assert out.loc[minute_index[1], "range_available_time"] == pd.Timestamp("2024-01-01 00:00:30")
