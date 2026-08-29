from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r17 import (
    R17Config,
    build_first_passage_paths,
    build_structural_state,
    r17_causal_audit,
    summarize_path_models,
)


def _bars(n: int = 120, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="1min")
    price = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.1,
            "low": price - 0.1,
            "close": price,
            "volume": 1.0,
        },
        index=index,
    )


def _setup(direction: str = "LONG", setup_id: str = "s1") -> pd.DataFrame:
    sign = 1 if direction == "LONG" else -1
    entry = 100.0
    stop = 99.0 if sign > 0 else 101.0
    target = 102.0 if sign > 0 else 98.0
    return pd.DataFrame(
        [
            {
                "setup_id": setup_id,
                "direction": direction,
                "trade_direction": sign,
                "research_split": "discovery",
                "setup_status": "executable",
                "pullback_available_time": pd.Timestamp("2024-01-01 00:05"),
                "reclaim_15m_available_time": pd.Timestamp("2024-01-01 00:08"),
                "signal_available_time": pd.Timestamp("2024-01-01 00:10"),
                "entry_time": pd.Timestamp("2024-01-01 00:10"),
                "entry_price": entry,
                "stop_price": stop,
                "risk_distance_pct": 0.01,
                "structural_target_price": target,
                "structural_target_available_time": pd.Timestamp("2024-01-01 00:00"),
                "structural_runway_pct": 0.02,
                "structural_reward_risk": 2.0,
                "trend_1d_available_time_at_pullback": pd.Timestamp("2024-01-01 00:00"),
                "trend_4h_available_time_at_pullback": pd.Timestamp("2024-01-01 00:00"),
                "trend_1d_available_time_at_signal": pd.Timestamp("2024-01-01 00:00"),
                "trend_4h_available_time_at_signal": pd.Timestamp("2024-01-01 00:00"),
                "trend_1d_age_bars_at_signal": 3.0,
                "trend_4h_age_bars_at_signal": 5.0,
                "pullback_bar_range_atr": 0.8,
                "signal_delay_minutes": 5.0,
            }
        ]
    )


def test_r17_predeclared_stop_ceiling_is_validated():
    with pytest.raises(ValueError):
        R17Config(max_stop_distance_pct=0.0).validate()


def test_structural_pivot_is_available_only_after_right_bar_closes():
    bars = _bars(10)
    bars["high"] = [100.0, 101.0, 103.0, 102.0, 101.0, 102.0, 104.0, 103.0, 102.0, 101.0]
    bars["low"] = [99.0, 99.5, 100.0, 99.8, 98.0, 99.0, 100.0, 99.5, 98.5, 98.0]
    states, pivots = build_structural_state(
        bars, minutes=1, timeframe="1m", atr_window=2, atr_min_periods=1
    )
    high = pivots.loc[(pivots["pivot_side"] == "high") & (pivots["pivot_pos_htf"] == 2)].iloc[0]
    assert high["pivot_available_time"] == bars.index[4]
    assert states.loc[states["state_available_time"] < bars.index[4], "latest_high_price"].isna().all()
    assert float(states.loc[states["state_available_time"] == bars.index[4], "latest_high_price"].iloc[0]) == 103.0


def test_later_bar_mutation_does_not_change_already_available_structure():
    bars = _bars(12)
    bars["high"] = [100, 101, 103, 102, 101, 102, 104, 103, 102, 101, 100, 99]
    bars["low"] = [99, 99.5, 100, 99.8, 98, 99, 100, 99.5, 98.5, 98, 97, 96]
    first, _ = build_structural_state(bars, minutes=1, timeframe="1m", atr_window=2, atr_min_periods=1)
    changed = bars.copy()
    changed.loc[changed.index[10:], ["high", "low", "close"]] = [200.0, 1.0, 150.0]
    second, _ = build_structural_state(changed, minutes=1, timeframe="1m", atr_window=2, atr_min_periods=1)
    cutoff = bars.index[10]
    cols = ["structural_direction", "latest_high_price", "latest_low_price"]
    pd.testing.assert_frame_equal(
        first.loc[first["state_available_time"] <= cutoff, cols].reset_index(drop=True),
        second.loc[second["state_available_time"] <= cutoff, cols].reset_index(drop=True),
    )


def test_same_bar_target_and_stop_is_pessimistically_stop_first():
    bars = _bars()
    entry_time = pd.Timestamp("2024-01-01 00:10")
    bars.loc[entry_time, ["high", "low"]] = [103.0, 98.0]
    paths = build_first_passage_paths(bars, _setup())
    assert set(paths["outcome"]) == {"sl_first"}
    assert np.allclose(paths["gross_r"], -1.0)


def test_horizon_exit_uses_observed_close_and_cost_stress():
    bars = _bars()
    cfg = R17Config(path_horizon_minutes=20).validate()
    paths = build_first_passage_paths(bars, _setup(), config=cfg)
    r3 = paths.loc[paths["target_model"].eq("R3")].iloc[0]
    assert r3["outcome"] == "horizon_exit"
    assert float(r3["gross_return"]) == 0.0
    assert float(r3["net_return_cost2x"]) == pytest.approx(-0.0022)


def test_primary_summary_never_pools_long_and_short():
    bars = _bars()
    long_paths = build_first_passage_paths(bars, _setup("LONG", "long"))
    short_paths = build_first_passage_paths(bars, _setup("SHORT", "short"))
    summary = summarize_path_models(pd.concat([long_paths, short_paths], ignore_index=True))
    assert set(summary["direction"]) == {"LONG", "SHORT"}
    assert not summary["direction"].eq("ALL").any()


def test_r17_causal_audit_passes_for_valid_setup_and_paths():
    bars = _bars()
    setup = _setup()
    paths = build_first_passage_paths(bars, setup)
    audit = r17_causal_audit(setup, paths)
    assert int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) == 0
