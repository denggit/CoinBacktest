from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research_common.swing_liquidity_zone_study.config import ZoneStudyConfig
from src.research_common.swing_liquidity_zone_study.controls import build_matched_controls
from src.research_common.swing_liquidity_zone_study.features import (
    attach_causal_market_features,
    build_causal_market_feature_frame,
)
from src.research_common.swing_liquidity_zone_study.outcomes import attach_structural_path_outcomes
from src.research_common.swing_liquidity_zone_study.reports import causal_audit
from src.research_common.swing_liquidity_zone_study.zones import build_sweep_zone_events


def _bars(periods: int = 600, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    x = np.arange(periods, dtype=float)
    close = 100.0 + np.sin(x / 31.0) * 0.5 + np.sin(x / 7.0) * 0.1
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "notional": 1_000.0, "trades_count": 10.0,
            "buy_notional": 500.0, "sell_notional": 500.0, "delta_notional": 0.0,
        },
        index=index,
    )


def _level(level_id: int, price: float, sweep_pos: int, timeframe: str = "1H", minutes: int = 60) -> dict[str, object]:
    event_time = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=sweep_pos + 1)
    return {
        "level_id": level_id,
        "source_timeframe": timeframe,
        "source_timeframe_min": minutes,
        "pivot_time": pd.Timestamp("2023-12-31"),
        "initial_available_time": event_time - pd.Timedelta(hours=2),
        "level_price": price,
        "sweep_pos": sweep_pos,
        "sweep_available_time": event_time,
        "age_minutes_at_sweep": 120.0,
        "touch_episode_count_before_sweep": 0,
        "approach_episode_count_before_sweep": 1,
        "confirmed_order_at_sweep": 2,
        "pivot_range_bp": 50.0,
        "pivot_close_location": 0.7,
        "pivot_lower_wick_fraction": 0.3,
        "left_high_range_20_bp": 200.0,
        "confirmation_reaction_close_bp": 80.0,
        "pivot_notional_vs_past20": 1.2,
        "pivot_trades_count_vs_past20": 1.1,
        "pivot_delta_ratio": -0.2,
    }


def test_same_bar_near_levels_merge_but_distant_level_stays_separate() -> None:
    bars = _bars()
    pos = 200
    bars.iloc[pos, bars.columns.get_loc("low")] = 98.0
    lifecycle = pd.DataFrame([
        _level(1, 100.00, pos, "15m", 15),
        _level(2, 100.05, pos, "1H", 60),
        _level(3, 102.00, pos, "4H", 240),
    ])
    zones = build_sweep_zone_events(lifecycle, bars, ZoneStudyConfig(zone_merge_tolerance_bp=10.0))
    assert len(zones) == 2
    assert sorted(zones["zone_member_count"].tolist()) == [1, 2]
    merged = zones.loc[zones["zone_member_count"].eq(2)].iloc[0]
    assert merged["zone_timeframe_count"] == 2
    assert bool(merged["zone_has_1H"])


def test_zone_rejects_member_not_available_by_closed_sweep_bar() -> None:
    bars = _bars()
    pos = 200
    bars.iloc[pos, bars.columns.get_loc("low")] = 98.0
    row = _level(1, 100.0, pos)
    row["initial_available_time"] = bars.index[pos] + pd.Timedelta(minutes=2)
    with pytest.raises(RuntimeError, match="unavailable"):
        build_sweep_zone_events(pd.DataFrame([row]), bars, ZoneStudyConfig())


def test_online_impulse_rule_only_uses_current_and_past_events() -> None:
    bars = _bars()
    for pos in (200, 203, 220):
        bars.iloc[pos, bars.columns.get_loc("low")] = 98.0
    lifecycle = pd.DataFrame([
        _level(1, 100.0, 200),
        _level(2, 100.1, 203),
        _level(3, 100.0, 220),
    ])
    zones = build_sweep_zone_events(lifecycle, bars, ZoneStudyConfig(impulse_gap_bars=5, impulse_price_tolerance_bp=50.0))
    assert zones["is_impulse_first_event"].tolist() == [True, False, True]
    assert zones["impulse_observation_number"].tolist() == [1, 2, 1]


def test_pre_atr_is_not_contaminated_by_current_sweep_bar() -> None:
    bars = _bars(2_000)
    pos = 1_500
    cfg = ZoneStudyConfig()
    before = build_causal_market_feature_frame(bars, cfg)
    changed = bars.copy()
    changed.iloc[pos, changed.columns.get_loc("high")] += 50.0
    changed.iloc[pos, changed.columns.get_loc("low")] -= 50.0
    after = build_causal_market_feature_frame(changed, cfg)
    assert before.iloc[pos]["pre_atr_60m_abs"] == after.iloc[pos]["pre_atr_60m_abs"]
    assert before.iloc[pos]["pre_atr_240m_abs"] == after.iloc[pos]["pre_atr_240m_abs"]


def test_sweep_depth_is_normalized_by_pre_event_atr() -> None:
    bars = _bars(2_000)
    pos = 1_500
    bars.iloc[pos, bars.columns.get_loc("low")] = 98.0
    lifecycle = pd.DataFrame([_level(1, 100.0, pos)])
    cfg = ZoneStudyConfig()
    zones = build_sweep_zone_events(lifecycle, bars, cfg)
    feature_frame = build_causal_market_feature_frame(bars, cfg)
    out = attach_causal_market_features(zones, bars, cfg, feature_frame=feature_frame).iloc[0]
    expected = (100.0 - 98.0) / out["pre_atr_60m_abs"]
    assert abs(out["sweep_depth_to_pre_atr_60m"] - expected) < 1e-9


def test_outcome_enters_next_open_and_uses_high_low_for_mfe_mae() -> None:
    bars = _bars(30)
    pos = 5
    bars.iloc[pos + 1, bars.columns.get_loc("open")] = 100.0
    bars.iloc[pos + 1 : pos + 6, bars.columns.get_loc("high")] = [101, 103, 102, 104, 101]
    bars.iloc[pos + 1 : pos + 6, bars.columns.get_loc("low")] = [99, 98, 99.5, 99, 100]
    bars.iloc[pos + 5, bars.columns.get_loc("close")] = 102.0
    event = pd.DataFrame({
        "zone_event_id": ["z"], "event_kind": ["swing_zone_sweep"], "event_pos": [pos],
        "event_available_time": [bars.index[pos] + pd.Timedelta(minutes=1)],
        "zone_floor_price": [99.5], "zone_ceiling_price": [100.0], "sweep_low": [98.5],
    })
    out = attach_structural_path_outcomes(event, bars, ZoneStudyConfig(path_horizons=(5,), tp_returns=(0.01,))).iloc[0]
    assert out["entry_reference_time"] == bars.index[pos + 1]
    assert abs(out["mfe_high_5m"] - 0.04) < 1e-12
    assert abs(out["mae_low_5m"] + 0.02) < 1e-12
    assert abs(out["close_return_5m"] - 0.02) < 1e-12


def test_structural_low_break_and_tp_order_are_recorded() -> None:
    bars = _bars(40)
    pos = 5
    bars.iloc[pos + 1, bars.columns.get_loc("open")] = 100.0
    bars.iloc[pos + 1 : pos + 8, bars.columns.get_loc("high")] = [100.5, 101.2, 102.0, 101.0, 100.5, 100.0, 99.5]
    bars.iloc[pos + 1 : pos + 8, bars.columns.get_loc("low")] = [99.5, 99.2, 99.0, 98.8, 98.4, 97.9, 97.0]
    event = pd.DataFrame({
        "zone_event_id": ["z"], "event_kind": ["swing_zone_sweep"], "event_pos": [pos],
        "event_available_time": [bars.index[pos] + pd.Timedelta(minutes=1)],
        "zone_floor_price": [99.0], "zone_ceiling_price": [100.0], "sweep_low": [98.5],
    })
    out = attach_structural_path_outcomes(event, bars, ZoneStudyConfig(path_horizons=(5, 15), tp_returns=(0.01, 0.03))).iloc[0]
    assert out["first_lower_low_pos"] == pos + 5
    assert bool(out["tp_1_before_lower_low_15m"])
    assert not bool(out["tp_3_before_lower_low_15m"])
    assert not bool(out["structural_low_survival_5m"])


def test_slow_multi_hour_rise_without_lower_low_is_preserved() -> None:
    bars = _bars(800)
    pos = 100
    bars.iloc[pos + 1, bars.columns.get_loc("open")] = 100.0
    rising = np.linspace(100.0, 110.0, 500)
    bars.iloc[pos + 1 : pos + 501, bars.columns.get_loc("high")] = rising + 0.1
    bars.iloc[pos + 1 : pos + 501, bars.columns.get_loc("low")] = rising - 0.2
    bars.iloc[pos + 1 : pos + 501, bars.columns.get_loc("close")] = rising
    event = pd.DataFrame({
        "zone_event_id": ["z"], "event_kind": ["swing_zone_sweep"], "event_pos": [pos],
        "event_available_time": [bars.index[pos] + pd.Timedelta(minutes=1)],
        "zone_floor_price": [99.0], "zone_ceiling_price": [100.0], "sweep_low": [98.0],
    })
    out = attach_structural_path_outcomes(event, bars, ZoneStudyConfig(path_horizons=(60, 360), tp_returns=(0.05,))).iloc[0]
    assert bool(out["structural_low_survival_360m"])
    assert out["first_lower_low_pos"] == -1
    assert out["mfe_before_lower_low_360m"] > 0.07
    assert bool(out["tp_5_before_lower_low_360m"])


def test_microsecond_index_keeps_next_open_timing() -> None:
    bars = _bars(100)
    bars.index = pd.DatetimeIndex(bars.index.to_numpy(dtype="datetime64[us]"))
    event = pd.DataFrame({
        "zone_event_id": ["z"], "event_kind": ["swing_zone_sweep"], "event_pos": [10],
        "event_available_time": [pd.Timestamp(bars.index[10]) + pd.Timedelta(minutes=1)],
        "zone_floor_price": [99.0], "zone_ceiling_price": [100.0], "sweep_low": [98.0],
    })
    out = attach_structural_path_outcomes(event, bars, ZoneStudyConfig(path_horizons=(5,), tp_returns=(0.01,))).iloc[0]
    assert out["entry_reference_time"] == pd.Timestamp(bars.index[11])


def test_matched_controls_exclude_every_raw_sweep_neighbourhood() -> None:
    bars = _bars(12_000)
    cfg = ZoneStudyConfig(path_horizons=(5, 60), control_exclusion_bars=5, control_min_downside_atr=0.05)
    # Create many similar downside bars so an exact bucket control exists.
    for pos in range(2_000, 10_000, 300):
        bars.iloc[pos, bars.columns.get_loc("low")] -= 0.5
    feature_frame = build_causal_market_feature_frame(bars, cfg)
    zone_pos = 5_000
    zone = pd.DataFrame({
        "zone_event_id": ["SZ_1"], "event_kind": ["swing_zone_sweep"], "event_pos": [zone_pos],
        "event_bar_time": [bars.index[zone_pos]], "event_available_time": [bars.index[zone_pos] + pd.Timedelta(minutes=1)],
        "zone_floor_price": [float(bars["low"].iloc[zone_pos])], "zone_ceiling_price": [float(bars["low"].iloc[zone_pos])],
        "zone_center_price": [float(bars["low"].iloc[zone_pos])], "sweep_low": [float(bars["low"].iloc[zone_pos])],
        "is_impulse_first_event": [True], "zone_primary_timeframe": ["1H"], "zone_max_timeframe_min": [60],
    })
    zone = attach_causal_market_features(zone, bars, cfg, feature_frame=feature_frame)
    lifecycle = pd.DataFrame({"sweep_pos": [zone_pos, 6_000]})
    controls = build_matched_controls(
        zone, lifecycle, bars, cfg,
        research_start=bars.index[1_500], research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        feature_frame=feature_frame,
    )
    assert not controls.empty
    for pos in controls["event_pos"].astype(int):
        assert abs(pos - zone_pos) > 5
        assert abs(pos - 6_000) > 5


def test_feature_label_tables_have_no_outcome_leakage() -> None:
    script_path = Path(__file__).resolve().parents[2] / "research" / "liquidity" / "03_swing_liquidity_zone_sweep_mechanism.py"
    spec = importlib.util.spec_from_file_location("r03_script", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bars = _bars(100)
    event = pd.DataFrame({
        "zone_event_id": ["z"], "event_kind": ["swing_zone_sweep"], "event_pos": [10],
        "event_available_time": [bars.index[10] + pd.Timedelta(minutes=1)],
        "zone_latest_level_available_time": [bars.index[5]],
        "zone_floor_price": [99.0], "zone_ceiling_price": [100.0], "sweep_low": [98.0],
        "causal_feature": [1.0],
    })
    outcomes = attach_structural_path_outcomes(event, bars, ZoneStudyConfig(path_horizons=(5,), tp_returns=(0.01,)))
    features, labels = module._feature_label_split(outcomes)
    assert "causal_feature" in features.columns
    assert not any("mfe" in c.lower() or "mae" in c.lower() or c.startswith("close_return_") for c in features.columns)
    audit = causal_audit(features, labels)
    assert int(audit["violations"].sum()) == 0
