#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.liquidity_magnet import (
    LiquidityMagnetConfig,
    attach_risk_frontier_outcomes,
    build_liquidity_magnet_universe,
    causal_audit,
)
from src.research_common.structured_stop_pool import FAMILY_COLUMNS


def _bars(periods: int = 500) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="1min")
    x = np.arange(periods, dtype=float)
    close = 105.0 - 0.015 * x + 0.3 * np.sin(x / 9.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.08
    low = np.minimum(open_, close) - 0.08
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=index,
    )


def _lifecycle() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "level_id": [1, 2],
            "source_timeframe": ["15m", "1H"],
            "source_timeframe_min": [15, 60],
            "level_price": [100.00, 99.95],
            "pivot_time": pd.to_datetime(["2023-01-01 00:30", "2023-01-01 00:45"]),
            "initial_available_time": pd.to_datetime(["2023-01-01 01:00", "2023-01-01 01:15"]),
            "active_pos": [60, 75],
            "sweep_pos": [350, 350],
            "sweep_available_time": pd.to_datetime(["2023-01-01 05:51", "2023-01-01 05:51"]),
            "confirmed_order_at_sweep": [2, 3],
            "confirmation_reaction_close_bp": [80.0, 100.0],
            "confirmation_reaction_high_bp": [120.0, 180.0],
            "left_high_range_20_bp": [220.0, 300.0],
            "left_low_gap_20_bp": [40.0, 50.0],
            "pivot_notional_vs_past20": [1.2, 1.5],
            "pivot_trades_count_vs_past20": [1.1, 1.4],
        }
    )


def _features() -> pd.DataFrame:
    out = pd.DataFrame({"level_id": [1, 2]})
    for i, family in enumerate(FAMILY_COLUMNS):
        out[family] = [i == 0, i in (0, 5)]
    return out


def test_candidate_is_next_open_and_members_are_already_available() -> None:
    bars = _bars()
    cfg = LiquidityMagnetConfig(distance_bands_bp=(100.0,), horizon_minutes=60).validate()
    candidates = build_liquidity_magnet_universe(
        _lifecycle(),
        _features(),
        bars,
        cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    assert not candidates.empty
    assert pd.to_datetime(candidates["entry_time"]).equals(pd.to_datetime(candidates["event_available_time"]))
    assert (
        pd.to_datetime(candidates["pool_member_initial_available_time_max"])
        <= pd.to_datetime(candidates["event_available_time"])
    ).all()
    assert (candidates["front_run_target_price"] < candidates["entry_price"]).all()


def test_same_bar_target_and_stop_is_conservative_stop() -> None:
    index = pd.date_range("2023-01-01", periods=20, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=index)
    bars.loc[index[5], "high"] = 101.5
    bars.loc[index[5], "low"] = 98.5
    candidate = pd.DataFrame(
        {
            "pool_event_id": ["X"],
            "distance_band_bp": [100.0],
            "event_available_time": [index[5]],
            "entry_pos": [5],
            "entry_time": [index[5]],
            "entry_price": [100.0],
            "front_run_target_price": [99.0],
            "stop_equal_distance": [101.0],
            "stop_local_high_15m": [101.0],
            "stop_local_high_60m": [101.0],
            "period": ["EARLY_2023_2024"],
        }
    )
    out = attach_risk_frontier_outcomes(
        candidate,
        bars,
        LiquidityMagnetConfig(distance_bands_bp=(100.0,), horizon_minutes=10),
        show_progress=False,
    )
    assert len(out) == 3
    assert out["outcome"].eq("STOP_CONSERVATIVE_SAME_BAR").all()
    assert out["net_return_1x_cost"].lt(0).all()


def test_equal_distance_target_rate_replay_has_no_invalid_rows() -> None:
    bars = _bars()
    cfg = LiquidityMagnetConfig(distance_bands_bp=(100.0, 50.0), horizon_minutes=120).validate()
    candidates = build_liquidity_magnet_universe(
        _lifecycle(),
        _features(),
        bars,
        cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    outcomes = attach_risk_frontier_outcomes(candidates, bars, cfg, show_progress=False)
    assert len(outcomes) == len(candidates) * 3
    assert not outcomes["outcome"].eq("INVALID").any()
    assert outcomes.loc[outcomes["stop_model"].eq("EQUAL_DISTANCE"), "nominal_reward_risk"].round(8).eq(1.0).all()


def test_microsecond_datetime_precision_does_not_shift_entry() -> None:
    bars = _bars()
    lifecycle = _lifecycle()
    lifecycle["initial_available_time"] = lifecycle["initial_available_time"].to_numpy(dtype="datetime64[us]")
    cfg = LiquidityMagnetConfig(distance_bands_bp=(100.0,), horizon_minutes=60).validate()
    candidates = build_liquidity_magnet_universe(
        lifecycle,
        _features(),
        bars,
        cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    audit = causal_audit(candidates, attach_risk_frontier_outcomes(candidates, bars, cfg, show_progress=False))
    assert audit.loc[audit["status"].eq("FAIL")].empty


def test_candidate_crossing_before_missing_minute_is_discarded() -> None:
    bars = _bars()
    # Remove the strict next minute after a known candidate signal.  A later
    # observed bar must not be treated as the next-open execution because the
    # intervening path is unknown.
    cfg = LiquidityMagnetConfig(distance_bands_bp=(100.0,), horizon_minutes=60).validate()
    baseline = build_liquidity_magnet_universe(
        _lifecycle(), _features(), bars, cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    assert not baseline.empty
    signal_bar = pd.Timestamp(baseline.iloc[0]["event_bar_time"])
    missing_next = signal_bar + pd.Timedelta(minutes=1)
    gapped = bars.drop(index=missing_next)
    candidates = build_liquidity_magnet_universe(
        _lifecycle(), _features(), gapped, cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    if not candidates.empty:
        assert (pd.to_datetime(candidates["entry_time"]) == pd.to_datetime(candidates["event_available_time"])).all()
        assert not (pd.to_datetime(candidates["event_bar_time"]) == signal_bar).any()
