#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.multiscale_absorption import (
    AbsorptionFeatureConfig,
    attach_forward_outcomes,
    build_absorption_features,
    extract_events,
    resample_trade_bars,
)


def _bars(n: int = 1200, *, freq: str = "1min") -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq)
    ret = rng.normal(0.0, 0.00015, n)
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.00005, 0.00025, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.00005, 0.00025, n))
    notional = rng.uniform(500_000.0, 1_500_000.0, n)
    delta = rng.normal(0.0, 0.12, n) * notional
    buy = (notional + delta) / 2.0
    sell = (notional - delta) / 2.0
    trades = rng.integers(80, 200, n)
    buy_trades = np.clip(np.round(trades * (buy / notional)), 1, trades - 1).astype(int)
    sell_trades = trades - buy_trades
    volume = notional / close
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "delta_notional": delta,
            "trades_count": trades,
            "buy_trades_count": buy_trades,
            "sell_trades_count": sell_trades,
            "buy_volume": volume * buy / notional,
            "sell_volume": volume * sell / notional,
        },
        index=idx,
    )


def _cfg() -> AbsorptionFeatureConfig:
    return AbsorptionFeatureConfig(
        process_window=5,
        baseline_bars=240,
        baseline_min_periods=120,
        floor_lookback=60,
        defense_lookback=60,
        reclaim_bars=3,
        atr_lookback=30,
    )


def test_future_perturbation_does_not_change_past_features() -> None:
    bars = _bars()
    base = build_absorption_features(bars, _cfg())
    cutoff = bars.index[950]
    changed = bars.copy()
    mask = changed.index > cutoff
    changed.loc[mask, ["open", "high", "low", "close"]] *= 1.8
    changed.loc[mask, ["notional", "buy_notional", "sell_notional", "delta_notional"]] *= 20.0
    alt = build_absorption_features(changed, _cfg())
    cols = [
        "pressure",
        "pressure_z",
        "flow_persistence",
        "price_response_norm",
        "pressure_retention",
        "prior_floor",
        "prior_ceiling",
        "prior_defense_count_long",
        "spring_reclaim_long",
        "spring_reclaim_short",
    ]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], alt.loc[:cutoff, cols])


def test_persistence_uses_current_window_flow_side() -> None:
    bars = _bars(500)
    # Last five bars have four sell-delta bars and one small buy bar. Aggregate
    # flow is sell, so persistence must be 4/5 rather than compare to each row's
    # own rolling side.
    loc = bars.index[-5:]
    total = bars.loc[loc, "notional"].to_numpy()
    ratios = np.array([-0.20, -0.20, -0.20, 0.02, -0.20])
    delta = total * ratios
    bars.loc[loc, "delta_notional"] = delta
    bars.loc[loc, "buy_notional"] = (total + delta) / 2.0
    bars.loc[loc, "sell_notional"] = (total - delta) / 2.0
    features = build_absorption_features(bars, _cfg())
    assert features.iloc[-1]["flow_side"] == -1
    assert np.isclose(features.iloc[-1]["flow_persistence"], 0.8)


def test_resample_preserves_orderflow_sums_and_ohlc() -> None:
    bars = _bars(20)
    out = resample_trade_bars(bars, "5min")
    first = bars.iloc[:5]
    row = out.iloc[0]
    assert np.isclose(row["open"], first.iloc[0]["open"])
    assert np.isclose(row["high"], first["high"].max())
    assert np.isclose(row["low"], first["low"].min())
    assert np.isclose(row["close"], first.iloc[-1]["close"])
    assert np.isclose(row["notional"], first["notional"].sum())
    assert np.isclose(row["delta_notional"], first["delta_notional"].sum())


def test_repeated_floor_tests_and_spring_are_causal() -> None:
    bars = _bars(700)
    # Build a stable floor around 1900 in the last ~100 bars, with multiple
    # defended tests, then one final flush below and same-bar reclaim.
    base = 1900.0
    start = 560
    bars.iloc[start:, bars.columns.get_loc("open")] = base * 1.004
    bars.iloc[start:, bars.columns.get_loc("close")] = base * 1.004
    bars.iloc[start:, bars.columns.get_loc("high")] = base * 1.014
    bars.iloc[start:, bars.columns.get_loc("low")] = base * 1.010
    for pos in (580, 610, 640):
        bars.iloc[pos, bars.columns.get_loc("low")] = base * 0.9998
        bars.iloc[pos, bars.columns.get_loc("close")] = base * 1.003
    flush_pos = 670
    bars.iloc[flush_pos, bars.columns.get_loc("low")] = base * 0.995
    bars.iloc[flush_pos, bars.columns.get_loc("close")] = base * 1.004
    # Force prior floor to be near base instead of a much lower random legacy low.
    for col in ("open", "high", "low", "close"):
        bars.iloc[500:560, bars.columns.get_loc(col)] = base * (1.004 if col != "low" else 1.0)

    features = build_absorption_features(bars, _cfg())
    row = features.iloc[flush_pos]
    assert row["prior_defense_count_long"] >= 2
    assert bool(row["spring_same_bar_long"])


def test_extract_events_uses_bar_close_as_signal_and_next_bar_open_entry() -> None:
    bars = _bars(700)
    features = build_absorption_features(bars, _cfg())
    # Force one already-computed causal floor touch to simplify extraction.
    pos = 600
    features.iloc[pos, features.columns.get_loc("near_floor")] = True
    features.iloc[pos - 1, features.columns.get_loc("near_floor")] = False
    features.iloc[pos, features.columns.get_loc("feature_ready")] = True
    events = extract_events(
        features,
        scale="1m",
        bar_delta=pd.Timedelta(minutes=1),
        floor_lookback_label="1h",
    )
    event = events.loc[
        (events["pattern"] == "floor_retest")
        & (pd.to_datetime(events["signal_bar_start"]) == bars.index[pos])
    ].iloc[0]
    assert pd.Timestamp(event["signal_time"]) == bars.index[pos] + pd.Timedelta(minutes=1)
    assert pd.Timestamp(event["entry_time"]) == bars.index[pos] + pd.Timedelta(minutes=1)


def test_short_fixed_horizon_return_is_linear_inverse_direction() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="1min")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 100, 90, 90, 90],
            "high": [101, 101, 101, 91, 91, 91],
            "low": [99, 99, 89, 89, 89, 89],
            "close": [100, 100, 90, 90, 90, 90],
        },
        index=idx,
    )
    events = pd.DataFrame(
        {
            "signal_bar_start": [idx[0]],
            "trade_side": [-1],
        }
    )
    out = attach_forward_outcomes(events, bars, horizons=(2,), round_trip_cost=0.0011)
    # Entry is next open=100; 2-bar exit close=90 => short gross +10%.
    assert np.isclose(out.iloc[0]["gross_h2"], 0.10)
    assert np.isclose(out.iloc[0]["net_h2"], 0.0989)


def test_continuous_time_near_floor_is_not_counted_as_many_distinct_tests() -> None:
    bars = _bars(700)
    base = 1950.0
    for col, mult in (("open", 1.002), ("close", 1.002), ("high", 1.004), ("low", 1.0005)):
        bars.iloc[560:650, bars.columns.get_loc(col)] = base * mult
    # Establish the causal rolling floor immediately before the near-floor run.
    bars.iloc[500:560, bars.columns.get_loc("low")] = base
    bars.iloc[500:560, bars.columns.get_loc("open")] = base * 1.002
    bars.iloc[500:560, bars.columns.get_loc("close")] = base * 1.002
    bars.iloc[500:560, bars.columns.get_loc("high")] = base * 1.004
    features = build_absorption_features(bars, _cfg())
    # A long uninterrupted stay in the zone is one touch episode, not dozens.
    count = float(features.iloc[645]["prior_defense_count_long"])
    assert count <= 2.0
