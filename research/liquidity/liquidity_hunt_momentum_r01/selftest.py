#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast deterministic smoke test for Liquidity Hunt Momentum R01."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity.liquidity_hunt_momentum_r01.core import (
    LiquidityHuntConfig,
    StrategyVariant,
    align_book_features_to_times,
    build_events,
    build_range_features,
    simulate_events,
)


def _synthetic_m1_frame() -> tuple[pd.DataFrame, LiquidityHuntConfig]:
    cfg = LiquidityHuntConfig(
        support_lookback_bars=3,
        notional_median_bars=3,
        notional_min_periods=2,
        cooldown_minutes=0,
    )
    count = 12
    starts = pd.date_range("2026-01-01 10:00:01", periods=count, freq="1min")
    ends = starts + pd.Timedelta(seconds=40)
    open_ = np.array([101, 101, 101, 101, 101, 100, 101, 101, 102, 103, 104, 105], dtype=float)
    close = np.array([101.2, 101.1, 101.2, 101.1, 101.2, 98.0, 101.2, 101.5, 102.5, 103.5, 104.5, 105.5])
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    low[:5] = 100.0
    high[:5] = 102.0
    low[5], high[5] = 98.0, 101.0
    low[6], high[6] = 99.0, 101.3
    notional = np.array([100, 100, 100, 100, 100, 250, 100, 100, 100, 100, 100, 100], dtype=float)
    volume = notional.copy()
    buy_ratio = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
    direction = np.sign(close - open_).astype(int)
    raw = pd.DataFrame(
        {
            "bar_id": np.arange(count, dtype=np.int64),
            "start_ts": starts,
            "end_ts": ends,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "direction": direction,
            "duration_seconds": 40.0,
            "volume": volume,
            "notional": notional,
            "buy_notional": notional * buy_ratio,
            "sell_notional": notional * (1.0 - buy_ratio),
            "taker_buy_ratio": buy_ratio,
        }
    )
    frame = build_range_features(raw, cfg)
    frame["book_available_time"] = frame["signal_time"]
    frame["book_context_missing_flag"] = False
    frame["book_available_after_signal_flag"] = False
    frame["footprint_missing_flag"] = False
    book_columns = (
        "book_obi_5s",
        "book_obi_5s_min",
        "book_obi_5s_max",
        "book_ask_depth_25bps_ref_ratio",
        "book_bid_depth_25bps_ref_ratio",
        "book_ask_to_bid_depth_25bps",
        "book_bid_to_ask_depth_25bps",
        "book_nearest_large_bid_price",
        "book_nearest_large_ask_price",
        "book_nearest_large_bid_depth_base",
        "book_nearest_large_ask_depth_base",
        "book_top_bid_wall_price",
        "book_top_ask_wall_price",
        "book_top_bid_wall_depth_base",
        "book_top_ask_wall_depth_base",
        "book_estimated_bid_replenished_base_5s",
        "book_estimated_ask_replenished_base_5s",
        "book_bid_replenish_to_consume",
        "book_ask_replenish_to_consume",
    )
    for column in book_columns:
        frame[column] = np.nan
    frame.loc[5, ["book_obi_5s", "book_obi_5s_min", "book_obi_5s_max"]] = [-0.40, -0.45, -0.35]
    frame.loc[6, ["book_obi_5s", "book_obi_5s_min", "book_obi_5s_max"]] = [0.30, 0.25, 0.35]
    frame.loc[6, ["book_nearest_large_bid_price", "book_nearest_large_bid_depth_base"]] = [100.0, 80.0]
    frame.loc[6, ["book_nearest_large_ask_price", "book_nearest_large_ask_depth_base"]] = [104.0, 80.0]
    return frame, cfg


def run_self_test() -> None:
    source = pd.DataFrame(
        {
            "available_time": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 10:00:05"]),
            "book_valid": [1, 1],
            "bid_depth_5bps_base": [100.0, 120.0],
            "ask_depth_5bps_base": [120.0, 80.0],
            "bid_depth_25bps_base": [200.0, 240.0],
            "ask_depth_25bps_base": [220.0, 160.0],
        }
    ).set_index("available_time", drop=False)
    aligned = align_book_features_to_times(
        pd.to_datetime(["2026-01-01 10:00:04", "2026-01-01 10:00:06"]),
        source,
        tolerance=pd.Timedelta(seconds=10),
    )
    if pd.Timestamp(aligned.loc[0, "book_available_time"]) != pd.Timestamp("2026-01-01 10:00:00"):
        raise AssertionError("book alignment used a future row")
    if pd.Timestamp(aligned.loc[1, "book_available_time"]) != pd.Timestamp("2026-01-01 10:00:05"):
        raise AssertionError("book alignment did not use latest causal row")

    frame, cfg = _synthetic_m1_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    strict = events.loc[events["stage"] == "M1_FLOW_RECLAIM_OBI_REBUILD"]
    if len(strict) != 1:
        raise AssertionError(f"expected one strict M1 event, got {len(strict)}")
    trades = simulate_events(strict, frame, cfg, StrategyVariant(name="self_test"))
    if len(trades) != 1:
        raise AssertionError(f"expected one simulated trade, got {len(trades)}")
    trade = trades.iloc[0]
    if not trade["entry_time"] > trade["signal_time"]:
        raise AssertionError("entry did not occur after the completed signal bar")
    if bool(trade["book_available_after_signal_flag"]):
        raise AssertionError("future Books context reached the trade")
    print("[self-test] liquidity_hunt_momentum_r01 passed", flush=True)
