from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.config import AtlasConfig
from src.research_common.swing_liquidity_atlas.lifecycle import (
    attach_active_confluence,
    build_event_table,
    build_level_lifecycle,
)
from src.research_common.swing_liquidity_atlas.outcomes import attach_forward_paths
from src.research_common.swing_liquidity_atlas.pivots import build_swing_low_universe
from src.research_common.swing_liquidity_atlas.reports import causal_audit


def _bars(periods: int = 240, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    open_ = close.copy()
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "notional": 1_000.0,
            "trades_count": 10.0,
            "buy_notional": 500.0,
            "sell_notional": 500.0,
            "delta_notional": 0.0,
        },
        index=index,
    )


def test_15m_pivot_is_available_only_after_right_bar_closes() -> None:
    bars = _bars(90)
    # 15m lows: 99, 90, 98, ... => pivot starts at 00:15 and is known at 00:45.
    bars.loc["2024-01-01 00:15":"2024-01-01 00:29", "low"] = 90.0
    bars.loc["2024-01-01 00:30":"2024-01-01 00:44", "low"] = 98.0
    cfg = AtlasConfig(timeframes=(("15m", 15),), confirmation_orders=(1, 2)).validate()
    levels = build_swing_low_universe(bars, cfg)
    pivot = levels.loc[levels["pivot_time"].eq(pd.Timestamp("2024-01-01 00:15:00"))].iloc[0]
    assert pivot["pivot_bar_end_time"] == pd.Timestamp("2024-01-01 00:30:00")
    assert pivot["initial_available_time"] == pd.Timestamp("2024-01-01 00:45:00")


def test_higher_order_confirmation_is_not_backfilled() -> None:
    bars = _bars(180)
    # Make the 01:00 15m bar the low across enough neighbours for order 3.
    bars["low"] = 99.0
    bars.loc["2024-01-01 01:00":"2024-01-01 01:14", "low"] = 80.0
    cfg = AtlasConfig(timeframes=(("15m", 15),), confirmation_orders=(1, 2, 3)).validate()
    levels = build_swing_low_universe(bars, cfg)
    pivot = levels.loc[levels["pivot_time"].eq(pd.Timestamp("2024-01-01 01:00:00"))].iloc[0]
    assert pivot["initial_available_time"] == pd.Timestamp("2024-01-01 01:30:00")
    assert pivot["order_3_available_time"] == pd.Timestamp("2024-01-01 02:00:00")


def test_daily_swing_low_is_included() -> None:
    bars = _bars(60 * 24 * 4)
    bars.loc["2024-01-02", "low"] = 80.0
    cfg = AtlasConfig(timeframes=(("1D", 1440),), confirmation_orders=(1,)).validate()
    levels = build_swing_low_universe(bars, cfg)
    assert not levels.empty
    assert set(levels["source_timeframe"]) == {"1D"}
    assert pd.Timestamp("2024-01-04 00:00:00") in set(pd.to_datetime(levels["initial_available_time"]))


def test_old_level_has_no_arbitrary_expiry_and_sweeps_days_later() -> None:
    bars = _bars(60 * 24 * 5)
    levels = pd.DataFrame(
        {
            "level_id": [1],
            "source_timeframe": ["4H"],
            "source_timeframe_min": [240],
            "pivot_time": [pd.Timestamp("2024-01-01")],
            "pivot_bar_end_time": [pd.Timestamp("2024-01-01 04:00")],
            "initial_available_time": [pd.Timestamp("2024-01-01 08:00")],
            "level_price": [90.0],
            "future_max_eventual_order_label": [1],
            "order_1_available_time": [pd.Timestamp("2024-01-01 08:00")],
        }
    )
    sweep_time = pd.Timestamp("2024-01-05 12:00")
    bars.loc[sweep_time, "low"] = 89.0
    bars.loc[sweep_time, "close"] = 91.0
    cfg = AtlasConfig(timeframes=(("4H", 240),), confirmation_orders=(1,)).validate()
    lifecycle = build_level_lifecycle(bars, levels, cfg, show_progress=False)
    assert lifecycle.iloc[0]["sweep_available_time"] == sweep_time + pd.Timedelta(minutes=1)
    assert lifecycle.iloc[0]["support_resolution_180m"] == "same_bar_reclaim"
    assert bool(lifecycle.iloc[0]["clean_reclaim_by_30m"])


def test_touch_does_not_consume_but_first_sweep_does() -> None:
    bars = _bars(120)
    level = 95.0
    bars.loc["2024-01-01 00:30", "low"] = 95.02
    bars.loc["2024-01-01 00:50", "low"] = 94.90
    bars.loc["2024-01-01 00:50", "close"] = 95.20
    levels = pd.DataFrame(
        {
            "level_id": [1], "source_timeframe": ["15m"], "source_timeframe_min": [15],
            "pivot_time": [pd.Timestamp("2024-01-01 00:00")],
            "pivot_bar_end_time": [pd.Timestamp("2024-01-01 00:15")],
            "initial_available_time": [pd.Timestamp("2024-01-01 00:20")],
            "level_price": [level], "future_max_eventual_order_label": [1],
            "order_1_available_time": [pd.Timestamp("2024-01-01 00:20")],
        }
    )
    cfg = AtlasConfig(timeframes=(("15m", 15),), confirmation_orders=(1,), touch_distance_bp=5.0).validate()
    lifecycle = build_level_lifecycle(bars, levels, cfg, show_progress=False).iloc[0]
    assert lifecycle["touch_available_time"] == pd.Timestamp("2024-01-01 00:31")
    assert lifecycle["sweep_available_time"] == pd.Timestamp("2024-01-01 00:51")
    assert lifecycle["stop_liquidity_state"] == "consumed_first_sweep"


def test_all_levels_are_tracked_not_only_latest() -> None:
    bars = _bars(180)
    levels = pd.DataFrame(
        {
            "level_id": [1, 2],
            "source_timeframe": ["1H", "4H"],
            "source_timeframe_min": [60, 240],
            "pivot_time": [pd.Timestamp("2023-12-31"), pd.Timestamp("2023-12-30")],
            "pivot_bar_end_time": [pd.Timestamp("2024-01-01 00:10"), pd.Timestamp("2024-01-01 00:10")],
            "initial_available_time": [pd.Timestamp("2024-01-01 00:20"), pd.Timestamp("2024-01-01 00:20")],
            "level_price": [95.0, 90.0],
            "future_max_eventual_order_label": [1, 1],
            "order_1_available_time": [pd.Timestamp("2024-01-01 00:20"), pd.Timestamp("2024-01-01 00:20")],
        }
    )
    bars.loc["2024-01-01 01:00", "low"] = 89.0
    cfg = AtlasConfig(timeframes=(("1H", 60), ("4H", 240)), confirmation_orders=(1,)).validate()
    lifecycle = build_level_lifecycle(bars, levels, cfg, show_progress=False)
    assert len(lifecycle) == 2
    assert lifecycle["sweep_pos"].ge(0).sum() == 2


def test_forward_path_uses_next_open_after_closed_event() -> None:
    bars = _bars(20)
    bars.loc["2024-01-01 00:06", "open"] = 100.0
    bars.loc["2024-01-01 00:06":"2024-01-01 00:10", "close"] = [101, 102, 103, 104, 105]
    event = pd.DataFrame(
        {
            "event_id": ["x"], "event_stage": ["sweep"], "event_pos": [5],
            "event_available_time": [pd.Timestamp("2024-01-01 00:06")],
            "initial_available_time": [pd.Timestamp("2024-01-01 00:00")],
        }
    )
    cfg = AtlasConfig(timeframes=(("15m", 15),), confirmation_orders=(1,), forward_horizons=(5,)).validate()
    out = attach_forward_paths(event, bars, cfg).iloc[0]
    assert out["entry_reference_time"] == pd.Timestamp("2024-01-01 00:06")
    assert abs(out["close_return_5m"] - 0.05) < 1e-12


def test_confluence_counts_active_cross_timeframe_levels() -> None:
    bars = _bars(120)
    lifecycle = pd.DataFrame(
        {
            "level_id": [1, 2], "level_price": [100.0, 100.1],
            "source_timeframe": ["1H", "4H"], "active_pos": [10, 10], "sweep_pos": [50, 60],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["a"], "event_pos": [20], "level_price": [100.0],
        }
    )
    cfg = AtlasConfig(timeframes=(("1H", 60), ("4H", 240)), confirmation_orders=(1,)).validate()
    out = attach_active_confluence(events, lifecycle, cfg).iloc[0]
    assert out["active_level_count_25p0bp"] == 2
    assert out["active_timeframe_count_25p0bp"] == 2


def test_end_to_end_causal_audit_zero() -> None:
    bars = _bars(1_200)
    x = np.arange(len(bars), dtype=float)
    bars["close"] = 100 + np.sin(x / 30) * 5
    bars["open"] = bars["close"].shift(1).fillna(bars["close"])
    bars["high"] = np.maximum(bars["open"], bars["close"]) + 0.5
    bars["low"] = np.minimum(bars["open"], bars["close"]) - 0.5
    cfg = AtlasConfig(timeframes=(("15m", 15), ("30m", 30)), confirmation_orders=(1, 2, 3)).validate()
    levels = build_swing_low_universe(bars, cfg)
    lifecycle = build_level_lifecycle(bars, levels, cfg, show_progress=False)
    events = build_event_table(lifecycle, bars, cfg)
    events = attach_forward_paths(events, bars, cfg)
    audit = causal_audit(levels, lifecycle, events, cfg)
    assert int(audit["violations"].sum()) == 0


def test_microsecond_datetime_index_does_not_break_searchsorted_units() -> None:
    bars = _bars(180)
    bars.index = pd.DatetimeIndex(bars.index.to_numpy(dtype="datetime64[us]"))
    bars.loc[pd.Timestamp("2024-01-01 01:00"), "low"] = 89.0
    bars.loc[pd.Timestamp("2024-01-01 01:00"), "close"] = 91.0
    levels = pd.DataFrame(
        {
            "level_id": [1], "source_timeframe": ["1H"], "source_timeframe_min": [60],
            "pivot_time": [pd.Timestamp("2023-12-31")],
            "pivot_bar_end_time": [pd.Timestamp("2024-01-01 00:10")],
            "initial_available_time": [pd.Timestamp("2024-01-01 00:20")],
            "level_price": [90.0], "future_max_eventual_order_label": [1],
            "order_1_available_time": [pd.Timestamp("2024-01-01 00:20")],
        }
    )
    cfg = AtlasConfig(timeframes=(("1H", 60),), confirmation_orders=(1,)).validate()
    lifecycle = build_level_lifecycle(bars, levels, cfg, show_progress=False)
    assert lifecycle.iloc[0]["sweep_available_time"] == pd.Timestamp("2024-01-01 01:01")


def test_late_revisit_after_acceptance_is_not_clean_reclaim() -> None:
    bars = _bars(120)
    bars.loc["2024-01-01 00:30", ["low", "close"]] = [89.0, 89.5]
    bars.loc["2024-01-01 00:31", "close"] = 89.4
    bars.loc["2024-01-01 00:32", "close"] = 89.3
    bars.loc["2024-01-01 00:40", "close"] = 91.0
    levels = pd.DataFrame(
        {
            "level_id": [1], "source_timeframe": ["4H"], "source_timeframe_min": [240],
            "pivot_time": [pd.Timestamp("2023-12-31")],
            "pivot_bar_end_time": [pd.Timestamp("2024-01-01 00:10")],
            "initial_available_time": [pd.Timestamp("2024-01-01 00:20")],
            "level_price": [90.0], "future_max_eventual_order_label": [1],
            "order_1_available_time": [pd.Timestamp("2024-01-01 00:20")],
        }
    )
    cfg = AtlasConfig(timeframes=(("4H", 240),), confirmation_orders=(1,)).validate()
    row = build_level_lifecycle(bars, levels, cfg, show_progress=False).iloc[0]
    assert row["support_resolution_180m"] == "accepted_below"
    assert bool(row["close_revisited_level_by_30m"])
    assert not bool(row["clean_reclaim_by_30m"])
