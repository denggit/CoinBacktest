from __future__ import annotations

import pandas as pd

from src.research_common.ict.htf_liquidity import (
    HTFLiquidityConfig,
    attach_first_consumption_time,
    build_htf_swing_catalog,
    build_remote_htf_levels_for_days,
    dedupe_same_family_sweeps,
)

TZ = "America/New_York"


def _premarket_context(day="2026-06-10"):
    return pd.DataFrame([
        {
            "ny_date": day,
            "level_type": "premarket_extreme",
            "premarket_high": 110.0,
            "premarket_low": 90.0,
            "premarket_range": 20.0,
            "premarket_range_pct": 0.2,
            "premarket_close": 100.0,
            "premarket_median_15m_range": 1.0,
        }
    ])


def test_first_consumption_time_is_after_causal_confirmation() -> None:
    idx = pd.date_range("2026-06-10 08:29", periods=4, freq="1min", tz=TZ)
    bars = pd.DataFrame(
        {
            "open": [95, 95, 95, 95],
            "high": [99, 99.5, 101, 101],
            "low": [91, 89, 89, 89],
            "close": [95, 95, 95, 95],
        },
        index=idx,
    )
    catalog = pd.DataFrame([
        {
            "liquidity_side": "high", "level_price": 100.0,
            "level_available_time": pd.Timestamp("2026-06-10 08:00", tz=TZ),
        },
        {
            "liquidity_side": "low", "level_price": 90.0,
            "level_available_time": pd.Timestamp("2026-06-10 08:00", tz=TZ),
        },
    ])
    out = attach_first_consumption_time(bars, catalog)
    high = out.loc[out["liquidity_side"] == "high"].iloc[0]
    low = out.loc[out["liquidity_side"] == "low"].iloc[0]
    assert pd.Timestamp(high["first_consumed_time"]) == pd.Timestamp("2026-06-10 08:32", tz=TZ)
    assert pd.Timestamp(low["first_consumed_time"]) == pd.Timestamp("2026-06-10 08:31", tz=TZ)


def test_all_active_remote_levels_are_retained_not_only_nearest() -> None:
    day = pd.Timestamp("2026-06-10").date()
    base = {
        "htf_timeframe": "1h", "level_type": "remote_1h_swing",
        "liquidity_family": "remote_1h_swing", "liquidity_strength": "causal_unconsumed_htf_swing",
        "tradable_level": True, "local_prominence_abs": 1.0, "two_sided_excursion_abs": 2.0,
    }
    catalog = pd.DataFrame([
        {**base, "liquidity_side": "high", "level_price": 103.0, "source_bar_time": pd.Timestamp("2026-06-05 10:00", tz=TZ), "level_available_time": pd.Timestamp("2026-06-05 13:00", tz=TZ), "first_consumed_time": pd.NaT},
        {**base, "liquidity_side": "high", "level_price": 108.0, "source_bar_time": pd.Timestamp("2026-06-03 11:00", tz=TZ), "level_available_time": pd.Timestamp("2026-06-03 14:00", tz=TZ), "first_consumed_time": pd.NaT},
        {**base, "liquidity_side": "low", "level_price": 96.0, "source_bar_time": pd.Timestamp("2026-06-04 11:00", tz=TZ), "level_available_time": pd.Timestamp("2026-06-04 14:00", tz=TZ), "first_consumed_time": pd.NaT},
    ])
    out = build_remote_htf_levels_for_days(catalog, _premarket_context(), [day])
    assert len(out) == 3
    highs = out.loc[out["liquidity_side"] == "high"].sort_values("active_rank_nearest")
    assert highs["level_price"].tolist() == [103.0, 108.0]
    assert highs["active_rank_nearest"].tolist() == [1, 2]


def test_level_consumed_before_0830_is_excluded_but_after_0830_is_active() -> None:
    day = pd.Timestamp("2026-06-10").date()
    base = {
        "htf_timeframe": "4h", "level_type": "remote_4h_swing", "liquidity_family": "remote_4h_swing",
        "liquidity_strength": "causal_unconsumed_htf_swing", "tradable_level": True,
        "liquidity_side": "high", "source_bar_time": pd.Timestamp("2026-06-01 08:00", tz=TZ),
        "level_available_time": pd.Timestamp("2026-06-02 16:00", tz=TZ),
        "local_prominence_abs": 2.0, "two_sided_excursion_abs": 3.0,
    }
    catalog = pd.DataFrame([
        {**base, "level_price": 104.0, "first_consumed_time": pd.Timestamp("2026-06-10 08:29", tz=TZ)},
        {**base, "level_price": 108.0, "first_consumed_time": pd.Timestamp("2026-06-10 08:31", tz=TZ)},
    ])
    out = build_remote_htf_levels_for_days(catalog, _premarket_context(), [day])
    assert out["level_price"].tolist() == [108.0]


def test_same_family_same_minute_sweeps_are_one_physical_event() -> None:
    t = pd.Timestamp("2026-06-10 09:01", tz=TZ)
    sweeps = pd.DataFrame([
        {"ny_date": "2026-06-10", "trade_side": "SHORT", "sweep_time": t, "liquidity_family": "remote_1h_swing", "level_type": "remote_1h_swing", "level_price": 103.0},
        {"ny_date": "2026-06-10", "trade_side": "SHORT", "sweep_time": t, "liquidity_family": "remote_1h_swing", "level_type": "remote_1h_swing", "level_price": 105.0},
        {"ny_date": "2026-06-10", "trade_side": "SHORT", "sweep_time": t, "liquidity_family": "remote_4h_swing", "level_type": "remote_4h_swing", "level_price": 104.0},
    ])
    out = dedupe_same_family_sweeps(sweeps)
    assert len(out) == 2
    one_h = out.loc[out["liquidity_family"] == "remote_1h_swing"].iloc[0]
    assert float(one_h["level_price"]) == 105.0
    assert int(one_h["same_family_levels_swept"]) == 2
    assert int(one_h["htf_confluence_count"]) == 2


def test_daily_swing_confirmation_uses_right_side_closed_days() -> None:
    rows = []
    days = pd.date_range("2026-06-01", periods=5, freq="D")
    highs = [100, 102, 110, 103, 101]
    lows = [90, 91, 92, 91, 90]
    for d, hi, lo in zip(days, highs, lows):
        ts = pd.Timestamp(d.date()).tz_localize(TZ) + pd.Timedelta(hours=4)
        rows.append((ts, lo + 2, hi, lo, lo + 3))
    bars = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).set_index("timestamp")
    catalog = build_htf_swing_catalog(
        bars,
        config=HTFLiquidityConfig(timeframes=("1d",), pivot_left=2, pivot_right=2, daily_min_rows=1),
    )
    high = catalog.loc[catalog["liquidity_side"] == "high"].iloc[0]
    assert pd.Timestamp(high["source_bar_time"]).date() == pd.Timestamp("2026-06-03").date()
    assert pd.Timestamp(high["level_available_time"]) == pd.Timestamp("2026-06-05 16:30", tz=TZ)
