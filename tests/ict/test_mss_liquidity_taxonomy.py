from __future__ import annotations

import numpy as np
import pandas as pd

from research.ict.mss.common.liquidity import enrich_level_sweeps_with_causal_quality
from research.ict.mss.common.structure import build_displacement_fvgs
from research.ict.mss.common.time_context import add_calendar_session_context


def _bars(n: int, freq: str = "1min", start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0},
        index=idx,
    )


def test_native_2m_fvg_is_not_available_until_completion_bar_closes() -> None:
    bars = _bars(10, freq="2min")
    bars.iloc[2] = [100.0, 101.0, 99.0, 100.0]
    bars.iloc[3] = [100.0, 106.0, 99.5, 105.5]
    bars.iloc[4] = [103.0, 107.0, 102.0, 105.0]
    fvgs = build_displacement_fvgs(bars, rolling_window=20, bar_minutes=2)
    row = fvgs.loc[fvgs["fvg_completion_pos"].eq(4) & fvgs["side"].eq(1)].iloc[0]
    assert row["fvg_completion_time"] == bars.index[4]
    assert row["fvg_available_time"] == bars.index[4] + pd.Timedelta(minutes=2)


def test_liquidity_excursion_stops_before_sweep_bar() -> None:
    bars = _bars(12)
    bars.iloc[2:8, bars.columns.get_loc("high")] = [106, 107, 108, 109, 110, 108]
    # If the sweep bar were illegally included this 200 high would dominate.
    bars.iloc[8, bars.columns.get_loc("high")] = 200.0
    bars.iloc[8, bars.columns.get_loc("low")] = 99.0

    levels = pd.DataFrame(
        [
            {
                "level_id": 1,
                "liquidity_side": "sell_side",
                "level_price": 100.0,
                "initial_available_time": bars.index[2],
                "source_timeframe_min": 60,
                "future_max_eventual_order_label": 5,
            },
            # A same-price level not confirmed until after the event must not be
            # counted as active clustered liquidity.
            {
                "level_id": 2,
                "liquidity_side": "sell_side",
                "level_price": 100.05,
                "initial_available_time": bars.index[10],
                "source_timeframe_min": 15,
                "future_max_eventual_order_label": 5,
            },
        ]
    )
    sweeps = pd.DataFrame(
        [
            {
                "level_id": 1,
                "liquidity_side": "sell_side",
                "level_price": 100.0,
                "initial_available_time": bars.index[2],
                "source_timeframe_min": 60,
                "active_pos": 2,
                "sweep_pos": 8,
                "sweep_bar_time": bars.index[8],
                "level_age_minutes_at_sweep": 6.0,
                "confirmed_order_at_sweep": 3,
                "confirmed_prominence_bp_at_sweep": 4.0,
                "pivot_range_bp": 30.0,
                "pivot_rejection_fraction": 0.7,
            }
        ]
    )
    out = enrich_level_sweeps_with_causal_quality(
        bars,
        levels,
        sweeps,
        cluster_tolerances_bp=(5.0, 10.0, 25.0),
        show_progress=False,
    ).iloc[0]
    assert np.isclose(out["max_excursion_away_bp_before_sweep"], 1000.0)
    assert int(out["quality_feature_last_pos"]) == 7
    assert int(out["cluster_count_10bp"]) == 1


def test_new_york_cash_open_uses_dst_not_fixed_utc_hour() -> None:
    # CoinBacktest legacy OHLC is +8.  NY cash open is 22:30 Beijing in winter
    # and 21:30 Beijing in summer because New York observes DST.
    frame = pd.DataFrame(
        {
            "t": pd.to_datetime([
                "2026-01-05 22:30:00",  # Monday, 09:30 EST
                "2026-06-15 21:30:00",  # Monday, 09:30 EDT
            ])
        }
    )
    out = add_calendar_session_context(frame, timestamp_col="t", project_offset_hours=8)
    assert out["us_cash_open_90m"].tolist() == [True, True]
    assert out["is_weekday_utc"].tolist() == [True, True]
    assert np.allclose(out["new_york_hour"].to_numpy(), [9.5, 9.5])
