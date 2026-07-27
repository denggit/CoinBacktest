from __future__ import annotations

import pandas as pd

from src.market_state.causal_alignment import causal_merge_context, timeframe_to_timedelta


def test_timeframe_parser_accepts_project_presets() -> None:
    assert timeframe_to_timedelta("1s") == pd.Timedelta(seconds=1)
    assert timeframe_to_timedelta("5m") == pd.Timedelta(minutes=5)
    assert timeframe_to_timedelta("4H") == pd.Timedelta(hours=4)
    assert timeframe_to_timedelta("1D") == pd.Timedelta(days=1)


def test_left_labeled_context_is_not_visible_before_close() -> None:
    primary_index = pd.date_range("2026-01-01 00:00:00", periods=11, freq="1min")
    primary = pd.DataFrame({"close": range(len(primary_index))}, index=primary_index)
    context = pd.DataFrame(
        {"trend": [10.0, 20.0]},
        index=pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00"]),
    )

    merged = causal_merge_context(primary, context, context_bar_duration="5min")

    assert pd.isna(merged.loc["2026-01-01 00:04:00", "trend_ctx"])
    assert merged.loc["2026-01-01 00:05:00", "trend_ctx"] == 10.0
    assert merged.loc["2026-01-01 00:09:00", "trend_ctx"] == 10.0
    assert merged.loc["2026-01-01 00:10:00", "trend_ctx"] == 20.0
