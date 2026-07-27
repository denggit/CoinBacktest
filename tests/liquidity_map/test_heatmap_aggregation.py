from __future__ import annotations

import pandas as pd
import pytest

from src.liquidity_map.aggregation import (
    aggregate_heatmap_cells,
    seconds_to_timeframe,
    timeframe_to_seconds,
)


def _frame() -> pd.DataFrame:
    start = 1780272000000
    rows = []
    for minute, depth in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
        ts = start + minute * 60_000
        rows.append(
            {
                "bucket_start_ms": ts,
                "bucket_end_ms": ts + 60_000,
                "price_index": 1800,
                "side_code": 1,
                "flow_valid": 1,
                "depth_base": depth,
                "depth_usd": depth * 1800,
                "order_count": minute + 1,
                "local_depth_ratio": 1.0,
                "added_base": 1.0,
                "removed_base": 2.0,
                "executed_base": 1.5,
                "cancelled_base": 0.5,
                "consumed_base": 1.5,
                "replenished_base": 1.0,
                "side": "bid",
                "price_low": 1800.0,
                "price_high": 1801.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["price_step"] = 1.0
    frame.attrs["heatmap_seconds"] = 60
    return frame


def test_timeframe_parser_supports_chart_intervals() -> None:
    assert timeframe_to_seconds("3m") == 180
    assert timeframe_to_seconds("1H") == 3600
    assert seconds_to_timeframe(900) == "15m"


def test_five_minute_depth_is_time_weighted_average_and_flow_is_sum() -> None:
    out = aggregate_heatmap_cells(_frame(), target_seconds="5m", target_price_step=1.0)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["depth_base"] == pytest.approx(30.0)
    assert row["order_count"] == 3
    assert row["added_base"] == pytest.approx(5.0)
    assert row["removed_base"] == pytest.approx(10.0)
    assert row["bucket_end_ms"] - row["bucket_start_ms"] == 300_000
    assert out.attrs["source_heatmap_seconds"] == 60
    assert out.attrs["heatmap_seconds"] == 300


def test_price_bins_are_summed_before_time_average() -> None:
    frame = _frame().iloc[:1].copy()
    second = frame.copy()
    second["price_index"] = 1801
    second["price_low"] = 1801.0
    second["price_high"] = 1802.0
    second["depth_base"] = 20.0
    second["depth_usd"] = 20.0 * 1801
    combined = pd.concat([frame, second], ignore_index=True)
    combined.attrs["price_step"] = 1.0
    combined.attrs["heatmap_seconds"] = 60
    out = aggregate_heatmap_cells(combined, target_seconds="1m", target_price_step=2.0)
    assert len(out) == 1
    assert out.iloc[0]["price_low"] == 1800.0
    assert out.iloc[0]["price_high"] == 1802.0
    assert out.iloc[0]["depth_base"] == pytest.approx(30.0)


def test_canonical_one_minute_heatmap_cannot_fabricate_30_second_cells() -> None:
    with pytest.raises(ValueError, match="finer than canonical"):
        aggregate_heatmap_cells(_frame(), target_seconds="30s")
