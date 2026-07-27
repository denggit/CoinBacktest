from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader


def _write_day(root: Path) -> None:
    loader = OKXLiquidityMapLoader(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=str(root),
    )
    paths = loader.store.paths_for_day("2026-01-01")
    paths.heatmap.parent.mkdir(parents=True, exist_ok=True)
    starts = []
    ends = []
    prices = []
    sides = []
    depth = []
    for slot in range(6):
        ts = 1767225600000 + slot * 5_000
        for price, side, amount in ((3000, 1, 10.0 + slot), (3001, -1, 20.0 + slot)):
            starts.append(ts)
            ends.append(ts + 5_000)
            prices.append(price)
            sides.append(side)
            depth.append(amount)
    count = len(starts)
    arrays = {
        "bucket_start_ms": np.asarray(starts, dtype=np.int64),
        "bucket_end_ms": np.asarray(ends, dtype=np.int64),
        "price_index": np.asarray(prices, dtype=np.int32),
        "side_code": np.asarray(sides, dtype=np.int8),
        "flow_valid": np.ones(count, dtype=np.uint8),
        "depth_base": np.asarray(depth, dtype=np.float32),
        "depth_usd": np.asarray(depth, dtype=np.float32) * np.asarray(prices, dtype=np.float32),
        "order_count": np.ones(count, dtype=np.int32),
        "local_depth_ratio": np.ones(count, dtype=np.float32),
        "end_depth_base": np.asarray(depth, dtype=np.float32),
        "end_depth_usd": np.asarray(depth, dtype=np.float32) * np.asarray(prices, dtype=np.float32),
        "end_order_count": np.ones(count, dtype=np.int32),
        "end_local_depth_ratio": np.ones(count, dtype=np.float32),
        "added_base": np.ones(count, dtype=np.float32),
        "removed_base": np.zeros(count, dtype=np.float32),
        "executed_base": np.zeros(count, dtype=np.float32),
        "cancelled_base": np.zeros(count, dtype=np.float32),
        "consumed_base": np.zeros(count, dtype=np.float32),
        "replenished_base": np.zeros(count, dtype=np.float32),
    }
    with paths.heatmap.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    paths.features.touch()
    paths.metadata.write_text(
        json.dumps(
            {
                "day": "2026-01-01",
                "config": {"price_step": 1.0, "heatmap_seconds": 5},
                "stats": {"heatmap_cells": count},
            }
        ),
        encoding="utf-8",
    )


def test_period_end_cache_builds_once_and_uses_last_snapshot(tmp_path):
    _write_day(tmp_path)
    loader = OKXLiquidityMapLoader(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=str(tmp_path),
    )
    first = list(
        loader.iter_period_end_snapshot_days(
            pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 00:00:30", tz="UTC"),
            timeframe=15,
            price_step=1.0,
            project_time=False,
        )
    )
    assert len(first) == 1
    frame = first[0]
    assert frame.attrs["cache_hit"] is False
    assert frame["bucket_start_ms"].nunique() == 2
    first_bucket = frame.loc[frame["bucket_start_ms"] == 1767225600000].set_index("side_code")
    assert first_bucket.loc[1, "end_depth_base"] == 12.0
    assert first_bucket.loc[-1, "end_depth_base"] == 22.0
    cache_path = Path(frame.attrs["cache_path"])
    assert cache_path.exists()

    second = list(
        loader.iter_period_end_snapshot_days(
            pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 00:00:30", tz="UTC"),
            timeframe=15,
            price_step=1.0,
            project_time=False,
        )
    )
    assert second[0].attrs["cache_hit"] is True
    pd.testing.assert_frame_equal(
        first[0].reset_index(drop=True),
        second[0].reset_index(drop=True),
        check_dtype=True,
    )


def test_period_end_cache_aggregates_price_step(tmp_path):
    _write_day(tmp_path)
    loader = OKXLiquidityMapLoader(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=str(tmp_path),
    )
    frame = loader.load_period_end_snapshots(
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:00:15", tz="UTC"),
        timeframe=15,
        price_step=2.0,
        project_time=False,
    )
    # Bid 3000 and ask 3001 share the same $2 price bucket but remain separate sides.
    assert set(frame["price_index"].tolist()) == {1500}
    assert set(frame["side_code"].tolist()) == {1, -1}
    assert frame.attrs["price_step"] == 2.0


def test_feature_alignment_reads_only_selected_rows(tmp_path):
    loader = OKXLiquidityMapLoader(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=str(tmp_path),
    )
    paths = loader.store.paths_for_day("2026-01-01")
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    available = np.asarray([1767225601000, 1767225602000, 1767225603000], dtype=np.int64)
    with paths.features.open("wb") as handle:
        np.savez_compressed(
            handle,
            available_time_ms=available,
            bucket_end_ms=available - 1,
            book_valid=np.ones(3, dtype=np.uint8),
            mid_price=np.asarray([3000.0, 3001.0, 3002.0], dtype=np.float64),
        )
    paths.heatmap.touch()
    paths.metadata.write_text(
        json.dumps({"config": {"price_step": 1.0, "heatmap_seconds": 5}}),
        encoding="utf-8",
    )
    times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 00:00:02.500", tz="UTC"),
            pd.Timestamp("2026-01-01 00:00:03.500", tz="UTC"),
        ]
    )
    aligned = loader.align_features_to_times(
        times,
        project_time=False,
        tolerance="5s",
        columns=["bucket_end_ms", "book_valid", "mid_price"],
    )
    assert aligned["mid_price"].tolist() == [3001.0, 3002.0]
    assert aligned["book_valid"].tolist() == [1.0, 1.0]


def test_period_end_cache_keeps_shallow_positive_last_snapshot_bins(tmp_path):
    loader = OKXLiquidityMapLoader(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=str(tmp_path),
    )
    paths = loader.store.paths_for_day("2026-01-02")
    paths.heatmap.parent.mkdir(parents=True, exist_ok=True)
    base = 1767312000000
    prices = np.asarray([2999, 3000, 3001], dtype=np.int32)
    sides = np.asarray([1, 1, -1], dtype=np.int8)
    depths = np.asarray([0.001, 100.0, 80.0], dtype=np.float32)
    count = len(prices)
    arrays = {
        "bucket_start_ms": np.full(count, base, dtype=np.int64),
        "bucket_end_ms": np.full(count, base + 5_000, dtype=np.int64),
        "price_index": prices,
        "side_code": sides,
        "flow_valid": np.ones(count, dtype=np.uint8),
        "end_depth_base": depths,
        "end_depth_usd": depths * prices.astype(np.float32),
        "end_order_count": np.ones(count, dtype=np.int32),
        "added_base": np.zeros(count, dtype=np.float32),
        "removed_base": np.zeros(count, dtype=np.float32),
        "executed_base": np.zeros(count, dtype=np.float32),
        "cancelled_base": np.zeros(count, dtype=np.float32),
        "consumed_base": np.zeros(count, dtype=np.float32),
        "replenished_base": np.zeros(count, dtype=np.float32),
    }
    with paths.heatmap.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    paths.features.touch()
    paths.metadata.write_text(
        json.dumps(
            {
                "day": "2026-01-02",
                "config": {"price_step": 1.0, "heatmap_seconds": 5},
                "stats": {"heatmap_cells": count},
            }
        ),
        encoding="utf-8",
    )

    frame = loader.load_period_end_snapshots(
        pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-02 00:00:15", tz="UTC"),
        timeframe=15,
        price_step=1.0,
        project_time=False,
    )
    shallow = frame.loc[frame["price_index"] == 2999, "end_depth_base"]
    assert len(shallow) == 1
    assert float(shallow.iloc[0]) == pytest.approx(0.001)
