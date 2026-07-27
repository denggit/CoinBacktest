from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader
from src.liquidity_map.builder import OfflineLiquidityMapBuilder
from src.liquidity_map.models import BookEvent, BookLevel, LiquidityBuildStats, LiquidityMapConfig
from src.liquidity_map.store import LiquidityFeatureStore


def _ms(day: date, seconds: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000) + seconds * 1000


def test_builder_is_causal_and_attributes_consumption() -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    config = LiquidityMapConfig(
        price_step=1.0,
        feature_seconds=60,
        heatmap_seconds=60,
        contract_value_base=0.1,
        max_book_staleness_seconds=120,
        max_levels_per_side=20,
    )
    builder = OfflineLiquidityMapBuilder(config)
    trades = pd.DataFrame(
        {
            "ts_ms": [start + 20_000],
            "price": [1801.0],
            "size": [5.0],
            "side": ["buy"],
        }
    )
    stats = LiquidityBuildStats(day=day.isoformat())
    by_price, by_time = builder.aggregate_trades([trades], stats=stats)
    events = [
        BookEvent(start, "snapshot", bids=(BookLevel(1800, 10),), asks=(BookLevel(1801, 10),), seq_id=1),
        BookEvent(start + 30_000, "update", asks=(BookLevel(1801, 5),), seq_id=2, prev_seq_id=1),
    ]
    features, heatmap, stats, _sources = builder.build_day(
        day,
        book_events=events,
        trade_by_price=by_price,
        trade_by_time=by_time,
        stats=stats,
        progress_every_events=0,
    )
    assert features["book_valid"][0] == 1
    assert features["available_time_ms"][0] == features["bucket_end_ms"][0] + 1000
    assert features["estimated_ask_consumed_base"][0] == 0.5
    assert features["estimated_ask_cancel_base"][0] == 0.0
    assert len(heatmap["bucket_start_ms"]) > 0
    # No events after 30 seconds; stale snapshots must not be extended forever.
    assert features["book_valid"][3] == 0




def test_heatmap_keeps_flow_for_fully_removed_level() -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    config = LiquidityMapConfig(
        price_step=1.0,
        feature_seconds=60,
        heatmap_seconds=60,
        contract_value_base=0.1,
        max_book_staleness_seconds=120,
        min_store_depth_base=0.05,
    )
    builder = OfflineLiquidityMapBuilder(config)
    trades = pd.DataFrame(
        {
            "ts_ms": [start + 20_000],
            "price": [1801.0],
            "size": [10.0],
            "side": ["buy"],
        }
    )
    by_price, by_time = builder.aggregate_trades([trades])
    events = [
        BookEvent(
            start,
            "snapshot",
            bids=(BookLevel(1800, 10),),
            asks=(BookLevel(1801, 10), BookLevel(1802, 10)),
            seq_id=1,
        ),
        BookEvent(start + 30_000, "update", asks=(BookLevel(1801, 0),), seq_id=2, prev_seq_id=1),
    ]
    _features, heatmap, _stats, _sources = builder.build_day(
        day,
        book_events=events,
        trade_by_price=by_price,
        trade_by_time=by_time,
        progress_every_events=0,
    )
    rows = pd.DataFrame(heatmap)
    removed = rows.loc[(rows["side_code"] == -1) & (rows["price_index"] == 1801)]
    assert not removed.empty
    assert removed.iloc[0]["removed_base"] == 1.0
    assert removed.iloc[0]["consumed_base"] == 1.0

def test_books_only_marks_trade_attribution_invalid() -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    config = LiquidityMapConfig(
        price_step=1.0,
        feature_seconds=60,
        heatmap_seconds=60,
        contract_value_base=0.1,
        max_book_staleness_seconds=120,
    )
    builder = OfflineLiquidityMapBuilder(config)
    events = [
        BookEvent(start, "snapshot", bids=(BookLevel(1800, 10),), asks=(BookLevel(1801, 10),), seq_id=1),
        BookEvent(start + 30_000, "update", asks=(BookLevel(1801, 5),), seq_id=2, prev_seq_id=1),
    ]
    features, heatmap, _stats, _sources = builder.build_day(
        day,
        book_events=events,
        progress_every_events=0,
        trade_attribution_valid=False,
    )
    assert features["trade_attribution_valid"][0] == 0
    assert features["book_removed_ask_base"][0] == 0.5
    assert features["estimated_ask_cancel_base"][0] == 0.0
    assert all(value == 0 for value in heatmap["flow_valid"])

def test_store_and_public_loader_use_available_time(tmp_path) -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    cfg = LiquidityMapConfig(price_step=1.0, feature_seconds=1, heatmap_seconds=15)
    feature = {
        "bucket_start_ms": start,
        "bucket_end_ms": start + 1000,
        "available_time_ms": start + 2000,
        "book_valid": 1,
        "best_bid": 1800.0,
        "best_ask": 1801.0,
        "mid_price": 1800.5,
    }
    heat = {
        "bucket_start_ms": start,
        "bucket_end_ms": start + 15_000,
        "price_index": 1800,
        "side_code": 1,
        "depth_base": 20.0,
        "depth_usd": 36000.0,
        "local_depth_ratio": 1.0,
    }
    store = LiquidityFeatureStore(data_dir=tmp_path)
    store.save_day(
        day,
        config=cfg,
        feature_rows=[feature],
        heatmap_rows=[heat],
        stats=LiquidityBuildStats(day=day.isoformat(), book_feature_rows=1, heatmap_cells=1),
    )
    loader = OKXLiquidityMapLoader(data_dir=str(tmp_path))
    features = loader.load_features("2026-06-01 08:00:00", "2026-06-01 08:01:00")
    assert len(features) == 1
    assert features.index[0] == pd.Timestamp("2026-06-01 08:00:02")
    cells = loader.load_heatmap("2026-06-01 08:00:00", "2026-06-01 08:01:00")
    assert len(cells) == 1
    assert cells.iloc[0]["price_low"] == 1800.0
    assert cells.iloc[0]["start_timestamp"] == pd.Timestamp("2026-06-01 08:00:00")


def test_public_loader_aggregates_canonical_heatmap_without_new_artifact(tmp_path) -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    cfg = LiquidityMapConfig(price_step=1.0, feature_seconds=1, heatmap_seconds=60)
    rows = []
    for minute, depth in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
        rows.append(
            {
                "bucket_start_ms": start + minute * 60_000,
                "bucket_end_ms": start + (minute + 1) * 60_000,
                "price_index": 1800,
                "side_code": 1,
                "flow_valid": 1,
                "depth_base": depth,
                "depth_usd": depth * 1800,
                "order_count": minute + 1,
                "local_depth_ratio": 1.0,
                "added_base": 1.0,
            }
        )
    store = LiquidityFeatureStore(data_dir=tmp_path)
    store.save_day(
        day,
        config=cfg,
        feature_rows=[],
        heatmap_rows=rows,
        stats=LiquidityBuildStats(day=day.isoformat(), heatmap_cells=len(rows)),
    )
    loader = OKXLiquidityMapLoader(data_dir=str(tmp_path))
    out = loader.load_heatmap_aggregated(
        "2026-06-01 08:00:00",
        "2026-06-01 08:05:00",
        timeframe="5m",
        price_step=1.0,
    )
    assert len(out) == 1
    assert out.iloc[0]["depth_base"] == 30.0
    assert out.iloc[0]["added_base"] == 5.0
    assert out.attrs["source_heatmap_seconds"] == 60
    assert out.attrs["heatmap_seconds"] == 300
    assert out.iloc[0]["start_timestamp"] == pd.Timestamp("2026-06-01 08:00:00")
    assert out.iloc[0]["end_timestamp"] == pd.Timestamp("2026-06-01 08:05:00")


def test_broad_config_allows_unlimited_price_bins() -> None:
    from src.liquidity_map.models import LiquidityMapConfig

    cfg = LiquidityMapConfig(books_depth=5000, heatmap_seconds=5, max_levels_per_side=0, min_store_ratio=0.0)
    cfg.validate()
    assert cfg.max_levels_per_side == 0


def test_heatmap_stores_exact_end_snapshot_separately_from_average() -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    config = LiquidityMapConfig(
        price_step=1.0,
        feature_seconds=1,
        heatmap_seconds=5,
        contract_value_base=0.1,
        max_book_staleness_seconds=30,
        max_levels_per_side=0,
        min_store_depth_base=0.0,
        min_store_ratio=0.0,
    )
    builder = OfflineLiquidityMapBuilder(config)
    events = [
        BookEvent(
            start,
            "snapshot",
            bids=(BookLevel(1800, 10),),
            asks=(BookLevel(1801, 10), BookLevel(1802, 20)),
            seq_id=1,
        ),
        # Remove the best ask before the 5-second boundary.  It must remain in
        # the within-bucket average/flow history but be absent from end depth.
        BookEvent(
            start + 3_000,
            "update",
            asks=(BookLevel(1801, 0),),
            seq_id=2,
            prev_seq_id=1,
        ),
    ]
    features, heatmap, _stats, _sources = builder.build_day(
        day,
        book_events=events,
        progress_every_events=0,
    )
    rows = pd.DataFrame(heatmap)
    first_bucket = rows.loc[rows["bucket_end_ms"] == start + 5_000]
    removed = first_bucket.loc[(first_bucket["side_code"] == -1) & (first_bucket["price_index"] == 1801)]
    surviving = first_bucket.loc[(first_bucket["side_code"] == -1) & (first_bucket["price_index"] == 1802)]
    assert not removed.empty
    assert removed.iloc[0]["depth_base"] > 0.0
    assert removed.iloc[0]["end_depth_base"] == 0.0
    assert not surviving.empty
    assert surviving.iloc[0]["end_depth_base"] == 2.0
    assert surviving.iloc[0]["end_order_count"] == 0
    # Feature best ask at the same boundary must agree with end snapshot.
    assert features["best_ask"][4] == 1802.0


def test_store_marks_interrupted_rebuild_incomplete_and_cleans_parts(tmp_path, monkeypatch) -> None:
    day = date(2026, 6, 1)
    start = _ms(day)
    cfg = LiquidityMapConfig(price_step=1.0, feature_seconds=1, heatmap_seconds=5)
    feature = {
        "bucket_start_ms": start,
        "bucket_end_ms": start + 1000,
        "available_time_ms": start + 2000,
        "book_valid": 1,
        "trade_attribution_valid": 1,
    }
    heat = {
        "bucket_start_ms": start,
        "bucket_end_ms": start + 5000,
        "price_index": 1800,
        "side_code": 1,
        "flow_valid": 1,
    }
    stats = LiquidityBuildStats(day=day.isoformat(), book_feature_rows=1, heatmap_cells=1)
    store = LiquidityFeatureStore(data_dir=tmp_path)
    store.save_day(day, config=cfg, feature_rows=[feature], heatmap_rows=[heat], stats=stats)
    assert store.has_day(day)

    original = store._write_npz_file
    calls = 0

    def fail_second_npz(path, arrays):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interrupted heatmap write")
        original(path, arrays)

    monkeypatch.setattr(store, "_write_npz_file", fail_second_npz)
    try:
        store.save_day(day, config=cfg, feature_rows=[feature], heatmap_rows=[heat], stats=stats)
    except RuntimeError as exc:
        assert "simulated interrupted" in str(exc)
    else:  # pragma: no cover - the injected failure must fire
        raise AssertionError("expected simulated interrupted write")

    # Metadata is the completion checkpoint.  The next prebuild must rebuild
    # this UTC day instead of skipping a mixed or partial artifact set.
    assert not store.has_day(day)
    paths = store.paths_for_day(day)
    assert not paths.metadata.exists()
    assert not list(paths.metadata.parent.glob("*.part"))
