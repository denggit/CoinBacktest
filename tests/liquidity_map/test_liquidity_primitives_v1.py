from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_liquidity_primitives import (
    CausalPrimitiveReference,
    LiquidityPrimitiveConfig,
    OKXLiquidityPrimitiveStore,
    build_liquidity_primitive_day,
)
from src.research_common.liquidity_wall_discovery import (
    CausalDepthReference,
    WallDiscoveryConfig,
    WallDiscoveryEngine,
    extract_snapshot_candidates,
    extract_primitive_snapshot_candidates,
)


def _snapshot(timestamp: str, *, wall_start: int = 90) -> pd.DataFrame:
    start = pd.Timestamp(timestamp, tz="UTC")
    start_ms = int(start.timestamp() * 1000)
    rows = []
    for side_code, prices in ((1, range(80, 100)), (-1, range(100, 120))):
        for price in prices:
            depth = 45.0 if side_code == 1 and wall_start <= price < wall_start + 5 else 10.0
            rows.append(
                {
                    "bucket_start_ms": start_ms,
                    "bucket_end_ms": start_ms + 5_000,
                    "price_index": price,
                    "side_code": side_code,
                    "end_depth_base": depth,
                    "depth_base": depth,
                    "added_base": 1.0,
                    "removed_base": 0.5,
                    "executed_base": 0.2,
                    "cancelled_base": 0.3,
                    "consumed_base": 0.2,
                    "replenished_base": 0.8,
                    "flow_valid": 1,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["heatmap_seconds"] = 5
    frame.attrs["price_step"] = 1.0
    frame.attrs["utc_day"] = start.date().isoformat()
    return frame


def _wall_cfg() -> WallDiscoveryConfig:
    return WallDiscoveryConfig(
        price_step=1.0,
        candidate_widths=(1, 3, 5, 8),
        maximum_distance_bps=3000.0,
        maximum_candidates_per_side=8,
        minimum_bin_events=1,
    )


def test_primitive_cache_preserves_neutral_depth_and_relative_summaries() -> None:
    frames = pd.concat(
        [_snapshot("2026-01-01 00:00:00"), _snapshot("2026-01-01 00:00:05", wall_start=91)],
        ignore_index=True,
    )
    frames.attrs.update({"heatmap_seconds": 5, "price_step": 1.0, "utc_day": "2026-01-01"})
    day = build_liquidity_primitive_day(
        frames,
        reference=CausalPrimitiveReference(window_hours=24),
        config=LiquidityPrimitiveConfig(),
    )
    assert day.snapshot_count == 2
    assert day.cell_count == 80
    first = day.snapshot(0)
    assert first.bucket_end_ms > first.bucket_start_ms
    assert first.bid_q50 == 10.0
    assert first.bid_max == 45.0
    assert first.bid_total > first.ask_total
    assert first.causal_q99 >= first.bid_q50
    assert np.array_equal(first.price_index[first.side_code == 1], np.arange(80, 100))


def test_primitive_candidate_path_matches_dataframe_path() -> None:
    frame = _snapshot("2026-01-01 00:00:00")
    cfg = _wall_cfg()
    old = extract_snapshot_candidates(
        frame,
        cfg,
        CausalDepthReference(window_hours=24, quantile=0.99),
    )
    day = build_liquidity_primitive_day(
        frame,
        reference=CausalPrimitiveReference(window_hours=24),
    )
    new = extract_primitive_snapshot_candidates(day.snapshot(0), cfg)
    old_key = [(x.side, x.low_bin, x.high_bin, x.morphology) for x in old]
    new_key = [(x.side, x.low_bin, x.high_bin, x.morphology) for x in new]
    assert new_key == old_key
    assert np.allclose(
        [x.average_local_multiple for x in new],
        [x.average_local_multiple for x in old],
        rtol=1e-6,
        atol=1e-6,
    )


def test_primitive_engine_tracks_without_dataframe_snapshots() -> None:
    frames = pd.concat(
        [_snapshot(f"2026-01-01 00:00:{second:02d}") for second in (0, 5, 10)],
        ignore_index=True,
    )
    frames.attrs.update({"heatmap_seconds": 5, "price_step": 1.0, "utc_day": "2026-01-01"})
    day = build_liquidity_primitive_day(
        frames,
        reference=CausalPrimitiveReference(window_hours=24),
    )
    engine = WallDiscoveryEngine(_wall_cfg(), source_seconds=5)
    state_count = 0
    for snapshot in day.iter_snapshots():
        _, states = engine.process_primitive_snapshot(snapshot)
        state_count += len(states)
    tracks = engine.finish()
    assert state_count > 0
    assert tracks


def test_primitive_store_is_atomic_and_reloadable(tmp_path: Path) -> None:
    frame = _snapshot("2026-01-01 00:00:00")
    day = build_liquidity_primitive_day(
        frame,
        reference=CausalPrimitiveReference(window_hours=24),
    )
    store = OKXLiquidityPrimitiveStore(data_dir=tmp_path, cache_version="test")
    paths = store.save_day("2026-01-01", arrays=day.arrays, metadata=day.metadata)
    assert paths.primitives.exists()
    assert paths.metadata.exists()
    assert not paths.primitives.with_name(paths.primitives.name + ".part").exists()
    loaded = store.load_day("2026-01-01")
    assert loaded.snapshot_count == 1
    assert loaded.cell_count == 40
    assert loaded.metadata["semantics"]["future_outcomes"] == "not included"


def test_reference_only_reload_matches_full_day(tmp_path: Path) -> None:
    frames = pd.concat(
        [_snapshot("2026-01-01 00:00:00"), _snapshot("2026-01-01 00:00:05", wall_start=91)],
        ignore_index=True,
    )
    frames.attrs.update({"heatmap_seconds": 5, "price_step": 1.0, "utc_day": "2026-01-01"})
    day = build_liquidity_primitive_day(
        frames,
        reference=CausalPrimitiveReference(window_hours=24),
    )
    store = OKXLiquidityPrimitiveStore(data_dir=tmp_path, cache_version="reference-only")
    store.save_day(
        "2026-01-01",
        arrays=day.arrays,
        metadata=day.metadata,
        compression_level=1,
    )

    full_reference = CausalPrimitiveReference(window_hours=24)
    full_reference.replay_day(store.load_day("2026-01-01"))

    compact_reference = CausalPrimitiveReference(window_hours=24)
    compact_reference.replay_arrays(*store.load_reference_arrays("2026-01-01"))

    next_timestamp = int(day.arrays["bucket_end_ms"][-1]) + 5_000
    assert compact_reference.update(next_timestamp, 17.0, 19.0) == full_reference.update(
        next_timestamp, 17.0, 19.0
    )
    assert store.load_metadata("2026-01-01")["compression_level"] == 1
