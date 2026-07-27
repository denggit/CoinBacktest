from __future__ import annotations

import pandas as pd

from src.liquidity_map.wall_detector import PersistentWallConfig, detect_persistent_liquidity_walls


def _bar_snapshots(*, start: str, bars: int, wall_fn, step_minutes: int = 15) -> pd.DataFrame:
    rows: list[dict] = []
    origin = pd.Timestamp(start, tz="UTC")
    for bar in range(bars):
        bucket_start = int((origin + pd.Timedelta(minutes=bar * step_minutes)).timestamp() * 1000)
        bucket_end = bucket_start + step_minutes * 60_000
        for price in range(80, 100):
            depth = float(wall_fn(bar, price, "bid") or 5.0)
            rows.append({
                "bucket_start_ms": bucket_start,
                "bucket_end_ms": bucket_end,
                "price_index": price,
                "side_code": 1,
                "price_low": float(price),
                "price_high": float(price + 1),
                "end_depth_base": depth,
            })
        for price in range(100, 120):
            depth = float(wall_fn(bar, price, "ask") or 5.0)
            rows.append({
                "bucket_start_ms": bucket_start,
                "bucket_end_ms": bucket_end,
                "price_index": price,
                "side_code": -1,
                "price_low": float(price),
                "price_high": float(price + 1),
                "end_depth_base": depth,
            })
    frame = pd.DataFrame(rows)
    frame.attrs["heatmap_seconds"] = step_minutes * 60
    frame.attrs["price_step"] = 1.0
    return frame


def _config(**overrides) -> PersistentWallConfig:
    values = {
        "reference_window_hours": 24,
        "reference_snapshot_quantile": 1.0,
        "lookback_hours": 2.0,
        "minimum_history_bars": 4,
        "strong_depth_ratio": 0.50,
        "zone_depth_ratio": 0.30,
        "support_depth_ratio": 0.15,
        "isolated_point_ratio": 0.50,
        "minimum_support_time_coverage": 0.60,
        "minimum_zone_time_coverage": 0.45,
        "minimum_core_time_coverage": 0.15,
        "minimum_average_depth_ratio": 0.12,
        "minimum_current_depth_ratio": 0.08,
        "minimum_zone_band_points": 3,
        "minimum_zone_support_points": 4,
        "minimum_zone_density_mass": 0.90,
        "minimum_zone_price_coverage": 0.45,
        "minimum_zone_strong_points": 1,
        "strongless_zone_min_band_points": 4,
        "point_minimum_zone_coverage": 0.55,
        "point_minimum_core_coverage": 0.20,
        "maximum_distance_bps": 3000,
        "maximum_missing_price_bins": 2,
        "maximum_cluster_span_bins": 18,
        "minimum_confirm_bars": 1,
        "persistent_after_minutes": 60,
        "major_after_minutes": 240,
        "maximum_missing_bars": 2,
        "minimum_match_overlap": 0.35,
        "maximum_center_drift_bins": 2,
        "boundary_smoothing_bars": 4,
        "minimum_strength_score": 0,
        "maximum_walls": 100,
    }
    values.update(overrides)
    return PersistentWallConfig(**values)


def test_persistent_single_line_becomes_point_wall() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side == "bid" and price == 85:
            return 100.0
        if side == "bid" and price == 95 and 1 <= bar <= 10:
            return 60.0
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-01", bars=12, wall_fn=wall_fn), config=_config()
    )
    point = [wall for wall in walls if wall.side == "bid" and wall.wall_type == "POINT" and wall.price_low == 95]
    assert point
    assert point[0].fields["detector_version"] == "v2_5_4_strict_rectangular_market_side"
    assert point[0].fields["input_semantics"] == "one final/latest order-book snapshot per chart bar"


def test_deep_and_light_rows_persisting_together_form_one_main_wall() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "bid":
            return None
        if price == 85:
            return 100.0
        if 1 <= bar <= 10:
            return {90: 55.0, 91: 18.0, 92: 35.0, 93: 40.0}.get(price)
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-02", bars=12, wall_fn=wall_fn), config=_config()
    )
    mains = [wall for wall in walls if wall.side == "bid" and wall.wall_type == "MAIN" and wall.price_low <= 90]
    assert len(mains) == 1
    active = [item for item in mains[0].fields["timeline"] if item["status"] != "FADING"]
    assert active
    assert active[-1]["price_low"] == 90.0
    assert active[-1]["price_high"] == 94.0
    assert set(active[-1]["price_indices"]) == {90, 91, 92, 93}


def test_one_bar_flash_does_not_become_wall() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side == "ask" and price == 115:
            return 100.0
        if side == "ask" and price == 105 and bar == 3:
            return 90.0
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-03", bars=10, wall_fn=wall_fn), config=_config()
    )
    assert not [wall for wall in walls if wall.side == "ask" and wall.price_low <= 105 < wall.price_high]


def test_two_missing_bars_keep_one_wall_identity_with_fading_gap() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "ask":
            return None
        if price == 115:
            return 100.0
        if 1 <= bar <= 7 or 10 <= bar <= 15:
            return {105: 55.0, 106: 35.0, 107: 32.0, 108: 18.0}.get(price)
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-04", bars=17, wall_fn=wall_fn),
        config=_config(maximum_missing_bars=2),
    )
    matching = [wall for wall in walls if wall.side == "ask" and wall.price_low <= 105 < wall.price_high]
    assert len(matching) == 1
    assert "FADING" in {item["status"] for item in matching[0].fields["timeline"]}


def test_moving_single_line_is_not_rewritten_as_one_wide_wall() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side == "bid" and price == 85:
            return 100.0
        if side == "bid" and price == 90 + min(bar, 8):
            return 70.0
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-05", bars=12, wall_fn=wall_fn),
        config=_config(maximum_center_drift_bins=1),
    )
    for wall in walls:
        if wall.side != "bid" or wall.price_low == 85:
            continue
        for item in wall.fields["timeline"]:
            assert item["price_high"] - item["price_low"] <= 2.0


def test_later_expansion_does_not_backfill_earlier_wall_bounds() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "bid":
            return None
        if price == 85:
            return 100.0
        if 1 <= bar <= 14:
            values = {90: 55.0, 91: 20.0, 92: 35.0, 93: 40.0}
            if bar >= 9:
                values[94] = 35.0
            return values.get(price)
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-06", bars=16, wall_fn=wall_fn), config=_config()
    )
    main = [wall for wall in walls if wall.side == "bid" and wall.wall_type == "MAIN" and wall.price_low <= 90][0]
    active = [item for item in main.fields["timeline"] if item["status"] != "FADING"]
    assert active
    # Timeline remains available for causal audit, but the public wall uses one
    # stable rectangle extracted from price bins that recur through the lifecycle.
    assert main.fields["rectangle_price_low"] == main.price_low
    assert main.fields["rectangle_price_high"] == main.price_high
    assert main.price_high - main.price_low <= 5.0


def test_config_still_accepts_old_fields_for_patch_compatibility() -> None:
    config = PersistentWallConfig(
        snapshot_depth_quantile=0.9,
        band_reference_quantile=0.75,
        minimum_confirm_seconds=30,
        maximum_fade_minutes=10,
    )
    config.validate()
    assert config.lookback_hours == 4.0


def test_price_band_wall_survives_two_bin_snapshot_drift() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "bid":
            return None
        if price == 85:
            return 100.0
        center = 94 + (bar % 3) - 1  # 93, 94, 95, still one visual band
        if 1 <= bar <= 12 and price in {center, center + 1}:
            return 60.0 if price == center else 35.0
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-08", bars=14, wall_fn=wall_fn),
        config=_config(history_price_tolerance_bins=2, maximum_center_drift_bins=2),
    )
    matching = [wall for wall in walls if wall.side == "bid" and wall.price_low <= 94 < wall.price_high]
    assert matching
    assert len({wall.wall_id for wall in matching}) == 1
    assert matching[0].fields["history_price_tolerance_bins"] == 2


def test_sparse_price_band_is_not_a_main_wall() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "bid":
            return None
        if price == 85:
            return 100.0
        if 1 <= bar <= 12:
            return {90: 55.0, 92: 40.0, 94: 35.0}.get(price)
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-09", bars=14, wall_fn=wall_fn),
        config=_config(
            history_price_tolerance_bins=0,
            maximum_missing_price_bins=1,
            minimum_zone_price_coverage=0.70,
            minimum_rectangle_support_occupancy=0.65,
        ),
    )
    assert not [wall for wall in walls if wall.side == "bid" and wall.wall_type == "MAIN" and 90 <= wall.center_price <= 95]


def test_crossed_book_level_cannot_become_wall_around_market_price() -> None:
    rows = []
    origin = pd.Timestamp("2026-01-10", tz="UTC")
    for bar in range(8):
        start = int((origin + pd.Timedelta(minutes=15 * bar)).timestamp() * 1000)
        end = start + 15 * 60_000
        # Deliberately crossed/invalid book: the deep bid is above the ask, so
        # its price rectangle would contain the reconstructed market midpoint.
        for price, side_code, depth in [(101, 1, 100.0), (100, -1, 100.0)]:
            rows.append({
                "bucket_start_ms": start, "bucket_end_ms": end,
                "price_index": price, "side_code": side_code,
                "price_low": float(price), "price_high": float(price + 1),
                "end_depth_base": depth,
            })
    frame = pd.DataFrame(rows)
    frame.attrs["heatmap_seconds"] = 900
    frame.attrs["price_step"] = 1.0
    walls = detect_persistent_liquidity_walls(
        frame,
        config=_config(minimum_market_clearance_bins=1),
    )
    assert walls == []


def test_final_wall_bounds_are_one_fixed_persistent_rectangle() -> None:
    def wall_fn(bar: int, price: int, side: str):
        if side != "bid":
            return None
        if price == 85:
            return 100.0
        if 1 <= bar <= 13:
            center = 92 + (bar % 2)
            return {center: 60.0, center + 1: 45.0, center + 2: 35.0, center + 3: 25.0}.get(price)
        return None

    walls = detect_persistent_liquidity_walls(
        _bar_snapshots(start="2026-01-11", bars=15, wall_fn=wall_fn),
        config=_config(
            history_price_tolerance_bins=2,
            rectangle_price_persistence=0.60,
            minimum_rectangle_support_occupancy=0.50,
            minimum_rectangle_zone_occupancy=0.25,
            minimum_current_support_occupancy=0.50,
        ),
    )
    main = [wall for wall in walls if wall.side == "bid" and wall.wall_type == "MAIN" and 91 <= wall.center_price <= 96]
    assert main
    wall = main[0]
    assert wall.price_low == wall.fields["rectangle_price_low"]
    assert wall.price_high == wall.fields["rectangle_price_high"]
    assert wall.price_high > wall.price_low
