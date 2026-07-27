from __future__ import annotations

import pandas as pd
import pytest

from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.plugins import build_default_registry
from analyze_tool.plugins import orderbook_liquidity_heatmap as module
from analyze_tool.plugins.orderbook_liquidity_heatmap import OrderBookLiquidityHeatmapPlugin


class FakeLoader:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def coverage(self):
        return []

    def load_heatmap(self, start, end, project_time=True):
        # 2026-06-01 00:00 UTC == project 08:00.  Twelve five-second
        # snapshots form one 1-minute candle.  The final snapshot deliberately
        # has different depth so period-end selection is observable.
        base = 1780272000000
        rows = []
        for slot in range(12):
            ts = base + slot * 5_000
            bid_depth = 10.0 + slot
            ask_depth = 20.0 + slot * 2
            rows.extend(
                [
                    {
                        "bucket_start_ms": ts,
                        "bucket_end_ms": ts + 5_000,
                        "price_index": 1800,
                        "side_code": 1,
                        "depth_base": bid_depth,
                        "depth_usd": bid_depth * 1800,
                        "order_count": 10 + slot,
                        "local_depth_ratio": 1.0,
                        "end_depth_base": bid_depth,
                        "end_depth_usd": bid_depth * 1800,
                        "end_order_count": 10 + slot,
                        "end_local_depth_ratio": 1.0,
                        "added_base": 1.0,
                        "removed_base": 0.5,
                        "executed_base": 0.4,
                        "cancelled_base": 0.1,
                        "consumed_base": 0.4,
                        "replenished_base": 0.4,
                        "flow_valid": 1,
                        "side": "bid",
                        "price_low": 1800.0,
                        "price_high": 1801.0,
                    },
                    {
                        "bucket_start_ms": ts,
                        "bucket_end_ms": ts + 5_000,
                        "price_index": 1801,
                        "side_code": -1,
                        "depth_base": ask_depth,
                        "depth_usd": ask_depth * 1801,
                        "order_count": 20 + slot,
                        "local_depth_ratio": 1.0,
                        "end_depth_base": ask_depth,
                        "end_depth_usd": ask_depth * 1801,
                        "end_order_count": 20 + slot,
                        "end_local_depth_ratio": 1.0,
                        "added_base": 2.0,
                        "removed_base": 3.0,
                        "executed_base": 2.0,
                        "cancelled_base": 1.0,
                        "consumed_base": 2.0,
                        "replenished_base": 2.0,
                        "flow_valid": 1,
                        "side": "ask",
                        "price_low": 1801.0,
                        "price_high": 1802.0,
                    },
                ]
            )
        frame = pd.DataFrame(rows)
        frame.attrs["price_step"] = 1.0
        frame.attrs["heatmap_seconds"] = 5
        return frame

    def load_features(self, *args, **kwargs):
        base = 1780272000000
        return pd.DataFrame(
            {
                "bucket_end_ms": [base + 60_000],
                "available_time": [pd.Timestamp("2026-06-01 08:01:00.001")],
                "book_valid": [1],
                "best_bid": [1800.5],
                "best_ask": [1801.5],
                "mid_price": [1801.0],
                "spread_bps": [5.55],
                "depth_imbalance_25bps": [0.2],
                "top_bid_wall_price": [1800.0],
                "top_bid_wall_depth_base": [21.0],
                "top_ask_wall_price": [1801.0],
                "top_ask_wall_depth_base": [42.0],
                "large_bid_depth_base": [21.0],
                "large_ask_depth_base": [42.0],
                "aggressive_buy_base": [2.0],
                "aggressive_sell_base": [1.0],
                "estimated_bid_cancel_base": [0.1],
                "estimated_ask_cancel_base": [1.0],
                "estimated_bid_consumed_base": [0.4],
                "estimated_ask_consumed_base": [2.0],
                "estimated_bid_replenished_base": [0.4],
                "estimated_ask_replenished_base": [2.0],
            }
        )


def _bars(timeframe: str = "1m") -> tuple[pd.DataFrame, PluginRunContext]:
    index = pd.DatetimeIndex([pd.Timestamp("2026-06-01 08:00:00")])
    bars = pd.DataFrame(
        {"open": [1800.0], "high": [1802.0], "low": [1799.0], "close": [1801.0], "volume": [1.0]},
        index=index,
    )
    return bars, PluginRunContext(
        display_df=bars,
        visible_df=bars,
        request={"timeframe": timeframe},
        meta={"symbol": "ETH-USDT-SWAP"},
    )


def test_plugin_registered_and_defaults_to_period_end(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context, {"normalization": "manual", "manual_max": 100, "large_window_hours": 1}
    )
    assert result.summary["display_mode"] == "period_end"
    assert result.summary["source_heatmap_seconds"] == 5
    assert result.summary["effective_render_seconds"] == 60
    assert len(result.heatmap) == 2
    assert {cell.start_timestamp for cell in result.heatmap} == {"2026-06-01 08:00:00"}
    assert {cell.end_timestamp for cell in result.heatmap} == {"2026-06-01 08:01:00"}
    ids = {item["id"] for item in build_default_registry().list_plugins()}
    assert "offline_orderbook_liquidity_heatmap_v1" in ids


def test_period_end_uses_last_completed_snapshot_and_exact_price_bin(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context, {"normalization": "manual", "manual_max": 100, "large_window_hours": 1}
    )
    by_side = {cell.side: cell for cell in result.heatmap}
    assert by_side["bid"].fields["depth"] == 21.0
    assert by_side["ask"].fields["depth"] == 42.0
    assert by_side["bid"].price_low == 1800.0
    assert by_side["bid"].price_high == 1801.0
    assert by_side["ask"].price_low == 1801.0
    assert by_side["ask"].price_high == 1802.0
    assert by_side["bid"].fields["source_snapshot_end"] == "2026-06-01 08:01:00"
    assert by_side["bid"].fields["source_lag_ms"] == 0


def test_alignment_audit_checks_time_and_price_bins(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context, {"normalization": "manual", "manual_max": 100, "large_window_hours": 1}
    )
    audit = result.summary["alignment_audit"]
    assert audit["time_alignment_ok"] is True
    assert audit["price_alignment_ok"] is True
    assert audit["price_mismatches"] == 0
    assert audit["status"] == "pass"
    assert audit["close_mid_median_bps"] == 0.0


def test_micro_detail_mode_remains_available(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context,
        {
            "display_mode": "micro_detail",
            "time_aggregation": "5s",
            "normalization": "manual",
            "manual_max": 100,
        },
    )
    assert result.summary["display_mode"] == "micro_detail"
    assert result.summary["effective_render_seconds"] == 5
    assert len({cell.start_timestamp for cell in result.heatmap}) == 12


def test_period_end_does_not_carry_disappeared_level_across_utc_day(monkeypatch):
    class CrossDayLoader(FakeLoader):
        def iter_heatmap_days(self, start, end, project_time=True):
            # One 15m project-time bar 07:55 -> 08:10 crosses UTC midnight.
            # The first daily frame contains an extra bid level that is absent
            # from the actual final snapshot in the second daily frame.
            common = {
                "side_code": 1,
                "depth_usd": 0.0,
                "order_count": 1,
                "local_depth_ratio": 1.0,
                "end_order_count": 1,
                "end_local_depth_ratio": 1.0,
                "added_base": 0.0,
                "removed_base": 0.0,
                "executed_base": 0.0,
                "cancelled_base": 0.0,
                "consumed_base": 0.0,
                "replenished_base": 0.0,
                "flow_valid": 1,
                "side": "bid",
            }
            first = pd.DataFrame(
                [
                    {**common, "bucket_start_ms": 1780271990000, "bucket_end_ms": 1780271995000, "price_index": 1799, "price_low": 1799.0, "price_high": 1800.0, "depth_base": 99.0, "end_depth_base": 99.0, "end_depth_usd": 99.0 * 1799},
                    {**common, "bucket_start_ms": 1780271990000, "bucket_end_ms": 1780271995000, "price_index": 1800, "price_low": 1800.0, "price_high": 1801.0, "depth_base": 10.0, "end_depth_base": 10.0, "end_depth_usd": 10.0 * 1800},
                ]
            )
            second = pd.DataFrame(
                [
                    {**common, "bucket_start_ms": 1780272590000, "bucket_end_ms": 1780272595000, "price_index": 1800, "price_low": 1800.0, "price_high": 1801.0, "depth_base": 20.0, "end_depth_base": 20.0, "end_depth_usd": 20.0 * 1800},
                ]
            )
            for frame in (first, second):
                frame.attrs["price_step"] = 1.0
                frame.attrs["heatmap_seconds"] = 5
                yield frame

        def load_features(self, *args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", CrossDayLoader)
    index = pd.DatetimeIndex([pd.Timestamp("2026-06-01 07:55:00")])
    bars = pd.DataFrame(
        {"open": [1800.0], "high": [1802.0], "low": [1799.0], "close": [1801.0], "volume": [1.0]},
        index=index,
    )
    context = PluginRunContext(display_df=bars, visible_df=bars, request={"timeframe": "15m"}, meta={"symbol": "ETH-USDT-SWAP"})
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context, {"normalization": "manual", "manual_max": 100, "large_window_hours": 1}
    )
    assert all(cell.price_low != 1799.0 for cell in result.heatmap)
    assert len(result.heatmap) == 1
    assert result.heatmap[0].fields["depth"] == 20.0


def test_period_end_rejects_legacy_average_only_artifact(monkeypatch):
    class LegacyLoader(FakeLoader):
        def load_heatmap(self, start, end, project_time=True):
            frame = super().load_heatmap(start, end, project_time=project_time)
            return frame.drop(columns=[
                "end_depth_base", "end_depth_usd", "end_order_count", "end_local_depth_ratio"
            ])

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", LegacyLoader)
    _, context = _bars()
    import pytest
    with pytest.raises(ValueError, match="force-rebuild"):
        OrderBookLiquidityHeatmapPlugin().run_with_context(
            context, {"normalization": "manual", "manual_max": 100, "large_window_hours": 1}
        )


def test_v254_wall_overlay_draws_one_deep_blue_fixed_rectangle(monkeypatch):
    from src.liquidity_map.wall_detector import PersistentLiquidityWall

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    base = 1780272000000
    wall = PersistentLiquidityWall(
        wall_id=7,
        side="bid",
        wall_type="ZONE",
        first_seen_ms=base,
        confirmed_at_ms=base + 20_000,
        last_seen_ms=base + 45_000,
        end_ms=base + 50_000,
        price_low=1798.0,
        price_high=1801.0,
        center_price=1799.5,
        duration_minutes=50 / 60,
        confirmed_duration_minutes=30 / 60,
        time_coverage=0.9,
        price_coverage=0.75,
        unique_price_bins=4,
        span_price_bins=3,
        average_depth=100.0,
        peak_depth=140.0,
        average_threshold_ratio=4.0,
        peak_threshold_ratio=7.0,
        maximum_observed_fade_minutes=0.0,
        observations=10,
        strength_score=80.0,
        active_at_end=True,
        fields={
            "detector_version": "v2_realtime_snapshot",
            "timeline": [
                {
                    "start_ms": base,
                    "end_ms": base + 20_000,
                    "price_low": 1799.0,
                    "price_high": 1801.0,
                    "status": "FORMING",
                    "depth_sum": 80.0,
                    "peak_depth": 50.0,
                    "median_depth": 10.0,
                    "peak_snapshot_ratio": 5.0,
                    "zone_snapshot_ratio": 4.0,
                    "peak_local_contrast": 4.0,
                    "price_indices": [1799, 1800],
                    "core_indices": [1800],
                },
                {
                    "start_ms": base + 20_000,
                    "end_ms": base + 35_000,
                    "price_low": 1799.0,
                    "price_high": 1801.0,
                    "status": "CONFIRMED",
                    "depth_sum": 100.0,
                    "peak_depth": 70.0,
                    "median_depth": 10.0,
                    "peak_snapshot_ratio": 7.0,
                    "zone_snapshot_ratio": 5.0,
                    "peak_local_contrast": 6.0,
                    "price_indices": [1799, 1800],
                    "core_indices": [1800],
                },
                {
                    "start_ms": base + 35_000,
                    "end_ms": base + 50_000,
                    "price_low": 1798.0,
                    "price_high": 1800.0,
                    "status": "CONFIRMED",
                    "depth_sum": 110.0,
                    "peak_depth": 75.0,
                    "median_depth": 10.0,
                    "peak_snapshot_ratio": 7.5,
                    "zone_snapshot_ratio": 5.5,
                    "peak_local_contrast": 6.5,
                    "price_indices": [1798, 1799],
                    "core_indices": [1799],
                },
            ],
        },
    )
    monkeypatch.setattr(module, "detect_persistent_liquidity_walls", lambda *args, **kwargs: [wall])
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context,
        {
            "normalization": "manual",
            "manual_max": 100,
            "large_window_hours": 1,
            "wall_show_forming": "no",
        },
    )
    assert len(result.price_regions) == 1
    region = result.price_regions[0]
    assert (region.price_low, region.price_high) == (1798.0, 1801.0)
    assert region.color == "#00AEEF"
    assert region.fields["rectangular_wall"] is True


def test_default_color_scale_is_causal_24h_global_max_with_50pct_ui_saturation(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context,
        {
            "color_window_hours": 24,
            "large_window_hours": 1,
            "wall_min_strength_score": 0,
        },
    )
    by_side = {cell.side: cell for cell in result.heatmap}
    # Final period-end bid/ask depths are 21 and 42 ETH.  The shared robust
    # P99 reference is approximately the ask depth, so bid remains near 50%
    # instead of making both sides 100%.
    assert by_side["bid"].intensity == pytest.approx(0.5, abs=0.01)
    assert by_side["ask"].intensity == 1.0
    assert by_side["bid"].fields["causal_color_cap"] == pytest.approx(41.79, abs=0.01)
    assert by_side["ask"].fields["causal_color_cap"] == pytest.approx(41.79, abs=0.01)
    assert result.summary["color_reference_semantics"] == "rolling max of per-snapshot robust high quantile"
    assert result.summary["ui"]["heatmap_color_max_pct"] == 50


def test_wall_detector_receives_one_final_snapshot_per_chart_bar(monkeypatch):
    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLoader)
    captured = {}

    def capture(frame, *, depth_column, config):
        captured["frame"] = frame.copy()
        captured["attrs"] = dict(frame.attrs)
        captured["depth_column"] = depth_column
        captured["config"] = config
        return []

    monkeypatch.setattr(module, "detect_persistent_liquidity_walls", capture)
    _, context = _bars("1m")
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context,
        {"normalization": "manual", "manual_max": 100, "large_window_hours": 1},
    )
    wall_frame = captured["frame"]
    assert captured["attrs"]["heatmap_seconds"] == 60
    assert wall_frame["bucket_start_ms"].nunique() == 1
    by_side = wall_frame.set_index("side_code")
    assert by_side.loc[1, "end_depth_base"] == 21.0
    assert by_side.loc[-1, "end_depth_base"] == 42.0
    assert result.summary["wall_detector_version"] == "v2_5_4_strict_rectangular_market_side"
    assert result.summary["wall_input_semantics"] == "historical bar final snapshot; live bar latest snapshot"


def test_period_end_plugin_uses_compact_cache_fast_path(monkeypatch):
    class CachedLoader(FakeLoader):
        def load_heatmap(self, *args, **kwargs):
            raise AssertionError("raw heatmap path must not be used when compact cache is available")

        def iter_period_end_snapshot_days(self, start, end, *, timeframe, price_step, project_time=True):
            base = 1780272000000
            frame = pd.DataFrame(
                [
                    {
                        "bucket_start_ms": base,
                        "bucket_end_ms": base + 60_000,
                        "source_bucket_start_ms": base + 55_000,
                        "source_bucket_end_ms": base + 60_000,
                        "price_index": 1800,
                        "side_code": 1,
                        "flow_valid": 1,
                        "end_depth_base": 21.0,
                        "end_depth_usd": 21.0 * 1800,
                        "end_order_count": 10,
                        "added_base": 1.0,
                        "removed_base": 0.5,
                        "executed_base": 0.4,
                        "cancelled_base": 0.1,
                        "consumed_base": 0.4,
                        "replenished_base": 0.4,
                        "side": "bid",
                        "price_low": 1800.0,
                        "price_high": 1801.0,
                    },
                    {
                        "bucket_start_ms": base,
                        "bucket_end_ms": base + 60_000,
                        "source_bucket_start_ms": base + 55_000,
                        "source_bucket_end_ms": base + 60_000,
                        "price_index": 1801,
                        "side_code": -1,
                        "flow_valid": 1,
                        "end_depth_base": 42.0,
                        "end_depth_usd": 42.0 * 1801,
                        "end_order_count": 20,
                        "added_base": 2.0,
                        "removed_base": 3.0,
                        "executed_base": 2.0,
                        "cancelled_base": 1.0,
                        "consumed_base": 2.0,
                        "replenished_base": 2.0,
                        "side": "ask",
                        "price_low": 1801.0,
                        "price_high": 1802.0,
                    },
                ]
            )
            frame.attrs.update(
                {
                    "source_price_step": 1.0,
                    "source_heatmap_seconds": 5,
                    "price_step": 1.0,
                    "heatmap_seconds": 60,
                    "source_row_count": 24,
                    "cache_hit": True,
                    "cache_path": "fake-period-end-cache.npz",
                    "utc_day": "2026-06-01",
                }
            )
            yield frame

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", CachedLoader)
    _, context = _bars()
    result = OrderBookLiquidityHeatmapPlugin().run_with_context(
        context,
        {"normalization": "manual", "manual_max": 100, "large_window_hours": 1},
    )
    assert len(result.heatmap) == 2
    assert result.summary["period_end_cache_enabled"] is True
    assert result.summary["period_end_cache_hits"] == 1
    assert result.summary["period_end_cache_misses"] == 0
