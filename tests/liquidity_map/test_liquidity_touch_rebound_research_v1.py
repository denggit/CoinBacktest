from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity.liquidity_touch_rebound_v1.core import (
    Band,
    LiquidityResearchConfig,
    Snapshot,
    SnapshotSide,
    Variant,
    build_touch_events,
    enrich_touch_events_with_book_features,
    extract_bands,
    history_features,
    prepare_bar_features,
    simulate_variant,
)


def _side(values: dict[int, float]) -> SnapshotSide:
    indices = np.asarray(sorted(values), dtype=np.int32)
    ratios = np.asarray([values[int(i)] for i in indices], dtype=np.float32)
    return SnapshotSide(
        price_indices=indices,
        ratios=ratios,
        depth_base=ratios * 100.0,
        depth_usd=ratios * 300_000.0,
    )


def _snapshot(ts: str, bid: dict[int, float], ask: dict[int, float] | None = None) -> Snapshot:
    t = pd.Timestamp(ts)
    return Snapshot(
        available_time=t,
        available_time_ms=int((t - pd.Timedelta(hours=8)).tz_localize("UTC").timestamp() * 1000),
        target_bar_end=t,
        target_bar_end_ms=int((t - pd.Timedelta(hours=8)).tz_localize("UTC").timestamp() * 1000),
        reference_depth=100.0,
        snapshot_p99_depth=80.0,
        bid=_side(bid),
        ask=_side(ask or {}),
    )


def test_single_deep_line_is_not_main_band_but_wide_medium_band_is() -> None:
    cfg = LiquidityResearchConfig(
        support_ratio=0.10,
        body_ratio=0.22,
        core_ratio=0.45,
        min_band_width_bins=2,
        min_body_bins=2,
    )
    single = extract_bands(_side({100: 0.95}), side_name="bid", cfg=cfg)
    assert single == []

    wide = extract_bands(
        _side({100: 0.25, 101: 0.31, 102: 0.27, 103: 0.42, 104: 0.18}),
        side_name="bid",
        cfg=cfg,
    )
    assert len(wide) == 1
    assert wide[0].width_bins == 5
    assert wide[0].body_count == 4
    assert wide[0].core_count == 0


def test_history_fixedness_penalizes_directional_drift() -> None:
    cfg = LiquidityResearchConfig(history_bars=8)
    band = Band(
        side="bid",
        low_bin=100,
        high_bin=104,
        price_step=1.0,
        support_count=5,
        body_count=4,
        core_count=1,
        support_occupancy=1.0,
        body_occupancy=0.8,
        core_occupancy=0.2,
        mean_ratio=0.35,
        p90_ratio=0.5,
        peak_ratio=0.6,
        normalized_mass=1.75,
        depth_base_sum=100.0,
        depth_usd_sum=300_000.0,
        weighted_center_bin=102.0,
        hole_ratio=0.0,
    )
    fixed = [_snapshot(f"2026-01-01 {i:02d}:00", {100: 0.3, 101: 0.35, 102: 0.5, 103: 0.3, 104: 0.25}) for i in range(8)]
    drifting = []
    for i in range(8):
        center = 99 + i
        drifting.append(_snapshot(f"2026-01-02 {i:02d}:00", {center: 0.3, center + 1: 0.5, center + 2: 0.3}))

    fixed_features = history_features(band, fixed, cfg=cfg)
    drift_features = history_features(band, drifting, cfg=cfg)
    assert fixed_features["history_fixedness_score"] > drift_features["history_fixedness_score"]
    assert drift_features["history_ghost_drift_score"] > fixed_features["history_ghost_drift_score"]


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 10:00", periods=12, freq="1min")
    frame = pd.DataFrame(
        {
            "open": [105, 104, 103, 102, 101, 101, 102, 103, 104, 105, 106, 107],
            "high": [106, 105, 104, 103, 102, 102, 103, 104, 105, 106, 107, 108],
            "low": [104, 103, 102, 101, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            "close": [104.5, 103.5, 102.5, 101.5, 101.2, 101.8, 102.8, 103.8, 104.8, 105.8, 106.8, 107.8],
            "volume": 100.0,
            "buy_notional": 40.0,
            "sell_notional": 60.0,
            "delta_notional": -20.0,
            "large_buy_notional": 0.0,
            "large_sell_notional": 10.0,
            "max_trade_notional": 10.0,
        },
        index=idx,
    )
    return prepare_bar_features(frame)


def test_touch_and_upper_liquidity_target_simulation() -> None:
    bars = _bars()
    cfg = LiquidityResearchConfig(
        touch_horizon_minutes=10,
        max_holding_minutes=10,
        round_trip_fee=0.0011,
        minimum_raw_rr=0.5,
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": 1,
                "snapshot_time": pd.Timestamp("2026-01-01 10:00"),
                "target_bar_end": pd.Timestamp("2026-01-01 10:00"),
                "bid_price_low": 99.0,
                "bid_price_high": 101.0,
                "bid_center_price": 100.0,
                "bid_width_bins": 2,
                "bid_body_occupancy": 1.0,
                "history_weighted_continuity": 0.8,
                "history_fixedness_score": 0.9,
                "history_ghost_drift_score": 0.0,
                "history_recent_retention": 1.1,
                "history_recent_slope": 0.1,
                "target_available": True,
                "ask_price_low": 106.0,
                "ask_price_high": 108.0,
                "raw_target_distance_bps": 495.0,
                "macro_uptrend": True,
                "snapshot_available_time_ms": 0,
                "snapshot_reference_depth": 1.0,
                "snapshot_p99_depth": 1.0,
                "market_price": 105.0,
                "distance_to_bid_band_bps": 380.0,
                "atr14_at_signal": 1.0,
                "atr_pct_at_signal": 0.01,
                "ret_15_at_signal": 0.0,
                "ret_60_at_signal": 0.0,
            }
        ]
    )
    events = build_touch_events(candidates, bars, cfg=cfg)
    assert len(events) == 1
    assert events.iloc[0]["touch_time"] == pd.Timestamp("2026-01-01 10:03")

    variant = Variant(
        name="test",
        entry_mode="limit",
        stop_buffer_bins=1,
        take_profit_buffer_bins=0,
        stop_buffer_wall_fraction=0.0,
        max_holding_minutes=10,
        fee_rate=0.0011,
        minimum_raw_rr=0.5,
    )
    trades = simulate_variant(events, bars, cfg=cfg, variant=variant)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "tp_upper_liquidity"
    assert trades.iloc[0]["target_price"] == 106.0
    assert trades.iloc[0]["stop_price"] == 98.0


def test_passive_limit_counts_touch_bar_stop_conservatively() -> None:
    bars = _bars()
    cfg = LiquidityResearchConfig(
        touch_horizon_minutes=10,
        max_holding_minutes=10,
        round_trip_fee=0.0011,
        minimum_raw_rr=0.5,
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": 2,
                "snapshot_time": pd.Timestamp("2026-01-01 10:00"),
                "target_bar_end": pd.Timestamp("2026-01-01 10:00"),
                "bid_price_low": 99.5,
                "bid_price_high": 100.0,
                "bid_center_price": 99.75,
                "bid_width_bins": 2,
                "bid_body_occupancy": 1.0,
                "history_weighted_continuity": 0.8,
                "history_fixedness_score": 0.9,
                "history_ghost_drift_score": 0.0,
                "history_recent_retention": 1.1,
                "history_recent_slope": 0.1,
                "target_available": True,
                "ask_price_low": 106.0,
                "ask_price_high": 108.0,
                "raw_target_distance_bps": 495.0,
                "macro_uptrend": True,
                "snapshot_available_time_ms": 0,
                "snapshot_reference_depth": 1.0,
                "snapshot_p99_depth": 1.0,
                "market_price": 105.0,
                "distance_to_bid_band_bps": 380.0,
                "atr14_at_signal": 1.0,
                "atr_pct_at_signal": 0.01,
                "ret_15_at_signal": 0.0,
                "ret_60_at_signal": 0.0,
            }
        ]
    )
    events = build_touch_events(candidates, bars, cfg=cfg)
    assert len(events) == 1
    variant = Variant(
        name="same_bar_stop",
        entry_mode="limit",
        stop_buffer_bins=0,
        take_profit_buffer_bins=0,
        stop_buffer_wall_fraction=0.0,
        max_holding_minutes=10,
        fee_rate=0.0011,
        minimum_raw_rr=0.5,
    )
    trades = simulate_variant(events, bars, cfg=cfg, variant=variant)
    assert len(trades) == 1
    assert trades.iloc[0]["stop_price"] == 99.5
    assert trades.iloc[0]["exit_reason"] == "sl_touch_bar_conservative"
    assert trades.iloc[0]["exit_time"] == pd.Timestamp("2026-01-01 10:04")



def test_exact_touch_book_flow_enrichment_is_causal() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": 0,
                "touch_time": pd.Timestamp("2026-01-01 10:04:00"),
                "bid_price_high": 100.0,
                "touch_penetration_fraction": 0.5,
            }
        ]
    )
    times = pd.date_range("2026-01-01 10:03:00", periods=121, freq="1s")
    features = pd.DataFrame(
        {
            "available_time": times,
            "book_valid": 1,
            "best_ask": np.where(times < pd.Timestamp("2026-01-01 10:04:20"), 101.0, 100.0),
            "aggressive_buy_base": 1.0,
            "aggressive_sell_base": 2.0,
            "trade_delta_base": -1.0,
            "book_added_bid_base": 3.0,
            "book_removed_bid_base": 2.0,
            "estimated_bid_cancel_base": 0.5,
            "estimated_bid_consumed_base": 1.0,
            "estimated_bid_replenished_base": 2.0,
            "spread_bps": 1.0,
            "bid_depth_25bps_base": 100.0,
            "ask_depth_25bps_base": 80.0,
            "depth_imbalance_25bps": np.linspace(0.1, 0.3, len(times)),
            "top_bid_wall_depth_base": 20.0,
            "top_bid_wall_ratio": 0.5,
            "top_bid_wall_distance_bps": 5.0,
            "nearest_large_bid_depth_base": 10.0,
        }
    )
    enriched = enrich_touch_events_with_book_features(events, features, pre_touch_seconds=60)
    row = enriched.iloc[0]
    assert bool(row["touch_exact_found"])
    assert row["touch_exact_time"] == pd.Timestamp("2026-01-01 10:04:20")
    assert row["touch_exact_seconds_from_bar_start"] == 20.0
    # The 60-second pre window ends before the exact contact second.
    assert row["pre_aggressive_sell_base"] == 120.0
    assert row["pre_bid_replenish_to_consume"] == 2.0
    assert row["touch_bid_replenish_to_consume"] == 2.0
    assert row["touch_depth_imbalance_change"] > 0
