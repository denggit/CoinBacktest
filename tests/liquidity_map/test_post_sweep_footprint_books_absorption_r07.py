from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.research_common.post_sweep_footprint_books.books import _event_book_features
from src.research_common.post_sweep_footprint_books.config import PostSweepFootprintBooksConfig
from src.research_common.post_sweep_footprint_books.footprint import (
    aggregate_footprint_bars,
    attach_footprint_context,
)
from src.research_common.post_sweep_footprint_books.reports import (
    causal_audit,
    feature_outcome_auc,
    frozen_quantile_lift,
    pair_overlap_summary,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_id": [1, 2],
            "start_ts": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "end_ts": pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "duration_seconds": [60.0, 60.0],
            "open": [100.0, 99.8],
            "high": [100.1, 99.9],
            "low": [99.8, 99.6],
            "close": [99.8, 99.7],
            "direction": [-1.0, -1.0],
            "notional": [10e6, 12e6],
            "buy_notional": [3e6, 3e6],
            "sell_notional": [7e6, 9e6],
            "delta_notional": [-4e6, -6e6],
            "large_buy_notional": [0.2e6, 0.2e6],
            "large_sell_notional": [0.5e6, 0.8e6],
            "large_delta_notional": [-0.3e6, -0.6e6],
            "max_trade_notional": [100_000.0, 150_000.0],
        }
    )


def _footprints() -> pd.DataFrame:
    rows = []
    for bar_id, low in ((1, 99.8), (2, 99.6)):
        for offset in range(5):
            sell = float((5 - offset) * 1_000_000)
            buy = float((offset + 1) * 200_000)
            rows.append(
                {
                    "bar_id": bar_id,
                    "price_bucket": low + offset,
                    "notional": buy + sell,
                    "trades_count": 20,
                    "buy_notional": buy,
                    "sell_notional": sell,
                    "delta_notional": buy - sell,
                    "large_buy_notional": 0.0,
                    "large_sell_notional": sell * 0.1,
                    "large_delta_notional": -sell * 0.1,
                    "max_trade_notional": 50_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_footprint_aggregation_builds_low_zone_absorption_features() -> None:
    cfg = PostSweepFootprintBooksConfig().validate()
    out = aggregate_footprint_bars(_bars(), _footprints(), cfg)
    assert len(out) == 2
    assert (out["fp_low3_sell_share"] > 0.5).all()
    assert (out["fp_low3_notional_share"] > 0).all()
    assert out.loc[1, "fp_new_low_extension_bp_vs_prev"] > 0
    assert out.loc[1, "fp_prev_down_bar_id"] == 1


def test_footprint_asof_never_uses_unfinished_bar() -> None:
    cfg = PostSweepFootprintBooksConfig().validate()
    bars = aggregate_footprint_bars(_bars(), _footprints(), cfg)
    events = pd.DataFrame(
        {
            "checkpoint_id": ["E1", "E2"],
            "checkpoint_available_time": pd.to_datetime(["2025-01-01 00:01:30", "2025-01-01 00:02:00"]),
        }
    )
    out = attach_footprint_context(events, bars)
    assert out["fp_bar_id"].tolist() == [1, 2]
    assert (out["fp_end_ts"] <= out["checkpoint_available_time"]).all()
    assert out["fp_causal_valid"].all()


def test_pair_overlap_reports_same_completed_bar() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["P1", "P1", "P2", "P2"],
            "period": ["H", "H", "H", "H"],
            "cohort": ["ORACLE_TURN", "PRIOR_FAILED_ATTEMPT"] * 2,
            "fp_bar_id": [10, 10, 12, 11],
            "fp_end_ts": pd.to_datetime(["2025-01-01"] * 4),
        }
    )
    out = pair_overlap_summary(frame)
    all_row = out.loc[out["period"] == "ALL"].iloc[0]
    assert all_row["pairs"] == 2
    assert all_row["same_completed_footprint_bar_rate"] == 0.5


def test_outcome_auc_and_frozen_threshold_are_not_refit_in_holdout() -> None:
    features = pd.DataFrame(
        {
            "checkpoint_id": [f"E{i}" for i in range(400)],
            "period": ["EARLY_2023_2024"] * 200 + ["LATE"] * 200,
            "fp_low3_sell_share": np.tile(np.linspace(0, 1, 200), 2),
        }
    )
    labels = pd.DataFrame(
        {
            "checkpoint_id": features["checkpoint_id"],
            "period": features["period"],
            "future_no_lower_low_60m": features["fp_low3_sell_share"] > 0.5,
        }
    )
    auc = feature_outcome_auc(
        features,
        labels,
        metrics=("fp_low3_sell_share",),
        outcomes=("future_no_lower_low_60m",),
        minimum_events=50,
    )
    assert (auc["separation_auc"] > 0.9).all()
    lift = frozen_quantile_lift(
        features,
        labels,
        reference_period="EARLY_2023_2024",
        metrics=("fp_low3_sell_share",),
        outcomes=("future_no_lower_low_60m",),
        minimum_events=50,
    )
    assert lift["frozen_threshold"].nunique() == 1
    assert set(lift["period"]) == {"EARLY_2023_2024", "LATE"}


def test_books_features_use_only_rows_available_by_event() -> None:
    times = pd.date_range("2025-01-01 00:00:00", periods=4, freq="5s")
    frame = pd.DataFrame(
        {
            "available_time": times,
            "book_valid": 1,
            "trade_attribution_valid": 1,
            "bid_depth_5bps_base": [100, 50, 80, 9999],
            "ask_depth_5bps_base": [100, 100, 90, 1],
            "bid_depth_10bps_base": 100,
            "ask_depth_10bps_base": 100,
            "bid_depth_25bps_base": 200,
            "ask_depth_25bps_base": 200,
            "depth_imbalance_25bps": [0.0, -0.2, 0.1, 0.9],
            "top_bid_wall_ratio": 2.0,
            "top_bid_wall_distance_bps": 3.0,
            "top_bid_wall_depth_base": 50.0,
            "top_ask_wall_ratio": 2.0,
            "top_ask_wall_distance_bps": 3.0,
            "top_ask_wall_depth_base": 50.0,
            "aggressive_buy_base": [1, 1, 1, 1000],
            "aggressive_sell_base": [10, 10, 10, 1000],
            "book_added_bid_base": [0, 20, 30, 1000],
            "book_added_ask_base": 0,
            "book_removed_bid_base": [0, 50, 0, 1000],
            "book_removed_ask_base": 0,
            "estimated_bid_cancel_base": [0, 10, 0, 1000],
            "estimated_ask_cancel_base": 0,
            "estimated_bid_consumed_base": [0, 40, 0, 1000],
            "estimated_ask_consumed_base": 0,
            "estimated_bid_replenished_base": [0, 20, 30, 1000],
            "estimated_ask_replenished_base": 0,
        }
    )
    cfg = PostSweepFootprintBooksConfig(books_lookback_seconds=60).validate()
    result = _event_book_features(frame, pd.Timestamp("2025-01-01 00:00:10"), cfg)
    assert result["book_metric_time"] == pd.Timestamp("2025-01-01 00:00:10")
    assert result["book_bid_depth_5bps_base"] == 80
    assert result["book_aggressive_sell_base_lookback"] == 30
    assert result["book_bid_replenished_base_lookback"] == 50
    assert result["books_causal_valid"] is True


def test_causal_audit_fails_future_footprint_or_books() -> None:
    frame = pd.DataFrame(
        {
            "checkpoint_available_time": pd.to_datetime(["2025-01-01 00:00:00"]),
            "fp_end_ts": pd.to_datetime(["2025-01-01 00:00:01"]),
            "book_metric_time": pd.to_datetime(["2025-01-01 00:00:02"]),
        }
    )
    out = causal_audit(frame, frame)
    assert (out["status"] == "FAIL").sum() == 4


def test_range_footprint_loader_supports_column_and_bar_range_pruning(tmp_path: Path) -> None:
    loader = OKXRangeFootprintLoader(data_dir=tmp_path, range_pct=0.002, price_step=1.0)
    with sqlite3.connect(loader.db_path) as conn:
        conn.executemany(
            f"""
            INSERT INTO {loader.table_name}
            (bar_id, start_ts, end_ts, price_bucket, volume, notional, trades_count,
             buy_volume, sell_volume, buy_notional, sell_notional, buy_trades_count,
             sell_trades_count, delta_volume, delta_notional, large_buy_notional,
             large_sell_notional, large_delta_notional, large_trades_count, max_trade_notional)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "2025-01-01 00:00:00.000", "2025-01-01 00:01:00.000", 100.0, 1, 10, 1, 0, 1, 0, 10, 0, 1, -1, -10, 0, 0, 0, 0, 10),
                (2, "2025-01-01 00:01:00.000", "2025-01-01 00:02:00.000", 99.0, 1, 20, 1, 0, 1, 0, 20, 0, 1, -1, -20, 0, 0, 0, 0, 20),
            ],
        )
        conn.commit()
    out = loader.load_local_data(
        bar_id_min=2,
        bar_id_max=2,
        columns=["bar_id", "price_bucket", "sell_notional"],
    )
    assert out.to_dict("records") == [{"bar_id": 2, "price_bucket": 99.0, "sell_notional": 20.0}]


def test_attach_books_context_uses_compact_loader_without_raw_rescan() -> None:
    from dataclasses import dataclass
    from src.research_common.post_sweep_footprint_books.books import attach_books_context

    @dataclass
    class Coverage:
        day: str
        features: int
        heatmap_cells: int
        metadata: str

    class FakeLoader:
        def coverage(self):
            return [Coverage("2024-12-31", 3, 0, "fake")]

        def load_features(self, start, end, *, project_time, index_mode, valid_only):
            times = pd.date_range("2025-01-01 00:00:00", periods=3, freq="5s")
            return pd.DataFrame(
                {
                    "available_time": times,
                    "book_valid": 1,
                    "trade_attribution_valid": 1,
                    "bid_depth_5bps_base": [100, 50, 80],
                    "ask_depth_5bps_base": [100, 100, 90],
                    "bid_depth_10bps_base": 100,
                    "ask_depth_10bps_base": 100,
                    "bid_depth_25bps_base": 200,
                    "ask_depth_25bps_base": 200,
                    "depth_imbalance_25bps": [0.0, -0.2, 0.1],
                    "top_bid_wall_ratio": 2.0,
                    "top_bid_wall_distance_bps": 3.0,
                    "top_bid_wall_depth_base": 50.0,
                    "top_ask_wall_ratio": 2.0,
                    "top_ask_wall_distance_bps": 3.0,
                    "top_ask_wall_depth_base": 50.0,
                    "aggressive_buy_base": [1, 1, 1],
                    "aggressive_sell_base": [10, 10, 10],
                    "book_added_bid_base": [0, 20, 30],
                    "book_added_ask_base": 0,
                    "book_removed_bid_base": [0, 50, 0],
                    "book_removed_ask_base": 0,
                    "estimated_bid_cancel_base": [0, 10, 0],
                    "estimated_ask_cancel_base": 0,
                    "estimated_bid_consumed_base": [0, 40, 0],
                    "estimated_ask_consumed_base": 0,
                    "estimated_bid_replenished_base": [0, 20, 30],
                    "estimated_ask_replenished_base": 0,
                }
            )

    events = pd.DataFrame(
        {
            "checkpoint_id": ["E1"],
            "checkpoint_available_time": pd.to_datetime(["2025-01-01 00:00:10"]),
        }
    )
    result = attach_books_context(
        events,
        loader=FakeLoader(),  # type: ignore[arg-type]
        config=PostSweepFootprintBooksConfig().validate(),
        progress=False,
    )
    assert len(result.context) == 1
    assert result.context.loc[0, "books_causal_valid"]
    assert result.context.loc[0, "book_bid_replenished_base_lookback"] == 50


def test_books_event_lookup_is_stable_for_microsecond_datetime_storage() -> None:
    times = pd.Series(
        np.array(
            [
                "2025-01-01T00:00:00.000000",
                "2025-01-01T00:00:05.000000",
                "2025-01-01T00:00:10.000000",
            ],
            dtype="datetime64[us]",
        )
    )
    frame = pd.DataFrame(
        {
            "available_time": times,
            "book_valid": 1,
            "trade_attribution_valid": 1,
            "bid_depth_5bps_base": [100.0, 50.0, 80.0],
            "ask_depth_5bps_base": [100.0, 100.0, 90.0],
            "bid_depth_10bps_base": 100.0,
            "ask_depth_10bps_base": 100.0,
            "bid_depth_25bps_base": 200.0,
            "ask_depth_25bps_base": 200.0,
            "depth_imbalance_25bps": [0.0, -0.2, 0.1],
            "top_bid_wall_ratio": 2.0,
            "top_bid_wall_distance_bps": 3.0,
            "top_bid_wall_depth_base": 50.0,
            "top_ask_wall_ratio": 2.0,
            "top_ask_wall_distance_bps": 3.0,
            "top_ask_wall_depth_base": 50.0,
            "aggressive_buy_base": [1.0, 1.0, 1.0],
            "aggressive_sell_base": [10.0, 10.0, 10.0],
            "book_added_bid_base": [0.0, 20.0, 30.0],
            "book_added_ask_base": 0.0,
            "book_removed_bid_base": [0.0, 50.0, 0.0],
            "book_removed_ask_base": 0.0,
            "estimated_bid_cancel_base": [0.0, 10.0, 0.0],
            "estimated_ask_cancel_base": 0.0,
            "estimated_bid_consumed_base": [0.0, 40.0, 0.0],
            "estimated_ask_consumed_base": 0.0,
            "estimated_bid_replenished_base": [0.0, 20.0, 30.0],
            "estimated_ask_replenished_base": 0.0,
        }
    )
    assert str(frame["available_time"].dtype) == "datetime64[us]"
    cfg = PostSweepFootprintBooksConfig(books_lookback_seconds=60).validate()
    result = _event_book_features(frame, pd.Timestamp("2025-01-01 00:00:10"), cfg)
    assert result["book_metric_time"] == pd.Timestamp("2025-01-01 00:00:10")
    assert result["book_window_rows"] == 3
    assert result["book_aggressive_sell_base_lookback"] == 30.0
    assert result["books_causal_valid"] is True


def test_books_events_outside_utc_coverage_are_filled_without_loading() -> None:
    from dataclasses import dataclass
    from src.research_common.post_sweep_footprint_books.books import attach_books_context

    @dataclass
    class Coverage:
        day: str
        features: int
        heatmap_cells: int
        metadata: str

    class FakeLoader:
        def coverage(self):
            return [Coverage("2025-11-01", 10, 0, "fake")]

        def load_features(self, *args, **kwargs):
            raise AssertionError("out-of-coverage events must not trigger compact Books loads")

    events = pd.DataFrame(
        {
            "checkpoint_id": ["EARLY"],
            "checkpoint_available_time": pd.to_datetime(["2024-01-01 12:00:00"]),
        }
    )
    result = attach_books_context(
        events,
        loader=FakeLoader(),  # type: ignore[arg-type]
        config=PostSweepFootprintBooksConfig().validate(),
        progress=False,
    )
    assert len(result.context) == 1
    assert not bool(result.context.loc[0, "books_causal_valid"])
    assert result.audit.loc[0, "status"] == "events_outside_compact_books_coverage"
