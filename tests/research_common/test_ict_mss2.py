from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2 import (
    MSS2Config,
    aggregate_bars,
    attach_execution_outcomes,
    attach_sweep_baseline_outcomes,
    build_execution_pivots,
    build_first_sweep_lifecycle,
    build_mss_fvg_events,
    causal_audit,
    classify_liquidity,
    split_features_and_labels,
)


def _bars(rows: list[tuple[float, float, float, float]], start: str = "2025-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_2m_aggregation_is_left_labelled_and_complete_only() -> None:
    bars = _bars([(100, 101, 99, 100)] * 5)
    out = aggregate_bars(bars, 2)
    assert list(out.index) == [pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-01-01 00:02")]
    assert list(out["bar_end_time"]) == [pd.Timestamp("2025-01-01 00:02"), pd.Timestamp("2025-01-01 00:04")]


def test_execution_pivot_is_available_only_after_right_bar_close() -> None:
    bars = _bars(
        [
            (99, 100, 98, 99),
            (100, 105, 99, 102),  # high pivot center
            (101, 103, 100, 101),  # right confirmation bar
            (101, 102, 100, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 101),
            (101, 102, 100, 101),
        ]
    )
    cfg = MSS2Config(execution_confirmation_orders=(1, 2, 3))
    pivots = build_execution_pivots(aggregate_bars(bars, 1), 1, cfg)
    row = pivots.loc[(pivots["pivot_side"] == "high") & (pivots["pivot_time"] == pd.Timestamp("2025-01-01 00:01"))].iloc[0]
    assert row["initial_available_time"] == pd.Timestamp("2025-01-01 00:03")
    assert row["pivot_bar_end_time"] == pd.Timestamp("2025-01-01 00:02")


def test_first_sweep_cannot_happen_before_liquidity_is_available() -> None:
    bars = _bars(
        [
            (100, 101, 94, 100),  # penetrates before level becomes active
            (100, 101, 96, 100),
            (100, 101, 96, 100),
            (100, 101, 94, 100),  # first legal sweep
            (100, 101, 96, 100),
            (100, 101, 96, 100),
            (100, 101, 96, 100),
            (100, 101, 96, 100),
            (100, 101, 96, 100),
            (100, 101, 96, 100),
        ]
    )
    levels = pd.DataFrame(
        [
            {
                "level_id": 1,
                "pivot_side": "low",
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "pivot_time": pd.Timestamp("2024-12-31 23:30"),
                "pivot_bar_end_time": pd.Timestamp("2024-12-31 23:45"),
                "level_price": 95.0,
                "initial_available_time": pd.Timestamp("2025-01-01 00:02"),
                "order_1_available_time": pd.Timestamp("2025-01-01 00:02"),
                "order_2_available_time": pd.NaT,
                "order_3_available_time": pd.NaT,
                "order_5_available_time": pd.NaT,
                "external_20_flag": 0,
                "external_50_flag": 0,
                "liquidity_side": "sell_side",
                "trade_direction": 1,
            }
        ]
    )
    out = build_first_sweep_lifecycle(bars, levels, MSS2Config())
    assert int(out.iloc[0]["sweep_pos_1m"]) == 3
    assert out.iloc[0]["sweep_available_time_1m"] == pd.Timestamp("2025-01-01 00:04")


def _long_mss_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        (98, 99, 97, 98),
        (98, 100, 97, 99),
        (99, 101, 98, 100),
        (100, 105, 99, 102),  # pre-sweep high pivot
        (102, 103, 99, 100),
        (100, 101, 98, 99),
        (99, 99, 90, 96),     # sweep 95 sell-side liquidity
        (96, 98, 95, 97),
        (100, 104, 100, 103), # closes through 105? no -> adjust below
        (103, 104, 100, 102),
        (102, 103, 101, 102),
        (102, 103, 101, 102),
        (102, 103, 101, 102),
        (102, 103, 101, 102),
        (102, 103, 101, 102),
        (102, 103, 101, 102),
    ]
    # Make MSS bar close above reference 105 and leave bullish FVG vs bar 6 high 99.
    rows[8] = (101, 108, 101, 107)
    # Next bar retraces through proximal 101 so limit can fill.
    rows[9] = (105, 106, 100, 104)
    bars = _bars(rows)
    lifecycle = pd.DataFrame(
        [
            {
                "level_id": 1,
                "pivot_side": "low",
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "pivot_time": pd.Timestamp("2024-12-30 00:00"),
                "pivot_bar_end_time": pd.Timestamp("2024-12-30 00:15"),
                "level_price": 95.0,
                "initial_available_time": pd.Timestamp("2024-12-30 00:30"),
                "order_1_available_time": pd.Timestamp("2024-12-30 00:30"),
                "order_2_available_time": pd.Timestamp("2024-12-30 00:45"),
                "order_3_available_time": pd.NaT,
                "order_5_available_time": pd.NaT,
                "future_eventual_order_1_label": 1,
                "future_eventual_order_2_label": 1,
                "future_eventual_order_3_label": 0,
                "future_eventual_order_5_label": 0,
                "external_20_flag": 1,
                "external_50_flag": 0,
                "liquidity_side": "sell_side",
                "trade_direction": 1,
                "active_pos_1m": 0,
                "sweep_pos_1m": 6,
                "sweep_bar_time_1m": pd.Timestamp("2025-01-01 00:06"),
                "sweep_available_time_1m": pd.Timestamp("2025-01-01 00:07"),
                "age_minutes_at_sweep": 3000.0,
                "sweep_depth_bp": 500.0,
                "confirmed_order_at_sweep": 2,
                "active_same_side_level_count_10p0bp": 1,
                "active_same_side_timeframe_count_10p0bp": 1,
                "liquidity_structural_score": 5,
                "old_remote_flag_6h": 1,
                "old_remote_flag_24h": 1,
                "old_remote_flag_72h": 0,
                "age_bucket": "1-3d",
                "liquidity_class": "structural_external",
                "quality_tier": "B",
            },
            {
                # Opposing unswept 15m buy-side liquidity target.
                "level_id": 2,
                "pivot_side": "high",
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "pivot_time": pd.Timestamp("2024-12-30 01:00"),
                "pivot_bar_end_time": pd.Timestamp("2024-12-30 01:15"),
                "level_price": 112.0,
                "initial_available_time": pd.Timestamp("2024-12-30 01:30"),
                "order_1_available_time": pd.Timestamp("2024-12-30 01:30"),
                "order_2_available_time": pd.NaT,
                "order_3_available_time": pd.NaT,
                "order_5_available_time": pd.NaT,
                "future_eventual_order_1_label": 1,
                "future_eventual_order_2_label": 0,
                "future_eventual_order_3_label": 0,
                "future_eventual_order_5_label": 0,
                "external_20_flag": 0,
                "external_50_flag": 0,
                "liquidity_side": "buy_side",
                "trade_direction": -1,
                "active_pos_1m": 0,
                "sweep_pos_1m": -1,
                "sweep_bar_time_1m": pd.NaT,
                "sweep_available_time_1m": pd.NaT,
                "age_minutes_at_sweep": np.nan,
                "sweep_depth_bp": np.nan,
                "confirmed_order_at_sweep": 0,
                "active_same_side_level_count_10p0bp": 0,
                "active_same_side_timeframe_count_10p0bp": 0,
                "liquidity_structural_score": 1,
                "old_remote_flag_6h": 0,
                "old_remote_flag_24h": 0,
                "old_remote_flag_72h": 0,
                "age_bucket": pd.NA,
                "liquidity_class": "minor_swing_candidate",
                "quality_tier": "D",
            },
        ]
    )
    return bars, lifecycle


def test_mss_is_close_confirmed_uses_pre_sweep_reference_and_entry_is_next_bar_or_later() -> None:
    bars, lifecycle = _long_mss_fixture()
    cfg = MSS2Config(max_mss_minutes=10, max_entry_wait_minutes=10, max_outcome_minutes=10)
    events = build_mss_fvg_events(
        bars,
        lifecycle,
        execution_minutes=1,
        reference_mode="recent",
        config=cfg,
        project_timezone="+8",
    )
    assert len(events) == 1
    event = events.iloc[0]
    assert int(event["mss_reference_pivot_pos"]) == 3
    assert int(event["mss_pos"]) == 8
    assert int(event["entry_fill_pos_1m"]) == 9
    assert event["entry_time"] >= event["mss_available_time"]
    assert int(event["has_displacement_fvg"]) == 1
    audit = causal_audit(lifecycle, events)
    assert audit["mss_reference_not_pre_sweep"] == 0
    assert audit["entry_not_after_mss"] == 0


def test_2m_cannot_react_until_sweep_containing_bar_has_closed() -> None:
    bars, lifecycle = _long_mss_fixture()
    cfg = MSS2Config(max_mss_minutes=20, max_entry_wait_minutes=20, max_outcome_minutes=20)
    events = build_mss_fvg_events(
        bars,
        lifecycle,
        execution_minutes=2,
        reference_mode="recent",
        config=cfg,
        project_timezone="+8",
    )
    if not events.empty:
        event = events.iloc[0]
        assert event["sweep_exec_available_time"] >= pd.Timestamp("2025-01-01 00:08")
        assert event["mss_available_time"] > event["sweep_exec_available_time"]


def test_old_remote_liquidity_is_not_expired_and_is_classified() -> None:
    bars = _bars([(100, 101, 99, 100)] * 10)
    lifecycle = pd.DataFrame(
        [
            {
                "level_id": 1,
                "pivot_side": "low",
                "source_timeframe": "4H",
                "source_timeframe_min": 240,
                "pivot_time": pd.Timestamp("2024-12-01"),
                "pivot_bar_end_time": pd.Timestamp("2024-12-01 04:00"),
                "level_price": 95.0,
                "initial_available_time": pd.Timestamp("2024-12-01 08:00"),
                "order_1_available_time": pd.Timestamp("2024-12-01 08:00"),
                "order_2_available_time": pd.Timestamp("2024-12-01 12:00"),
                "order_3_available_time": pd.Timestamp("2024-12-01 16:00"),
                "order_5_available_time": pd.NaT,
                "future_eventual_order_1_label": 1,
                "future_eventual_order_2_label": 1,
                "future_eventual_order_3_label": 1,
                "future_eventual_order_5_label": 0,
                "external_20_flag": 1,
                "external_50_flag": 1,
                "liquidity_side": "sell_side",
                "trade_direction": 1,
                "active_pos_1m": 0,
                "sweep_pos_1m": 5,
                "sweep_bar_time_1m": bars.index[5],
                "sweep_available_time_1m": bars.index[5] + pd.Timedelta(minutes=1),
                "age_minutes_at_sweep": 50_000.0,
                "sweep_depth_bp": 10.0,
                "confirmed_order_at_sweep": 3,
            }
        ]
    )
    out = classify_liquidity(lifecycle, MSS2Config())
    assert int(out.iloc[0]["old_remote_flag_72h"]) == 1
    assert out.iloc[0]["age_bucket"] == ">=7d"
    assert out.iloc[0]["liquidity_class"] in {"major_external", "multi_tf_pool", "same_price_pool"}


def test_future_labels_are_physically_excluded_from_model_feature_table() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["x"],
            "level_id": [1],
            "execution_minutes": [1],
            "reference_mode": ["recent"],
            "displacement_atr": [1.2],
            "filled_flag": [1],
            "mfe_r_180m": [2.0],
            "r1p0_gross_return": [0.01],
        }
    )
    features, labels = split_features_and_labels(frame)
    assert "displacement_atr" in features.columns
    assert "filled_flag" not in features.columns
    assert not any(col.startswith("entry_") for col in features.columns)
    assert "mfe_r_180m" not in features.columns
    assert "r1p0_gross_return" not in features.columns
    assert "filled_flag" in labels.columns


def test_same_bar_stop_target_is_pessimistically_stop_first() -> None:
    bars, lifecycle = _long_mss_fixture()
    # Create enough excursion on the fill bar to touch both stop and 1R target.
    bars.loc[pd.Timestamp("2025-01-01 00:09"), "low"] = 89.0
    bars.loc[pd.Timestamp("2025-01-01 00:09"), "high"] = 115.0
    cfg = MSS2Config(max_mss_minutes=10, max_entry_wait_minutes=10, max_outcome_minutes=10)
    events = build_mss_fvg_events(bars, lifecycle, execution_minutes=1, reference_mode="recent", config=cfg)
    outcomes = attach_execution_outcomes(bars, lifecycle, events, execution_minutes=1, config=cfg)
    assert len(outcomes) == 1
    assert outcomes.iloc[0]["r1p0_outcome"] == "stop"


def test_sweep_only_baseline_starts_at_next_1m_open() -> None:
    bars, lifecycle = _long_mss_fixture()
    out = attach_sweep_baseline_outcomes(bars, lifecycle, config=MSS2Config(max_outcome_minutes=5))
    row = out.loc[out["level_id"].eq(1)].iloc[0]
    assert int(row["sweep_baseline_entry_pos_1m"]) == 7
    assert row["sweep_baseline_entry_time"] == pd.Timestamp("2025-01-01 00:07")
    assert np.isfinite(float(row["sweep_close_return_5m"]))
