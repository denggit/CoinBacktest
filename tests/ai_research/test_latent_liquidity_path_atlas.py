from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.ai_research.latent_liquidity_path_atlas.candidates import (
    _assign_release_episodes,
    build_candidate_frame,
    normalize_second_bars,
    select_candidates,
)
from src.ai_research.latent_liquidity_path_atlas.clustering import (
    assign_path_clusters,
    fit_path_clusters,
)
from src.ai_research.latent_liquidity_path_atlas.config import LatentLiquidityPathAtlasConfig
from src.ai_research.latent_liquidity_path_atlas.features import event_feature_table
from src.ai_research.latent_liquidity_path_atlas.macro import attach_macro_path_context, build_macro_path_context
from src.ai_research.latent_liquidity_path_atlas.outcomes import attach_outcomes
from src.ai_research.latent_liquidity_path_atlas.pipeline import _assign_global_release_episodes
from src.ai_research.latent_liquidity_path_atlas.unswept_swings import attach_unswept_swing_inventory


def _bars(periods: int = 5000, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1s")
    close = np.full(periods, 100.0)
    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close.copy(),
            "notional": np.full(periods, 10_000.0),
            "buy_notional": np.full(periods, 5_000.0),
            "sell_notional": np.full(periods, 5_000.0),
            "delta_notional": np.zeros(periods),
            "trades_count": np.full(periods, 20.0),
            "large_buy_notional": np.zeros(periods),
            "large_sell_notional": np.zeros(periods),
            "large_delta_notional": np.zeros(periods),
            "max_trade_notional": np.full(periods, 1_000.0),
        },
        index=index,
    )


def _minute_bars(periods: int = 12050) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1min")
    close = np.linspace(100.0, 110.0, len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "notional": 1000.0,
            "buy_notional": 520.0,
            "sell_notional": 480.0,
            "delta_notional": 40.0,
            "trades_count": 10.0,
        },
        index=index,
    )


def _config(**kwargs) -> LatentLiquidityPathAtlasConfig:
    base = LatentLiquidityPathAtlasConfig(
        warmup_start="2023-12-31 00:00:00",
        research_start="2024-01-01 00:00:00",
        research_end="2024-01-02 00:00:00",
        pre_context_seconds=3600,
        post_label_seconds=600,
    )
    return replace(base, **kwargs)


def test_config_forbids_sub_15m_swing_timeframes() -> None:
    with pytest.raises(ValueError, match="minimum Swing timeframe is 15m"):
        _config(swing_timeframes=(("5m", 5),)).validate()



def test_release_episodes_do_not_merge_opposite_sides() -> None:
    index = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 00:00:01",
        "2024-01-01 00:00:02",
    ])
    events = pd.DataFrame(
        {"event_side": ["DOWN", "UP", "DOWN"]},
        index=index,
    )
    assigned = _assign_release_episodes(events, gap_seconds=45)
    assert assigned.iloc[0]["release_episode_id"] == assigned.iloc[2]["release_episode_id"]
    assert assigned.iloc[0]["release_episode_id"] != assigned.iloc[1]["release_episode_id"]
    assert assigned.iloc[0]["release_episode_size"] == 2
    assert assigned.iloc[1]["release_episode_size"] == 1

    global_input = assigned.reset_index(names="event_time")
    global_assigned = _assign_global_release_episodes(global_input, gap_seconds=45)
    assert global_assigned.iloc[0]["release_episode_id"] == global_assigned.iloc[2]["release_episode_id"]
    assert global_assigned.iloc[0]["release_episode_id"] != global_assigned.iloc[1]["release_episode_id"]
    assert global_assigned.iloc[0]["release_episode_size"] == 2
    assert global_assigned.iloc[1]["release_episode_size"] == 1


def test_flow_burst_can_enter_without_swing_admission() -> None:
    raw = _bars()
    event_time = raw.index[4000]
    raw.loc[event_time, ["low", "close"]] = [99.95, 99.97]
    raw.loc[event_time, "sell_notional"] = 1_500_000.0
    raw.loc[event_time, "notional"] = 1_600_000.0
    raw.loc[event_time, "delta_notional"] = -1_400_000.0
    raw.loc[event_time, "trades_count"] = 800.0
    frame = build_candidate_frame(normalize_second_bars(raw, _config()), _config())
    events = select_candidates(frame, event_time, event_time, _config())
    assert len(events) == 1
    assert events.iloc[0]["event_side"] == "DOWN"
    assert bool(events.iloc[0]["source_flow_burst_down"])
    assert "release_episode_id" in events


def test_boundary_event_is_not_required_to_have_flow_burst() -> None:
    raw = _bars()
    x = np.arange(len(raw))
    close = 100.0 + 0.005 * np.sin(x / 11.0)
    raw["open"] = close
    raw["close"] = close
    raw["high"] = close + 0.001
    raw["low"] = close - 0.001
    event_time = raw.index[4000]
    raw.loc[event_time, ["low", "close"]] = [99.90, 99.91]
    frame = build_candidate_frame(normalize_second_bars(raw, _config()), _config())
    events = select_candidates(frame, event_time, event_time, _config())
    assert len(events) == 1
    assert bool(events.iloc[0]["source_boundary_down"])
    assert not bool(events.iloc[0]["source_flow_burst_down"])


def test_current_burst_is_not_in_its_own_baseline() -> None:
    raw = _bars()
    event_time = raw.index[4000]
    raw.loc[event_time, "notional"] = 1_000_000.0
    frame = build_candidate_frame(normalize_second_bars(raw, _config()), _config())
    assert float(frame.loc[event_time, "z_notional"]) > 20.0
    next_time = event_time + pd.Timedelta(seconds=1)
    assert float(frame.loc[next_time, "z_notional"]) < float(frame.loc[event_time, "z_notional"])


def test_no_micro_swing_columns_are_created() -> None:
    raw = _bars()
    event_time = raw.index[4000]
    raw.loc[event_time, "sell_notional"] = 1_500_000.0
    raw.loc[event_time, "notional"] = 1_600_000.0
    raw.loc[event_time, "delta_notional"] = -1_400_000.0
    raw.loc[event_time, "trades_count"] = 800.0
    frame = build_candidate_frame(normalize_second_bars(raw, _config()), _config())
    events = select_candidates(frame, event_time, event_time, _config())
    features = event_feature_table(frame, events, _config())
    assert len(features) == 1
    assert not any("pivot" in name.lower() for name in features.columns)
    assert "path_pressure_without_progress_60s" in features


def test_shallow_immediate_reversal_label() -> None:
    raw = _bars(5200)
    event_time = raw.index[4000]
    raw.loc[event_time, ["low", "close"]] = [99.90, 99.92]
    future = pd.date_range(event_time + pd.Timedelta(seconds=1), periods=600, freq="1s")
    path_close = np.r_[np.linspace(99.93, 100.25, 10), np.full(590, 100.25)]
    raw.loc[future, "open"] = path_close
    raw.loc[future, "close"] = path_close
    raw.loc[future, "high"] = path_close + 0.002
    raw.loc[future, "low"] = path_close - 0.002
    bars = normalize_second_bars(raw, _config())
    events = pd.DataFrame({"event_id": ["E1"], "event_time": [event_time], "event_side": ["DOWN"]})
    labels = attach_outcomes(bars, events, _config())
    assert labels.iloc[0]["outcome_type"] == "SHALLOW_IMMEDIATE_REVERSAL"


def test_extend_stabilize_reversal_label() -> None:
    raw = _bars(5200)
    event_time = raw.index[4000]
    raw.loc[event_time, ["low", "close"]] = [99.95, 99.96]
    future = pd.date_range(event_time + pd.Timedelta(seconds=1), periods=600, freq="1s")
    path_close = np.r_[
        np.linspace(99.94, 99.70, 40),
        np.full(20, 99.70),
        np.linspace(99.70, 100.10, 100),
        np.full(440, 100.10),
    ]
    raw.loc[future, "open"] = path_close
    raw.loc[future, "close"] = path_close
    raw.loc[future, "high"] = path_close + 0.001
    raw.loc[future, "low"] = path_close - 0.001
    bars = normalize_second_bars(raw, _config())
    events = pd.DataFrame({"event_id": ["E2"], "event_time": [event_time], "event_side": ["DOWN"]})
    labels = attach_outcomes(bars, events, _config())
    assert labels.iloc[0]["outcome_type"] == "EXTEND_STABILIZE_REVERSAL"


def test_cluster_fit_is_frozen_to_pre_cutoff_rows() -> None:
    rng = np.random.default_rng(4)
    rows = 300
    features = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(rows)],
            "event_time": pd.date_range("2024-01-01", periods=rows, freq="1h"),
            "event_side": ["DOWN"] * rows,
            "period": ["TRAIN_2023_2024"] * rows,
            "causal_feature_time": pd.date_range("2024-01-01", periods=rows, freq="1h"),
            "path_ret_5s": rng.normal(size=rows),
            "path_efficiency_5s": rng.normal(size=rows),
        }
    )
    cfg = _config(cluster_count=6, minimum_cluster_rows=120, cluster_train_end="2024-01-10")
    model = fit_path_clusters(features, cfg)
    assert model is not None
    centers = model.model.cluster_centers_.copy()
    later = features.copy()
    later["event_time"] = pd.date_range("2025-01-01", periods=rows, freq="1h")
    later["path_ret_5s"] = 100.0
    assigned = assign_path_clusters(later, model)
    assert len(assigned) == rows
    np.testing.assert_allclose(centers, model.model.cluster_centers_)


def test_macro_context_uses_only_completed_minute_bar_and_liquidity_features() -> None:
    minute = _minute_bars()
    cfg = _config()
    context = build_macro_path_context(minute, cfg)
    event_time = pd.Timestamp("2024-01-08 00:00:30")
    events = pd.DataFrame(
        {
            "event_id": ["M1"],
            "event_time": [event_time],
            "event_side": ["DOWN"],
            "period": ["TRAIN_2023_2024"],
            "causal_feature_time": [event_time],
            "pre_path_available_time": [event_time - pd.Timedelta(seconds=1)],
        }
    )
    attached = attach_macro_path_context(events, context, cfg)
    assert pd.Timestamp(attached.iloc[0]["macro_available_time"]) <= event_time
    assert pd.Timestamp(attached.iloc[0]["macro_bar_start_time"]) == pd.Timestamp("2024-01-07 23:59:00")
    assert "macro_turnover_per_range_intensity_1440m" in attached.columns
    assert "macro_pressure_without_progress_240m" in attached.columns
    assert not any("pivot" in name.lower() for name in attached.columns)


def test_all_unswept_15m_plus_levels_are_attached_not_only_last() -> None:
    event_time = pd.Timestamp("2024-02-01 00:00:30")
    features = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_time": [event_time],
            "event_side": ["DOWN"],
            "macro_pre_event_close": [100.0],
        }
    )
    levels = pd.DataFrame(
        {
            "level_id": [1, 2, 3, 4],
            "level_side": ["LOW", "LOW", "LOW", "HIGH"],
            "source_timeframe": ["4H", "15m", "1H", "1D"],
            "source_timeframe_min": [240, 15, 60, 1440],
            "pivot_time": pd.to_datetime(["2023-01-01", "2024-01-31", "2024-01-20", "2023-12-01"]),
            "level_price": [99.0, 99.8, 98.0, 101.0],
            "initial_available_time": pd.to_datetime(["2023-01-02", "2024-01-31 12:00", "2024-01-20 02:00", "2023-12-03"], format="mixed"),
            # Level 3 was already swept and must be absent.
            "sweep_available_time": pd.to_datetime([pd.NaT, pd.NaT, "2024-01-25", pd.NaT]),
            "pivot_range_bp": [100.0, 30.0, 50.0, 120.0],
            "pivot_notional_vs_past20": [1.1, 1.3, 0.9, 1.5],
        }
    )
    attached = attach_unswept_swing_inventory(features, levels, _config())
    row = attached.iloc[0]
    assert row["unswept_relevant_count"] == 2
    assert row["unswept_15m_relevant_count"] == 1
    assert row["unswept_4H_relevant_count"] == 1
    assert row["unswept_nearest_distance_bp"] == pytest.approx(20.0)
    assert row["unswept_oldest_age_minutes"] > 500_000
    assert row["unswept_confluence_count_25bp"] == 1


def test_future_or_consumed_swing_levels_are_not_visible() -> None:
    event_time = pd.Timestamp("2024-02-01")
    features = pd.DataFrame(
        {"event_id": ["E1"], "event_time": [event_time], "event_side": ["UP"], "macro_pre_event_close": [100.0]}
    )
    levels = pd.DataFrame(
        {
            "level_id": [1, 2],
            "level_side": ["HIGH", "HIGH"],
            "source_timeframe": ["15m", "1D"],
            "source_timeframe_min": [15, 1440],
            "pivot_time": pd.to_datetime(["2024-01-20", "2024-01-01"]),
            "level_price": [101.0, 102.0],
            "initial_available_time": pd.to_datetime(["2024-02-02", "2024-01-03"]),
            "sweep_available_time": pd.to_datetime([pd.NaT, "2024-01-20"]),
            "pivot_range_bp": [20.0, 100.0],
            "pivot_notional_vs_past20": [1.0, 1.0],
        }
    )
    attached = attach_unswept_swing_inventory(features, levels, _config())
    assert attached.iloc[0]["unswept_relevant_count"] == 0


def test_pre_event_path_excludes_release_second() -> None:
    raw = _bars()
    event_time = raw.index[4000]
    raw.loc[event_time, "close"] = 90.0
    raw.loc[event_time, "low"] = 89.9
    raw.loc[event_time, "sell_notional"] = 5_000_000.0
    raw.loc[event_time, "notional"] = 5_100_000.0
    raw.loc[event_time, "delta_notional"] = -4_900_000.0
    raw.loc[event_time, "trades_count"] = 2000.0
    cfg = _config()
    frame = build_candidate_frame(normalize_second_bars(raw, cfg), cfg)
    events = select_candidates(frame, event_time, event_time, cfg)
    features = event_feature_table(frame, events, cfg)
    assert abs(float(features.iloc[0]["path_ret_5s"])) < 1e-12
    assert pd.Timestamp(features.iloc[0]["pre_path_available_time"]) == event_time - pd.Timedelta(seconds=1)


def test_cluster_contract_is_liquidity_first_and_excludes_absolute_scale() -> None:
    from src.ai_research.latent_liquidity_path_atlas.features import model_feature_columns

    frame = pd.DataFrame(
        {
            "path_ret_5s": [0.1],
            "path_notional_5s": [1_000_000.0],
            "path_notional_intensity_5s": [2.0],
            "path_pressure_without_progress_5s": [0.4],
            "z_notional": [9.0],
            "down_event_score": [10.0],
            "open": [2000.0],
            "macro_notional_60m": [1e9],
            "macro_notional_intensity_60m": [1.3],
            "unswept_relevant_count": [4],
            "unswept_max_level_available_time": [pd.Timestamp("2024-01-01")],
        }
    )
    columns = set(model_feature_columns(frame))
    assert "path_ret_5s" in columns
    assert "path_notional_intensity_5s" in columns
    assert "path_pressure_without_progress_5s" in columns
    assert "macro_notional_intensity_60m" in columns
    assert "unswept_relevant_count" in columns
    assert "z_notional" not in columns
    assert "down_event_score" not in columns
    assert "open" not in columns
    assert "path_notional_5s" not in columns
    assert "macro_notional_60m" not in columns
    assert "unswept_max_level_available_time" not in columns


def test_macro_alignment_normalizes_microsecond_and_nanosecond_keys() -> None:
    minute = _minute_bars()
    minute.index = minute.index.astype("datetime64[us]")
    cfg = _config()
    context = build_macro_path_context(minute, cfg)
    assert str(context["macro_available_time"].dtype) == "datetime64[ns]"
    event_time = pd.Timestamp("2024-01-08 00:00:30")
    events = pd.DataFrame(
        {
            "event_id": ["MIXED_PRECISION"],
            "event_time": pd.Series(np.array([event_time.to_datetime64()], dtype="datetime64[ns]")),
            "event_side": ["DOWN"],
            "period": ["TRAIN_2023_2024"],
            "causal_feature_time": [event_time],
            "pre_path_available_time": [event_time - pd.Timedelta(seconds=1)],
        }
    )
    attached = attach_macro_path_context(events, context, cfg)
    assert len(attached) == 1
    assert str(attached["event_time"].dtype) == "datetime64[ns]"
    assert str(attached["macro_available_time"].dtype) == "datetime64[ns]"


def test_second_and_minute_normalizers_force_nanosecond_axes() -> None:
    second = _bars()
    second.index = second.index.astype("datetime64[us]")
    normalized_second = normalize_second_bars(second, _config())
    assert str(normalized_second.index.dtype) == "datetime64[ns]"
    minute = _minute_bars()
    minute.index = minute.index.astype("datetime64[us]")
    context = build_macro_path_context(minute, _config())
    assert str(context["macro_bar_start_time"].dtype) == "datetime64[ns]"
    assert str(context["macro_available_time"].dtype) == "datetime64[ns]"


def test_empty_discovery_frames_fail_cleanly_without_schema_keyerror(tmp_path) -> None:
    from src.ai_research.latent_liquidity_path_atlas.reports import write_all_reports

    empty = pd.DataFrame()
    assignments = assign_path_clusters(empty, None)
    reports = write_all_reports(tmp_path, _config(), empty, empty, assignments, empty, 0)
    quality = reports["01_data_quality.csv"].set_index("check")
    assert quality.loc["feature_rows_positive", "status"] == "FAIL"
    assert quality.loc["macro_context_coverage", "status"] == "FAIL"


def test_data_quality_rejects_missing_macro_alignment() -> None:
    from src.ai_research.latent_liquidity_path_atlas.reports import data_quality

    features = pd.DataFrame(
        {"event_id": ["E1"], "macro_available_time": [pd.NaT], "release_episode_id": ["P1"]}
    )
    labels = pd.DataFrame({"event_id": ["E1"]})
    assignments = pd.DataFrame({"path_cluster": [-1]})
    swing_levels = pd.DataFrame({"level_id": [1]})
    quality = data_quality(features, labels, assignments, swing_levels).set_index("check")
    assert quality.loc["macro_context_coverage", "status"] == "FAIL"


def test_global_episode_assignment_mutates_wide_frame_without_sort_copy() -> None:
    rows = 200
    frame = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(rows)],
            "event_time": pd.date_range("2024-01-01", periods=rows, freq="1s"),
            "event_side": np.where(np.arange(rows) % 2 == 0, "DOWN", "UP"),
        }
    )
    # Deliberately create many small numeric blocks like the full feature atlas.
    for idx in range(80):
        frame[f"wide_{idx}"] = np.arange(rows, dtype=float) + idx
    result = _assign_global_release_episodes(frame, gap_seconds=45)
    assert result["release_episode_id"].nunique() == 2
    assert result["release_episode_size"].eq(100).all()
    assert result["release_episode_weight"].dtype == np.float32
    assert result["wide_79"].iloc[-1] == pytest.approx(278.0)


def test_cluster_training_and_assignment_are_bounded() -> None:
    rng = np.random.default_rng(42)
    rows = 1200
    features = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(rows)],
            "event_time": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "event_side": np.where(np.arange(rows) % 2 == 0, "DOWN", "UP"),
            "period": ["TRAIN_2023_2024"] * rows,
            "path_ret_5s": rng.normal(size=rows).astype(np.float32),
            "path_efficiency_5s": rng.normal(size=rows).astype(np.float32),
            "path_pressure_without_progress_5s": rng.normal(size=rows).astype(np.float32),
        }
    )
    cfg = _config(
        cluster_count=6,
        minimum_cluster_rows=120,
        cluster_train_sample_cap=240,
        cluster_assign_batch_rows=125,
        cluster_train_end="2024-12-31",
    )
    model = fit_path_clusters(features, cfg)
    assert model is not None
    assert model.train_rows == 240
    assert model.eligible_train_rows == rows
    assigned = assign_path_clusters(features, model, batch_rows=cfg.cluster_assign_batch_rows)
    assert len(assigned) == rows
    assert assigned["path_cluster"].ge(0).all()
    assert assigned["cluster_distance"].dtype == np.float32


def test_large_report_uses_bounded_descriptive_sample(tmp_path) -> None:
    from src.ai_research.latent_liquidity_path_atlas.reports import write_all_reports

    rows = 300
    event_time = pd.date_range("2024-01-01", periods=rows, freq="1min")
    event_id = [f"E{i}" for i in range(rows)]
    features = pd.DataFrame(
        {
            "event_id": event_id,
            "event_time": event_time,
            "event_side": np.where(np.arange(rows) % 2 == 0, "DOWN", "UP"),
            "period": ["TRAIN_2023_2024"] * rows,
            "release_episode_id": [f"P{i // 3}" for i in range(rows)],
            "release_episode_number": np.arange(rows) // 3 + 1,
            "release_episode_ordinal": np.arange(rows) % 3 + 1,
            "release_episode_size": np.full(rows, 3),
            "release_episode_weight": np.full(rows, 1 / 3, dtype=np.float32),
            "candidate_source_count": np.ones(rows),
            "macro_available_time": event_time,
            "causal_feature_time": event_time,
            "pre_path_available_time": event_time - pd.Timedelta(seconds=1),
            "unswept_max_level_available_time": event_time - pd.Timedelta(minutes=15),
            "unswept_relevant_count": np.ones(rows),
            "unswept_nearest_distance_bp": np.ones(rows),
            "unswept_oldest_age_minutes": np.ones(rows),
            "unswept_confluence_count_10bp": np.ones(rows),
            "unswept_confluence_count_25bp": np.ones(rows),
            "unswept_confluence_count_50bp": np.ones(rows),
            "unswept_confluence_count_100bp": np.ones(rows),
            "path_pressure_without_progress_5s": np.arange(rows, dtype=np.float32),
        }
    )
    labels = pd.DataFrame(
        {
            "event_id": event_id,
            "event_time": event_time,
            "event_side": features["event_side"],
            "label_start_time": event_time + pd.Timedelta(seconds=1),
            "label_end_time": event_time + pd.Timedelta(seconds=600),
            "outcome_type": np.where(np.arange(rows) % 3 == 0, "ACCEPT_CONTINUATION", "EXTEND_STABILIZE_REVERSAL"),
            "favorable_reversal": np.arange(rows) % 3 != 0,
            "future_extension_bp": np.ones(rows),
            "future_reversal_after_extreme_bp": np.ones(rows),
            "future_time_to_extreme_seconds": np.ones(rows),
        }
    )
    assignments = pd.DataFrame(
        {
            "event_id": event_id,
            "event_time": event_time,
            "event_side": features["event_side"],
            "period": features["period"],
            "path_cluster": np.zeros(rows, dtype=np.int16),
            "cluster_distance": np.zeros(rows, dtype=np.float32),
        }
    )
    swing_levels = pd.DataFrame(
        {
            "level_side": ["LOW"],
            "source_timeframe": ["15m"],
            "sweep_available_time": [pd.NaT],
            "lifetime_minutes": [100.0],
            "pivot_time": [pd.Timestamp("2023-12-31")],
        }
    )
    cfg = _config(descriptive_sample_cap=100, csv_write_chunk_rows=100)
    reports = write_all_reports(tmp_path, cfg, features, labels, assignments, swing_levels, 100, 300)
    family = reports["07_liquidity_feature_family_summary.csv"]
    assert family["sample_rows"].eq(100).all()
    assert family["population_rows"].eq(rows).all()
    assert (tmp_path / "12_feature_table.csv.gz").exists()


def test_chunk_cache_round_trip(tmp_path) -> None:
    from src.ai_research.latent_liquidity_path_atlas.pipeline import _load_chunk_cache, _save_chunk_cache

    features = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_time": [pd.Timestamp("2024-01-01")],
            "event_side": ["DOWN"],
            "path_ret_5s": np.array([0.1], dtype=np.float32),
        }
    )
    labels = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_time": [pd.Timestamp("2024-01-01")],
            "event_side": ["DOWN"],
            "favorable_reversal": [True],
        }
    )
    path = tmp_path / "chunk.pkl.gz"
    _save_chunk_cache(path, features, labels)
    loaded_features, loaded_labels = _load_chunk_cache(path)
    pd.testing.assert_frame_equal(loaded_features, features)
    pd.testing.assert_frame_equal(loaded_labels, labels)
