from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.state_context_ablation.config import (
    ABLATION_VARIANTS,
    StateContextAblationConfig,
)
from src.ai_research.state_context_ablation.modeling import (
    ALL_STATE_COLUMNS,
    AblationPeriodData,
    _align_probability,
    build_uplift_table,
    default_ablation_folds,
    select_stable_uplift,
    variant_matrix,
)
from src.ai_research.state_context_ablation.outcomes import build_opening_outcome_frame
from src.ai_research.swing_baseline.dataset import SwingYearShard


def _fake_shard() -> SwingYearShard:
    minute_index = pd.date_range("2024-01-01 00:00", periods=500, freq="1min")
    price = 100.0 + np.arange(len(minute_index), dtype=float) * 0.01
    ohlc = np.column_stack([price, price + 0.05, price - 0.04, price + 0.01])
    decisions = pd.DatetimeIndex([minute_index[0], minute_index[15], minute_index[30]])
    entry_times = decisions + pd.Timedelta(minutes=1)
    entry_positions = minute_index.get_indexer(entry_times)
    return SwingYearShard(
        path=Path("fake"),
        decision_times_ns=decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        features=np.zeros((3, 2), dtype=np.float32),
        context=np.zeros((3, 1), dtype=np.float64),
        labels=np.zeros((3, 1), dtype=np.float32),
        entry_times_ns=entry_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entry_prices=ohlc[entry_positions, 0],
        minute_times_ns=minute_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        minute_ohlc=ohlc,
        high_feature_columns=("a",),
        full_feature_columns=("a", "b"),
        context_columns=("c",),
        label_columns=("label",),
    )


def test_outcomes_start_at_next_minute_entry_and_use_future_only() -> None:
    config = StateContextAblationConfig(horizons_hours=(3, 6), primary_horizon_hours=6)
    shard = _fake_shard()
    frame = build_opening_outcome_frame(shard, config)
    entry = shard.entry_prices[0]
    start = 1
    end = start + 3 * 60
    expected_high = np.max(shard.minute_ohlc[start:end, 1])
    expected_low = np.min(shard.minute_ohlc[start:end, 2])
    expected_close = shard.minute_ohlc[end - 1, 3]
    assert np.isclose(frame.iloc[0]["long_mfe_h3"], expected_high / entry - 1.0)
    assert np.isclose(frame.iloc[0]["long_mae_h3"], 1.0 - expected_low / entry)
    assert np.isclose(frame.iloc[0]["future_close_return_h3"], expected_close / entry - 1.0)


def _period(probability: bool = True) -> AblationPeriodData:
    rows = 8
    return AblationPeriodData(
        timestamps_ns=pd.date_range("2024-01-01", periods=rows, freq="15min").to_numpy(dtype="datetime64[ns]").astype(np.int64),
        base_x=np.ones((rows, 3), dtype=np.float32),
        state_x=np.arange(rows * len(ALL_STATE_COLUMNS), dtype=np.float32).reshape(rows, -1),
        outcomes={
            "long_mfe_h3": np.ones(rows),
            "long_mae_h3": np.ones(rows),
            "short_mfe_h3": np.ones(rows),
            "short_mae_h3": np.ones(rows),
            "future_close_return_h3": np.ones(rows),
            "long_utility_h3": np.ones(rows),
            "short_utility_h3": np.ones(rows),
            "long_mfe_h6": np.ones(rows),
            "long_mae_h6": np.ones(rows),
            "short_mfe_h6": np.ones(rows),
            "short_mae_h6": np.ones(rows),
            "future_close_return_h6": np.ones(rows),
            "long_utility_h6": np.ones(rows),
            "short_utility_h6": np.ones(rows),
        },
        base_columns=("b1", "b2", "b3"),
        state_columns=ALL_STATE_COLUMNS,
        activity_persist_probability=np.linspace(0.1, 0.8, rows) if probability else None,
    )


def test_all_ablation_variants_build_distinct_matrices() -> None:
    data = _period()
    shapes = {}
    for variant in ABLATION_VARIANTS:
        matrix, columns = variant_matrix(variant, data)
        assert matrix.shape[0] == len(data.timestamps_ns)
        assert matrix.shape[1] == len(columns)
        shapes[variant] = matrix.shape[1]
    assert shapes["base_plus_all_state"] > shapes["base_multiframe"]
    assert shapes["base_plus_all_state_and_activity_persist"] == shapes["base_plus_all_state"] + 1
    assert shapes["state_only"] == len(ALL_STATE_COLUMNS)


def test_probability_alignment_is_exact_and_does_not_forward_fill() -> None:
    target = np.array([10, 20, 30, 40], dtype=np.int64)
    source = np.array([10, 30], dtype=np.int64)
    aligned = _align_probability(target, source, np.array([0.2, 0.8]))
    assert np.isclose(aligned[0], 0.2)
    assert np.isnan(aligned[1])
    assert np.isclose(aligned[2], 0.8)
    assert np.isnan(aligned[3])


def test_folds_keep_2026_sealed_and_use_prior_calibration() -> None:
    config = StateContextAblationConfig()
    folds = default_ablation_folds(config)
    assert [fold.fold_id for fold in folds] == ["WF_2024", "WF_2025"]
    for fold in folds:
        assert fold.fit_end < fold.calibration_start < fold.test_start
        assert fold.test_end < pd.Timestamp(config.sealed_holdout_start)


def test_uplift_and_stable_candidate_require_both_folds() -> None:
    metrics = pd.DataFrame(
        [
            {"fold_id": fold, "variant": variant, "direction_rank_ic": rank, "utility_direction_rank_ic": rank, "mean_utility_rank_ic": rank, "meaningful_direction_accuracy": 0.55}
            for fold in ("WF_2024", "WF_2025")
            for variant, rank in (("base_multiframe", 0.10), ("base_plus_all_state", 0.12))
        ]
    )
    signals = pd.DataFrame(
        [
            {"fold_id": fold, "variant": variant, "quantile": 0.90, "cost_multiplier": 1.0, "signals": 500, "mean_net_close_return": net, "profit_factor": pf, "mean_mae": 0.01, "mean_mfe": 0.02}
            for fold in ("WF_2024", "WF_2025")
            for variant, net, pf in (("base_multiframe", 0.001, 1.1), ("base_plus_all_state", 0.0015, 1.2))
        ]
    )
    uplift = build_uplift_table(metrics, signals)
    stable = select_stable_uplift(uplift, StateContextAblationConfig())
    assert len(stable) == 1
    assert bool(stable.iloc[0]["passes"])


def test_outcomes_do_not_cross_sealed_2026_boundary() -> None:
    minute_index = pd.date_range("2025-12-31 17:50", periods=800, freq="1min")
    price = 100.0 + np.arange(len(minute_index), dtype=float) * 0.001
    ohlc = np.column_stack([price, price + 0.01, price - 0.01, price])
    decisions = pd.DatetimeIndex([pd.Timestamp("2025-12-31 17:45"), pd.Timestamp("2025-12-31 18:15")])
    entry_times = decisions + pd.Timedelta(minutes=1)
    entry_positions = minute_index.searchsorted(entry_times)
    shard = SwingYearShard(
        path=Path("fake"),
        decision_times_ns=decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        features=np.zeros((2, 1), dtype=np.float32),
        context=np.zeros((2, 1), dtype=np.float64),
        labels=np.zeros((2, 1), dtype=np.float32),
        entry_times_ns=entry_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entry_prices=ohlc[entry_positions, 0],
        minute_times_ns=minute_index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        minute_ohlc=ohlc,
        high_feature_columns=("a",),
        full_feature_columns=("a",),
        context_columns=("c",),
        label_columns=("label",),
    )
    frame = build_opening_outcome_frame(shard, StateContextAblationConfig())
    assert np.isfinite(frame.iloc[0]["long_mfe_h6"])
    assert np.isnan(frame.iloc[1]["long_mfe_h6"])
