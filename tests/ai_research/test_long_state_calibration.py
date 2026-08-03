from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.config import LongStateCalibrationConfig
from src.ai_research.long_state_calibration.modeling import (
    ACTIVITY_SOFT_COLUMNS,
    ALL_SOFT_COLUMNS,
    STRATEGIC_SOFT_COLUMNS,
    build_expanding_oof_blocks,
    empirical_percentile,
    meta_matrix,
    select_episode_peaks,
    select_stable_candidates,
    state_rank_multiplier,
)
from src.ai_research.state_context_ablation.modeling import AblationPeriodData


def _period(rows: int = 20000) -> AblationPeriodData:
    columns = tuple(dict.fromkeys((*ALL_SOFT_COLUMNS, "tactical_state", "entry_state", "tactical_score", "entry_score")))
    state = np.arange(rows * len(columns), dtype=np.float32).reshape(rows, len(columns)) / 1000.0
    return AblationPeriodData(
        timestamps_ns=pd.date_range("2023-01-01", periods=rows, freq="15min").to_numpy(dtype="datetime64[ns]").astype(np.int64),
        base_x=np.ones((rows, 3), dtype=np.float32),
        state_x=state,
        outcomes={
            "long_utility_h6": np.linspace(-0.02, 0.03, rows),
            "long_mfe_h6": np.linspace(0.0, 0.05, rows),
            "long_mae_h6": np.linspace(0.03, 0.0, rows),
            "future_close_return_h6": np.linspace(-0.01, 0.02, rows),
        },
        base_columns=("a", "b", "c"),
        state_columns=columns,
    )


def test_soft_state_contract_excludes_tactical_and_entry_direction() -> None:
    forbidden = {"tactical_state", "entry_state", "tactical_score", "entry_score"}
    assert not forbidden.intersection(ALL_SOFT_COLUMNS)
    assert set(STRATEGIC_SOFT_COLUMNS).issubset(ALL_SOFT_COLUMNS)
    assert set(ACTIVITY_SOFT_COLUMNS).issubset(ALL_SOFT_COLUMNS)


def test_meta_matrices_only_add_frozen_score_and_soft_state() -> None:
    data = _period(100)
    score = np.linspace(-1, 1, 100)
    score_only, score_names = meta_matrix("score_only_meta", score, data)
    combined, names = meta_matrix("score_plus_strategic_activity_meta", score, data)
    state_only, state_names = meta_matrix("soft_state_only_meta", score, data)
    assert score_only.shape == (100, 1)
    assert score_names == ("base_long_score",)
    assert combined.shape[1] == 1 + len(ALL_SOFT_COLUMNS)
    assert names[0] == "base_long_score"
    assert state_only.shape[1] == len(ALL_SOFT_COLUMNS)
    assert all("tactical_state" not in name and "entry_state" not in name for name in state_names)


def test_expanding_oof_blocks_are_strictly_prior_with_embargo() -> None:
    config = LongStateCalibrationConfig(oof_min_train_days=60, oof_blocks=4, oof_embargo_hours=18)
    index = pd.date_range("2023-01-01", "2023-09-30", freq="15min")
    blocks = build_expanding_oof_blocks(index, config)
    assert len(blocks) == 4
    for block in blocks:
        assert block.train_end < block.prediction_start
        assert block.prediction_start - block.train_end == pd.Timedelta(hours=18)


def test_episode_peaks_merge_dense_signals_and_enforce_six_hour_cooldown() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="15min")
    scores = np.zeros(40)
    signal = np.zeros(40, dtype=bool)
    signal[[0, 1, 2, 20, 21, 30]] = True
    scores[[0, 1, 2, 20, 21, 30]] = [1, 3, 2, 4, 5, 6]
    positions = select_episode_peaks(
        index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        scores,
        signal,
        merge_gap_minutes=30,
        cooldown_hours=6,
    )
    assert positions.tolist() == [1, 30]


def test_rank_multiplier_is_calibration_based_and_bounded() -> None:
    calibration_base = np.array([0.0, 1.0, 2.0, 3.0])
    calibration_variant = np.array([0.0, 1.0, 2.0, 3.0])
    test_base = np.array([-10.0, 1.5, 10.0])
    test_variant = np.array([10.0, 1.5, -10.0])
    multiplier = state_rank_multiplier(calibration_base, calibration_variant, test_base, test_variant)
    assert np.all(multiplier >= 0.5)
    assert np.all(multiplier <= 1.5)
    pct = empirical_percentile(calibration_base, np.array([0.5, 2.5]))
    assert np.allclose(pct, [0.25, 0.75])


def test_stable_candidate_requires_both_oos_years() -> None:
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        rows.append(
            {
                "fold_id": fold,
                "variant": "score_plus_activity_meta",
                "delta_long_utility_rank_ic_vs_score_only_meta": 0.005,
                "delta_rerank_mean_long_utility_vs_score_only_meta": 0.001,
                "delta_rerank_mean_mae_vs_score_only_meta": -0.0001,
                "delta_weighted_mean_net_close_return_vs_score_only_meta": 0.0002,
                "delta_weighted_mean_mae_vs_score_only_meta": -0.0001,
                "independent_events": 100,
            }
        )
    stable = select_stable_candidates(pd.DataFrame(rows), LongStateCalibrationConfig())
    assert bool(stable.iloc[0]["passes_calibration"])
    assert bool(stable.iloc[0]["passes_risk_scaling"])
