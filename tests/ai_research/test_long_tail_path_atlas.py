from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline
from src.ai_research.long_tail_path_atlas.atlas import (
    CLUSTER_FEATURES,
    assign_clusters,
    extract_event_path,
    fit_path_cluster_model,
    semantic_path_labels,
)
from src.ai_research.long_tail_path_atlas.config import LongTailPathAtlasConfig


def _minute_frame(periods: int = 4000, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close.copy(),
        },
        index=index,
    )


def _timeline(start: str = "2024-01-01", periods: int = 400) -> tuple[ScoreTimeline, np.ndarray]:
    index = pd.date_range(start, periods=periods, freq="15min")
    scores = np.linspace(0.0, 1.0, periods)
    calibration = np.linspace(-0.2, 1.2, 1000)
    thresholds = {q: float(np.quantile(calibration, q)) for q in (0.50, 0.60, 0.70, 0.90, 0.95)}
    return (
        ScoreTimeline(
            decision_times_ns=index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            scores=scores,
            calibration_thresholds=thresholds,
        ),
        calibration,
    )


def _event(timestamp: str = "2024-01-01 04:00") -> EventCandidate:
    return EventCandidate(
        event_id="q90_test",
        decision_time_ns=int(pd.Timestamp(timestamp).value),
        score=1.0,
        signal_quantile=0.90,
    )


def test_extracts_exact_complete_48h_path_and_six_hour_features() -> None:
    frame = _minute_frame(5000)
    entry = pd.Timestamp("2024-01-01 04:01")
    # Smooth +2% move over the first six hours.
    positions = frame.index.get_indexer(pd.date_range(entry, periods=360, freq="1min"))
    trajectory = np.linspace(100.0, 102.0, 360)
    frame.iloc[positions, frame.columns.get_loc("open")] = trajectory
    frame.iloc[positions, frame.columns.get_loc("close")] = trajectory
    frame.iloc[positions, frame.columns.get_loc("high")] = trajectory + 0.02
    frame.iloc[positions, frame.columns.get_loc("low")] = trajectory - 0.02
    path = prepare_minute_path_frame(frame)
    timeline, calibration = _timeline(periods=500)
    extraction = extract_event_path(
        event=_event(),
        fold_id="WF_2024",
        phase="oos",
        path=path,
        timeline=timeline,
        calibration_scores=calibration,
        config=LongTailPathAtlasConfig(),
    )
    assert extraction is not None
    assert len(extraction.points) == 48 * 60
    assert extraction.summary["path_rows"] == 48 * 60
    assert float(extraction.summary["ret_360m"]) > 0.019
    assert bool(extraction.summary["fixed6h_positive_expectancy_event"])


def test_score_asof_does_not_use_the_next_15m_decision_early() -> None:
    frame = _minute_frame(5000)
    path = prepare_minute_path_frame(frame)
    times = pd.to_datetime(["2024-01-01 04:00", "2024-01-01 04:15", "2024-01-01 04:30"])
    timeline = ScoreTimeline(
        decision_times_ns=times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        scores=np.array([0.1, 0.9, 0.8]),
        calibration_thresholds={0.50: 0.5, 0.60: 0.6, 0.70: 0.7, 0.90: 0.9, 0.95: 0.95},
    )
    extraction = extract_event_path(
        event=_event(),
        fold_id="WF_2024",
        phase="oos",
        path=path,
        timeline=timeline,
        calibration_scores=np.linspace(0.0, 1.0, 101),
        config=LongTailPathAtlasConfig(),
    )
    assert extraction is not None
    points = extraction.points
    assert points.loc[points["timestamp"] == pd.Timestamp("2024-01-01 04:14"), "base_score"].iloc[0] == 0.1
    assert points.loc[points["timestamp"] == pd.Timestamp("2024-01-01 04:15"), "base_score"].iloc[0] == 0.9


def test_incomplete_or_gapped_48h_path_is_rejected() -> None:
    frame = _minute_frame(3000)
    frame = frame.drop(pd.Timestamp("2024-01-02 00:00"))
    path = prepare_minute_path_frame(frame)
    timeline, calibration = _timeline(periods=500)
    extraction = extract_event_path(
        event=_event(),
        fold_id="WF_2024",
        phase="oos",
        path=path,
        timeline=timeline,
        calibration_scores=calibration,
        config=LongTailPathAtlasConfig(),
    )
    assert extraction is None


def _base_semantic_row() -> dict[str, object]:
    return {
        "fixed6h_positive_expectancy_event": True,
        "time_to_up_1p0pct": 30.0,
        "mae_before_first_1pct": 0.002,
        "mae_360m": 0.003,
        "longest_underwater_360m": 20.0,
        "mfe_180m": 0.018,
        "mfe_360m": 0.020,
        "peak_giveback_360m": 0.003,
        "time_to_mfe_360m": 100.0,
        "ret_1440m": 0.015,
        "mfe_1440m": 0.025,
        "post6_mfe_increment_1440m": 0.005,
        "q90_reconfirmations_360m": 3.0,
        "score_percentile_min_360m": 0.75,
    }


def test_semantic_types_cover_immediate_delayed_spike_and_failures() -> None:
    config = LongTailPathAtlasConfig()
    immediate = semantic_path_labels(_base_semantic_row(), config)
    assert immediate["semantic_path_type"] == "immediate_clean_winner"

    delayed_row = _base_semantic_row()
    delayed_row.update(time_to_up_1p0pct=220.0, mae_before_first_1pct=0.012, mae_360m=0.015, longest_underwater_360m=180.0)
    delayed = semantic_path_labels(delayed_row, config)
    assert delayed["semantic_path_type"] == "delayed_recovery_winner"

    spike_row = _base_semantic_row()
    spike_row.update(time_to_up_1p0pct=120.0, mae_before_first_1pct=0.003, peak_giveback_360m=0.012)
    spike = semantic_path_labels(spike_row, config)
    assert spike["semantic_path_type"] == "early_spike_giveback_winner"

    rescue_row = _base_semantic_row()
    rescue_row.update(fixed6h_positive_expectancy_event=False, ret_1440m=0.02, mfe_1440m=0.03, mfe_360m=0.005, peak_giveback_360m=0.002)
    rescue = semantic_path_labels(rescue_row, config)
    assert rescue["semantic_path_type"] == "late_rescue_after_6h"

    failure_row = _base_semantic_row()
    failure_row.update(fixed6h_positive_expectancy_event=False, ret_1440m=-0.02, mfe_1440m=0.005, mfe_360m=0.004, peak_giveback_360m=0.001)
    failure = semantic_path_labels(failure_row, config)
    assert failure["semantic_path_type"] == "persistent_failure"


def test_cluster_model_is_fit_only_from_discovery_frame() -> None:
    rng = np.random.default_rng(7)
    discovery = pd.DataFrame(rng.normal(size=(60, len(CLUSTER_FEATURES))), columns=CLUSTER_FEATURES)
    config = LongTailPathAtlasConfig(minimum_discovery_events=36)
    model = fit_path_cluster_model(discovery, config)
    assert model is not None
    centers_before = model.model.cluster_centers_.copy()
    oos = pd.DataFrame(rng.normal(loc=50.0, size=(20, len(CLUSTER_FEATURES))), columns=CLUSTER_FEATURES)
    assigned = assign_clusters(oos, model)
    assert len(assigned) == len(oos)
    np.testing.assert_allclose(centers_before, model.model.cluster_centers_)


def test_configuration_formally_excludes_abandoned_state_model() -> None:
    payload = str(LongTailPathAtlasConfig().to_dict()).lower()
    assert "strategic_state" not in payload
    assert "tactical_state" not in payload
    assert "entry_state" not in payload
    assert "activity_state" not in payload
