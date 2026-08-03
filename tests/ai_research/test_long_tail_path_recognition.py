from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline
from src.ai_research.long_tail_path_atlas.atlas import extract_event_path
from src.ai_research.long_tail_path_atlas.config import LongTailPathAtlasConfig
from src.ai_research.long_tail_path_recognition.config import LongTailPathRecognitionConfig
from src.ai_research.long_tail_path_recognition.features import build_checkpoint_row, feature_sets
from src.ai_research.long_tail_path_recognition.modeling import (
    causal_oof_probabilities,
    stable_signal_candidates,
)


def _minute_frame(periods: int = 5000, start: str = "2024-01-01") -> pd.DataFrame:
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
    scores = 0.9 - np.linspace(0.0, 0.5, periods)
    calibration = np.linspace(0.0, 1.0, 1001)
    return (
        ScoreTimeline(
            decision_times_ns=index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
            scores=scores,
            calibration_thresholds={q: float(np.quantile(calibration, q)) for q in (0.50, 0.60, 0.70, 0.90, 0.95)},
        ),
        calibration,
    )


def _event() -> EventCandidate:
    return EventCandidate(
        event_id="q70_test",
        decision_time_ns=int(pd.Timestamp("2024-01-01 04:00").value),
        score=0.95,
        signal_quantile=0.70,
    )


def _extraction(frame: pd.DataFrame):
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
    return extraction, path


def test_checkpoint_features_ignore_future_path_after_cutoff() -> None:
    base = _minute_frame()
    entry = pd.Timestamp("2024-01-01 04:01")
    first60 = base.index.get_indexer(pd.date_range(entry, periods=60, freq="1min"))
    trajectory = np.linspace(100.0, 100.5, 60)
    for column in ("open", "close"):
        base.iloc[first60, base.columns.get_loc(column)] = trajectory
    base.iloc[first60, base.columns.get_loc("high")] = trajectory + 0.02
    base.iloc[first60, base.columns.get_loc("low")] = trajectory - 0.02

    winner = base.copy()
    future = winner.index.get_indexer(pd.date_range(entry + pd.Timedelta(minutes=60), periods=1380, freq="1min"))
    up = np.linspace(100.5, 104.0, len(future))
    for column in ("open", "close"):
        winner.iloc[future, winner.columns.get_loc(column)] = up
    winner.iloc[future, winner.columns.get_loc("high")] = up + 0.02
    winner.iloc[future, winner.columns.get_loc("low")] = up - 0.02

    loser = base.copy()
    down = np.linspace(100.5, 96.0, len(future))
    for column in ("open", "close"):
        loser.iloc[future, loser.columns.get_loc(column)] = down
    loser.iloc[future, loser.columns.get_loc("high")] = down + 0.02
    loser.iloc[future, loser.columns.get_loc("low")] = down - 0.02

    winner_extraction, winner_path = _extraction(winner)
    loser_extraction, loser_path = _extraction(loser)
    config = LongTailPathRecognitionConfig()
    winner_row = build_checkpoint_row(winner_extraction, checkpoint_minutes=60, path=winner_path, config=config)
    loser_row = build_checkpoint_row(loser_extraction, checkpoint_minutes=60, path=loser_path, config=config)

    feature_columns = [column for column in winner_row if column.startswith("x_")]
    for column in feature_columns:
        left = winner_row[column]
        right = loser_row[column]
        if pd.isna(left) and pd.isna(right):
            continue
        assert np.isclose(float(left), float(right), atol=1e-12), column
    assert winner_row["net_24h_1x"] > loser_row["net_24h_1x"]


def test_score_ablation_has_price_only_and_score_only_variants() -> None:
    frame, path = _extraction(_minute_frame())
    row = build_checkpoint_row(frame, checkpoint_minutes=60, path=path, config=LongTailPathRecognitionConfig())
    sets = {item.name: item.columns for item in feature_sets(pd.DataFrame([row]))}
    assert all(column.startswith("x_path__") for column in sets["path_structure_logistic"])
    assert all(column.startswith("x_score__") for column in sets["score_only_logistic"])
    assert not any(column.startswith("x_score__") for column in sets["path_structure_logistic"])


def test_checkpoint_structure_detects_break_and_reclaim_without_future() -> None:
    frame = _minute_frame()
    entry = pd.Timestamp("2024-01-01 04:01")
    positions = frame.index.get_indexer(pd.date_range(entry, periods=180, freq="1min"))
    trajectory = np.concatenate([np.linspace(100.0, 99.0, 90), np.linspace(99.0, 100.3, 90)])
    for column in ("open", "close"):
        frame.iloc[positions, frame.columns.get_loc(column)] = trajectory
    frame.iloc[positions, frame.columns.get_loc("high")] = trajectory + 0.02
    frame.iloc[positions, frame.columns.get_loc("low")] = trajectory - 0.02
    extraction, path = _extraction(frame)
    row = build_checkpoint_row(extraction, checkpoint_minutes=180, path=path, config=LongTailPathRecognitionConfig())
    assert row["x_path__reclaimed_entry_after_drawdown"] == 1.0
    assert row["x_path__recovery_from_trough"] > 0.01
    assert row["x_path__current_return"] > 0


def test_causal_oof_leaves_early_rows_unpredicted() -> None:
    rng = np.random.default_rng(11)
    rows = 160
    times = pd.date_range("2023-01-01", periods=rows, freq="12h")
    frame = pd.DataFrame(
        {
            "decision_time": times,
            "x_path__current_return": rng.normal(size=rows),
            "x_path__current_mfe": rng.normal(size=rows),
            "x_path__current_mae": rng.normal(size=rows),
            "x_path__peak_giveback": rng.normal(size=rows),
            "x_path__underwater_fraction": rng.uniform(size=rows),
            "x_path__capture_of_mfe": rng.normal(size=rows),
            "x_score__entry_score_percentile": rng.uniform(size=rows),
        }
    )
    target = (frame["x_path__current_return"].to_numpy() + rng.normal(scale=0.8, size=rows) > 0).astype(np.int8)
    config = LongTailPathRecognitionConfig(minimum_train_rows=30, minimum_class_rows=8, oof_splits=3)
    feature_set = feature_sets(frame.assign(**{
        "x_path__recovery_from_trough": 0.0,
        "x_score__score_percentile_end": 0.5,
    }))[0]
    result = causal_oof_probabilities(
        frame,
        target=target,
        task="persistent_failure",
        feature_set=feature_set,
        config=config,
    )
    assert np.isnan(result.probabilities[:20]).all()
    assert result.valid_mask.sum() > 20
    assert result.folds_used >= 1


def test_stable_candidate_requires_both_oos_years() -> None:
    metrics = pd.DataFrame(
        [
            {"fold_id": "WF_2024", "task": "persistent_failure", "checkpoint_minutes": 180, "feature_set": "path_structure_logistic", "scope": "primary_q90", "roc_auc": 0.72, "top_probability_decile_lift": 2.0, "average_precision_lift": 1.4},
            {"fold_id": "WF_2025", "task": "persistent_failure", "checkpoint_minutes": 180, "feature_set": "path_structure_logistic", "scope": "primary_q90", "roc_auc": 0.70, "top_probability_decile_lift": 1.8, "average_precision_lift": 1.3},
            {"fold_id": "WF_2024", "task": "post6_continuation", "checkpoint_minutes": 360, "feature_set": "score_only_logistic", "scope": "primary_q90", "roc_auc": 0.80, "top_probability_decile_lift": 2.0, "average_precision_lift": 1.5},
        ]
    )
    stable = stable_signal_candidates(metrics)
    row = stable.loc[stable["task"] == "persistent_failure"].iloc[0]
    assert bool(row["stable_signal"])
    assert not (stable["task"] == "post6_continuation").any()


def test_configuration_excludes_abandoned_state_model() -> None:
    payload = str(LongTailPathRecognitionConfig().to_dict()).lower()
    for token in ("strategic_state", "tactical_state", "entry_state", "activity_state"):
        assert token not in payload
