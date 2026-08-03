from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.future_process_forecast.alert_audit import audit_alert_episodes, build_alert_episodes
from src.ai_research.future_process_forecast.alert_audit_config import DEFAULT_PROCESS_ALERT_AUDIT_CONFIG


def _event(process: str, start: str, end: str = "2024-01-01 12:00:00") -> pd.DataFrame:
    row = {
        "process": process,
        "event_uid": f"{process}:{start}",
        "start_time": pd.Timestamp(start),
        "end_time": pd.Timestamp(end),
        "start_price": 100.0,
        "target_move": 0.05 if process != "volatile_range" else 0.06,
        "mfe_72h": 0.08,
        "up_excursion": 0.03,
        "down_excursion": 0.03,
    }
    return pd.DataFrame([row])


def test_continuous_signal_points_are_one_first_alert_episode() -> None:
    times = pd.date_range("2024-01-01", periods=12, freq="15min").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    scores = np.array([0.1, 0.95, 0.96, 0.94, 0.2, 0.1, 0.1, 0.1, 0.1, 0.93, 0.94, 0.1])
    episodes = build_alert_episodes(times, scores, 0.9, merge_gap_hours=1.0)
    assert len(episodes) == 2
    assert episodes.iloc[0]["signal_points"] == 3
    assert episodes.iloc[0]["first_alert_time"] == pd.Timestamp("2024-01-01 00:15:00")


def test_pre_start_and_early_start_can_be_actionable_but_late_alert_is_not() -> None:
    config = DEFAULT_PROCESS_ALERT_AUDIT_CONFIG
    times = pd.date_range("2024-01-01 08:00:00", periods=21, freq="15min")
    prices = np.full(len(times), 100.0)
    episodes = pd.DataFrame(
        [
            {"episode_id": 0, "first_pos": 0, "last_pos": 0, "first_alert_time": times[0], "last_alert_time": times[0], "signal_points": 1, "duration_hours": 0.0, "first_score": 0.9, "peak_score": 0.9},
            {"episode_id": 1, "first_pos": 9, "last_pos": 9, "first_alert_time": times[9], "last_alert_time": times[9], "signal_points": 1, "duration_hours": 0.0, "first_score": 0.9, "peak_score": 0.9},
            {"episode_id": 2, "first_pos": 20, "last_pos": 20, "first_alert_time": times[20], "last_alert_time": times[20], "signal_points": 1, "duration_hours": 0.0, "first_score": 0.9, "peak_score": 0.9},
        ]
    )
    result = audit_alert_episodes(
        episodes,
        process="up_expansion",
        horizon_hours=6,
        process_events=_event("up_expansion", "2024-01-01 10:00:00", "2024-01-01 16:00:00"),
        decision_prices=prices,
        fold_start=pd.Timestamp("2024-01-01"),
        fold_end=pd.Timestamp("2024-01-31"),
        config=config,
    )
    assert result.episodes.iloc[0]["classification"] == "actionable_pre_start"
    assert result.episodes.iloc[1]["classification"] == "actionable_early_start"
    assert result.episodes.iloc[2]["classification"] == "late_or_spent_ongoing"


def test_event_coverage_uses_first_actionable_alert_and_remaining_space() -> None:
    config = DEFAULT_PROCESS_ALERT_AUDIT_CONFIG
    times = pd.date_range("2024-01-01 08:00:00", periods=9, freq="15min")
    episodes = pd.DataFrame(
        [
            {"episode_id": 0, "first_pos": 0, "last_pos": 0, "first_alert_time": times[0], "last_alert_time": times[0], "signal_points": 1, "duration_hours": 0.0, "first_score": 0.9, "peak_score": 0.9},
            {"episode_id": 1, "first_pos": 4, "last_pos": 4, "first_alert_time": times[4], "last_alert_time": times[4], "signal_points": 1, "duration_hours": 0.0, "first_score": 0.95, "peak_score": 0.95},
        ]
    )
    prices = np.full(len(times), 100.0)
    result = audit_alert_episodes(
        episodes,
        process="volatile_range",
        horizon_hours=6,
        process_events=_event("volatile_range", "2024-01-01 10:00:00"),
        decision_prices=prices,
        fold_start=pd.Timestamp("2024-01-01"),
        fold_end=pd.Timestamp("2024-01-31"),
        config=config,
    )
    assert result.event_metrics["event_coverage"] == 1.0
    assert result.event_metrics["mean_actionable_alerts_per_covered_event"] == 2.0
    assert result.event_metrics["median_first_alert_remaining_opportunity"] >= 0.05
