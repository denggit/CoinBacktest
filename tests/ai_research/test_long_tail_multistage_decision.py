from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline
from src.ai_research.long_tail_multistage_decision.config import LongTailMultistageConfig
from src.ai_research.long_tail_multistage_decision.features import (
    build_checkpoint_row,
    extract_extended_event_path,
)
from src.ai_research.long_tail_multistage_decision.policy import (
    PolicyThresholds,
    enforce_non_overlap,
    simulate_policy_event,
    stable_policy_candidates,
)


def _frame(periods: int = 8000, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    return pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02, "close": close}, index=index)


def _timeline(start: str = "2024-01-01", periods: int = 600) -> ScoreTimeline:
    index = pd.date_range(start, periods=periods, freq="15min")
    scores = np.linspace(0.92, 0.30, periods)
    return ScoreTimeline(
        decision_times_ns=index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        scores=scores,
        calibration_thresholds={0.50: 0.50, 0.60: 0.60, 0.70: 0.70, 0.90: 0.90, 0.95: 0.95},
    )


def _event() -> EventCandidate:
    return EventCandidate(
        event_id="q70_test",
        decision_time_ns=int(pd.Timestamp("2024-01-01 04:00").value),
        score=0.91,
        signal_quantile=0.70,
    )


def _extract(frame: pd.DataFrame):
    path = prepare_minute_path_frame(frame)
    extraction = extract_extended_event_path(
        event=_event(),
        fold_id="WF_2024",
        phase="oos",
        scope="broad_q70",
        path=path,
        timeline=_timeline(),
        config=LongTailMultistageConfig(),
    )
    assert extraction is not None
    return extraction, path


def test_checkpoint_features_do_not_change_when_future_changes() -> None:
    base = _frame()
    entry = pd.Timestamp("2024-01-01 04:01")
    first = base.index.get_indexer(pd.date_range(entry, periods=180, freq="1min"))
    trajectory = np.linspace(100.0, 100.4, len(first))
    for column in ("open", "close"):
        base.iloc[first, base.columns.get_loc(column)] = trajectory
    base.iloc[first, base.columns.get_loc("high")] = trajectory + 0.02
    base.iloc[first, base.columns.get_loc("low")] = trajectory - 0.02

    winner = base.copy()
    loser = base.copy()
    future_index = pd.date_range(entry + pd.Timedelta(minutes=180), periods=7000, freq="1min")
    positions = winner.index.get_indexer(future_index)
    up = np.linspace(100.4, 108.0, len(positions))
    down = np.linspace(100.4, 92.0, len(positions))
    for frame, values in ((winner, up), (loser, down)):
        for column in ("open", "close"):
            frame.iloc[positions, frame.columns.get_loc(column)] = values
        frame.iloc[positions, frame.columns.get_loc("high")] = values + 0.02
        frame.iloc[positions, frame.columns.get_loc("low")] = values - 0.02

    win_extract, win_path = _extract(winner)
    lose_extract, lose_path = _extract(loser)
    config = LongTailMultistageConfig()
    win_row = build_checkpoint_row(win_extract, checkpoint_minutes=180, path=win_path, timeline=_timeline(), config=config)
    lose_row = build_checkpoint_row(lose_extract, checkpoint_minutes=180, path=lose_path, timeline=_timeline(), config=config)
    for column in [name for name in win_row if name.startswith("x_")]:
        left, right = win_row[column], lose_row[column]
        if pd.isna(left) and pd.isna(right):
            continue
        assert np.isclose(float(left), float(right), atol=1e-12), column
    assert win_row["label_persistent_failure"] != lose_row["label_persistent_failure"]


def _event_row() -> pd.Series:
    values = {
        "event_id": "event",
        "fold_id": "WF_2024",
        "scope": "broad_q70",
        "decision_time": pd.Timestamp("2024-01-01"),
        "entry_time": pd.Timestamp("2024-01-01 00:01"),
        "entry_price_delay_1m": 100.0,
        "entry_price_delay_3m": 100.1,
        "entry_price_delay_5m": 100.2,
        "open_after_180m": 100.5,
        "close_price_180m": 99.0,
        "close_price_360m": 102.0,
        "close_price_1440m": 104.0,
        "close_price_7200m": 108.0,
        "mfe_360m": 0.025,
        "mae_360m": 0.012,
        "mfe_1440m": 0.05,
        "mae_1440m": 0.015,
        "mfe_7200m": 0.10,
        "mae_7200m": 0.02,
        "event_score_percentile": 0.80,
        "weak_now_180": True,
        "weak_now_360": False,
        "path_class_360": "healthy_hold",
        "p_failure_180": 0.90,
        "p_recovery_180": 0.10,
        "p_failure_360": 0.20,
        "p_recovery_360": 0.80,
        "p_continuation_360": 0.80,
        "p_longhold_1440": 0.80,
    }
    return pd.Series(values)


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(0.8, 0.4, 0.2, 0.7, 0.8, 0.2, 0.7, 0.7)


def test_early_exit_requires_high_failure_and_low_recovery() -> None:
    row = _event_row()
    config = LongTailMultistageConfig()
    trade = simulate_policy_event(row, policy="full_multistage", delay_minutes=1, thresholds=_thresholds(), config=config)
    assert trade is not None
    assert trade["exit_reason"] == "confirmed_failure_t180"
    recovered = row.copy()
    recovered["p_recovery_180"] = 0.9
    trade2 = simulate_policy_event(recovered, policy="full_multistage", delay_minutes=1, thresholds=_thresholds(), config=config)
    assert trade2 is not None
    assert trade2["exit_reason"] == "five_day_longhold"


def test_half_probe_adds_only_after_healthy_t180() -> None:
    row = _event_row()
    row["p_failure_180"] = 0.2
    row["p_recovery_180"] = 0.8
    trade = simulate_policy_event(row, policy="half_probe_then_add", delay_minutes=1, thresholds=_thresholds(), config=LongTailMultistageConfig())
    assert trade is not None
    assert trade["position_added_at_180"] is True
    assert trade["cost_weight"] == 1.0


def test_non_overlap_skips_events_during_open_position() -> None:
    frame = pd.DataFrame(
        [
            {"executed": True, "entry_time": pd.Timestamp("2024-01-01"), "exit_time": pd.Timestamp("2024-01-02"), "decision_time": pd.Timestamp("2024-01-01")},
            {"executed": True, "entry_time": pd.Timestamp("2024-01-01 06:00"), "exit_time": pd.Timestamp("2024-01-01 12:00"), "decision_time": pd.Timestamp("2024-01-01 06:00")},
            {"executed": True, "entry_time": pd.Timestamp("2024-01-03"), "exit_time": pd.Timestamp("2024-01-04"), "decision_time": pd.Timestamp("2024-01-03")},
        ]
    )
    kept, skipped = enforce_non_overlap(frame)
    assert len(kept) == 2
    assert skipped == 1


def test_stable_policy_requires_both_years_and_positive_expectancy() -> None:
    summary = pd.DataFrame(
        [
            {"fold_id": "WF_2024", "scope": "broad_q70", "policy": "full_multistage", "delay_minutes": 1, "cost_multiplier": 2.0, "mean_net_return": 0.003, "profit_factor": 1.5, "trades": 100, "max_drawdown": -0.1, "top10_profit_share": 0.4, "mean_net_without_top10": 0.001, "total_compounded_return": 0.35},
            {"fold_id": "WF_2025", "scope": "broad_q70", "policy": "full_multistage", "delay_minutes": 1, "cost_multiplier": 2.0, "mean_net_return": 0.004, "profit_factor": 1.6, "trades": 110, "max_drawdown": -0.12, "top10_profit_share": 0.45, "mean_net_without_top10": 0.002, "total_compounded_return": 0.45},
        ]
    )
    periods = pd.DataFrame(
        [
            {"fold_id": fold, "scope": "broad_q70", "policy": "full_multistage", "delay_minutes": 1, "cost_multiplier": 2.0, "quarter": quarter, "mean_net_return": 0.001}
            for fold in ("WF_2024", "WF_2025")
            for quarter in ("Q1", "Q2", "Q3", "Q4")
        ]
    )
    stable = stable_policy_candidates(summary, periods, LongTailMultistageConfig())
    assert bool(stable.iloc[0]["stable_positive_expectancy"])


def test_config_does_not_restore_abandoned_state_model() -> None:
    text = str(LongTailMultistageConfig().to_dict()).lower()
    for token in ("strategic_state", "tactical_state", "entry_state", "activity_state"):
        assert token not in text


def test_rolling_entry_oof_uses_separate_calibration_blocks(monkeypatch) -> None:
    from src.ai_research.long_tail_multistage_decision import entry_oof
    from src.ai_research.long_tail_exit_audit.config import LongTailExitAuditConfig
    from src.ai_research.state_context_ablation.modeling import AblationPeriodData

    rows = 30000
    index = pd.date_range("2023-01-01", periods=rows, freq="15min")
    x = np.linspace(-1.0, 1.0, rows, dtype=np.float32).reshape(-1, 1)
    data = AblationPeriodData(
        timestamps_ns=index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        base_x=x,
        state_x=np.empty((rows, 0), dtype=np.float32),
        outcomes={"long_utility_h6": x[:, 0].astype(float)},
        base_columns=("x",),
        state_columns=(),
    )

    class FakeModel:
        def predict(self, matrix):
            return np.asarray(matrix[:, 0], dtype=float)

    monkeypatch.setattr(entry_oof, "fit_base_long_model", lambda data, config: FakeModel())
    config = LongTailMultistageConfig(
        entry_oof_min_train_days=60,
        entry_oof_calibration_days=10,
        entry_oof_blocks=3,
        minimum_train_rows=30,
        minimum_class_rows=5,
    )
    result = entry_oof.build_rolling_oof_entry_timeline(
        data,
        event_builder_config=LongTailExitAuditConfig(),
        config=config,
    )
    assert len(result.audit) >= 3
    assert result.valid_rows > 5000
    assert len(result.events) > 0
    assert (result.audit["maximum_train_time_ns"] < result.audit["minimum_calibration_time_ns"]).all()
    assert (result.audit["minimum_calibration_time_ns"] < result.audit["minimum_validation_time_ns"]).all()
