from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.config import (
    EXIT_RECIPES,
    ExitRecipe,
    LongTailExitAuditConfig,
)
from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.modeling import select_stable_candidates
from src.ai_research.long_tail_exit_audit.simulator import (
    EventCandidate,
    ScoreTimeline,
    simulate_event,
    simulate_sequential_events,
    structural_stop_at,
)


def _path(periods: int = 4000, *, start: str = "2024-01-01"):
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    frame = pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close.copy(),
        },
        index=index,
    )
    # Stable causal structure at 99.0 before entries.
    frame.loc[index[:300], "low"] = 99.0
    return frame


def _timeline(start: str = "2024-01-01", periods: int = 400) -> ScoreTimeline:
    index = pd.date_range(start, periods=periods, freq="15min")
    score = np.full(periods, 1.0)
    return ScoreTimeline(
        decision_times_ns=index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        scores=score,
        calibration_thresholds={0.50: 0.5, 0.60: 0.6, 0.70: 0.7, 0.90: 0.9, 0.95: 0.95},
    )


def _event(start: str = "2024-01-01 04:00") -> EventCandidate:
    return EventCandidate(
        event_id="e1",
        decision_time_ns=int(pd.Timestamp(start).value),
        score=1.0,
        signal_quantile=0.90,
    )


def test_structural_stop_uses_only_pre_entry_minutes() -> None:
    frame = _path(600)
    path = prepare_minute_path_frame(frame)
    entry_time = pd.Timestamp("2024-01-01 04:01")
    position = path.locate_exact(entry_time)
    assert position is not None
    config = LongTailExitAuditConfig()
    before = structural_stop_at(path, position, 100.0, 60, config)
    assert before is not None
    # A large future drop must not alter the already-computed pre-entry stop.
    frame.loc[entry_time + pd.Timedelta(minutes=5), "low"] = 80.0
    changed = prepare_minute_path_frame(frame)
    after = structural_stop_at(changed, position, 100.0, 60, config)
    assert after is not None
    assert before.stop_price == after.stop_price


def test_same_minute_stop_and_target_uses_conservative_stop_first() -> None:
    frame = _path(2000)
    event = _event()
    entry_time = pd.Timestamp(event.decision_time_ns, unit="ns") + pd.Timedelta(minutes=1)
    # Prior structure creates about 1% risk. This minute touches both stop and TP.
    frame.loc[entry_time, "high"] = 103.0
    frame.loc[entry_time, "low"] = 98.0
    path = prepare_minute_path_frame(frame)
    recipe = next(item for item in EXIT_RECIPES if item.name == "s60_tp_1p5r")
    trade = simulate_event(
        event=event,
        recipe=recipe,
        delay_minutes=1,
        path=path,
        timeline=_timeline(),
        config=LongTailExitAuditConfig(),
    )
    assert trade is not None
    assert trade.exit_reason == "hard_stop"
    assert trade.gross_return < 0


def test_trailing_activation_is_deferred_until_next_minute() -> None:
    frame = _path(2000)
    event = _event()
    entry_time = pd.Timestamp(event.decision_time_ns, unit="ns") + pd.Timedelta(minutes=1)
    # First minute activates the trail and dips through the would-be trail. It
    # must not exit because intraminute ordering is unknown.
    frame.loc[entry_time, ["high", "low", "close"]] = [101.6, 100.2, 101.0]
    frame.loc[entry_time + pd.Timedelta(minutes=1), ["high", "low", "close"]] = [101.2, 100.8, 100.9]
    path = prepare_minute_path_frame(frame)
    recipe = next(item for item in EXIT_RECIPES if item.name == "s60_trail_a1p0_g0p5")
    trade = simulate_event(
        event=event,
        recipe=recipe,
        delay_minutes=1,
        path=path,
        timeline=_timeline(),
        config=LongTailExitAuditConfig(),
    )
    assert trade is not None
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_time_ns == int((entry_time + pd.Timedelta(minutes=1)).value)


def test_rolling_renewal_exits_one_minute_after_failed_checkpoint() -> None:
    frame = _path(4000)
    event = _event()
    timeline = _timeline(periods=500)
    checkpoint = pd.Timestamp(event.decision_time_ns, unit="ns") + pd.Timedelta(hours=6)
    scores = timeline.scores.copy()
    position = int(np.searchsorted(timeline.decision_times_ns, int(checkpoint.value)))
    scores[position:] = 0.6  # below q70, above hard invalidation threshold
    timeline = ScoreTimeline(timeline.decision_times_ns, scores, timeline.calibration_thresholds)
    path = prepare_minute_path_frame(frame)
    recipe = next(item for item in EXIT_RECIPES if item.name == "s60_renew_q70_trail")
    trade = simulate_event(
        event=event,
        recipe=recipe,
        delay_minutes=1,
        path=path,
        timeline=timeline,
        config=LongTailExitAuditConfig(),
    )
    assert trade is not None
    assert trade.exit_reason == "model_not_renewed"
    assert trade.exit_time_ns == int((checkpoint + pd.Timedelta(minutes=1)).value)


def test_sequential_simulation_skips_overlapping_positions() -> None:
    frame = _path(4000)
    path = prepare_minute_path_frame(frame)
    timeline = _timeline(periods=500)
    first = _event()
    second = EventCandidate(
        event_id="e2",
        decision_time_ns=first.decision_time_ns + int(pd.Timedelta(hours=1).value),
        score=1.1,
        signal_quantile=0.90,
    )
    recipe = ExitRecipe(name="long_safety", stop_lookback_minutes=60, safety_cap_hours=24)
    trades, audit = simulate_sequential_events(
        events=(first, second),
        recipe=recipe,
        delay_minutes=1,
        path=path,
        timeline=timeline,
        config=LongTailExitAuditConfig(recipes=(recipe,)),
    )
    assert len(trades) == 1
    assert audit["skipped_overlap"] == 1


def test_positive_expectancy_gate_rejects_one_negative_year_even_with_high_win_rate() -> None:
    config = LongTailExitAuditConfig(minimum_trades_per_year=10, minimum_positive_quarters=2)
    rows = []
    for recipe in ("fixed_6h_diagnostic", "candidate"):
        for fold in ("WF_2024", "WF_2025"):
            for cost in (1.0, 2.0, 3.0):
                expectancy = 0.004 if recipe == "fixed_6h_diagnostic" else (0.003 if fold == "WF_2024" else -0.0001)
                rows.append(
                    {
                        "fold_id": fold,
                        "signal_quantile": 0.90,
                        "recipe": recipe,
                        "delay_minutes": 1,
                        "cost_multiplier": cost,
                        "trades": 100,
                        "mean_net_return": expectancy,
                        "profit_factor": 2.0,
                        "risk_sized_max_drawdown": -0.05,
                        "safety_cap_share": 0.0,
                        "win_rate": 0.90,
                    }
                )
            rows.append(
                {
                    "fold_id": fold,
                    "signal_quantile": 0.90,
                    "recipe": recipe,
                    "delay_minutes": 3,
                    "cost_multiplier": 1.0,
                    "trades": 100,
                    "mean_net_return": 0.001,
                    "profit_factor": 1.5,
                    "risk_sized_max_drawdown": -0.05,
                    "safety_cap_share": 0.0,
                    "win_rate": 0.90,
                }
            )
    periods = pd.DataFrame(
        [
            {"fold_id": fold, "signal_quantile": 0.90, "recipe": "candidate", "delay_minutes": 1, "period_kind": "quarter", "period": f"{fold[-4:]}Q{i}", "cost_multiplier": 2.0, "mean_net_return": 0.001}
            for fold in ("WF_2024", "WF_2025")
            for i in (1, 2, 3, 4)
        ]
    )
    concentration = pd.DataFrame(
        [
            {"fold_id": fold, "signal_quantile": 0.90, "recipe": "candidate", "delay_minutes": 1, "top10_profit_share_1x": 0.3, "mean_net_without_top10_1x": 0.001, "mean_net_without_top10_2x": 0.0005}
            for fold in ("WF_2024", "WF_2025")
        ]
    )
    stable = select_stable_candidates(pd.DataFrame(rows), periods, concentration, config)
    candidate = stable.loc[stable["recipe"] == "candidate"].iloc[0]
    assert not bool(candidate["passes_positive_expectancy_gate"])
    assert not bool(candidate["positive_1x_both_years"])


def test_configuration_formally_excludes_state_model() -> None:
    config = LongTailExitAuditConfig()
    payload = config.to_dict()
    text = str(payload).lower()
    assert "strategic_state" not in text
    assert "tactical_state" not in text
    assert "entry_state" not in text
