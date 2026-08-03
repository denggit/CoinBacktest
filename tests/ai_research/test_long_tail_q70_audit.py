from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate
from src.ai_research.long_tail_q70_audit.analysis import (
    comparison_table,
    empirical_percentile,
    enforce_non_overlap,
    simulate_fixed_horizon_event,
    summarize,
)
from src.ai_research.long_tail_q70_audit.config import Q70CrossYearAuditConfig
from src.ai_research.long_tail_q70_audit.pipeline import _stable_candidate


def _path(periods: int = 1000) -> object:
    index = pd.date_range("2024-01-01", periods=periods, freq="1min")
    close = np.linspace(100.0, 104.0, periods)
    frame = pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05, "close": close},
        index=index,
    )
    return prepare_minute_path_frame(frame)


def test_empirical_percentile_uses_prior_reference_only() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([0.5, 2.0, 4.5])
    result = empirical_percentile(reference, values)
    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_fixed_horizon_uses_next_minute_open_and_complete_360_rows() -> None:
    path = _path()
    event = EventCandidate(
        event_id="q70_test",
        decision_time_ns=int(pd.Timestamp("2024-01-01 01:00").value),
        score=0.8,
        signal_quantile=0.70,
    )
    trade = simulate_fixed_horizon_event(
        event,
        fold_id="WF_2024",
        scope="broad_q70",
        delay_minutes=1,
        score_percentile=0.80,
        path=path,
        config=Q70CrossYearAuditConfig(),
    )
    assert trade is not None
    assert trade["entry_time"] == pd.Timestamp("2024-01-01 01:01")
    assert trade["holding_minutes"] == 360
    assert trade["score_band"] == "q70_to_q90"


def test_incomplete_minute_path_is_rejected() -> None:
    index = pd.date_range("2024-01-01", periods=700, freq="1min").delete(300)
    close = np.full(len(index), 100.0)
    path = prepare_minute_path_frame(pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=index))
    event = EventCandidate("event", int(pd.Timestamp("2024-01-01 01:00").value), 0.8, 0.70)
    trade = simulate_fixed_horizon_event(
        event,
        fold_id="WF_2024",
        scope="broad_q70",
        delay_minutes=1,
        score_percentile=0.8,
        path=path,
        config=Q70CrossYearAuditConfig(),
    )
    assert trade is None


def test_non_overlap_is_enforced_after_delay() -> None:
    frame = pd.DataFrame(
        [
            {"entry_time": pd.Timestamp("2024-01-01 00:01"), "exit_time": pd.Timestamp("2024-01-01 06:00"), "decision_time": pd.Timestamp("2024-01-01"), "score": 0.8},
            {"entry_time": pd.Timestamp("2024-01-01 06:00"), "exit_time": pd.Timestamp("2024-01-01 11:59"), "decision_time": pd.Timestamp("2024-01-01 05:59"), "score": 0.9},
            {"entry_time": pd.Timestamp("2024-01-01 06:01"), "exit_time": pd.Timestamp("2024-01-01 12:00"), "decision_time": pd.Timestamp("2024-01-01 06:00"), "score": 0.7},
        ]
    )
    kept, skipped = enforce_non_overlap(frame)
    assert len(kept) == 2
    assert skipped == 1


def test_summary_preserves_positive_expectancy_after_top10() -> None:
    rows = []
    for i in range(40):
        gross = 0.012 if i < 28 else -0.008
        rows.append({"entry_time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=7 * i), "gross_return": gross, "mfe": 0.02, "mae": -0.01})
    metrics = summarize(pd.DataFrame(rows), cost_multiplier=2.0, config=Q70CrossYearAuditConfig())
    assert metrics["mean_net_return"] > 0
    assert metrics["profit_factor"] > 1
    assert metrics["mean_net_without_top10"] > 0


def _summary_rows() -> pd.DataFrame:
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for scope, trades, total in (("broad_q70", 300, 2.0), ("primary_q90", 210, 1.2)):
            for delay in (1, 5):
                for cost in (2.0, 3.0):
                    rows.append(
                        {
                            "fold_id": fold, "scope": scope, "delay_minutes": delay, "cost_multiplier": cost,
                            "trades": trades, "mean_net_return": 0.004, "profit_factor": 1.8,
                            "total_compounded_return": total, "max_drawdown": -0.12,
                            "top10_profit_share": 0.40, "mean_net_without_top10": 0.002,
                        }
                    )
    return pd.DataFrame(rows)


def test_stable_q70_requires_incremental_band_and_both_years() -> None:
    summary = _summary_rows()
    comparison = comparison_table(summary)
    bands = pd.DataFrame(
        [
            {"fold_id": fold, "scope": "broad_q70", "delay_minutes": 1, "score_band": "q70_to_q90", "cost_multiplier": 2.0, "mean_net_return": 0.003, "profit_factor": 1.4}
            for fold in ("WF_2024", "WF_2025")
        ]
    )
    periods = pd.DataFrame(
        [
            {"fold_id": fold, "scope": "broad_q70", "delay_minutes": 1, "cost_multiplier": 2.0, "period_kind": "quarter", "period": f"{year}Q{q}", "mean_net_return": 0.002}
            for fold, year in (("WF_2024", 2024), ("WF_2025", 2025))
            for q in range(1, 5)
        ]
    )
    stable = _stable_candidate(summary, periods, bands, comparison, Q70CrossYearAuditConfig())
    assert bool(stable.iloc[0]["stable_q70_expansion"])


def test_config_does_not_depend_on_holding_or_state_models() -> None:
    text = str(Q70CrossYearAuditConfig().to_dict()).lower()
    for token in ("recoverable_drawdown", "post6_continuation", "strategic_state", "tactical_state"):
        assert token not in text
