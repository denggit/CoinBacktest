from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai_research.long_tail_state_gate_diagnostic.analysis import (
    _ohlc_resample,
    build_attribution_findings,
    classify_combined_state,
    classify_trend,
    counterfactual_gate_summary,
    monthly_market_vs_c2,
    summarize_c2_by_state,
    summarize_fixed6h_by_state,
)
from src.ai_research.long_tail_state_gate_diagnostic.config import StateGateDiagnosticConfig


def test_config_freezes_stage_contract() -> None:
    config = StateGateDiagnosticConfig()
    config.validate()
    assert config.anchor_delay_minutes == 1
    assert config.anchor_cost_multiplier == 2.0
    assert config.q70_quantile == 0.70
    with pytest.raises(ValueError):
        StateGateDiagnosticConfig(q70_quantile=0.75).validate()


def test_completed_bar_availability_is_shifted() -> None:
    index = pd.date_range("2026-01-01", periods=8 * 60, freq="1min")
    values = np.arange(len(index), dtype=float) + 100.0
    minute = pd.DataFrame(
        {"open": values, "high": values + 1, "low": values - 1, "close": values + 0.5},
        index=index,
    )
    bars = _ohlc_resample(minute, "4h", pd.Timedelta(hours=4), "tf4h")
    assert bars.index[0] == pd.Timestamp("2026-01-01 04:00:00")
    assert bars.iloc[0]["tf4h_source_start"] == pd.Timestamp("2026-01-01 00:00:00")


def test_trend_and_combined_state_rules_are_deterministic() -> None:
    index = pd.RangeIndex(3)
    trend = classify_trend(
        pd.Series([0.1, -0.1, 0.1], index=index),
        pd.Series([0.1, -0.1, -0.1], index=index),
        pd.Series([0.1, -0.1, 0.1], index=index),
    )
    assert trend.tolist() == ["UP", "DOWN", "MIXED"]
    combined = classify_combined_state(
        pd.Series(["UP", "MIXED", "DOWN", "MIXED", "UP"]),
        pd.Series(["UP", "UP", "DOWN", "DOWN", "MIXED"]),
    )
    assert combined.tolist() == [
        "BULL_ALIGNED",
        "BULL_TACTICAL",
        "BEAR_ALIGNED",
        "BEAR_TACTICAL",
        "MIXED",
    ]


def _cycle_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "analysis_period": ["2024"] * 5 + ["2025"] * 5,
            "cycle_return": [0.02, -0.01, 0.03, -0.02, 0.01] * 2,
            "combined_state": ["BULL_ALIGNED", "BEAR_ALIGNED", "BULL_TACTICAL", "MIXED", "BEAR_TACTICAL"] * 2,
            "trend_1d": ["UP", "DOWN", "MIXED", "MIXED", "DOWN"] * 2,
            "trend_4h": ["UP", "DOWN", "UP", "MIXED", "DOWN"] * 2,
            "above_1d_ema50": [True, False, True, True, False] * 2,
        }
    )


def test_counterfactual_gate_keeps_all_baseline_and_discloses_coverage() -> None:
    out = counterfactual_gate_summary(_cycle_frame())
    baseline = out.loc[(out["analysis_period"] == "2024") & (out["gate"] == "G0_ALL")].iloc[0]
    assert baseline["accepted_trades"] == 5
    assert baseline["coverage"] == pytest.approx(1.0)
    filtered = out.loc[(out["analysis_period"] == "2024") & (out["gate"] == "G1_EXCLUDE_BEAR_ALIGNED")].iloc[0]
    assert filtered["accepted_trades"] == 4
    assert "future untouched validation" in filtered["interpretation"]


def test_fixed6h_summary_applies_frozen_two_x_cost() -> None:
    trades = pd.DataFrame(
        {
            "analysis_period": ["2026_H1", "2026_H1"],
            "combined_state": ["BULL_ALIGNED", "BULL_ALIGNED"],
            "gross_return": [0.01, -0.002],
            "mfe": [0.02, 0.01],
            "mae": [-0.005, -0.01],
        }
    )
    out = summarize_fixed6h_by_state(trades, StateGateDiagnosticConfig())
    row = out.iloc[0]
    expected = np.mean([0.01 - 0.0026, -0.002 - 0.0026])
    assert row["mean_return"] == pytest.approx(expected)


def test_attribution_can_identify_regime_and_score_drift() -> None:
    c2_summary = pd.DataFrame(
        [
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BEAR_ALIGNED", "trades": 10, "mean_return": -0.01},
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BULL_ALIGNED", "trades": 10, "mean_return": 0.01},
        ]
    )
    fixed = pd.DataFrame(
        [{"analysis_period": "2026_JULY", "trades": 10, "mean_return": -0.001}]
    )
    score = pd.DataFrame(
        [
            {"analysis_period": "CAL_Q4_2025", "combined_state": "BULL_ALIGNED", "q70_exceedance_rate": 0.30},
            {"analysis_period": "2026_H1", "combined_state": "BULL_ALIGNED", "q70_exceedance_rate": 0.55},
        ]
    )
    gates = pd.DataFrame(
        [
            {"analysis_period": "2026_JULY", "gate": "G0_ALL", "total_return": 0.05, "coverage": 1.0},
        ]
    )
    findings, decision, _ = build_attribution_findings(c2_summary, fixed, score, gates)
    assert decision == "DIAGNOSIS_REGIME_DEPENDENCE_AND_SCORE_DRIFT"
    assert findings.loc[findings["finding"] == "h1_regime_separation", "supported"].iloc[0]


def test_invalid_drawdown_bands_are_rejected() -> None:
    with pytest.raises(ValueError):
        StateGateDiagnosticConfig(near_90d_high_drawdown=-0.30, deep_90d_drawdown=-0.20).validate()


def test_state_timeline_uses_only_completed_context() -> None:
    from src.ai_research.long_tail_state_gate_diagnostic.analysis import build_state_timeline

    index = pd.date_range("2023-01-01", "2024-03-31 23:59", freq="1min")
    trend = np.linspace(100.0, 160.0, len(index))
    minute = pd.DataFrame(
        {"open": trend, "high": trend * 1.001, "low": trend * 0.999, "close": trend},
        index=index,
    )
    config = StateGateDiagnosticConfig(
        analysis_start="2024-03-01 00:00:00",
        analysis_end="2024-03-02 23:59:59",
    )
    state = build_state_timeline(minute, config)
    assert not state.empty
    assert state["context_available_time_flag"].all()
    assert (pd.to_datetime(state["tf4h_available_time"]) <= pd.to_datetime(state["decision_time"])).all()
    assert (pd.to_datetime(state["tf1d_available_time"]) <= pd.to_datetime(state["decision_time"])).all()


def test_score_summary_includes_h1_aggregate() -> None:
    from src.ai_research.long_tail_state_gate_diagnostic.analysis import summarize_score_state

    scores = pd.DataFrame(
        {
            "analysis_period": ["2026_Q1", "2026_Q2", "2026_JULY"],
            "combined_state": ["MIXED", "MIXED", "MIXED"],
            "score": [0.1, 0.2, 0.3],
        }
    )
    out = summarize_score_state(scores, 0.15)
    h1 = out.loc[out["analysis_period"] == "2026_H1"]
    assert len(h1) == 1
    assert h1.iloc[0]["decision_rows"] == 2
    assert h1.iloc[0]["q70_exceedance_rate"] == pytest.approx(0.5)


def test_align_state_normalizes_mixed_datetime_resolutions() -> None:
    from src.ai_research.long_tail_state_gate_diagnostic.analysis import align_state

    frame = pd.DataFrame(
        {
            "decision_time": pd.Series(
                np.array(["2026-01-01T00:15:00", "2026-01-01T00:30:00"], dtype="datetime64[us]")
            ),
            "value": [1, 2],
        }
    )
    state = pd.DataFrame(
        {
            "decision_time": pd.Series(
                np.array(["2026-01-01T00:00:00", "2026-01-01T00:30:00"], dtype="datetime64[ns]")
            ),
            "combined_state": ["MIXED", "BULL_ALIGNED"],
        }
    )
    out = align_state(frame, state)
    assert out["combined_state"].tolist() == ["MIXED", "BULL_ALIGNED"]
    assert str(out["decision_time"].dtype) == "datetime64[ns]"


def test_build_state_timeline_accepts_microsecond_index() -> None:
    from src.ai_research.long_tail_state_gate_diagnostic.analysis import build_state_timeline

    index = pd.date_range("2023-01-01", "2024-03-02 23:59", freq="1min").astype("datetime64[us]")
    trend = np.linspace(100.0, 160.0, len(index))
    minute = pd.DataFrame(
        {"open": trend, "high": trend * 1.001, "low": trend * 0.999, "close": trend},
        index=index,
    )
    config = StateGateDiagnosticConfig(
        analysis_start="2024-03-01 00:00:00",
        analysis_end="2024-03-02 23:59:59",
    )
    state = build_state_timeline(minute, config)
    assert not state.empty
    assert str(state["decision_time"].dtype) == "datetime64[ns]"


def test_monthly_market_summary_accepts_named_datetime_index() -> None:
    index = pd.date_range("2026-01-01", "2026-02-28 23:59", freq="1min", name="timestamp")
    close = np.linspace(100.0, 120.0, len(index))
    minute = pd.DataFrame({"close": close}, index=index)
    cycles = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-01-10", "2026-02-10"]),
            "cycle_return": [0.01, -0.005],
        }
    )
    scores = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-01-10", "2026-02-10"]),
            "score": [0.2, 0.1],
        }
    )
    state = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-01-10", "2026-02-10"]),
            "combined_state": ["BULL_ALIGNED", "MIXED"],
        }
    )

    exact_monthly = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "c2_account_return": [0.02, -0.01],
        }
    )
    out = monthly_market_vs_c2(
        minute, cycles, scores, state, threshold=0.15, account_monthly_returns=exact_monthly
    )

    assert "month_end" in out.columns
    assert out["month"].tolist() == ["2026-01", "2026-02"]
    assert out["dominant_state"].tolist() == ["BULL_ALIGNED", "MIXED"]
    assert out["c2_account_return"].tolist() == pytest.approx([0.02, -0.01])
    assert out["c2_entry_cohort_return"].tolist() == pytest.approx([0.01, -0.005])


def test_c2_keeps_trade_mae_distinct_from_account_drawdown() -> None:
    cycles = pd.DataFrame(
        {
            "analysis_period": ["2026_H1", "2026_H1"],
            "combined_state": ["MIXED", "MIXED"],
            "trend_1d": ["MIXED", "MIXED"],
            "trend_4h": ["MIXED", "MIXED"],
            "drawdown_state": ["CORRECTION", "CORRECTION"],
            "vol_state": ["VOL_NORMAL", "VOL_NORMAL"],
            "cycle_return": [0.01, -0.005],
            "hard_stop_exit": [False, True],
            "soft_failure_exit": [False, False],
            "full_mae": [np.nan, np.nan],
            "cycle_max_drawdown": [-0.01, -0.02],
        }
    )
    out = summarize_c2_by_state(cycles)
    row = out.loc[
        (out["analysis_period"] == "2026_H1")
        & (out["state_dimension"] == "combined_state")
        & (out["state_value"] == "MIXED")
    ].iloc[0]
    assert pd.isna(row["mean_cycle_mae"])
    assert row["mean_cycle_account_drawdown"] == pytest.approx(-0.015)


def test_attribution_detail_matches_actual_nonseparation() -> None:
    c2_summary = pd.DataFrame(
        [
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BEAR_ALIGNED", "trades": 10, "mean_return": 0.001},
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BULL_ALIGNED", "trades": 5, "mean_return": -0.005},
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BULL_TACTICAL", "trades": 5, "mean_return": 0.001},
        ]
    )
    fixed = pd.DataFrame(
        [{"analysis_period": "2026_JULY", "trades": 10, "mean_return": -0.001}]
    )
    score = pd.DataFrame(
        [
            {"analysis_period": "CAL_Q4_2025", "combined_state": "MIXED", "q70_exceedance_rate": 0.30},
            {"analysis_period": "2026_H1", "combined_state": "MIXED", "q70_exceedance_rate": 0.60},
        ]
    )
    gates = pd.DataFrame(
        [
            {"analysis_period": "2026_JULY", "gate": "G0_ALL", "total_return": 0.05, "coverage": 1.0},
        ]
    )
    findings, decision, _ = build_attribution_findings(c2_summary, fixed, score, gates)
    row = findings.loc[findings["finding"] == "h1_regime_separation"].iloc[0]
    assert not bool(row["supported"])
    assert "BEAR_ALIGNED mean=0.100%" in row["detail"]
    assert decision == "DIAGNOSIS_SCORE_DRIFT_DOMINANT"


def test_positive_gate_is_not_mislabeled_as_uplift() -> None:
    periods = ["2024", "2025", "2026_H1", "2026_JULY"]
    gate_rows = []
    for period, base, filtered in zip(periods, [0.8, 1.0, 0.05, 0.09], [0.5, 0.4, 0.002, 0.091]):
        gate_rows.append({"analysis_period": period, "gate": "G0_ALL", "total_return": base, "coverage": 1.0})
        gate_rows.append({"analysis_period": period, "gate": "G1_EXCLUDE_BEAR_ALIGNED", "total_return": filtered, "coverage": 0.7})
    c2_summary = pd.DataFrame(
        [
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BEAR_ALIGNED", "trades": 10, "mean_return": 0.001},
            {"analysis_period": "2026_H1", "state_dimension": "combined_state", "state_value": "BULL_ALIGNED", "trades": 10, "mean_return": -0.001},
        ]
    )
    fixed = pd.DataFrame([{"analysis_period": "2026_JULY", "trades": 1, "mean_return": -0.001}])
    score = pd.DataFrame(
        [
            {"analysis_period": "CAL_Q4_2025", "combined_state": "MIXED", "q70_exceedance_rate": 0.30},
            {"analysis_period": "2026_H1", "combined_state": "MIXED", "q70_exceedance_rate": 0.60},
        ]
    )
    findings, _, _ = build_attribution_findings(c2_summary, fixed, score, pd.DataFrame(gate_rows))
    positive = findings.loc[findings["finding"] == "simple_gate_positive_all_periods"].iloc[0]
    uplift = findings.loc[findings["finding"] == "simple_gate_uplift_all_periods"].iloc[0]
    assert bool(positive["supported"])
    assert not bool(uplift["supported"])
