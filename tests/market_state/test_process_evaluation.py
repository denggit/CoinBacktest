#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.process_evaluation import (
    ProcessEvaluationConfig,
    build_episode_outcomes,
    build_probability_calibration,
    build_process_registry,
    build_stage_information,
    build_stage_progression,
)
from src.market_state.process_map import ProcessMapConfig, ProcessMapEngine
from src.market_state.validity_audit import ValidityAuditConfig, build_forward_path_frame


def _state_frame(rows: int = 1600) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="1min")
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.00015, rows)
    close = 2000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0003,
            "low": np.minimum(open_, close) * 0.9997,
            "close": close,
            "available_time": idx + pd.Timedelta(minutes=1),
            "data_ready": True,
            "volatility_state": "normal",
            "flow_state": "balanced",
            "flow_score": 0.0,
            "flow_strength": 0.5,
            "impact_state": "neutral",
            "sell_absorption_score": 0.0,
            "buy_absorption_score": 0.0,
            "location_state": "middle_zone",
            "structural_location_score": 0.0,
            "atr_pct": 0.0005,
        },
        index=idx,
    )
    # Repeated complete long-reversal episodes, followed by a mild positive path.
    for base in range(80, 1400, 80):
        frame.loc[idx[base], ["flow_state", "flow_score", "impact_state"]] = ["sell_persistent", -0.4, "sell_effective"]
        frame.loc[idx[base + 3], ["impact_state", "sell_absorption_score"]] = ["sell_absorbed", 0.8]
        frame.loc[idx[base + 6], "location_state"] = "downside_sweep_reclaim"
        frame.loc[idx[base + 9], ["flow_state", "flow_score"]] = ["buy_building", 0.3]
        frame.loc[idx[base + 10: base + 25], "close"] *= 1.001
    return frame


def test_v3_evaluation_tables_are_role_aware() -> None:
    state = _state_frame()
    process_cfg = ProcessMapConfig(
        semantic_version="v3",
        probability_horizons_bars=(5, 15, 60),
        default_reversal_horizon_bars=15,
        default_breakout_horizon_bars=60,
        minimum_probability_samples=3,
    )
    result = ProcessMapEngine(process_cfg).compute(state)
    path = build_forward_path_frame(
        result.frame,
        ValidityAuditConfig(horizons_bars=(5, 15, 60), trap_horizon_bars=60, minimum_events=1),
    )
    eval_cfg = ProcessEvaluationConfig(
        horizons_bars=(5, 15, 60),
        holdout_start="2023-01-02",
        minimum_stage_samples=5,
        minimum_holdout_samples=2,
        minimum_years=1,
        minimum_profiles=1,
    )
    summary, yearly, periods, raw = build_stage_information(
        result.frame,
        result.stage_events,
        path,
        profile="base",
        config=eval_cfg,
    )
    progression = build_stage_progression(result.episodes, profile="base", process_config=process_cfg)
    outcomes = build_episode_outcomes(
        result.episodes,
        result.frame,
        path,
        profile="base",
        config=eval_cfg,
        process_config=process_cfg,
    )
    calibration, bins = build_probability_calibration(outcomes)
    registry = build_process_registry(
        summary,
        yearly,
        periods,
        progression,
        config=eval_cfg,
        process_config=process_cfg,
    )

    assert not summary.empty
    assert {"mean_return_uplift", "mean_win_rate_uplift"}.issubset(summary.columns)
    assert not progression.empty
    assert (progression["progression_rate"].dropna().between(0.0, 1.0)).all()
    assert not outcomes.empty
    assert {"probability", "actual_success"}.issubset(outcomes.columns)
    assert not calibration.empty
    assert not bins.empty
    assert set(registry["status"]).issubset(
        {"KEEP_PROCESS_CANDIDATE", "KEEP_STAGE_ONLY", "REVISE_PROCESS", "DROP_PROCESS"}
    )
