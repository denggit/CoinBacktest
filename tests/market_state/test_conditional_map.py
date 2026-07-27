from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.market_state.conditional_map import (
    ConditionDefinition,
    ConditionalMapConfig,
    attach_conditional_targets,
    build_condition_definitions,
    build_information_registry,
    build_ladder_incremental_summary,
    build_state_duration_summary,
    build_transition_matrix,
    evaluate_conditions,
)


def state_frame(rows: int = 900) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1min")
    trend = np.full(rows, "balanced", dtype=object)
    trend[150:450] = "up"
    trend[600:850] = "down"
    phase = np.full(rows, "balanced", dtype=object)
    phase[150:220] = "startup"
    phase[220:360] = "continuation"
    phase[360:420] = "mature"
    phase[420:450] = "decay"
    phase[600:670] = "startup"
    phase[670:780] = "continuation"
    phase[780:820] = "mature"
    phase[820:850] = "decay"
    quality = np.full(rows, "normal", dtype=object)
    quality[220:330] = "high_order"
    quality[360:450] = "noisy"
    volatility = np.full(rows, "normal", dtype=object)
    volatility[50:130] = "compression"
    volatility[300:380] = "expansion"
    volatility[700:730] = "shock"
    flow = np.full(rows, "balanced", dtype=object)
    flow[180:320] = "buy_persistent"
    flow[380:410] = "sell_pressure"
    flow[620:760] = "sell_persistent"
    impact = np.full(rows, "neutral", dtype=object)
    impact[200:300] = "buy_effective"
    impact[390:405] = "sell_absorbed"
    impact[650:740] = "sell_effective"
    impact[800:815] = "buy_absorbed"
    location = np.full(rows, "middle_zone", dtype=object)
    location[390:405] = "downside_sweep_reclaim"
    location[800:815] = "upside_sweep_reject"
    location[250:270] = "breakout_accept"
    location[700:720] = "breakdown_accept"
    context = np.full(rows, "wait", dtype=object)
    context[390:405] = "long_reversal_watch"
    context[800:815] = "short_reversal_watch"
    age = np.zeros(rows, dtype=int)
    for start, end in ((150, 450), (600, 850)):
        age[start:end] = np.arange(1, end - start + 1)
    flow_score = np.zeros(rows)
    flow_score[180:320] = 0.20
    flow_score[380:410] = -0.15
    flow_score[620:760] = -0.20
    flow_score[800:815] = 0.10
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "available_time": index + pd.Timedelta(minutes=1),
            "data_ready": True,
            "orderflow_available": True,
            "location_available": True,
            "trend_state": trend,
            "trend_phase": phase,
            "trend_quality_state": quality,
            "trend_state_age": age,
            "volatility_state": volatility,
            "activity_z": np.where(volatility == "shock", 2.5, np.where(volatility == "compression", -1.0, 0.0)),
            "flow_state": flow,
            "flow_score": flow_score,
            "impact_state": impact,
            "location_state": location,
            "trade_context_state": context,
            "signal_year": index.year,
            "signal_month": index.to_period("M").astype(str),
        },
        index=index,
    )


def path_frame(rows: int = 1200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="12h")
    condition = np.arange(rows) % 4 < 2
    holdout = index >= pd.Timestamp("2025-01-01")
    long_ret = np.where(condition, 0.0010, -0.0008) + np.where(holdout, 0.0001, 0.0)
    long_mfe = np.where(condition, 0.0020, 0.0010)
    long_mae = np.where(condition, -0.0008, -0.0015)
    frame = pd.DataFrame(
        {
            "available_time": index + pd.Timedelta(minutes=1),
            "signal_year": index.year,
            "volatility_state": np.where(np.arange(rows) % 3 == 0, "expansion", "normal"),
            "trend_state": "balanced",
            "long_return_h5": long_ret,
            "short_return_h5": -long_ret,
            "long_mfe_h5": long_mfe,
            "long_mae_h5": long_mae,
            "short_mfe_h5": -long_mae,
            "short_mae_h5": -long_mfe,
            "long_trap_h5": False,
            "short_trap_h5": False,
        },
        index=index,
    )
    return attach_conditional_targets(frame, (5,))


def test_condition_masks_are_append_invariant() -> None:
    full = state_frame(900)
    prefix = full.iloc[:700]
    cfg = ConditionalMapConfig(horizons_bars=(5,), sample_stride_bars=3, minimum_samples=1, minimum_holdout_samples=1)
    full_defs = {d.condition_name: d.mask.loc[prefix.index] for d in build_condition_definitions(full, cfg)}
    prefix_defs = {d.condition_name: d.mask for d in build_condition_definitions(prefix, cfg)}
    assert full_defs.keys() == prefix_defs.keys()
    for name in full_defs:
        pdt.assert_series_equal(prefix_defs[name], full_defs[name])


def test_role_aware_direction_and_range_evaluation() -> None:
    frame = path_frame()
    mask = pd.Series(np.arange(len(frame)) % 4 < 2, index=frame.index)
    range_mask = pd.Series(np.arange(len(frame)) % 3 == 0, index=frame.index)
    frame.loc[range_mask, "future_range_h5"] *= 1.5
    definitions = [
        ConditionDefinition(
            "known_long_bias",
            "test",
            "known directional effect",
            "direction_context",
            "directional_return",
            1,
            1,
            ("signal_year", "volatility_state"),
            mask,
        ),
        ConditionDefinition(
            "known_high_range",
            "test",
            "known future range effect",
            "future_range_high",
            "future_range",
            0,
            1,
            ("signal_year",),
            range_mask,
        ),
    ]
    cfg = ConditionalMapConfig(
        horizons_bars=(5,),
        minimum_samples=10,
        minimum_holdout_samples=10,
        minimum_years=1,
        holdout_start="2025-01-01",
        min_effect_size=0.0,
        min_direction_uplift=0.0,
        min_range_relative_uplift=0.0,
        minimum_supported_profiles=1,
        minimum_supported_horizons=1,
    )
    result = evaluate_conditions(frame, definitions, cfg, profile="base")
    summary = result.summary.set_index("condition_name")
    assert summary.loc["known_long_bias", "primary_uplift"] > 0.0
    assert summary.loc["known_long_bias", "primary_win_rate_uplift"] > 0.0
    assert summary.loc["known_high_range", "primary_relative_uplift"] > 0.0

    rows, registry = build_information_registry(result.summary, result.yearly, result.periods, cfg)
    assert rows["information_flag"].all()
    assert set(registry["evidence_status"]) == {"KEEP"}


def test_ladder_increment_reports_child_marginal_change() -> None:
    summary = pd.DataFrame(
        [
            {"profile": "base", "condition_name": "parent", "horizon_bars": 60, "samples": 1000, "primary_uplift": 0.0001, "primary_win_rate_uplift": 0.01, "ladder_name": "x", "ladder_stage": 1, "parent_condition": ""},
            {"profile": "base", "condition_name": "child", "horizon_bars": 60, "samples": 400, "primary_uplift": 0.0003, "primary_win_rate_uplift": 0.03, "ladder_name": "x", "ladder_stage": 2, "parent_condition": "parent"},
        ]
    )
    yearly = pd.DataFrame(
        [
            {"profile": "base", "condition_name": "parent", "horizon_bars": 60, "signal_year": 2024, "primary_uplift": 0.0001},
            {"profile": "base", "condition_name": "child", "horizon_bars": 60, "signal_year": 2024, "primary_uplift": 0.0003},
        ]
    )
    periods = pd.DataFrame(
        [
            {"profile": "base", "condition_name": "parent", "horizon_bars": 60, "period": "holdout", "primary_uplift": 0.0001},
            {"profile": "base", "condition_name": "child", "horizon_bars": 60, "period": "holdout", "primary_uplift": 0.00025},
        ]
    )
    out = build_ladder_incremental_summary(summary, yearly, periods)
    assert len(out) == 1
    assert np.isclose(out.iloc[0]["retention_ratio"], 0.4)
    assert out.iloc[0]["incremental_primary_uplift"] > 0.0
    assert out.iloc[0]["holdout_incremental_uplift"] > 0.0


def test_transition_and_duration_tables_are_well_formed() -> None:
    frame = state_frame(900)
    transition = build_transition_matrix(frame, profile="base", horizons_bars=(1, 5))
    duration = build_state_duration_summary(frame, profile="base")
    assert not transition.empty
    assert not duration.empty
    probs = transition.groupby(["profile", "axis", "horizon_bars", "current_state"])["transition_probability"].sum()
    assert np.allclose(probs.to_numpy(), 1.0)
    up = duration.loc[(duration["axis"] == "trend") & (duration["state"] == "up")]
    assert not up.empty
    assert int(up.iloc[0]["max_duration_bars"]) >= 300
