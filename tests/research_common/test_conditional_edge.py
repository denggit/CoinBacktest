#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.conditional_edge import (
    ConditionalEdgeConfig,
    add_base_uplift,
    assign_time_splits,
    build_tail_specs,
    evaluate_base_universes,
    evaluate_specs,
    feature_monotonicity,
    final_qualification,
    fit_discovery_quantiles,
    freeze_discovery_candidates,
    pivot_split_results,
    prepare_conditional_features,
)


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-01", "2026-06-30 23:00:00", freq="6h")
    n = len(dates)
    side = rng.choice([-1, 1], n)
    activity = rng.normal(0.0, 1.0, n)
    gross = rng.normal(0.0010, 0.0030, n) + (activity >= 0.5) * 0.0012
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n),
            "event_cluster_id": np.arange(n),
            "side": side,
            "side_name": np.where(side > 0, "LONG", "SHORT"),
            "signal_bar_start": dates,
            "pressure_window_bars": np.ones(n, dtype=int),
            "flow_ratio": side * rng.uniform(0.0, 1.0, n),
            "trade_imbalance": side * rng.uniform(0.0, 1.0, n),
            "large_flow_ratio": side * rng.uniform(0.0, 1.0, n),
            "pressure_z": rng.uniform(2.0, 4.0, n),
            "activity_z": activity,
            "notional_ratio": np.exp(activity * 0.4),
            "flow_persistence": rng.uniform(0.0, 1.0, n),
            "large_notional_share": rng.uniform(0.0, 0.5, n),
            "large_trade_share": rng.uniform(0.0, 0.2, n),
            "flow_concentration": rng.uniform(0.0, 1.0, n),
            "avg_trade_notional_ratio": rng.uniform(0.5, 2.0, n),
            "max_trade_notional_ratio": rng.uniform(0.5, 3.0, n),
            "price_response_norm": rng.normal(0.0, 1.0, n),
            "pressure_effectiveness": rng.normal(0.0, 1.0, n),
            "impact_bps_per_million": rng.normal(0.0, 5.0, n),
            "direction_close_location": rng.uniform(0.0, 1.0, n),
            "continuation_gross_h30": gross,
            "continuation_net_h30": gross - 0.0015,
            "reversal_gross_h30": -gross,
            "reversal_net_h30": -gross - 0.0015,
        }
    )
    return assign_time_splits(prepare_conditional_features(frame), ConditionalEdgeConfig())


def test_side_alignment_is_symmetric() -> None:
    raw = _events().head(50)
    assert np.allclose(raw["flow_ratio_aligned"], raw["side"] * raw["flow_ratio"])
    assert (raw["flow_ratio_aligned"] >= 0.0).all()


def test_discovery_quantiles_ignore_holdout_perturbation() -> None:
    events = _events()
    first = fit_discovery_quantiles(events, features=["activity_z"], tail_quantiles=[0.8, 0.9])
    changed = events.copy()
    changed.loc[changed["research_split"].eq("holdout"), "activity_z"] = 1_000_000.0
    second = fit_discovery_quantiles(changed, features=["activity_z"], tail_quantiles=[0.8, 0.9])
    pd.testing.assert_frame_equal(first, second)


def test_univariate_freeze_does_not_read_holdout_outcomes() -> None:
    events = _events()
    thresholds = fit_discovery_quantiles(events, features=["activity_z"], tail_quantiles=[0.5, 0.67, 0.8])
    specs = build_tail_specs(
        thresholds,
        feature_polarities={"activity_z": ("high",)},
        tail_quantiles=[0.5, 0.67, 0.8],
        horizons=[30],
    )
    base = pivot_split_results(evaluate_base_universes(events, horizons=[30]))
    scan = add_base_uplift(
        pivot_split_results(evaluate_specs(events, specs, splits=("discovery",))),
        base,
    )
    frozen = freeze_discovery_candidates(
        scan,
        feature_monotonicity(scan),
        ConditionalEdgeConfig(
            minimum_discovery_events=100,
            minimum_year_events=20,
            target_monthly_events_low=1,
            target_monthly_events_high=500,
            minimum_active_date_ratio=0.01,
            discovery_fdr_alpha=1.0,
        ),
    )
    changed = events.copy()
    holdout = changed["research_split"].eq("holdout")
    changed.loc[holdout, "continuation_net_h30"] *= -100.0
    changed_scan = add_base_uplift(
        pivot_split_results(evaluate_specs(changed, specs, splits=("discovery",))),
        base,
    )
    changed_frozen = freeze_discovery_candidates(
        changed_scan,
        feature_monotonicity(changed_scan),
        ConditionalEdgeConfig(
            minimum_discovery_events=100,
            minimum_year_events=20,
            target_monthly_events_low=1,
            target_monthly_events_high=500,
            minimum_active_date_ratio=0.01,
            discovery_fdr_alpha=1.0,
        ),
    )
    assert frozen.loc[frozen["frozen_discovery_flag"], "spec_id"].tolist() == changed_frozen.loc[
        changed_frozen["frozen_discovery_flag"], "spec_id"
    ].tolist()


def test_final_gate_rejects_small_sample_even_when_profitable() -> None:
    row = {
        "spec_id": "small",
        "full_events": 999,
        "discovery_events": 500,
        "validation_events": 250,
        "holdout_events": 249,
        "full_min_year_events": 100,
        "discovery_net_mean": 0.001,
        "validation_net_mean": 0.001,
        "holdout_net_mean": 0.001,
        "full_net_mean": 0.001,
        "full_net_profit_factor": 2.0,
        "discovery_net_profit_factor": 1.5,
        "validation_net_profit_factor": 1.5,
        "holdout_net_profit_factor": 1.5,
        "full_positive_month_ratio": 0.8,
        "full_positive_years": 4,
        "full_active_date_ratio": 0.8,
        "full_events_per_month": 60.0,
        "full_top5_winner_share": 0.1,
    }
    result = final_qualification(pd.DataFrame([row]), ConditionalEdgeConfig())
    assert not bool(result.loc[0, "sample_gate"])
    assert not bool(result.loc[0, "qualified_edge_flag"])
