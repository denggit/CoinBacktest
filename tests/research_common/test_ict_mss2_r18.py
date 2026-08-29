#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r18 import (
    R18Config,
    build_positioning_unwind_events,
    build_positioning_unwind_paths,
    r18_causal_audit,
    summarize_r18_paths,
)


def _long_bars() -> pd.DataFrame:
    index = pd.date_range("2023-01-01 00:00:00", periods=1_700, freq="1min")
    close = np.empty(len(index), dtype=float)
    close[:65] = np.linspace(120.0, 110.0, 65)
    close[65:70] = np.linspace(110.05, 110.95, 5)
    close[70:171] = np.linspace(111.0, 122.0, 101)
    close[171:] = 122.0
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.03,
            "low": np.minimum(open_, close) - 0.03,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _oi() -> pd.DataFrame:
    timestamp = pd.date_range("2023-01-01 00:05:00", periods=14, freq="5min")
    base = pd.Series([100.0 + i for i in range(13)] + [111.5])
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_time": timestamp + pd.Timedelta("1min"),
            "sum_open_interest": base,
            "sum_open_interest_value": base * 120.0,
            "oi_base_change_5m": base.pct_change(),
            "oi_base_change_1h": base / base.shift(12) - 1.0,
            "oi_baseline_age_seconds_5m": 300.0,
            "oi_baseline_age_seconds_1h": 3600.0,
        }
    )


def test_frozen_long_transition_is_causal_and_executable() -> None:
    events, seal, engineering = build_positioning_unwind_events(_long_bars(), _oi())
    assert len(events) == 1
    event = events.iloc[0]
    assert event["direction"] == "Long"
    assert event["setup_status"] == "executable"
    assert event["build_price_return_1h"] < 0
    assert event["build_oi_base_change_1h"] > 0
    assert event["prior_oi_base_change_5m"] >= 0
    assert event["release_oi_base_change_5m"] < 0
    assert event["stabilization_close"] > event["stabilization_prior_high"]
    assert event["entry_time"] == pd.Timestamp(event["signal_available_time"]).ceil("min")
    assert float(event["risk_distance_pct"]) <= R18Config().max_stop_distance_pct
    assert int(seal.loc[seal["check"].eq("holdout_outcome_rows_computed"), "value"].iloc[0]) == 0
    assert int(engineering.loc[engineering["check"].eq("visible_long_candidates"), "value"].iloc[0]) == 1


def test_future_price_mutation_cannot_change_admission() -> None:
    bars = _long_bars()
    original, _, _ = build_positioning_unwind_events(bars, _oi())
    mutated = bars.copy()
    cutoff = pd.Timestamp(original.iloc[0]["signal_available_time"]) + pd.Timedelta(minutes=10)
    mutated.loc[mutated.index >= cutoff, ["open", "high", "low", "close"]] *= 1.5
    replay, _, _ = build_positioning_unwind_events(mutated, _oi())
    columns = [
        "direction",
        "build_oi_metric_time",
        "release_oi_metric_time",
        "signal_available_time",
        "entry_time",
        "entry_price",
        "stop_price",
        "structural_target_price",
        "setup_status",
    ]
    pd.testing.assert_frame_equal(original[columns], replay[columns])


def test_future_or_oracle_columns_are_physically_rejected() -> None:
    oi = _oi()
    oi["future_oi_change_1h"] = 0.1
    with pytest.raises(RuntimeError, match="future leakage"):
        build_positioning_unwind_events(_long_bars(), oi)


def test_nonpositive_oi_and_long_gap_are_not_interpolated() -> None:
    zero = _oi()
    zero.loc[12, "sum_open_interest"] = 0.0
    events, _, _ = build_positioning_unwind_events(_long_bars(), zero)
    assert events.empty

    gap = _oi().drop(index=12).reset_index(drop=True)
    events, _, _ = build_positioning_unwind_events(_long_bars(), gap)
    assert events.empty


def test_first_passage_is_stop_first_on_same_bar() -> None:
    index = pd.date_range("2023-01-01 00:00:00", periods=3, freq="1min")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [102.0, 100.0, 100.0],
            "low": [98.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": 1.0,
        },
        index=index,
    )
    events = pd.DataFrame(
        [
            {
                "setup_id": "same_bar",
                "direction": "Long",
                "trade_direction": 1,
                "research_split": "discovery",
                "setup_status": "executable",
                "build_oi_available_time": index[0] - pd.Timedelta(minutes=5),
                "release_oi_available_time": index[0],
                "signal_available_time": index[0],
                "entry_time": index[0],
                "entry_price": 100.0,
                "stop_price": 99.0,
                "risk_distance_pct": 0.01,
                "structural_target_price": 101.0,
                "structural_runway_pct": 0.01,
                "structural_reward_risk": 1.0,
                "build_price_return_1h": -0.01,
                "build_oi_base_change_1h": 0.01,
                "release_oi_base_change_5m": -0.01,
            }
        ]
    )
    paths = build_positioning_unwind_paths(bars, events, config=R18Config(path_horizon_minutes=1))
    assert len(paths) == 4
    assert paths["outcome"].eq("sl_first").all()


def test_paths_cost_summary_and_causal_audit() -> None:
    events, _, _ = build_positioning_unwind_events(_long_bars(), _oi())
    paths = build_positioning_unwind_paths(_long_bars(), events)
    assert len(paths) == 4
    assert np.allclose(paths["net_return_cost2x"], paths["gross_return"] - 0.0022)
    score = summarize_r18_paths(paths)
    assert set(score["direction"]) == {"Long"}
    audit = r18_causal_audit(events, paths)
    assert int(audit["violations"].sum()) == 0
