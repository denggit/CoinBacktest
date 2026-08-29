#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r19 import (
    R19Config,
    build_positioning_rebuild_events,
    build_positioning_rebuild_paths,
    r19_causal_audit,
    summarize_r19_paths,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2023-01-01 00:00:00", periods=1_700, freq="1min")
    close = np.empty(len(index), dtype=float)
    close[:65] = np.linspace(100.0, 110.0, 65)
    close[65:70] = np.linspace(109.95, 109.50, 5)
    close[70:75] = np.linspace(109.55, 109.80, 5)
    close[75:80] = np.linspace(109.85, 110.20, 5)
    close[80:281] = np.linspace(110.25, 123.0, 201)
    close[281:] = 123.0
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.02,
            "low": np.minimum(open_, close) - 0.02,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _oi() -> pd.DataFrame:
    timestamp = pd.date_range("2023-01-01 00:05:00", periods=16, freq="5min")
    base = pd.Series([100.0 + i for i in range(13)] + [111.5, 111.0, 111.2])
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_time": timestamp + pd.Timedelta("1min"),
            "sum_open_interest": base,
            "sum_open_interest_value": base * 110.0,
            "oi_base_change_5m": base.pct_change(),
            "oi_base_change_1h": base / base.shift(12) - 1.0,
            "oi_baseline_age_seconds_5m": 300.0,
            "oi_baseline_age_seconds_1h": 3600.0,
        }
    )


def test_build_release_first_rebuild_continuation_is_executable() -> None:
    events, seal, engineering = build_positioning_rebuild_events(_bars(), _oi())
    assert len(events) == 1
    event = events.iloc[0]
    assert event["direction"] == "Long"
    assert event["setup_status"] == "executable"
    assert event["build_price_return_1h"] > 0
    assert event["build_oi_base_change_1h"] > 0
    assert event["release_oi_base_change_5m"] < 0
    assert event["rebuild_oi_base_change_5m"] >= 0
    assert event["rebuild_close"] > event["release_bar_high"]
    assert 0 < event["release_duration_minutes"] <= R19Config().rebuild_window_minutes
    assert event["entry_time"] == pd.Timestamp(event["signal_available_time"]).ceil("min")
    assert int(seal.loc[seal["check"].eq("holdout_outcome_rows_computed"), "value"].iloc[0]) == 0
    assert int(engineering.loc[engineering["check"].eq("successful_rebuild_breaks"), "value"].iloc[0]) == 1


def test_first_rebuild_without_break_cannot_be_rescued() -> None:
    bars = _bars()
    bars.loc[bars.index[75:80], ["open", "high", "low", "close"]] -= 1.0
    events, _, engineering = build_positioning_rebuild_events(bars, _oi())
    assert events.empty
    assert int(engineering.loc[engineering["check"].eq("successful_rebuild_breaks"), "value"].iloc[0]) == 0


def test_gap_inside_release_episode_invalidates_setup() -> None:
    oi = _oi().drop(index=14).reset_index(drop=True)
    events, _, _ = build_positioning_rebuild_events(_bars(), oi)
    assert events.empty


def test_right_edge_censor_does_not_decrement_prior_time_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aligned = pd.DataFrame(
        {
            "available_time": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 01:05", "2023-01-01 01:06", "2023-01-01 01:10"]
            ),
            "current_oi_valid": True,
            "build_oi_valid": True,
            "metric_gap_valid": True,
            "price_step_valid": True,
            "oi_base_change_5m": [-0.1, -0.1, -0.1, -0.1],
            "build_oi_base_change_5m": [0.1, -0.1, 0.1, -0.1],
            "build_oi_base_change_1h": 0.1,
            "build_price_return_1h": 0.1,
        }
    )
    monkeypatch.setattr(
        "src.research_common.ict_mss2.r19.prepare_positioning_alignment",
        lambda *_args, **_kwargs: aligned,
    )
    events, _, engineering = build_positioning_rebuild_events(_bars(), _oi())
    counts = engineering.set_index("check")["value"].astype(int)
    assert events.empty
    assert counts["release_episodes"] == 2
    assert counts["expired_after_60m"] == 1
    assert counts["right_edge_censored"] == 1


def test_nonfinite_build_price_cannot_admit_release(monkeypatch: pytest.MonkeyPatch) -> None:
    aligned = pd.DataFrame(
        {
            "available_time": pd.to_datetime(["2023-01-01 00:00"]),
            "current_oi_valid": True,
            "build_oi_valid": True,
            "metric_gap_valid": True,
            "price_step_valid": True,
            "oi_base_change_5m": [-0.1],
            "build_oi_base_change_5m": [0.1],
            "build_oi_base_change_1h": [0.1],
            "build_price_return_1h": [np.nan],
        }
    )
    monkeypatch.setattr(
        "src.research_common.ict_mss2.r19.prepare_positioning_alignment",
        lambda *_args, **_kwargs: aligned,
    )
    events, _, engineering = build_positioning_rebuild_events(_bars(), _oi())
    counts = engineering.set_index("check")["value"].astype(int)
    assert events.empty
    assert counts["release_episodes"] == 0


def test_future_mutation_does_not_change_event() -> None:
    events, _, _ = build_positioning_rebuild_events(_bars(), _oi())
    bars = _bars()
    cutoff = pd.Timestamp(events.iloc[0]["signal_available_time"]) + pd.Timedelta(minutes=10)
    bars.loc[bars.index >= cutoff, ["open", "high", "low", "close"]] *= 0.7
    replay, _, _ = build_positioning_rebuild_events(bars, _oi())
    columns = ["direction", "release_oi_metric_time", "rebuild_oi_metric_time", "signal_available_time", "entry_time", "entry_price", "stop_price", "setup_status"]
    pd.testing.assert_frame_equal(events[columns], replay[columns])


def test_paths_costs_and_causal_audit() -> None:
    events, _, _ = build_positioning_rebuild_events(_bars(), _oi())
    paths = build_positioning_rebuild_paths(_bars(), events)
    assert len(paths) == 4
    assert "H0_1H_VOLATILITY_RANGE" in set(paths["target_model"])
    assert np.allclose(paths["net_return_cost2x"], paths["gross_return"] - 0.0022)
    score = summarize_r19_paths(paths)
    assert len(score) == 4
    audit = r19_causal_audit(events, paths)
    assert int(audit["violations"].sum()) == 0
