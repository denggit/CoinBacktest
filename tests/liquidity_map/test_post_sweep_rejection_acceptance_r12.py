#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.research_common.post_sweep_acceptance import (
    PostSweepAcceptanceConfig,
    attach_checkpoint_outcomes,
    build_post_sweep_checkpoints,
    causal_audit,
    scorecard,
)
from src.research_common.structured_stop_pool import FAMILY_COLUMNS


def _bars(*, with_gap: bool = False) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=180, freq="1min")
    if with_gap:
        index = index.delete(31)
    close = np.full(len(index), 100.0, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    notional = np.full(len(index), 1_000_000.0)
    sell = np.full(len(index), 550_000.0)
    buy = notional - sell
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": notional / close, "notional": notional,
            "buy_notional": buy, "sell_notional": sell,
            "delta_notional": buy - sell, "trades_count": 100.0,
        },
        index=index,
    )


def _zone(bars: pd.DataFrame, *, pos: int = 30, floor: float = 99.0, ceiling: float = 100.0) -> pd.DataFrame:
    row = {
        "zone_event_id": "Z1",
        "event_kind": "swing_zone_sweep",
        "event_pos": pos,
        "event_bar_time": bars.index[pos],
        "event_available_time": bars.index[pos] + pd.Timedelta(minutes=1),
        "zone_latest_level_available_time": bars.index[pos] - pd.Timedelta(minutes=30),
        "zone_floor_price": floor,
        "zone_ceiling_price": ceiling,
        "zone_center_price": np.sqrt(floor * ceiling),
        "zone_timeframe_count": 2,
        "zone_max_timeframe_min": 60,
        "sweep_low": 98.5,
        "high_stop_release_label": True,
        "stop_release_score": 1.0,
    }
    for family in FAMILY_COLUMNS:
        row[family] = family.endswith("multitimeframe_confluence")
    return pd.DataFrame([row])


def _cfg(**kwargs) -> PostSweepAcceptanceConfig:
    values = {
        "checkpoints_minutes": (1, 3, 5, 10),
        "horizon_minutes": 60,
        "minimum_spec_events": 1,
        "minimum_period_events": 1,
        "minimum_promote_events": 1,
    }
    values.update(kwargs)
    return replace(PostSweepAcceptanceConfig(), **values).validate()


def test_rejection_and_next_open_are_causal() -> None:
    bars = _bars()
    bars.iloc[30, bars.columns.get_loc("low")] = 98.4
    bars.iloc[31, bars.columns.get_loc("close")] = 99.4
    bars.iloc[32, bars.columns.get_loc("close")] = 100.4
    bars.iloc[33, bars.columns.get_loc("close")] = 100.6
    bars.iloc[34, bars.columns.get_loc("close")] = 100.8
    checkpoints = build_post_sweep_checkpoints(
        _zone(bars), bars, _cfg(),
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    assert not checkpoints.empty
    assert set(checkpoints["state"]).intersection({"REJECT", "STRONG_REJECT"})
    assert (checkpoints["entry_pos"] == checkpoints["checkpoint_pos"] + 1).all()
    assert (pd.to_datetime(checkpoints["entry_time"]) == pd.to_datetime(checkpoints["checkpoint_available_time"])).all()


def test_persistent_acceptance_is_classified() -> None:
    bars = _bars()
    bars.iloc[30, bars.columns.get_loc("low")] = 98.4
    for pos, value in zip(range(31, 42), np.linspace(98.7, 96.5, 11)):
        bars.iloc[pos, bars.columns.get_loc("close")] = value
        bars.iloc[pos, bars.columns.get_loc("low")] = value - 0.2
        bars.iloc[pos, bars.columns.get_loc("high")] = value + 0.2
    checkpoints = build_post_sweep_checkpoints(
        _zone(bars), bars, _cfg(),
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    late = checkpoints.loc[checkpoints["checkpoint_minutes"].eq(10)].iloc[0]
    assert late["state"] == "PERSISTENT_ACCEPT"
    assert late["state_direction"] == "SHORT"


def test_gap_between_sweep_and_checkpoint_is_discarded() -> None:
    bars = _bars(with_gap=True)
    zones = _zone(bars, pos=30)
    checkpoints = build_post_sweep_checkpoints(
        zones, bars, _cfg(checkpoints_minutes=(1,)),
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    assert checkpoints.empty


def test_outcomes_use_natural_stop_and_conservative_same_bar() -> None:
    bars = _bars()
    bars.iloc[30, bars.columns.get_loc("low")] = 98.4
    bars.iloc[31, bars.columns.get_loc("close")] = 100.2
    checkpoints = build_post_sweep_checkpoints(
        _zone(bars), bars, _cfg(checkpoints_minutes=(1,)),
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    assert len(checkpoints) == 1
    entry_pos = int(checkpoints.iloc[0]["entry_pos"])
    # Force the next bar to touch both a tight natural stop and 1R target.
    entry = float(bars["open"].iloc[entry_pos])
    path_low = float(checkpoints.iloc[0]["path_low_visible"])
    stop = min(path_low * (1 - 5 / 10_000), entry * (1 - 5 / 10_000))
    risk = entry - stop
    bars.iloc[entry_pos, bars.columns.get_loc("low")] = stop - 0.01
    bars.iloc[entry_pos, bars.columns.get_loc("high")] = entry + risk + 0.01
    outcomes = attach_checkpoint_outcomes(checkpoints, bars, _cfg(checkpoints_minutes=(1,)), show_progress=False)
    long_row = outcomes.loc[outcomes["trade_direction"].eq("LONG")].iloc[0]
    assert long_row["r1p0_outcome"] == "STOP_CONSERVATIVE_SAME_BAR"
    assert long_row["natural_stop_distance_bp"] > 0


def test_causal_audit_passes_and_scorecard_is_explicit() -> None:
    bars = _bars()
    bars.iloc[30, bars.columns.get_loc("low")] = 98.4
    bars.iloc[31, bars.columns.get_loc("close")] = 100.2
    checkpoints = build_post_sweep_checkpoints(
        _zone(bars), bars, _cfg(checkpoints_minutes=(1,)),
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    outcomes = attach_checkpoint_outcomes(checkpoints, bars, _cfg(checkpoints_minutes=(1,)), show_progress=False)
    assert causal_audit(checkpoints, outcomes)["status"].eq("PASS").all()
    long_summary = pd.DataFrame(
        [{
            "quality_slice": "ALL", "trade_direction": "LONG", "checkpoint_minutes": 1,
            "state": "REJECT", "target_r": 1.0, "events": 10,
            "net_1x_mean_r": -0.1, "net_2x_mean_r": -0.2,
            "profit_factor_1x": 0.8,
        }]
    )
    periods = pd.DataFrame(
        [{
            "trade_direction": "LONG", "checkpoint_minutes": 1, "state": "REJECT",
            "target_r": 1.0, "period": "EARLY_2023_2024", "events": 10,
            "net_1x_mean_r": -0.1,
        }]
    )
    result = scorecard(long_summary, pd.DataFrame(), periods, _cfg())
    assert result.iloc[0]["decision"] == "rejected"


def test_all_report_builders_run_on_synthetic_output() -> None:
    from src.research_common.post_sweep_acceptance import (
        data_quality,
        design_table,
        direction_outcome_summary,
        family_timeframe_summary,
        period_stability,
        release_interaction,
        state_distribution,
        state_feature_profile,
        transition_matrix,
    )

    bars = _bars()
    bars.iloc[30, bars.columns.get_loc("low")] = 98.4
    bars.iloc[31, bars.columns.get_loc("close")] = 99.5
    bars.iloc[32, bars.columns.get_loc("close")] = 100.3
    bars.iloc[33, bars.columns.get_loc("close")] = 100.5
    cfg = _cfg()
    zones = _zone(bars)
    checkpoints = build_post_sweep_checkpoints(
        zones, bars, cfg,
        research_start=pd.Timestamp("2023-01-01"),
        research_end_exclusive=pd.Timestamp("2023-01-02"),
        show_progress=False,
    )
    outcomes = attach_checkpoint_outcomes(checkpoints, bars, cfg, show_progress=False)
    assert not design_table(cfg).empty
    assert data_quality(bars, zones, checkpoints, outcomes, cfg)["status"].ne("FAIL").all()
    assert not state_distribution(checkpoints).empty
    assert not state_feature_profile(checkpoints).empty
    long_summary = direction_outcome_summary(outcomes, cfg, "LONG")
    short_summary = direction_outcome_summary(outcomes, cfg, "SHORT")
    assert not long_summary.empty
    # Synthetic path may contain no preferred acceptance state; empty is valid.
    assert isinstance(short_summary, pd.DataFrame)
    assert not period_stability(outcomes, cfg).empty
    assert isinstance(transition_matrix(checkpoints), pd.DataFrame)
    assert not release_interaction(outcomes, cfg).empty
    assert not family_timeframe_summary(outcomes, cfg).empty
