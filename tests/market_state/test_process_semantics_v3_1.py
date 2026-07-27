#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.process_map import ProcessMapConfig, ProcessMapEngine, stage_event_mask


def _strict_frame(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=rows, freq="1min")
    close = 2000.0 + np.linspace(0.0, 4.0, rows)
    frame = pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.30,
            "low": close - 0.30,
            "close": close,
            "available_time": idx + pd.Timedelta(minutes=1),
            "data_ready": True,
            "volatility_state": "normal",
            "flow_state": "balanced",
            "flow_score": 0.0,
            "flow_fast_score": 0.0,
            "flow_strength": 0.5,
            "flow_acceleration": 0.0,
            "flow_intensity_z": 0.0,
            "price_move_score": 0.0,
            "flow_price_effectiveness": 0.0,
            "impact_state": "neutral",
            "sell_absorption_score": 0.0,
            "buy_absorption_score": 0.0,
            "location_state": "middle_zone",
            "structural_location_score": 0.0,
            "local_support": 1998.0,
            "local_resistance": 2002.0,
            "atr_pct": 0.0005,
        },
        index=idx,
    )
    return frame


def test_v31_recovery_requires_new_flow_price_and_effectiveness_confirmation() -> None:
    frame = _strict_frame()
    idx = frame.index
    frame.loc[idx[20], ["flow_state", "flow_score", "flow_fast_score", "impact_state"]] = [
        "sell_persistent", -0.35, -0.30, "sell_effective"
    ]
    frame.loc[idx[24], ["impact_state", "sell_absorption_score"]] = ["sell_absorbed", 0.80]
    frame.loc[idx[28], "location_state"] = "downside_sweep_reclaim"
    sweep_close = float(frame.loc[idx[28], "close"])

    # A weak natural bounce after the sweep must not complete the process.
    frame.loc[idx[30], [
        "flow_state", "flow_score", "flow_fast_score", "flow_acceleration",
        "flow_price_effectiveness", "price_move_score", "close",
    ]] = ["buy_pressure", 0.03, 0.04, 0.02, 0.02, 0.05, sweep_close + 0.20]

    # Later, genuinely new reverse flow, positive price response and reclaim do.
    frame.loc[idx[31], ["flow_score", "flow_fast_score"]] = [-0.01, 0.02]
    frame.loc[idx[32], [
        "flow_state", "flow_score", "flow_fast_score", "flow_acceleration",
        "flow_price_effectiveness", "price_move_score", "close",
    ]] = ["buy_building", 0.12, 0.16, 0.10, 0.22, 0.25, sweep_close + 0.50]

    result = ProcessMapEngine(ProcessMapConfig(minimum_probability_samples=2)).compute(frame)
    assert not bool(stage_event_mask(result.frame, "long_reversal", 4).iloc[30])
    assert bool(stage_event_mask(result.frame, "long_reversal", 4).iloc[32])
    completed = result.episodes.loc[
        result.episodes["family"].eq("long_reversal") & result.episodes["completed"].eq(True)
    ]
    assert len(completed) == 1
    assert completed.iloc[0]["stage_4_available_time"] == idx[32] + pd.Timedelta(minutes=1)


def test_v31_compression_does_not_almost_automatically_advance() -> None:
    frame = _strict_frame(90)
    idx = frame.index
    frame.loc[idx[10:18], "volatility_state"] = "compression"
    # Post-compression buy flow without expansion, intensity and level break is
    # intentionally insufficient.
    frame.loc[idx[18], [
        "flow_state", "flow_score", "flow_fast_score", "flow_intensity_z",
        "flow_price_effectiveness", "price_move_score", "impact_state",
    ]] = ["buy_pressure", 0.07, 0.08, 0.10, 0.10, 0.10, "buy_effective"]

    cfg = ProcessMapConfig(
        breakout_compression_min_bars=8,
        breakout_compression_to_impulse_bars=6,
        minimum_probability_samples=2,
    )
    result = ProcessMapEngine(cfg).compute(frame)
    episode = result.episodes.loc[result.episodes["family"].eq("long_breakout")].iloc[0]
    assert int(episode["max_stage_reached"]) == 1
    assert episode["status"] == "expired"


def test_v31_breakout_requires_fresh_impulse_then_later_retest_or_hold() -> None:
    frame = _strict_frame(100)
    idx = frame.index
    frame.loc[idx[20:28], "volatility_state"] = "compression"
    # Fresh expansion and level-breaking impulse after mature compression.
    frame.loc[idx[28], [
        "volatility_state", "flow_state", "flow_score", "flow_fast_score",
        "flow_acceleration", "flow_intensity_z", "price_move_score",
        "flow_price_effectiveness", "impact_state", "local_resistance",
        "close", "high", "low",
    ]] = [
        "expansion", "buy_building", 0.14, 0.18,
        0.10, 1.20, 0.35,
        0.35, "buy_effective", 2002.0,
        2002.30, 2002.50, 2001.90,
    ]
    # A later retest holds the impulse anchor; completion cannot occur on the
    # impulse bar or the immediately following bar.
    frame.loc[idx[29], ["close", "high", "low", "flow_score", "flow_price_effectiveness"]] = [
        2002.15, 2002.35, 2002.02, 0.05, 0.05
    ]
    frame.loc[idx[30], ["close", "high", "low", "flow_score", "flow_price_effectiveness"]] = [
        2002.20, 2002.40, 2001.98, 0.04, 0.04
    ]

    cfg = ProcessMapConfig(
        breakout_compression_min_bars=8,
        breakout_accept_min_delay_bars=2,
        breakout_accept_hold_bars=2,
        minimum_probability_samples=2,
    )
    result = ProcessMapEngine(cfg).compute(frame)
    assert bool(stage_event_mask(result.frame, "long_breakout", 1).iloc[27])
    assert bool(stage_event_mask(result.frame, "long_breakout", 2).iloc[28])
    assert not bool(stage_event_mask(result.frame, "long_breakout", 3).iloc[29])
    assert bool(stage_event_mask(result.frame, "long_breakout", 3).iloc[30])
    completed = result.episodes.loc[
        result.episodes["family"].eq("long_breakout") & result.episodes["completed"].eq(True)
    ]
    assert len(completed) == 1
