#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical sleeve definitions frozen by R02."""

from __future__ import annotations

from .contracts import SleeveId, SleeveSpec


SHORT_HORIZON_SPEC = SleeveSpec(
    sleeve_id="short_horizon",
    display_name="Short-horizon microstructure",
    decision_cadence="5s",
    intended_hold_min_minutes=5,
    intended_hold_max_minutes=60,
    target_move_min=0.003,
    target_move_max=0.008,
    direction_timeframes=("1m", "5m", "15m"),
    entry_timeframes=("1s", "1m", "5m"),
    exit_style="micro_dynamic_with_time_cap",
    max_hold_is_safety_only=False,
)

INTRADAY_TREND_SPEC = SleeveSpec(
    sleeve_id="intraday_trend",
    display_name="Intraday trend",
    decision_cadence="5m",
    intended_hold_min_minutes=60,
    intended_hold_max_minutes=720,
    target_move_min=0.010,
    target_move_max=0.025,
    direction_timeframes=("4H", "1H", "30m"),
    entry_timeframes=("30m", "15m", "5m", "1m"),
    exit_style="structure_state_trailing_with_time_cap",
    max_hold_is_safety_only=True,
)

SWING_SPEC = SleeveSpec(
    sleeve_id="swing",
    display_name="3%-5% target-centric swing entry",
    decision_cadence="15m",
    intended_hold_min_minutes=0,
    intended_hold_max_minutes=7_200,
    target_move_min=0.030,
    target_move_max=0.050,
    direction_timeframes=("1D", "4H", "1H"),
    entry_timeframes=("30m", "15m", "5m", "1m"),
    exit_style="structure_state_trailing_with_time_cap",
    max_hold_is_safety_only=True,
)

SLEEVE_SPECS: dict[SleeveId, SleeveSpec] = {
    spec.sleeve_id: spec
    for spec in (SHORT_HORIZON_SPEC, INTRADAY_TREND_SPEC, SWING_SPEC)
}


def get_sleeve_spec(sleeve_id: SleeveId) -> SleeveSpec:
    return SLEEVE_SPECS[sleeve_id]
