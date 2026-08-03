#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.structured_stop_pool import (
    FAMILY_COLUMNS,
    StructuredStopPoolConfig,
    attach_first_touch_outcomes,
    attach_stop_release_labels,
    attach_zone_hypotheses,
    build_level_structure_features,
    calibrate_release_score,
    causal_audit,
)


def _expand_htf(rows: list[tuple[float, float, float, float]], minutes: int = 15) -> pd.DataFrame:
    start = pd.Timestamp("2023-01-01")
    records = []
    index = []
    for i, (o, h, l, c) in enumerate(rows):
        for j in range(minutes):
            index.append(start + pd.Timedelta(minutes=i * minutes + j))
            records.append(
                {
                    "open": o if j == 0 else (o + c) / 2,
                    "high": h,
                    "low": l,
                    "close": c if j == minutes - 1 else (o + c) / 2,
                    "notional": 1_000_000.0,
                    "buy_notional": 500_000.0,
                    "sell_notional": 500_000.0,
                    "delta_notional": 0.0,
                    "trades_count": 100.0,
                    "large_sell_notional": 10_000.0,
                    "large_sell_trades_count": 1.0,
                    "max_trade_notional": 20_000.0,
                }
            )
    return pd.DataFrame(records, index=pd.DatetimeIndex(index))


def test_h1_h2_h8_are_causal_structural_families() -> None:
    htf = [
        (112, 116, 108, 113),
        (113, 117, 109, 114),
        (114, 118, 110, 115),
        (115, 119, 111, 116),
        (116, 120, 112, 117),
        (110, 115, 105, 110),
        (108, 112, 100, 105),  # older low
        (106, 120, 104, 116),  # reference high
        (115, 117, 90, 95),    # lower low
        (96, 125, 94, 105),    # failed-breakdown recovery + BOS
        (105, 112, 95, 100),   # current higher low
        (101, 115, 99, 110),
        (110, 116, 103, 108),
    ]
    bars = _expand_htf(htf)
    delta = pd.Timedelta(minutes=15)
    levels = pd.DataFrame(
        {
            "level_id": [1, 2, 3],
            "source_timeframe": ["15m"] * 3,
            "source_timeframe_min": [15] * 3,
            "pivot_pos_htf": [6, 8, 10],
            "pivot_time": [bars.index[90], bars.index[120], bars.index[150]],
            "level_price": [100.0, 90.0, 95.0],
            "initial_available_time": [bars.index[90] + 2 * delta, bars.index[120] + 2 * delta, bars.index[150] + 2 * delta],
            "confirmation_reaction_high_bp": [120.0, 300.0, 200.0],
        }
    )
    cfg = StructuredStopPoolConfig(timeframes=(("15m", 15),), atr_window_htf=5).validate()
    features, thresholds = build_level_structure_features(levels, bars, cfg)
    current = features.loc[features["level_id"].eq(3)].iloc[0]
    assert bool(current[FAMILY_COLUMNS[0]])
    assert bool(current[FAMILY_COLUMNS[1]])
    assert bool(current[FAMILY_COLUMNS[7]])
    assert pd.Timestamp(current["structure_available_time"]) == pd.Timestamp(current["initial_available_time"])
    assert not thresholds.empty


def test_zone_confluence_is_attached_without_changing_other_families() -> None:
    levels = pd.DataFrame(
        {
            "level_id": [1, 2],
            "structure_available_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "source_timeframe": ["15m", "1H"],
            **{family: [family == FAMILY_COLUMNS[0], False] for family in FAMILY_COLUMNS},
        }
    )
    zones = pd.DataFrame(
        {
            "zone_event_id": ["Z"],
            "zone_member_level_ids": ["1|2"],
            "zone_member_count": [2],
            "zone_timeframe_count": [2],
            "zone_formation_span_minutes": [180.0],
            "zone_max_timeframe_min": [60],
            "event_available_time": pd.to_datetime(["2023-01-03"]),
        }
    )
    out = attach_zone_hypotheses(zones, levels)
    assert bool(out.loc[0, FAMILY_COLUMNS[0]])
    assert bool(out.loc[0, FAMILY_COLUMNS[5]])
    assert bool(out.loc[0, "independent_multitimeframe_confluence"])


def test_release_score_uses_early_controls_and_freezes_forward() -> None:
    index = pd.date_range("2023-01-01", periods=500, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "notional": 1_000.0,
            "buy_notional": 500.0,
            "sell_notional": 500.0,
            "delta_notional": 0.0,
            "trades_count": 10.0,
            "large_sell_notional": 10.0,
            "large_sell_trades_count": 1.0,
            "max_trade_notional": 20.0,
        },
        index=index,
    )
    positions = np.arange(100, 360)
    bars.iloc[positions[::7], bars.columns.get_loc("sell_notional")] = 2_000.0
    bars.iloc[positions[::7], bars.columns.get_loc("notional")] = 2_500.0
    events = pd.DataFrame(
        {
            "zone_event_id": [f"E{i}" for i in range(len(positions))],
            "event_kind": ["non_zone_downside_control"] * len(positions),
            "event_pos": positions,
            "event_available_time": index[positions] + pd.Timedelta(minutes=1),
        }
    )
    cfg = StructuredStopPoolConfig(release_baseline_minutes=60, release_long_baseline_minutes=120).validate()
    attached = attach_stop_release_labels(events, bars, cfg)
    scored, calibration = calibrate_release_score(attached, cfg)
    assert scored["stop_release_score"].notna().sum() > 100
    assert calibration.loc[calibration["component"].eq("STOP_RELEASE_SCORE_THRESHOLD"), "calibration_source"].iloc[0] == "EARLY_MATCHED_CONTROLS"


def test_first_touch_same_bar_is_conservative_stop() -> None:
    index = pd.date_range("2023-01-01", periods=10, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=index)
    bars.loc[index[2], "high"] = 100.30
    bars.loc[index[2], "low"] = 99.70
    events = pd.DataFrame({"event_pos": [1], "event_available_time": [index[2]]})
    out = attach_first_touch_outcomes(events, bars, StructuredStopPoolConfig())
    assert out.loc[0, "tp15_sl15_outcome"] == "SL_CONSERVATIVE_SAME_BAR"
    assert out.loc[0, "tp15_sl15_net_return_1x_cost"] < 0


def test_causal_audit_rejects_late_structure() -> None:
    level = pd.DataFrame(
        {
            "structure_available_time": pd.to_datetime(["2023-01-02"]),
            "initial_available_time": pd.to_datetime(["2023-01-01"]),
        }
    )
    zone = pd.DataFrame(
        {
            "zone_member_structure_available_time_max": pd.to_datetime(["2023-01-01"]),
            "event_available_time": pd.to_datetime(["2023-01-02"]),
        }
    )
    outcome = pd.DataFrame(
        {
            "event_available_time": pd.to_datetime(["2023-01-02"]),
            "r09_entry_time": pd.to_datetime(["2023-01-02"]),
            "event_kind": ["swing_zone_sweep"],
            **{family: [False] for family in FAMILY_COLUMNS},
        }
    )
    audit = causal_audit(level, zone, outcome)
    row = audit.loc[audit["check"].eq("level_structure_available_after_level_available")].iloc[0]
    assert row["status"] == "FAIL"
