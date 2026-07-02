#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd

from src.research_common.event_study import first_touch_outcome, normalize_side


def test_normalize_side_accepts_common_encodings() -> None:
    s = pd.Series(["LONG", "short", "BUY", "sell", 1, -2, 0, "unknown"])

    out = normalize_side(s)

    assert out.tolist() == [1, -1, 1, -1, 1, -1, 0, 0]


def test_first_touch_conservative_same_bar_policy_marks_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="1min")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 102.0, 100.0],
            "low": [100.0, 98.0, 100.0],
            "close": [100.0, 101.0, 100.0],
        },
        index=idx,
    )
    side = pd.Series([1, 0, 0], index=idx)

    out = first_touch_outcome(bars, side, target_pct=0.01, stop_pct=0.01, horizon=1, same_bar_policy="conservative")

    assert out.iloc[0]["touch_result"] == "STOP"
    assert bool(out.iloc[0]["same_bar_both_hit_flag"]) is True
