from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg_v3 import _find_latest_fvg_in_leg


def test_fvg_may_precede_mss_break_bar() -> None:
    tz = "America/New_York"
    # Bullish FVG is completed at position 2: low[2] > high[0].  Signal/MSS is
    # later at position 4. R04 must still accept that FVG as part of the same
    # reversal displacement leg; no same-bar coupling is allowed.
    highs = np.array([100.0, 101.0, 103.0, 104.0, 106.0])
    lows = np.array([98.0, 99.0, 101.0, 100.5, 102.5])
    available = pd.date_range("2026-06-02 09:01", periods=5, freq="1min", tz=tz).asi8
    pos = _find_latest_fvg_in_leg(
        is_long=True,
        highs=highs,
        lows=lows,
        available_ns=available,
        terminal_time=pd.Timestamp("2026-06-02 09:01", tz=tz),
        signal_pos=4,
    )
    assert pos == 2
    assert pos != 4


def test_fvg_first_candle_may_contain_terminal_extreme() -> None:
    tz = "America/New_York"
    highs = np.array([100.0, 101.0, 104.0])
    lows = np.array([98.0, 99.0, 102.0])
    available = pd.date_range("2026-06-02 09:01", periods=3, freq="1min", tz=tz).asi8
    pos = _find_latest_fvg_in_leg(
        is_long=True,
        highs=highs,
        lows=lows,
        available_ns=available,
        terminal_time=pd.Timestamp("2026-06-02 09:01", tz=tz),
        signal_pos=2,
    )
    assert pos == 2

from src.research_common.ict.premarket_mss_fvg_v3 import (
    _select_inbound_anchor,
    _select_mss_reference,
)


def test_post_terminal_short_term_high_can_be_bullish_mss_reference() -> None:
    """Low raid -> final low -> new STH -> break STH is a valid MSS path."""
    tz = "America/New_York"
    pivots = pd.DataFrame(
        [
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 08:56", tz=tz),
                "pivot_price": 105.0,
                "confirmation_available_time": pd.Timestamp("2026-06-02 08:58", tz=tz),
            },
            {
                # This is the small STH that forms only AFTER the final low.
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 09:06", tz=tz),
                "pivot_price": 101.5,
                "confirmation_available_time": pd.Timestamp("2026-06-02 09:08", tz=tz),
            },
        ]
    )
    ref = _select_mss_reference(
        pivots,
        side="high",
        sweep_bar_start=pd.Timestamp("2026-06-02 09:00", tz=tz),
        terminal_available_time=pd.Timestamp("2026-06-02 09:03", tz=tz),
        signal_available_time=pd.Timestamp("2026-06-02 09:10", tz=tz),
    )
    assert ref is not None
    assert pd.Timestamp(ref["pivot_time"]) == pd.Timestamp("2026-06-02 09:06", tz=tz)
    assert float(ref["pivot_price"]) == 101.5
    assert ref["reference_relation"] == "post_terminal_dynamic"


def test_post_terminal_mss_reference_does_not_corrupt_inbound_displacement_anchor() -> None:
    tz = "America/New_York"
    pivots = pd.DataFrame(
        [
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 08:56", tz=tz),
                "pivot_price": 105.0,
                "confirmation_available_time": pd.Timestamp("2026-06-02 08:58", tz=tz),
            },
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 09:06", tz=tz),
                "pivot_price": 101.5,
                "confirmation_available_time": pd.Timestamp("2026-06-02 09:08", tz=tz),
            },
        ]
    )
    anchor = _select_inbound_anchor(
        pivots,
        side="high",
        terminal_available_time=pd.Timestamp("2026-06-02 09:03", tz=tz),
        signal_available_time=pd.Timestamp("2026-06-02 09:10", tz=tz),
        fallback_time=pd.Timestamp("2026-06-02 09:00", tz=tz),
        fallback_price=100.0,
    )
    assert pd.Timestamp(anchor["anchor_time"]) == pd.Timestamp("2026-06-02 08:56", tz=tz)
    assert float(anchor["anchor_price"]) == 105.0
    assert anchor["anchor_source"] == "pre_terminal_opposing_pivot"
