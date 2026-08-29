#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline D: causal swing market-structure trend following for ETH.

A swing uses 3 bars left + 3 bars right, but is made available only after the
three right-side bars have closed.  Bull trend requires the two latest confirmed
swing highs and lows to both be rising; bear trend is symmetric.  Entry occurs
when a closed candle breaks the latest confirmed swing in trend direction.
Stop is the latest confirmed opposite swing.  No future pivot information is
used at signal time.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.mf.trend_following.common import StrategySpec, atr, make_parser, run_spec

STRATEGY_NAME = "tf04_market_structure"
SWING_ORDER = 3


def _confirmed_swings(df: pd.DataFrame, order: int = SWING_ORDER) -> pd.DataFrame:
    """Return causal confirmed swing prices on each current/available row.

    At row t, the candidate pivot is t-order. The test window ends at t, hence
    all evidence is already closed/known. The returned ``new_*`` columns become
    non-null only at confirmation time, never at the historical pivot timestamp.
    """
    win = 2 * order + 1
    candidate_high = df["high"].shift(order)
    candidate_low = df["low"].shift(order)
    window_high = df["high"].rolling(win, min_periods=win).max()
    window_low = df["low"].rolling(win, min_periods=win).min()
    new_high = candidate_high.where(candidate_high >= window_high)
    new_low = candidate_low.where(candidate_low <= window_low)
    return pd.DataFrame({"new_swing_high": new_high, "new_swing_low": new_low}, index=df.index)


def _last_two_confirmed(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    last = series.ffill()
    # On a row with a newly confirmed swing, previous is the last confirmed
    # swing strictly before this row.  Then ffill that previous value forward.
    previous_at_event = last.shift(1).where(series.notna())
    previous = previous_at_event.ffill()
    return last, previous


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr14"] = atr(out, 14)
    swings = _confirmed_swings(out, SWING_ORDER)
    out = pd.concat([out, swings], axis=1)
    out["last_swing_high"], out["prev_swing_high"] = _last_two_confirmed(out["new_swing_high"])
    out["last_swing_low"], out["prev_swing_low"] = _last_two_confirmed(out["new_swing_low"])

    bull_structure = (out["last_swing_high"] > out["prev_swing_high"]) & (out["last_swing_low"] > out["prev_swing_low"])
    bear_structure = (out["last_swing_high"] < out["prev_swing_high"]) & (out["last_swing_low"] < out["prev_swing_low"])
    long_fire = bull_structure & (out["close"] > out["last_swing_high"]) & (out["close"].shift(1) <= out["last_swing_high"].shift(1))
    short_fire = bear_structure & (out["close"] < out["last_swing_low"]) & (out["close"].shift(1) >= out["last_swing_low"].shift(1))
    out["bull_structure"] = bull_structure
    out["bear_structure"] = bear_structure
    out["signal"] = np.select([long_fire, short_fire], [1, -1], default=0).astype("int8")
    out["stop"] = np.where(out["signal"] > 0, out["last_swing_low"], np.where(out["signal"] < 0, out["last_swing_high"], np.nan))
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=(
        "open", "high", "low", "close", "volume", "atr14",
        "new_swing_high", "new_swing_low", "last_swing_high", "prev_swing_high",
        "last_swing_low", "prev_swing_low", "bull_structure", "bear_structure", "signal", "stop",
    ),
    trailing_atr_mult=3.0,
    trail_after_r=1.0,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
