#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline C: established-trend pullback and rejoin for ETH.

Trend is EMA50/EMA200.  Long entry requires the previous close to be at/below
EMA20 and the current closed candle to reclaim EMA20 while remaining above
EMA200 with positive 12h momentum.  Short is symmetric.  The initial stop is
the tighter risk-valid choice between the recent 8-bar extreme and 2.5 ATR.
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

STRATEGY_NAME = "tf03_trend_pullback"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr14"] = atr(out, 14)
    out["ema20"] = out["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["mom_48"] = out["close"].pct_change(48)
    out["recent_low_8"] = out["low"].rolling(8, min_periods=8).min()
    out["recent_high_8"] = out["high"].rolling(8, min_periods=8).max()

    long_fire = (
        (out["ema50"] > out["ema200"])
        & (out["close"] > out["ema200"])
        & (out["mom_48"] > 0)
        & (out["close"].shift(1) <= out["ema20"].shift(1))
        & (out["close"] > out["ema20"])
    )
    short_fire = (
        (out["ema50"] < out["ema200"])
        & (out["close"] < out["ema200"])
        & (out["mom_48"] < 0)
        & (out["close"].shift(1) >= out["ema20"].shift(1))
        & (out["close"] < out["ema20"])
    )
    out["signal"] = np.select([long_fire, short_fire], [1, -1], default=0).astype("int8")
    # Keep the stop on the protective side while avoiding an unnecessarily wide
    # structural stop; the generic executor still rejects stops wider than 3%.
    long_stop = np.maximum(out["recent_low_8"], out["close"] - 2.5 * out["atr14"])
    short_stop = np.minimum(out["recent_high_8"], out["close"] + 2.5 * out["atr14"])
    out["stop"] = np.where(out["signal"] > 0, long_stop, np.where(out["signal"] < 0, short_stop, np.nan))
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=("open", "high", "low", "close", "volume", "atr14", "ema20", "ema50", "ema200", "mom_48", "recent_low_8", "recent_high_8", "signal", "stop"),
    trailing_atr_mult=2.75,
    trail_after_r=0.8,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
