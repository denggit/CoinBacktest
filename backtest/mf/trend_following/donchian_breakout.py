#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline A: Donchian / price-breakout trend following for ETH.

Signal: the just-closed 15m candle closes beyond the *previous* 24h high/low.
Entry: next 15m open.  Initial stop: 2 ATR.  Exit: ATR trail, opposite signal,
or maximum holding window.  No fixed nearby TP.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.mf.trend_following.common import StrategySpec, atr, atr_stop, make_parser, run_spec

STRATEGY_NAME = "tf01_donchian_breakout"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr14"] = atr(out, 14)
    out["donchian_high_96"] = out["high"].shift(1).rolling(96, min_periods=96).max()
    out["donchian_low_96"] = out["low"].shift(1).rolling(96, min_periods=96).min()
    long_fire = (out["close"] > out["donchian_high_96"]) & (out["close"].shift(1) <= out["donchian_high_96"].shift(1))
    short_fire = (out["close"] < out["donchian_low_96"]) & (out["close"].shift(1) >= out["donchian_low_96"].shift(1))
    out["signal"] = np.select([long_fire, short_fire], [1, -1], default=0).astype("int8")
    out["stop"] = atr_stop(out["close"], out["atr14"], out["signal"], mult=2.0)
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=("open", "high", "low", "close", "volume", "atr14", "donchian_high_96", "donchian_low_96", "signal", "stop"),
    trailing_atr_mult=3.0,
    trail_after_r=1.0,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
