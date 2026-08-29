#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline E: volatility compression -> expansion breakout trend following.

A setup requires prior ATR compression, then the current closed bar must expand
in true range and close through the *previous* 6h range.  Volume must also be
above its previous 12h mean.  Entry is the next 15m open.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backtest.mf.trend_following.common import StrategySpec, atr, atr_stop, make_parser, run_spec, true_range

STRATEGY_NAME = "tf05_volatility_expansion"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tr"] = true_range(out)
    out["atr14"] = atr(out, 14)
    out["atr96"] = atr(out, 96)
    out["atr_ratio"] = out["atr14"] / out["atr96"].replace(0, np.nan)
    out["pre_range_high_24"] = out["high"].shift(1).rolling(24, min_periods=24).max()
    out["pre_range_low_24"] = out["low"].shift(1).rolling(24, min_periods=24).min()
    out["prev_volume_mean_48"] = out["volume"].shift(1).rolling(48, min_periods=24).mean()
    out["prev_atr14"] = out["atr14"].shift(1)
    squeeze = out["atr_ratio"].shift(1) < 0.75
    expansion = out["tr"] > 1.5 * out["prev_atr14"]
    volume_ok = out["volume"] > out["prev_volume_mean_48"]
    long_fire = squeeze & expansion & volume_ok & (out["close"] > out["pre_range_high_24"])
    short_fire = squeeze & expansion & volume_ok & (out["close"] < out["pre_range_low_24"])
    out["squeeze"] = squeeze
    out["expansion"] = expansion
    out["signal"] = np.select([long_fire, short_fire], [1, -1], default=0).astype("int8")
    out["stop"] = atr_stop(out["close"], out["atr14"], out["signal"], mult=2.25)
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=(
        "open", "high", "low", "close", "volume", "tr", "atr14", "atr96", "atr_ratio",
        "pre_range_high_24", "pre_range_low_24", "prev_volume_mean_48", "squeeze", "expansion", "signal", "stop",
    ),
    trailing_atr_mult=3.25,
    trail_after_r=1.0,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
