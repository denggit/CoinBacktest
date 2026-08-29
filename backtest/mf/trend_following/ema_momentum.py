#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline B: EMA trend + time-series momentum for ETH.

Long regime: EMA50 > EMA200, close > EMA50 and trailing 12h return > +1%.
Short is symmetric. A trade is fired only when the regime becomes active.
All features use the just-closed bar; execution is next-bar open.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.mf.trend_following.common import StrategySpec, atr, atr_stop, make_parser, run_spec, state_entry_signal

STRATEGY_NAME = "tf02_ema_momentum"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr14"] = atr(out, 14)
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["mom_48"] = out["close"].pct_change(48)
    long_state = (out["ema50"] > out["ema200"]) & (out["close"] > out["ema50"]) & (out["mom_48"] > 0.01)
    short_state = (out["ema50"] < out["ema200"]) & (out["close"] < out["ema50"]) & (out["mom_48"] < -0.01)
    out["signal"] = state_entry_signal(long_state, short_state)
    out["stop"] = atr_stop(out["close"], out["atr14"], out["signal"], mult=2.0)
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=("open", "high", "low", "close", "volume", "atr14", "ema50", "ema200", "mom_48", "signal", "stop"),
    trailing_atr_mult=3.0,
    trail_after_r=1.0,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
