#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline F: price trend confirmed by OKX trade-bar order flow.

Uses trade-derived delta_notional / taker_buy_ratio / notional.  Long state
requires positive price trend, positive rolling aggressive-flow imbalance and
buyers controlling >52% of recent taker notional; short is symmetric.  This is
not a single-candle order-flow trigger: the confirmation is accumulated across
multiple 15m bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.mf.trend_following.common import StrategySpec, atr, atr_stop, make_parser, run_spec, state_entry_signal

STRATEGY_NAME = "tf06_orderflow_trend"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = ("delta_notional", "taker_buy_ratio", "notional")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "order-flow trend backtest requires OKX trade-bar fields "
            f"{required}; missing={missing}. Use OKXTradeBarLoader data."
        )
    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["atr14"] = atr(out, 14)
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["mom_48"] = out["close"].pct_change(48)
    out["delta_sum_16"] = out["delta_notional"].rolling(16, min_periods=12).sum()
    out["buy_ratio_mean_8"] = out["taker_buy_ratio"].rolling(8, min_periods=6).mean()
    out["prev_notional_mean_48"] = out["notional"].shift(1).rolling(48, min_periods=24).mean()
    activity_ok = out["notional"] > out["prev_notional_mean_48"]

    long_state = (
        (out["ema50"] > out["ema200"])
        & (out["close"] > out["ema50"])
        & (out["mom_48"] > 0)
        & (out["delta_sum_16"] > 0)
        & (out["buy_ratio_mean_8"] > 0.52)
        & activity_ok
    )
    short_state = (
        (out["ema50"] < out["ema200"])
        & (out["close"] < out["ema50"])
        & (out["mom_48"] < 0)
        & (out["delta_sum_16"] < 0)
        & (out["buy_ratio_mean_8"] < 0.48)
        & activity_ok
    )
    out["activity_ok"] = activity_ok
    out["signal"] = state_entry_signal(long_state, short_state)
    out["stop"] = atr_stop(out["close"], out["atr14"], out["signal"], mult=2.0)
    return out


SPEC = StrategySpec(
    strategy_name=STRATEGY_NAME,
    build_features=build_features,
    audit_cols=(
        "open", "high", "low", "close", "volume", "notional", "delta_notional", "taker_buy_ratio",
        "atr14", "ema50", "ema200", "mom_48", "delta_sum_16", "buy_ratio_mean_8",
        "prev_notional_mean_48", "activity_ok", "signal", "stop",
    ),
    trailing_atr_mult=3.0,
    trail_after_r=1.0,
)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = make_parser(__doc__ or STRATEGY_NAME, STRATEGY_NAME).parse_args(argv)
    return run_spec(args, SPEC)


if __name__ == "__main__":
    main()
