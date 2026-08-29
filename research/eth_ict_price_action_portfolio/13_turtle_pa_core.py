#!/usr/bin/env python
"""Causal standard Donchian/Turtle price-action mechanisms for ETH."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _stable_portfolio_bridge as stable
from research.eth_ict_price_action_portfolio._literature_trend_bridge import daily_bos_positions


RESULTS = Path(__file__).resolve().parent / "ict_pa_v3" / "results"


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous).abs(), (frame["low"] - previous).abs()], axis=1
    ).max(axis=1)


def turtle_state(frame: pd.DataFrame, entry: int, exit_: int) -> pd.Series:
    upper = frame["high"].shift(1).rolling(entry, min_periods=entry).max()
    lower = frame["low"].shift(1).rolling(entry, min_periods=entry).min()
    exit_low = frame["low"].shift(1).rolling(exit_, min_periods=exit_).min()
    exit_high = frame["high"].shift(1).rolling(exit_, min_periods=exit_).max()
    state = np.zeros(len(frame), dtype=float)
    current = 0.0
    for i in range(len(frame)):
        close = float(frame["close"].iloc[i])
        if current == 0.0:
            if np.isfinite(upper.iloc[i]) and close > float(upper.iloc[i]):
                current = 1.0
            elif np.isfinite(lower.iloc[i]) and close < float(lower.iloc[i]):
                current = -1.0
        elif current > 0.0:
            if np.isfinite(lower.iloc[i]) and close < float(lower.iloc[i]):
                current = -1.0
            elif np.isfinite(exit_low.iloc[i]) and close < float(exit_low.iloc[i]):
                current = 0.0
        else:
            if np.isfinite(upper.iloc[i]) and close > float(upper.iloc[i]):
                current = 1.0
            elif np.isfinite(exit_high.iloc[i]) and close > float(exit_high.iloc[i]):
                current = 0.0
        state[i] = current
    return pd.Series(state, index=frame.index)


def timeframe_position(bars: pd.DataFrame, frequency: str, entry: int, exit_: int, cap: float) -> pd.Series:
    frame = bars.resample(frequency, label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna()
    state = turtle_state(frame, entry, exit_)
    atr = true_range(frame).shift(1).rolling(20, min_periods=20).mean()
    # One quarter-percent account risk per 2 ATR stop.  This is deliberately
    # below the original Turtle risk and is identical across definitions.
    size = (0.0025 / (2.0 * atr / frame["close"])).clip(upper=cap).fillna(0.0)
    desired = state * size
    delta = pd.Timedelta(frequency)
    available = pd.Series(desired.to_numpy(), index=frame.index + delta + pd.Timedelta(minutes=15))
    return available.reindex(bars.index, method="ffill").fillna(0.0)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bars, _ = stable.load_inputs()
    daily20 = timeframe_position(bars, "1D", 20, 10, 0.50)
    daily55 = timeframe_position(bars, "1D", 55, 20, 0.50)
    four20 = timeframe_position(bars, "4h", 20, 10, 0.25)
    four55 = timeframe_position(bars, "4h", 55, 20, 0.25)
    bos = daily_bos_positions(bars)
    candidates = {
        "daily_turtle_20_10": {"daily20": daily20},
        "daily_turtle_55_20": {"daily55": daily55},
        "four_hour_turtle_20_10": {"four20": four20},
        "four_hour_turtle_55_20": {"four55": four55},
        "equal_daily_turtles": {"daily20": daily20 * 0.5, "daily55": daily55 * 0.5},
        "equal_four_hour_turtles": {"four20": four20 * 0.5, "four55": four55 * 0.5},
        "equal_all_turtles": {"daily20": daily20 * 0.25, "daily55": daily55 * 0.25, "four20": four20 * 0.25, "four55": four55 * 0.25},
        "bos_plus_equal_turtles": {"bos": bos * 0.5, "daily20": daily20 * 0.125, "daily55": daily55 * 0.125, "four20": four20 * 0.125, "four55": four55 * 0.125},
    }
    rows = []
    yearly_rows = []
    for name, mapping in candidates.items():
        replay = stable.simulate(bars, pd.DataFrame(mapping, index=bars.index))
        rows.append(stable.metrics(replay, name))
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = stable.metrics(local, f"{name}:{year}")
            row.update({"model": name, "year": year})
            yearly_rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "05_turtle_pa_screen.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(RESULTS / "06_turtle_pa_years.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
