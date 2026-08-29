#!/usr/bin/env python
"""Frozen mechanism-family screen for the clean ETH portfolio.

This is deliberately a small comparison of standard, predeclared trend
families.  It is not a parameter grid and does not promote the best row by
itself.  All families use the same volatility, execution, fee and carry model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "clean_causal_v1" / "results"
BARS_PER_DAY = 6
BARS_PER_YEAR = 365 * BARS_PER_DAY
ONE_WAY_COST = 0.00050
ANNUAL_CARRY = 0.05
TARGET_VOL = 0.12
GROSS_CAP = 1.50


def _load() -> pd.DataFrame:
    path = ROOT / "ohlcv_4h.csv"
    bars = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return bars


def _volatility_multiplier(bars: pd.DataFrame) -> pd.Series:
    realised = np.log(bars["close"]).diff().rolling(30 * BARS_PER_DAY, min_periods=30 * BARS_PER_DAY).std(ddof=0)
    return (TARGET_VOL / (realised * np.sqrt(BARS_PER_YEAR)).replace(0, np.nan)).clip(upper=GROSS_CAP)


def _daily_rebalance(frame: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(np.arange(len(frame)) % BARS_PER_DAY == 0, index=frame.index)
    return frame.where(mask, np.nan).ffill()


def tsmom_signals(bars: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(bars["close"])
    return pd.DataFrame(
        {f"mom_{days}d": np.sign(log_close.diff(days * BARS_PER_DAY)) for days in (7, 30, 90)},
        index=bars.index,
    )


def ema_crossover_signals(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    pairs = ((8, 32), (16, 64), (32, 128))
    return pd.DataFrame(
        {
            f"ema_{fast}_{slow}": np.sign(
                close.ewm(span=fast * BARS_PER_DAY, adjust=False, min_periods=slow * BARS_PER_DAY).mean()
                - close.ewm(span=slow * BARS_PER_DAY, adjust=False, min_periods=slow * BARS_PER_DAY).mean()
            )
            for fast, slow in pairs
        },
        index=bars.index,
    )


def donchian_signals(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    output: dict[str, pd.Series] = {}
    for entry_days, exit_days in ((20, 10), (60, 30), (120, 60)):
        entry = entry_days * BARS_PER_DAY
        exit_window = exit_days * BARS_PER_DAY
        prior_high = bars["high"].rolling(entry, min_periods=entry).max().shift(1)
        prior_low = bars["low"].rolling(entry, min_periods=entry).min().shift(1)
        exit_high = bars["high"].rolling(exit_window, min_periods=exit_window).max().shift(1)
        exit_low = bars["low"].rolling(exit_window, min_periods=exit_window).min().shift(1)
        state = 0
        values: list[int] = []
        for price, high, low, stop_high, stop_low in zip(close, prior_high, prior_low, exit_high, exit_low):
            if np.isfinite(high) and price > high:
                state = 1
            elif np.isfinite(low) and price < low:
                state = -1
            elif state > 0 and np.isfinite(stop_low) and price < stop_low:
                state = 0
            elif state < 0 and np.isfinite(stop_high) and price > stop_high:
                state = 0
            values.append(state)
        output[f"donchian_{entry_days}_{exit_days}"] = pd.Series(values, index=bars.index, dtype=float)
    return pd.DataFrame(output, index=bars.index)


def mean_reversion_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """Standard close-to-mean reversion states, entered at ±2 rolling sigma."""
    close = bars["close"]
    output: dict[str, pd.Series] = {}
    for days in (3, 5, 10):
        window = days * BARS_PER_DAY
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        zscore = (close - mean) / std.replace(0, np.nan)
        state = 0
        values: list[int] = []
        for z in zscore:
            if np.isfinite(z) and z <= -2.0:
                state = 1
            elif np.isfinite(z) and z >= 2.0:
                state = -1
            elif state > 0 and np.isfinite(z) and z >= 0.0:
                state = 0
            elif state < 0 and np.isfinite(z) and z <= 0.0:
                state = 0
            values.append(state)
        output[f"reversion_z2_{days}d"] = pd.Series(values, index=bars.index, dtype=float)
    return pd.DataFrame(output, index=bars.index)


def simulate(bars: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    vol = _volatility_multiplier(bars)
    targets = signals.mul(vol / signals.shape[1], axis=0)
    targets = _daily_rebalance(targets).shift(1).fillna(0.0)
    records: list[dict[str, object]] = []
    previous = np.zeros(signals.shape[1], dtype=float)
    equity = 1.0
    peak = 1.0
    for i in range(len(bars) - 1):
        positions = targets.iloc[i].to_numpy(dtype=float)
        gross = float(np.abs(positions).sum())
        if gross > GROSS_CAP:
            positions *= GROSS_CAP / gross
            gross = GROSS_CAP
        net = float(positions.sum())
        turnover = float(np.abs(positions - previous).sum())
        price_return = float(bars["open"].iloc[i + 1] / bars["open"].iloc[i] - 1.0)
        cost = turnover * ONE_WAY_COST
        carry = gross * ANNUAL_CARRY / BARS_PER_YEAR
        net_return = net * price_return - cost - carry
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        records.append(
            {
                "timestamp": bars.index[i],
                "net_return": net_return,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "gross_exposure": gross,
                "net_exposure": net,
                "long_short_overlap": bool((positions > 0).any() and (positions < 0).any()),
                "turnover": turnover,
            }
        )
        previous = positions
    return pd.DataFrame(records).set_index("timestamp")


def summarize(name: str, frame: pd.DataFrame) -> dict[str, object]:
    monthly = (1.0 + frame["net_return"]).groupby(frame.index.to_period("M")).prod() - 1.0
    elapsed = (frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400)
    final = float(frame["equity"].iloc[-1])
    return {
        "family": name,
        "total_return": final - 1.0,
        "cagr": final ** (1 / elapsed) - 1.0,
        "max_drawdown": float(frame["drawdown"].min()),
        "positive_month_rate": float((monthly > 0).mean()),
        "annual_turnover": float(frame["turnover"].sum() / elapsed),
        "long_short_overlap_rate": float(frame["long_short_overlap"].mean()),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
    }


def main() -> int:
    bars = _load()
    families = {
        "return_sign_7_30_90": tsmom_signals(bars),
        "ema_crossover_8_32_16_64_32_128": ema_crossover_signals(bars),
        "donchian_20_10_60_30_120_60": donchian_signals(bars),
        "mean_reversion_z2_3_5_10": mean_reversion_signals(bars),
        "donchian_plus_reversion": pd.concat(
            [donchian_signals(bars), mean_reversion_signals(bars)[["reversion_z2_5d"]]], axis=1
        ),
    }
    rows: list[dict[str, object]] = []
    for name, signals in families.items():
        replay = simulate(bars, signals)
        replay.to_csv(ROOT / f"family_{name}.csv")
        rows.append(summarize(name, replay))
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "frozen_family_screen.csv", index=False)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
