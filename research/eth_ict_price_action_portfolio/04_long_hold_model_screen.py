#!/usr/bin/env python
"""Frozen long-hold Price Action model-family screen.

No numerical optimiser is used.  The families are standard, pre-declared
continuous-state trend/structure definitions.  The purpose is to test the
user's Calmar >= 1 stability gate, not to choose the highest backtest return.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import (  # noqa: E402
    IctPaConfig,
    confirmed_pivots,
    resample_ohlcv,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


OUT = Path(__file__).resolve().parent / "ict_pa_v1" / "results"


def persistent(long_event: pd.Series, short_event: pd.Series) -> pd.Series:
    values: list[float] = []
    state = 0.0
    for long_now, short_now in zip(long_event.fillna(False), short_event.fillna(False)):
        if bool(long_now) and not bool(short_now):
            state = 1.0
        elif bool(short_now) and not bool(long_now):
            state = -1.0
        values.append(state)
    return pd.Series(values, index=long_event.index)


def streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def replay(daily: pd.DataFrame, close_state: pd.Series, cfg: IctPaConfig) -> dict[str, object]:
    log_return = np.log(daily["close"]).diff()
    vol = log_return.shift(1).rolling(30, min_periods=30).std(ddof=0) * np.sqrt(365)
    size = (cfg.core_target_volatility / vol.replace(0.0, np.nan)).clip(upper=cfg.core_notional_cap)
    position = (close_state * size).shift(1).fillna(0.0)
    price_return = daily["open"].shift(-1) / daily["open"] - 1.0
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    net = position * price_return - turnover * cfg.one_way_cost - position.abs() * cfg.annual_carry_drag / 365
    net = net.loc[pd.Timestamp(cfg.start):pd.Timestamp(cfg.end)].dropna()
    position = position.reindex(net.index).fillna(0.0)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    elapsed = max((net.index[-1] - net.index[0]).days / 365.25, 1 / 365.25)
    final = float(equity.iloc[-1])
    cagr = final ** (1 / elapsed) - 1.0
    max_dd = abs(float(drawdown.min()))
    return {
        "max_consecutive_flat_days": streak(position.abs() <= 1e-12),
        "max_consecutive_losing_days": streak(net < 0.0),
        "max_drawdown_abs": max_dd,
        "cagr": cagr,
        "total_return": final - 1.0,
        "calmar": cagr / max_dd if max_dd > 0 else np.nan,
        "positive_day_rate": float((net > 0.0).mean()),
        "mean_abs_exposure": float(position.abs().mean()),
        "turnover": float(turnover.reindex(net.index).sum()),
        "passes_calmar_gate": bool(cagr >= max_dd),
    }


def main() -> int:
    cfg = IctPaConfig()
    cache = Path(__file__).resolve().parent / "bars_15m.pkl"
    if cache.exists():
        bars = pd.read_pickle(cache)
    else:
        minute = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
        minute = minute.loc["2020-01-01":"2026-08-15 23:59:59", ["open", "high", "low", "close", "volume"]]
        bars = resample_ohlcv(minute, "15min")
    daily = resample_ohlcv(bars, "1D")
    close = daily["close"]

    pivots = confirmed_pivots(daily, 2, 2)
    high = pivots["confirmed_high"].shift(1)
    low = pivots["confirmed_low"].shift(1)
    bos = persistent(
        (close > high) & (close.shift(1) <= high.shift(1)),
        (close < low) & (close.shift(1) >= low.shift(1)),
    )

    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    ema_state = np.sign(ema50 - ema200)

    tsmom_state = np.sign(np.log(close).diff(365))

    monthly = resample_ohlcv(daily, "MS")
    monthly_mean10 = monthly["close"].rolling(10, min_periods=10).mean()
    monthly_state = np.sign(monthly["close"] - monthly_mean10)
    monthly_state.index = monthly_state.index + pd.offsets.MonthBegin(1)
    monthly_state = monthly_state.reindex(daily.index, method="ffill")

    high60 = daily["high"].shift(1).rolling(60, min_periods=60).max()
    low60 = daily["low"].shift(1).rolling(60, min_periods=60).min()
    breakout_state = persistent(close > high60, close < low60)

    families = {
        "confirmed_daily_BOS": bos,
        "EMA_50_200_continuous": pd.Series(ema_state, index=daily.index),
        "12_calendar_month_momentum": pd.Series(tsmom_state, index=daily.index),
        "10_month_price_vs_mean": pd.Series(monthly_state, index=daily.index),
        "60_day_price_channel": breakout_state,
        "equal_BOS_plus_12m_momentum": (bos + pd.Series(tsmom_state, index=daily.index)) / 2.0,
        "two_thirds_BOS_one_third_12m": (
            2.0 * bos + pd.Series(tsmom_state, index=daily.index).fillna(bos)
        ) / 3.0,
        "majority_BOS_12m_EMA": np.sign(
            bos
            + pd.Series(tsmom_state, index=daily.index).fillna(0.0)
            + pd.Series(ema_state, index=daily.index).fillna(0.0)
        ),
    }
    rows = [{"model_family": name, **replay(daily, state, cfg)} for name, state in families.items()]
    result = pd.DataFrame(rows).sort_values(
        ["max_consecutive_flat_days", "max_consecutive_losing_days", "max_drawdown_abs", "cagr", "total_return"],
        ascending=[True, True, True, False, False],
        kind="stable",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "long_hold_model_screen.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
