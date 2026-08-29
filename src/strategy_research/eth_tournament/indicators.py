from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minp = min_periods if min_periods is not None else max(20, window // 4)
    past = series.shift(1)
    mean = past.rolling(window, min_periods=minp).mean()
    std = past.rolling(window, min_periods=minp).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def bollinger(close: pd.Series, length: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def annualized_vol(close: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    ret = np.log(close / close.shift(1))
    return ret.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)
