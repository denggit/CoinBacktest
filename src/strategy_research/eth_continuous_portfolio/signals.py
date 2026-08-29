from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioSpec
from .data import ContinuousPortfolioData
from .indicators import adx, annualized_vol, atr


DONCHIAN_LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
TSMOM_LOOKBACKS = (21, 63, 126, 252)


def _with_available_index(values: pd.Series, bars: pd.DataFrame, name: str) -> pd.Series:
    out = pd.Series(values.to_numpy(float), index=pd.DatetimeIndex(bars["available_time"]), name=name)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _donchian_state(d: pd.DataFrame, lookback: int) -> pd.Series:
    hi = d["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lo = d["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    mid = (hi + lo) / 2.0
    close = d["close"].to_numpy(float)
    hi_v = hi.to_numpy(float)
    lo_v = lo.to_numpy(float)
    mid_v = mid.to_numpy(float)
    sig = np.zeros(len(d), dtype=float)
    state = 0
    trail = np.nan
    for i in range(len(d)):
        if not (np.isfinite(hi_v[i]) and np.isfinite(lo_v[i]) and np.isfinite(mid_v[i])):
            continue
        c = close[i]
        if state == 0:
            if c > hi_v[i]:
                state, trail = 1, mid_v[i]
            elif c < lo_v[i]:
                state, trail = -1, mid_v[i]
        elif state == 1:
            trail = max(float(trail), mid_v[i])
            if c < trail:
                state, trail = 0, np.nan
                if c < lo_v[i]:
                    state, trail = -1, mid_v[i]
        else:
            trail = min(float(trail), mid_v[i])
            if c > trail:
                state, trail = 0, np.nan
                if c > hi_v[i]:
                    state, trail = 1, mid_v[i]
        sig[i] = state
    return pd.Series(sig, index=d.index)


def _turtle_state(d: pd.DataFrame, entry_n: int = 55, exit_n: int = 20) -> pd.Series:
    entry_hi = d["high"].rolling(entry_n, min_periods=entry_n).max().shift(1)
    entry_lo = d["low"].rolling(entry_n, min_periods=entry_n).min().shift(1)
    exit_hi = d["high"].rolling(exit_n, min_periods=exit_n).max().shift(1)
    exit_lo = d["low"].rolling(exit_n, min_periods=exit_n).min().shift(1)
    state = 0
    out = np.zeros(len(d), dtype=float)
    for i, c in enumerate(d["close"].to_numpy(float)):
        if state == 0:
            if np.isfinite(entry_hi.iloc[i]) and c > entry_hi.iloc[i]:
                state = 1
            elif np.isfinite(entry_lo.iloc[i]) and c < entry_lo.iloc[i]:
                state = -1
        elif state == 1:
            if np.isfinite(exit_lo.iloc[i]) and c < exit_lo.iloc[i]:
                state = -1 if np.isfinite(entry_lo.iloc[i]) and c < entry_lo.iloc[i] else 0
        else:
            if np.isfinite(exit_hi.iloc[i]) and c > exit_hi.iloc[i]:
                state = 1 if np.isfinite(entry_hi.iloc[i]) and c > entry_hi.iloc[i] else 0
        out[i] = state
    return pd.Series(out, index=d.index)


def _supertrend_state(h: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.Series:
    a = atr(h, length).to_numpy(float)
    high = h["high"].to_numpy(float)
    low = h["low"].to_numpy(float)
    close = h["close"].to_numpy(float)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    final_upper = upper.copy()
    final_lower = lower.copy()
    state = np.zeros(len(h), dtype=float)
    direction = 0
    for i in range(1, len(h)):
        if not np.isfinite(a[i]):
            continue
        if np.isfinite(final_upper[i - 1]):
            if upper[i] >= final_upper[i - 1] and close[i - 1] <= final_upper[i - 1]:
                final_upper[i] = final_upper[i - 1]
        if np.isfinite(final_lower[i - 1]):
            if lower[i] <= final_lower[i - 1] and close[i - 1] >= final_lower[i - 1]:
                final_lower[i] = final_lower[i - 1]
        if direction <= 0 and close[i] > final_upper[i - 1]:
            direction = 1
        elif direction >= 0 and close[i] < final_lower[i - 1]:
            direction = -1
        state[i] = direction
    return pd.Series(state, index=h.index)


def _keltner_state(h: pd.DataFrame, ema_length: int = 20, atr_length: int = 20, mult: float = 2.0) -> pd.Series:
    mid = h["close"].ewm(span=ema_length, adjust=False, min_periods=ema_length).mean()
    a = atr(h, atr_length)
    upper = mid + mult * a
    lower = mid - mult * a
    out = np.zeros(len(h), dtype=float)
    state = 0
    c = h["close"].to_numpy(float)
    for i in range(len(h)):
        if not (np.isfinite(upper.iloc[i]) and np.isfinite(lower.iloc[i])):
            continue
        if c[i] > upper.iloc[i]:
            state = 1
        elif c[i] < lower.iloc[i]:
            state = -1
        out[i] = state
    return pd.Series(out, index=h.index)


def _adx_state(h: pd.DataFrame, adx_length: int = 14, threshold: float = 25.0, ema_length: int = 50) -> pd.Series:
    strength = adx(h, adx_length)
    ema = h["close"].ewm(span=ema_length, adjust=False, min_periods=ema_length).mean()
    direction = np.sign(h["close"] - ema)
    return direction.where(strength >= threshold, 0.0).fillna(0.0)


def build_sleeves(data: ContinuousPortfolioData) -> pd.DataFrame:
    d = data.bars("1D")
    h4 = data.bars("4h")

    don = pd.concat([_donchian_state(d, n) for n in DONCHIAN_LOOKBACKS], axis=1).mean(axis=1)
    turtle = _turtle_state(d)
    channel_family = (don + turtle) / 2.0

    ma_20_50 = np.sign(d["close"].rolling(20, min_periods=20).mean() - d["close"].rolling(50, min_periods=50).mean())
    ma_50_200 = np.sign(d["close"].rolling(50, min_periods=50).mean() - d["close"].rolling(200, min_periods=200).mean())
    ma_family = pd.concat([ma_20_50, ma_50_200], axis=1).mean(axis=1).fillna(0.0)

    tsmom_parts = [np.sign(d["close"] / d["close"].shift(n) - 1.0) for n in TSMOM_LOOKBACKS]
    tsmom_family = pd.concat(tsmom_parts, axis=1).mean(axis=1).fillna(0.0)

    st = _supertrend_state(h4)
    kel = _keltner_state(h4)
    adx_sig = _adx_state(h4)
    intraday_family = pd.concat([st, kel, adx_sig], axis=1).mean(axis=1).fillna(0.0)

    daily = pd.DataFrame(
        {
            "channel_family": _with_available_index(channel_family, d, "channel_family"),
            "ma_family": _with_available_index(ma_family, d, "ma_family"),
            "tsmom_family": _with_available_index(tsmom_family, d, "tsmom_family"),
        }
    )
    intraday = _with_available_index(intraday_family, h4, "intraday_family").to_frame()
    union = daily.index.union(intraday.index).sort_values()
    sleeves = daily.reindex(union).ffill().join(intraday.reindex(union).ffill(), how="outer")
    sleeves = sleeves.fillna(0.0).clip(-1.0, 1.0)
    sleeves["raw_signal"] = sleeves[["channel_family", "ma_family", "tsmom_family", "intraday_family"]].mean(axis=1)

    vol = annualized_vol(d["close"], window=90)
    vol_schedule = _with_available_index(vol, d, "realized_vol")
    sleeves = sleeves.join(vol_schedule.reindex(union).ffill(), how="left")
    return sleeves.sort_index()


def build_raw_target(sleeves: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    out = sleeves.copy()
    scale = spec.volatility_target / out["realized_vol"].replace(0.0, np.nan)
    # Volatility targeting may increase exposure in quiet regimes, but leverage is hard capped.
    out["vol_scale"] = scale.clip(lower=0.0, upper=spec.max_abs_exposure).fillna(0.0)
    out["raw_target"] = (out["raw_signal"] * out["vol_scale"]).clip(-spec.max_abs_exposure, spec.max_abs_exposure)
    return out
