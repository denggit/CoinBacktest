from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ZARATTINI_LOOKBACKS


@dataclass(frozen=True)
class RuleSchedule:
    strategy_id: str
    schedule: pd.DataFrame
    source_rule: str
    adaptation: str


def annualized_vol(close: pd.Series, window: int, periods_per_year: int = 365) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(float(periods_per_year))


def _donchian_state(close: pd.Series, lookback: int, *, allow_short: bool) -> pd.Series:
    """Zarattini/Pagani/Barbon state using rolling CLOSE channels and monotone midpoint trailing stop."""
    up = close.rolling(lookback, min_periods=lookback).max()
    down = close.rolling(lookback, min_periods=lookback).min()
    mid = (up + down) / 2.0
    c = close.to_numpy(float)
    u = up.to_numpy(float)
    d = down.to_numpy(float)
    m = mid.to_numpy(float)
    out = np.zeros(len(close), dtype=float)
    state = 0
    trail = np.nan
    for i in range(len(close)):
        if not (np.isfinite(c[i]) and np.isfinite(u[i]) and np.isfinite(d[i]) and np.isfinite(m[i])):
            continue
        is_up = np.isclose(c[i], u[i], rtol=1e-12, atol=1e-12)
        is_down = np.isclose(c[i], d[i], rtol=1e-12, atol=1e-12)
        if state == 0:
            if is_up:
                state, trail = 1, m[i]
            elif allow_short and is_down:
                state, trail = -1, m[i]
        elif state == 1:
            if is_up:
                state, trail = 1, max(float(trail), m[i])
            else:
                trail = max(float(trail), m[i])
                if c[i] <= trail:
                    state, trail = 0, np.nan
                    if allow_short and is_down:
                        state, trail = -1, m[i]
        else:
            if is_down:
                state, trail = -1, min(float(trail), m[i])
            else:
                trail = min(float(trail), m[i])
                if c[i] >= trail:
                    state, trail = 0, np.nan
                    if is_up:
                        state, trail = 1, m[i]
        out[i] = state
    return pd.Series(out, index=close.index, name=f"pos_{lookback}")


def build_zarattini(daily: pd.DataFrame, *, allow_short: bool) -> RuleSchedule:
    vol = annualized_vol(daily["close"], 90, 365)
    per_model: list[pd.Series] = []
    for n in ZARATTINI_LOOKBACKS:
        state = _donchian_state(daily["close"], n, allow_short=allow_short)
        magnitude = (0.25 / vol.replace(0.0, np.nan)).clip(upper=2.0)
        desired = (state * magnitude).fillna(0.0)

        # Paper's 20% rebalance threshold is applied only to volatility-driven sizing changes.
        # Signal changes (entry/exit/flip) execute immediately. We interpret "difference ... exceeds 20%"
        # as an absolute 0.20 portfolio-weight-point threshold; this interpretation is disclosed in reports.
        executed = np.zeros(len(desired), dtype=float)
        prev_state = 0.0
        prev_weight = 0.0
        for i, (s, w) in enumerate(zip(state.to_numpy(float), desired.to_numpy(float), strict=True)):
            if s != prev_state:
                prev_weight = float(w)
            elif abs(float(w) - prev_weight) > 0.20:
                prev_weight = float(w)
            executed[i] = prev_weight
            prev_state = float(s)
        per_model.append(pd.Series(executed, index=daily.index, name=f"w_{n}"))

    target = pd.concat(per_model, axis=1).mean(axis=1)
    schedule = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(daily["available_time"]),
            "raw_target": target.to_numpy(float),
            "realized_vol": vol.to_numpy(float),
        },
        index=daily.index,
    ).dropna(subset=["signal_time"])
    strategy_id = "SL01_ZARATTINI_LONG" if not allow_short else "SL02_ZARATTINI_LS"
    return RuleSchedule(
        strategy_id=strategy_id,
        schedule=schedule,
        source_rule=(
            "9 Donchian close-channel models (5/10/20/30/60/90/150/250/360D), monotone midpoint trailing stop, "
            "90D annualized vol, 25% target, 2x cap, equal-weight ensemble, 20% volatility-rebalance threshold"
        ),
        adaptation=(
            "Applied to ETH-USDT-SWAP instead of BTC/crypto cross-section; +8 daily boundary; project causal execution at first 1m open strictly after signal_time; "
            "20% threshold interpreted as 0.20 absolute weight points."
        ),
    )


def _ewma_vol_60(close: pd.Series) -> pd.Series:
    """Standard MOP-style ex-ante volatility implementation using 60-day EWMA daily returns."""
    ret = close.pct_change()
    # span=60 is the common published replication convention for the MOP ex-ante volatility estimator.
    var = ret.ewm(span=60, adjust=False, min_periods=60).var(bias=False)
    return np.sqrt(var * 365.0)


def build_mop_tsmom(daily: pd.DataFrame) -> RuleSchedule:
    """Single-ETH adaptation of Moskowitz/Ooi/Pedersen 12-month TSMOM, monthly rebalanced."""
    close = daily["close"]
    # Original rule is sign of past 12-month excess return. Crypto trades 365 days/year, so 365 daily bars represent 12 calendar months.
    mom = close / close.shift(365) - 1.0
    vol = _ewma_vol_60(close)
    desired = np.sign(mom) * (0.40 / vol.replace(0.0, np.nan))

    frame = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(daily["available_time"]),
            "raw_target": desired,
            "realized_vol": vol,
        },
        index=daily.index,
    ).dropna(subset=["signal_time", "raw_target"])

    # MOP's canonical strategy is monthly. Keep only the last fully known daily observation whose signal becomes available in each calendar month.
    month = pd.DatetimeIndex(frame["signal_time"]).to_period("M")
    frame = frame.assign(_month=month).groupby("_month", sort=True).head(1).drop(columns="_month")
    return RuleSchedule(
        strategy_id="SL03_MOP_TSMOM_12M",
        schedule=frame,
        source_rule="sign(past 12-month return) × 40% / ex-ante volatility; monthly holding/rebalance",
        adaptation=(
            "Single ETH perpetual rather than diversified 58-market futures portfolio; 12 months mapped to 365 crypto daily bars; "
            "60-day EWMA volatility uses 365-day annualization; no artificial leverage cap added."
        ),
    )


def turtle_n(daily: pd.DataFrame) -> pd.Series:
    """Original Turtle N: 20-day EMA of True Range, initialized by a 20-day simple average."""
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (prev_close - daily["low"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    values = tr.to_numpy(float)
    out = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    for i in range(19, len(values)):
        if i == 19:
            window = values[:20]
            if np.isfinite(window).all():
                out[i] = float(np.mean(window))
        elif np.isfinite(values[i]) and np.isfinite(out[i - 1]):
            out[i] = (19.0 * out[i - 1] + values[i]) / 20.0
    return pd.Series(out, index=daily.index, name="N")


def build_turtle_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Thresholds known after each +8 daily close, for resting System-2 stop orders."""
    return pd.DataFrame(
        {
            "available_time": pd.to_datetime(daily["available_time"]),
            "entry_high": daily["high"].rolling(55, min_periods=55).max(),
            "entry_low": daily["low"].rolling(55, min_periods=55).min(),
            "exit_high": daily["high"].rolling(20, min_periods=20).max(),
            "exit_low": daily["low"].rolling(20, min_periods=20).min(),
            "N": turtle_n(daily),
        },
        index=daily.index,
    ).dropna()
