from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import StrategySpec
from ..data import TournamentData
from ..indicators import annualized_vol


LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)


def build_target(data: TournamentData, spec: StrategySpec) -> pd.Series:
    d = data.bars("1D").copy()
    close = d["close"].astype(float)
    sleeve_signals: list[pd.Series] = []
    long_short = bool(spec.params.get("long_short", False))
    for n in LOOKBACKS:
        prior_high = d["high"].rolling(n, min_periods=n).max().shift(1)
        prior_low = d["low"].rolling(n, min_periods=n).min().shift(1)
        midpoint = (prior_high + prior_low) / 2.0
        sig = pd.Series(0.0, index=d.index)
        state = 0
        trail = np.nan
        for i in range(len(d)):
            c = float(close.iloc[i])
            hi = prior_high.iloc[i]
            lo = prior_low.iloc[i]
            mid = midpoint.iloc[i]
            if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(mid):
                continue
            if state == 0:
                if c > hi:
                    state, trail = 1, float(mid)
                elif long_short and c < lo:
                    state, trail = -1, float(mid)
            elif state == 1:
                trail = max(float(trail), float(mid))
                if c < trail:
                    state, trail = 0, np.nan
                    if long_short and c < lo:
                        state, trail = -1, float(mid)
            else:
                trail = min(float(trail), float(mid))
                if c > trail:
                    state, trail = 0, np.nan
                    if c > hi:
                        state, trail = 1, float(mid)
            sig.iloc[i] = float(state)
        sleeve_signals.append(sig)
    ensemble = pd.concat(sleeve_signals, axis=1).mean(axis=1)
    vol = annualized_vol(close, window=90, periods_per_year=365.25)
    scaler = (0.25 / vol).clip(lower=0.0, upper=2.0)
    target = (ensemble * scaler).clip(-2.0, 2.0)
    target.index = pd.DatetimeIndex(d["available_time"])
    return target
