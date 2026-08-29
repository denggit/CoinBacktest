from __future__ import annotations

import pandas as pd

from ..contracts import StrategySpec
from ..data import TournamentData
from ..indicators import annualized_vol


def build_target(data: TournamentData, spec: StrategySpec) -> pd.Series:
    d = data.bars("1D").copy()
    fast = int(spec.params["fast"])
    slow = int(spec.params["slow"])
    vol_window = int(spec.params["vol_window"])
    vol_target = float(spec.params["vol_target"])
    sma_fast = d["close"].rolling(fast, min_periods=fast).mean()
    sma_slow = d["close"].rolling(slow, min_periods=slow).mean()
    direction = pd.Series(0.0, index=d.index)
    direction[sma_fast > sma_slow] = 1.0
    direction[sma_fast < sma_slow] = -1.0
    vol = annualized_vol(d["close"], vol_window, 365.25)
    scale = (vol_target / vol).clip(lower=0.0, upper=2.0)
    target = (direction * scale).clip(-2.0, 2.0)
    target.index = pd.DatetimeIndex(d["available_time"])
    return target
