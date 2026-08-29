from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EntryEvent, StrategySignals, StrategySpec
from ..data import TournamentData
from ..indicators import atr, rolling_zscore


def build_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    b = data.bars("15min").copy()
    av = atr(b, 14)
    prior_high = b["high"].rolling(96, min_periods=96).max().shift(1)
    prior_low = b["low"].rolling(96, min_periods=96).min().shift(1)
    imbalance = b["flow_imbalance"]
    z = rolling_zscore(imbalance, window=672, min_periods=192)  # past 7d of 15m bars
    entries = []
    audit = []
    for i in range(len(b)):
        t = pd.Timestamp(b["available_time"].iloc[i])
        if t < pd.Timestamp(data.cfg.research_start) or t > pd.Timestamp(data.cfg.research_end):
            continue
        if not np.isfinite(av.iloc[i]) or not np.isfinite(z.iloc[i]):
            continue
        side = 0
        if b["close"].iloc[i] > prior_high.iloc[i] and z.iloc[i] >= 1.0:
            side = 1
        elif b["close"].iloc[i] < prior_low.iloc[i] and z.iloc[i] <= -1.0:
            side = -1
        if side:
            stop = float(av.iloc[i] * 2.0)
            entries.append(EntryEvent(t, side, stop_distance=stop, target_distance=float(av.iloc[i] * 3.0), max_hold_minutes=720, tag="FLOW_BREAKOUT"))
            audit.append({"signal_time": t, "side": side, "flow_imbalance": imbalance.iloc[i], "flow_z": z.iloc[i], "atr": av.iloc[i], "prior_high": prior_high.iloc[i], "prior_low": prior_low.iloc[i]})
    return StrategySignals(entries=entries, audit=pd.DataFrame(audit))
