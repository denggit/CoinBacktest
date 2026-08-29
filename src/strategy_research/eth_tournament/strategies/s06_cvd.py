from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EntryEvent, StrategySignals, StrategySpec
from ..data import TournamentData
from ..indicators import atr


def build_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    b = data.bars("15min").copy()
    cvd = b["delta_notional"].cumsum()
    av = atr(b, 14)
    prior_low = b["low"].rolling(12, min_periods=12).min().shift(1)
    prior_high = b["high"].rolling(12, min_periods=12).max().shift(1)
    prior_cvd_low = cvd.rolling(12, min_periods=12).min().shift(1)
    prior_cvd_high = cvd.rolling(12, min_periods=12).max().shift(1)
    entries = []
    audit = []
    for i in range(1, len(b)):
        t = pd.Timestamp(b["available_time"].iloc[i])
        if t < pd.Timestamp(data.cfg.research_start) or t > pd.Timestamp(data.cfg.research_end):
            continue
        if not np.isfinite(av.iloc[i]):
            continue
        side = 0
        # Price makes the extreme but CVD fails to confirm; close reclaims the prior extreme.
        if b["low"].iloc[i] < prior_low.iloc[i] and cvd.iloc[i] > prior_cvd_low.iloc[i] and b["close"].iloc[i] > prior_low.iloc[i]:
            side = 1
        elif b["high"].iloc[i] > prior_high.iloc[i] and cvd.iloc[i] < prior_cvd_high.iloc[i] and b["close"].iloc[i] < prior_high.iloc[i]:
            side = -1
        if side:
            stop = float(av.iloc[i] * 1.5)
            entries.append(EntryEvent(t, side, stop_distance=stop, target_distance=float(av.iloc[i] * 2.5), max_hold_minutes=360, tag="CVD_EXHAUSTION"))
            audit.append({"signal_time": t, "side": side, "cvd": cvd.iloc[i], "atr": av.iloc[i], "prior_low": prior_low.iloc[i], "prior_high": prior_high.iloc[i]})
    return StrategySignals(entries=entries, audit=pd.DataFrame(audit))
