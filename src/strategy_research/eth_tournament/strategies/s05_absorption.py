from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EntryEvent, StrategySignals, StrategySpec
from ..data import TournamentData


def build_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    f = data.absorption_features().copy()
    if f.empty:
        return StrategySignals()
    lower_q = f["lower_delta_ratio"].shift(1).rolling(200, min_periods=100).quantile(0.10)
    upper_q = f["upper_delta_ratio"].shift(1).rolling(200, min_periods=100).quantile(0.90)
    entries = []
    audit = []
    for i in range(len(f)):
        t = pd.Timestamp(f["available_time"].iloc[i])
        if t < pd.Timestamp(data.cfg.research_start) or t > pd.Timestamp(data.cfg.research_end):
            continue
        width = float(f["high"].iloc[i] - f["low"].iloc[i])
        if width <= 0 or not np.isfinite(width):
            continue
        ld = f["lower_delta_ratio"].iloc[i]
        ud = f["upper_delta_ratio"].iloc[i]
        cp = f["close_pos"].iloc[i]
        td = f["total_delta_ratio"].iloc[i]
        side = 0
        if np.isfinite(ld) and np.isfinite(lower_q.iloc[i]) and ld <= lower_q.iloc[i] and td < 0 and cp >= 0.65:
            side = 1
        elif np.isfinite(ud) and np.isfinite(upper_q.iloc[i]) and ud >= upper_q.iloc[i] and td > 0 and cp <= 0.35:
            side = -1
        if side:
            stop = width * 1.25
            entries.append(EntryEvent(t, side, stop_distance=stop, target_distance=2.0 * stop, max_hold_minutes=180, tag="FOOTPRINT_ABSORPTION", metadata={"bar_id": int(f["bar_id"].iloc[i])}))
            audit.append({"signal_time": t, "side": side, "bar_id": int(f["bar_id"].iloc[i]), "lower_delta_ratio": ld, "upper_delta_ratio": ud, "close_pos": cp, "total_delta_ratio": td, "stop_distance": stop})
    return StrategySignals(entries=entries, audit=pd.DataFrame(audit))
