from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EntryEvent, StrategySignals, StrategySpec
from ..data import TournamentData
from ..indicators import atr, rolling_zscore


def build_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    q = data.quarter_hour_opening_imbalance().copy()
    if q.empty:
        return StrategySignals()
    # 96 quarter-hours/day * 30d. Shifted rolling stats in rolling_zscore prohibit full-sample thresholds.
    z = rolling_zscore(q["imbalance"], 96 * 30, min_periods=96 * 10)
    b15 = data.bars("15min")
    av = atr(b15, 14)
    av_by_available = pd.Series(av.to_numpy(float), index=pd.DatetimeIndex(b15["available_time"]))
    entries = []
    audit = []
    hold = int(spec.params["hold_minutes"])
    for i in range(len(q)):
        t = pd.Timestamp(q["available_time"].iloc[i])
        if t < pd.Timestamp(data.cfg.research_start) or t > pd.Timestamp(data.cfg.research_end):
            continue
        zi = z.iloc[i]
        if not np.isfinite(zi) or abs(zi) < 1.5:
            continue
        pos = av_by_available.index.searchsorted(t, side="right") - 1
        if pos < 0:
            continue
        av_i = float(av_by_available.iloc[pos])
        if not np.isfinite(av_i) or av_i <= 0:
            continue
        side = 1 if zi > 0 else -1
        entries.append(EntryEvent(t, side, stop_distance=2.5 * av_i, target_distance=None, max_hold_minutes=hold, tag=f"QH_OI_{hold}M", metadata={"imbalance_z": float(zi)}))
        audit.append({"signal_time": t, "side": side, "imbalance": q["imbalance"].iloc[i], "z": zi, "atr15": av_i, "hold_minutes": hold})
    return StrategySignals(entries=entries, audit=pd.DataFrame(audit))
