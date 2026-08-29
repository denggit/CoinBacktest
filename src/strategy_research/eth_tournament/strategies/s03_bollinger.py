from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EntryEvent, ExitEvent, StrategySignals, StrategySpec
from ..data import TournamentData
from ..indicators import atr, bollinger, rsi


def build_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    tf = str(spec.params["timeframe"])
    mode = str(spec.params["mode"])
    b = data.bars(tf).copy()
    mid, upper, lower = bollinger(b["close"], 20, 2.0)
    rv = rsi(b["close"], 14)
    av = atr(b, 14)
    entries: list[EntryEvent] = []
    exits: list[ExitEvent] = []
    audit_rows = []
    for i in range(1, len(b)):
        t = pd.Timestamp(b["available_time"].iloc[i])
        if t < pd.Timestamp(data.cfg.research_start) or t > pd.Timestamp(data.cfg.research_end):
            continue
        c = float(b["close"].iloc[i])
        if not all(np.isfinite(x) for x in [mid.iloc[i], upper.iloc[i], lower.iloc[i], av.iloc[i]]):
            continue
        side = 0
        if mode == "mean_reversion":
            if c < lower.iloc[i] and b["close"].iloc[i - 1] >= lower.iloc[i - 1] and rv.iloc[i] < 30:
                side = 1
            elif c > upper.iloc[i] and b["close"].iloc[i - 1] <= upper.iloc[i - 1] and rv.iloc[i] > 70:
                side = -1
            if c >= mid.iloc[i] or rv.iloc[i] >= 50:
                exits.append(ExitEvent(t, 1, "BB_MID_OR_RSI"))
            if c <= mid.iloc[i] or rv.iloc[i] <= 50:
                exits.append(ExitEvent(t, -1, "BB_MID_OR_RSI"))
            stop_mult = 2.0
        else:
            if c > upper.iloc[i] and b["close"].iloc[i - 1] <= upper.iloc[i - 1]:
                side = 1
            elif c < lower.iloc[i] and b["close"].iloc[i - 1] >= lower.iloc[i - 1]:
                side = -1
            if c < mid.iloc[i]:
                exits.append(ExitEvent(t, 1, "BB_MID_FAIL"))
            if c > mid.iloc[i]:
                exits.append(ExitEvent(t, -1, "BB_MID_FAIL"))
            stop_mult = 3.0
        if side:
            entries.append(EntryEvent(t, side, stop_distance=float(av.iloc[i] * stop_mult), target_distance=None, max_hold_minutes=None, tag=mode))
            audit_rows.append({"signal_time": t, "side": side, "close": c, "mid": mid.iloc[i], "upper": upper.iloc[i], "lower": lower.iloc[i], "rsi": rv.iloc[i], "atr": av.iloc[i]})
    return StrategySignals(entries, exits, pd.DataFrame(audit_rows))
