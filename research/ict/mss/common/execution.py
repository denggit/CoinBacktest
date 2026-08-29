#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal FVG limit-entry and conservative outcome simulation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .structure import normalize_bars

EPS = 1e-12


def _first_touch_pair(
    *,
    side: int,
    entry: float,
    stop: float,
    target_r: float,
    start: int,
    end: int,
    low_index: SegmentThresholdIndex,
    high_index: SegmentThresholdIndex,
) -> tuple[str, int, float]:
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= EPS:
        return "INVALID_RISK", -1, np.nan
    target = entry + risk * float(target_r) if side == 1 else entry - risk * float(target_r)
    # A limit entry may happen anywhere inside the fill candle.  Its favorable
    # extreme can therefore predate the fill and must not count as a target.
    # Adverse movement through the stop on the fill candle is still valid: from
    # a non-gap open the path must cross the limit before reaching a farther
    # adverse stop.  This deliberately biases bare-OHLC simulation downward.
    target_start = int(start) + 1
    if side == 1:
        stop_pos = low_index.first_leq(start, end, stop)
        target_pos = high_index.first_geq(target_start, end, target) if target_start <= end else -1
    else:
        stop_pos = high_index.first_geq(start, end, stop)
        target_pos = low_index.first_leq(target_start, end, target) if target_start <= end else -1
    if stop_pos < 0 and target_pos < 0:
        return "TIMEOUT", -1, float(target)
    if stop_pos >= 0 and (target_pos < 0 or stop_pos <= target_pos):
        # Same 1m bar target+stop is conservatively a stop because intrabar path
        # is unknowable from bare OHLC.
        return "STOP", int(stop_pos), float(target)
    return "TARGET", int(target_pos), float(target)


def attach_limit_entry_and_outcomes(
    primary: pd.DataFrame,
    setups: pd.DataFrame,
    *,
    max_fill_wait_bars: int = 120,
    outcome_horizon_bars: int = 240,
    target_rs: Iterable[float] = (1.0, 1.5, 2.0, 3.0),
    round_trip_cost_pct: float = 0.0011,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Attach first FVG retest fill and fixed-R diagnostic outcomes.

    The limit order becomes active only on the bar after the FVG completion bar.
    Entry is intentionally priced at the requested FVG near edge even when a
    later bar gaps through it in the trader's favour; this is conservative.
    """

    bars = normalize_bars(primary)
    if setups.empty:
        return setups.copy()
    out = setups.copy().reset_index(drop=True)
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    n = len(bars)
    target_rs = tuple(float(v) for v in target_rs)

    result_rows: list[dict[str, object]] = []
    reporter = ProgressReporter(
        label="[ict-mss] FVG limit fills + R outcomes",
        total=len(out),
        every=max(1, len(out) // 200),
        enabled=bool(show_progress),
    )
    for done, row in enumerate(out.itertuples(index=False), start=1):
        side = int(row.side)
        completion = int(row.fvg_completion_pos)
        active_pos = completion + 1
        entry = float(row.fvg_near_price)
        stop = float(row.stop_extreme)
        fill_end = min(n - 1, completion + int(max_fill_wait_bars))
        if side == 1:
            fill_pos = low_index.first_leq(active_pos, fill_end, entry) if active_pos < n else -1
            structurally_valid = stop < entry
        else:
            fill_pos = high_index.first_geq(active_pos, fill_end, entry) if active_pos < n else -1
            structurally_valid = stop > entry
        risk_pct = abs(entry - stop) / entry if structurally_valid and entry > 0 else np.nan
        base: dict[str, object] = {
            "order_active_pos": int(active_pos),
            "order_active_time": bars.index[active_pos] if active_pos < n else pd.NaT,
            "first_fill_pos": int(fill_pos),
            "first_fill_time": bars.index[fill_pos] if fill_pos >= 0 else pd.NaT,
            "entry_price": entry,
            "stop_price": stop,
            "risk_pct": float(risk_pct) if np.isfinite(risk_pct) else np.nan,
            "entry_structure_valid": bool(structurally_valid),
            "gap_through_stop_on_fill_flag": False,
        }
        if fill_pos >= 0 and structurally_valid:
            if side == 1:
                gap_through = bool(open_[fill_pos] <= stop)
            else:
                gap_through = bool(open_[fill_pos] >= stop)
            base["gap_through_stop_on_fill_flag"] = gap_through
            end = min(n - 1, fill_pos + int(outcome_horizon_bars))
            if end >= fill_pos:
                # Favorable fill-candle extremes may have happened before the
                # intrabar limit fill, so MFE begins on the next candle.  MAE
                # keeps the fill candle because adverse traversal beyond the
                # entry is executable once the limit has been crossed.
                fav_start = fill_pos + 1
                path_high_fav = high[fav_start : end + 1] if fav_start <= end else np.asarray([], dtype=float)
                path_low_fav = low[fav_start : end + 1] if fav_start <= end else np.asarray([], dtype=float)
                path_high_adv = high[fill_pos : end + 1]
                path_low_adv = low[fill_pos : end + 1]
                if side == 1:
                    max_fav = float(np.max(path_high_fav)) if path_high_fav.size else entry
                    max_adv = float(np.min(path_low_adv)) if path_low_adv.size else np.nan
                    mfe_pct = max_fav / entry - 1.0 if np.isfinite(max_fav) else np.nan
                    mae_pct = max_adv / entry - 1.0 if np.isfinite(max_adv) else np.nan
                else:
                    min_fav = float(np.min(path_low_fav)) if path_low_fav.size else entry
                    max_adv = float(np.max(path_high_adv)) if path_high_adv.size else np.nan
                    mfe_pct = 1.0 - min_fav / entry if np.isfinite(min_fav) else np.nan
                    mae_pct = 1.0 - max_adv / entry if np.isfinite(max_adv) else np.nan
                base["mfe_pct_horizon"] = mfe_pct
                base["mae_pct_horizon"] = mae_pct
                base["mfe_r_horizon"] = mfe_pct / risk_pct if risk_pct > EPS else np.nan
                base["mae_r_horizon"] = mae_pct / risk_pct if risk_pct > EPS else np.nan
            for target_r in target_rs:
                token = str(target_r).replace(".", "p")
                result, exit_pos, target_price = _first_touch_pair(
                    side=side,
                    entry=entry,
                    stop=stop,
                    target_r=target_r,
                    start=fill_pos,
                    end=end,
                    low_index=low_index,
                    high_index=high_index,
                )
                if gap_through:
                    result = "GAP_STOP"
                    exit_pos = fill_pos
                    exit_price = float(open_[fill_pos])
                elif result == "STOP":
                    exit_price = stop
                elif result == "TARGET":
                    exit_price = target_price
                else:
                    exit_pos = end
                    exit_price = float(close[end])
                gross_pct = (exit_price / entry - 1.0) * side
                net_pct = gross_pct - float(round_trip_cost_pct)
                gross_r = gross_pct / risk_pct if risk_pct > EPS else np.nan
                net_r = net_pct / risk_pct if risk_pct > EPS else np.nan
                base[f"result_r{token}"] = result
                base[f"exit_pos_r{token}"] = int(exit_pos)
                base[f"exit_time_r{token}"] = bars.index[exit_pos] if exit_pos >= 0 else pd.NaT
                base[f"gross_return_pct_r{token}"] = float(gross_pct)
                base[f"net_return_pct_r{token}"] = float(net_pct)
                base[f"gross_r_r{token}"] = float(gross_r) if np.isfinite(gross_r) else np.nan
                base[f"net_r_r{token}"] = float(net_r) if np.isfinite(net_r) else np.nan
        else:
            base["mfe_pct_horizon"] = np.nan
            base["mae_pct_horizon"] = np.nan
            base["mfe_r_horizon"] = np.nan
            base["mae_r_horizon"] = np.nan
            for target_r in target_rs:
                token = str(target_r).replace(".", "p")
                result = "NO_FILL" if structurally_valid else "INVALID_STOP"
                base[f"result_r{token}"] = result
                base[f"exit_pos_r{token}"] = -1
                base[f"exit_time_r{token}"] = pd.NaT
                base[f"gross_return_pct_r{token}"] = np.nan
                base[f"net_return_pct_r{token}"] = np.nan
                base[f"gross_r_r{token}"] = np.nan
                base[f"net_r_r{token}"] = np.nan
        result_rows.append(base)
        reporter.update(done)
    reporter.close()
    attached = pd.concat([out, pd.DataFrame(result_rows)], axis=1)
    attached["fill_wait_bars"] = np.where(
        attached["first_fill_pos"].ge(0),
        attached["first_fill_pos"] - attached["fvg_completion_pos"],
        np.nan,
    )
    return attached
