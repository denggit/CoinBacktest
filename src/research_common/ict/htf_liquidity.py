#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal unconsumed higher-timeframe liquidity swings for ICT research.

The module deliberately keeps higher-timeframe liquidity discovery separate
from the intraday MSS/displacement engine.  A HTF swing may become a candidate
liquidity level only after its right-hand pivot confirmation is fully closed,
and remains active until the first completed 1m bar strictly trades through it.

No nearest-only rule is used: every still-active 1H/4H/1D swing is retained so
research can measure whether older / farther liquidity has different edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
import heapq
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, NY_TZ, aggregate_closed_bars, slice_ny_day
from .premarket_mss_fvg_v2 import confirmed_pivots_with_excursion

SESSION_START = dtime(4, 0)
SESSION_END = dtime(16, 30)
TRADE_ANCHOR = dtime(8, 30)


@dataclass(frozen=True)
class HTFLiquidityConfig:
    timeframes: tuple[str, ...] = ("1h", "4h", "1d")
    pivot_left: int = 2
    pivot_right: int = 2
    daily_min_rows: int = 740


def _tf_minutes(label: str) -> int | None:
    norm = str(label).strip().lower()
    if norm in {"1h", "60m", "60"}:
        return 60
    if norm in {"4h", "240m", "240"}:
        return 240
    if norm in {"1d", "d", "day", "daily"}:
        return None
    raise ValueError(f"unsupported HTF label: {label!r}")


def _canonical_label(label: str) -> str:
    mins = _tf_minutes(label)
    return "1d" if mins is None else ("1h" if mins == 60 else "4h")


def build_session_daily_bars(bars_ny: pd.DataFrame, *, min_rows: int = 740) -> pd.DataFrame:
    """Build causal NY-session daily bars from the 04:00-16:30 research tape.

    A session bar is available at 16:30 NY on that same date.  A daily swing
    therefore cannot be used until its right-side daily bars have themselves
    closed, as enforced later by the pivot confirmation timestamp.
    """
    if bars_ny.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(bars_ny.index)
    if idx.tz is None:
        raise ValueError("build_session_daily_bars expects timezone-aware NY bars")
    mins = idx.hour * 60 + idx.minute
    work = bars_ny.loc[(mins >= 240) & (mins < 990), [c for c in ("open", "high", "low", "close", "volume") if c in bars_ny.columns]].copy()
    if work.empty:
        return pd.DataFrame()
    work["_ny_date"] = pd.Index(work.index.date)
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    grouped = work.groupby("_ny_date", sort=True)
    out = grouped.agg(agg)
    counts = grouped.size().rename("bar_count")
    out = out.join(counts)
    out = out.loc[out["bar_count"] >= int(min_rows)].copy()
    if out.empty:
        return out
    starts = [pd.Timestamp(d).tz_localize(NY_TZ) + pd.Timedelta(hours=4) for d in out.index]
    available = [pd.Timestamp(d).tz_localize(NY_TZ) + pd.Timedelta(hours=16, minutes=30) for d in out.index]
    out.index = pd.DatetimeIndex(starts, name="bar_start_ny")
    out["available_time"] = pd.DatetimeIndex(available)
    out["timeframe_label"] = "1d"
    return out


def build_htf_frames(bars_ny: pd.DataFrame, config: HTFLiquidityConfig = HTFLiquidityConfig()) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for raw in config.timeframes:
        label = _canonical_label(raw)
        mins = _tf_minutes(raw)
        if mins is None:
            frame = build_session_daily_bars(bars_ny, min_rows=config.daily_min_rows)
        else:
            frame = aggregate_closed_bars(bars_ny, mins)
            if not frame.empty:
                # Drop any accidental overnight/partial-session anchors.  The
                # input is already clipped to the actual stock session used by
                # this research, but the explicit mask keeps the invariant local.
                anchor_mins = frame.index.hour * 60 + frame.index.minute
                frame = frame.loc[(anchor_mins >= 240) & (anchor_mins < 960)].copy()
                frame["timeframe_label"] = label
        frames[label] = frame
    return frames


def build_htf_swing_catalog(
    bars_ny: pd.DataFrame,
    *,
    config: HTFLiquidityConfig = HTFLiquidityConfig(),
) -> pd.DataFrame:
    """Build causally confirmed 1H/4H/1D swing catalog."""
    parts: list[pd.DataFrame] = []
    for label, frame in build_htf_frames(bars_ny, config).items():
        if frame.empty:
            continue
        piv = confirmed_pivots_with_excursion(frame, left=config.pivot_left, right=config.pivot_right)
        if piv.empty:
            continue
        piv = piv.copy()
        piv["htf_timeframe"] = label
        piv["level_type"] = f"remote_{label}_swing"
        piv["liquidity_family"] = piv["level_type"]
        piv["liquidity_side"] = piv["pivot_side"].astype(str)
        piv["level_price"] = pd.to_numeric(piv["pivot_price"], errors="coerce")
        piv["source_bar_time"] = pd.to_datetime(piv["pivot_time"])
        piv["level_available_time"] = pd.to_datetime(piv["confirmation_available_time"])
        piv["tradable_level"] = True
        piv["liquidity_strength"] = "causal_unconsumed_htf_swing"
        parts.append(piv)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["level_available_time", "htf_timeframe", "liquidity_side", "level_price"], kind="mergesort").reset_index(drop=True)


def attach_first_consumption_time(bars_ny: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach the first completed 1m bar that strictly sweeps each HTF swing.

    Uses two heaps and one chronological pass through 1m bars, avoiding an
    O(pivots * history) rescan on multi-year datasets.
    """
    if catalog.empty:
        out = catalog.copy()
        out["first_consumed_time"] = pd.Series(dtype="datetime64[ns]")
        return out
    work = catalog.copy().reset_index(drop=True)
    work["level_available_time"] = pd.to_datetime(work["level_available_time"])
    order = work.sort_values("level_available_time", kind="mergesort").index.to_numpy(int)
    consumed: list[pd.Timestamp | pd.NaT] = [pd.NaT] * len(work)

    high_heap: list[tuple[float, int]] = []  # lowest high level is consumed first as price rises
    low_heap: list[tuple[float, int]] = []   # negative level => highest low level consumed first as price falls
    cursor = 0
    idx = pd.DatetimeIndex(bars_ny.index)
    available = idx + pd.Timedelta(minutes=1)
    highs = pd.to_numeric(bars_ny["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(bars_ny["low"], errors="coerce").to_numpy(float)

    for pos, now in enumerate(available):
        now_ts = pd.Timestamp(now)
        while cursor < len(order):
            ridx = int(order[cursor])
            if pd.Timestamp(work.at[ridx, "level_available_time"]) > now_ts:
                break
            level = float(work.at[ridx, "level_price"])
            side = str(work.at[ridx, "liquidity_side"])
            if side == "high":
                heapq.heappush(high_heap, (level, ridx))
            else:
                heapq.heappush(low_heap, (-level, ridx))
            cursor += 1

        hi = highs[pos]
        lo = lows[pos]
        if np.isfinite(hi):
            while high_heap and high_heap[0][0] < hi:
                _level, ridx = heapq.heappop(high_heap)
                if pd.isna(consumed[ridx]):
                    consumed[ridx] = now_ts
        if np.isfinite(lo):
            while low_heap and (-low_heap[0][0]) > lo:
                _neg_level, ridx = heapq.heappop(low_heap)
                if pd.isna(consumed[ridx]):
                    consumed[ridx] = now_ts

    work["first_consumed_time"] = pd.Series(pd.DatetimeIndex(consumed, tz=NY_TZ), index=work.index)
    return work


def _premarket_context(premarket_levels: pd.DataFrame) -> pd.DataFrame:
    if premarket_levels.empty:
        return pd.DataFrame()
    cols = [
        "ny_date", "premarket_high", "premarket_low", "premarket_range",
        "premarket_range_pct", "premarket_close", "premarket_median_15m_range",
    ]
    existing = [c for c in cols if c in premarket_levels.columns]
    ctx = premarket_levels[existing].drop_duplicates("ny_date", keep="first").copy()
    return ctx.set_index("ny_date", drop=False)


def build_remote_htf_levels_for_days(
    catalog: pd.DataFrame,
    premarket_levels: pd.DataFrame,
    days: Sequence,
) -> pd.DataFrame:
    """Materialize every HTF swing still unconsumed at each day's 08:30 anchor.

    No nearest-level shortcut is used.  This intentionally preserves old/far
    levels so their age and distance can be tested rather than assumed away.
    """
    if catalog.empty or premarket_levels.empty:
        return pd.DataFrame()
    ctx = _premarket_context(premarket_levels)
    work = catalog.copy()
    work["level_available_time"] = pd.to_datetime(work["level_available_time"], utc=True).dt.tz_convert(NY_TZ)
    raw_consumed = work.get("first_consumed_time", pd.Series(pd.NaT, index=work.index))
    if pd.Series(raw_consumed).notna().any():
        work["first_consumed_time"] = pd.to_datetime(raw_consumed, utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    else:
        work["first_consumed_time"] = pd.Series(pd.NaT, index=work.index, dtype=f"datetime64[ns, {NY_TZ}]")
    rows: list[dict[str, object]] = []

    for day in days:
        day_text = str(pd.Timestamp(day).date())
        if day_text not in ctx.index:
            continue
        anchor = pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=8, minutes=30)
        active = work.loc[
            (work["level_available_time"] <= anchor)
            & (work["first_consumed_time"].isna() | (work["first_consumed_time"] > anchor))
        ].copy()
        if active.empty:
            continue
        day_ctx = ctx.loc[day_text]
        pm_close = float(day_ctx["premarket_close"])
        for rec in active.to_dict("records"):
            px = float(rec["level_price"])
            source = pd.Timestamp(rec["source_bar_time"])
            row = dict(rec)
            row.update({
                "ny_date": day_text,
                "premarket_high": float(day_ctx["premarket_high"]),
                "premarket_low": float(day_ctx["premarket_low"]),
                "premarket_range": float(day_ctx["premarket_range"]),
                "premarket_range_pct": float(day_ctx["premarket_range_pct"]),
                "premarket_close": pm_close,
                "premarket_median_15m_range": float(day_ctx["premarket_median_15m_range"]),
                "active_at_0830": True,
                "htf_age_calendar_days": float((anchor.normalize() - source.normalize()).total_seconds() / 86400.0),
                "htf_distance_from_premarket_close_pct": abs(px / pm_close - 1.0) if abs(pm_close) > EPS else np.nan,
                "rejection_reason": "",
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["active_rank_nearest"] = (
        out.groupby(["ny_date", "htf_timeframe", "liquidity_side"])["htf_distance_from_premarket_close_pct"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return out.sort_values(["ny_date", "htf_timeframe", "liquidity_side", "active_rank_nearest"], kind="mergesort").reset_index(drop=True)


def dedupe_same_family_sweeps(sweeps: pd.DataFrame) -> pd.DataFrame:
    """One physical same-minute sweep per liquidity family.

    If one 1m bar crosses several old swings from the same timeframe, counting
    each level as a separate trade would multiply one market event.  Keep the
    deepest crossed level and record how many levels were swept together.
    """
    if sweeps.empty:
        return sweeps.copy()
    work = sweeps.copy()
    work["liquidity_family"] = work.get("liquidity_family", work["level_type"]).astype(str)
    rows: list[dict[str, object]] = []
    keys = ["ny_date", "trade_side", "sweep_time", "liquidity_family"]
    for _, g in work.groupby(keys, sort=True, dropna=False):
        side = str(g["trade_side"].iloc[0])
        px = pd.to_numeric(g["level_price"], errors="coerce")
        choose_idx = px.idxmin() if side == "LONG" else px.idxmax()
        rec = dict(work.loc[choose_idx])
        rec["same_family_levels_swept"] = int(len(g))
        rec["same_family_swept_prices"] = ",".join(f"{float(x):.8f}" for x in sorted(px.dropna().tolist()))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Exact-minute cross-timeframe confluence is diagnostic only. Family PnL
    # remains separate and therefore cannot be summed as independent trades.
    confluence_keys = ["ny_date", "trade_side", "sweep_time"]
    families = out.groupby(confluence_keys)["liquidity_family"].agg(lambda s: sorted(set(map(str, s))))
    fam_map = {k: v for k, v in families.items()}
    all_count: list[int] = []
    htf_count: list[int] = []
    names: list[str] = []
    for r in out.itertuples(index=False):
        key = (str(r.ny_date), str(r.trade_side), r.sweep_time)
        fams = fam_map.get(key, [str(r.liquidity_family)])
        all_count.append(len(fams))
        htf_count.append(sum(f.startswith("remote_") for f in fams))
        names.append("|".join(fams))
    out["liquidity_confluence_count"] = all_count
    out["htf_confluence_count"] = htf_count
    out["confluent_liquidity_families"] = names
    return out.sort_values(["sweep_time", "trade_side", "liquidity_family"], kind="mergesort").reset_index(drop=True)
