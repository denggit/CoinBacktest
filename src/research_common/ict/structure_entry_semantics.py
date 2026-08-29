#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal structure hierarchy / FVG-train primitives for SOXL ICT R12.

R12 is deliberately an *atlas*, not a final entry filter.  It addresses three
semantic gaps discovered during manual 2026-08-05 replay review:

1. A mathematical 1/1 pivot is not automatically a meaningful MSS structure.
   We keep every causal pivot but attach a continuous visibility/structure
   score and several causal reference-selection labels rather than equating
   ``latest == important``.
2. An MSS displacement can leave a train of FVGs.  FVGs whose middle candle is
   before/on the actual structure-break candle belong to the break impulse;
   later continuation FVGs are diagnosed separately instead of being chased by
   default.
3. One liquidity raid can generate more than one entry attempt.  A weak micro
   break does not terminate the episode; a later, stronger structure break may
   still be traded.

All timestamps use closed-bar availability.  No pivot, FVG or target may be
used before its ``available_time``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import (
    EPS,
    NY_TZ,
    PREMARKET_START,
    TRADE_END,
    aggregate_closed_bars,
    slice_ny_day,
)
from .premarket_mss_fvg_v2 import confirmed_pivots_with_excursion


@dataclass(frozen=True)
class StructureSemanticConfig:
    execution_timeframes: tuple[int, ...] = (1, 2, 5)
    pivot_left: int = 1
    pivot_right: int = 1
    structure_lookback_minutes: int = 150
    range_context_bars: int = 20
    absolute_entry_buffer: float = 0.10
    early_start: dtime = dtime(4, 0)
    early_end: dtime = dtime(8, 30)
    late_start: dtime = dtime(8, 30)
    late_end: dtime = dtime(9, 30)


def _day_anchor(day, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp.combine(pd.Timestamp(day).date(), dtime(hh, mm)).tz_localize(NY_TZ)


def _as_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    # Pandas 3 can retain us-resolution.  Normalise before comparing with
    # Timestamp.value (always ns) to avoid the R05 1000x bug.
    return idx.as_unit("ns").asi8


def build_dual_session_liquidity_levels(
    bars_ny: pd.DataFrame,
    days: Sequence,
    *,
    config: StructureSemanticConfig = StructureSemanticConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two independent session-liquidity families.

    ``early_premarket_extreme`` is the frozen 04:00-08:30 range used by the
    original research.  ``late_premarket_extreme`` is the 08:30-09:30 range,
    only frozen/available from 09:30 onward.  A separate running-extreme table
    is returned so a setup before 09:30 may use only the late-session high/low
    that was already visible at that time.
    """
    frozen_rows: list[dict[str, object]] = []
    running_rows: list[dict[str, object]] = []
    for day in days:
        day_text = str(pd.Timestamp(day).date())
        early = slice_ny_day(bars_ny, pd.Timestamp(day).date(), config.early_start, config.early_end)
        late = slice_ny_day(bars_ny, pd.Timestamp(day).date(), config.late_start, config.late_end)
        if not early.empty:
            hi_t = pd.Timestamp(pd.to_numeric(early["high"], errors="coerce").idxmax())
            lo_t = pd.Timestamp(pd.to_numeric(early["low"], errors="coerce").idxmin())
            hi = float(early.loc[hi_t, "high"]); lo = float(early.loc[lo_t, "low"])
            avail = _day_anchor(day, 8, 30)
            common = {
                "ny_date": day_text,
                "session_family": "early_premarket_0400_0830",
                "session_high": hi,
                "session_low": lo,
                "level_available_time": avail,
                "tradable_level": True,
            }
            frozen_rows += [
                {**common, "liquidity_family": "early_premarket_extreme", "level_type": "early_premarket_high", "liquidity_side": "high", "level_price": hi, "source_bar_time": hi_t},
                {**common, "liquidity_family": "early_premarket_extreme", "level_type": "early_premarket_low", "liquidity_side": "low", "level_price": lo, "source_bar_time": lo_t},
            ]
        if not late.empty:
            high = -np.inf; low = np.inf
            high_t = pd.NaT; low_t = pd.NaT
            for ts, row in late.iterrows():
                h = float(row["high"]); l = float(row["low"])
                avail = pd.Timestamp(ts) + pd.Timedelta(minutes=1)
                if np.isfinite(h) and h > high:
                    high = h; high_t = pd.Timestamp(ts)
                    running_rows.append({"ny_date": day_text, "liquidity_side": "high", "level_price": high, "source_bar_time": high_t, "level_available_time": avail, "liquidity_family": "late_premarket_running_extreme"})
                if np.isfinite(l) and l < low:
                    low = l; low_t = pd.Timestamp(ts)
                    running_rows.append({"ny_date": day_text, "liquidity_side": "low", "level_price": low, "source_bar_time": low_t, "level_available_time": avail, "liquidity_family": "late_premarket_running_extreme"})
            if np.isfinite(high) and np.isfinite(low):
                avail = _day_anchor(day, 9, 30)
                common = {
                    "ny_date": day_text,
                    "session_family": "late_premarket_0830_0930",
                    "session_high": float(high),
                    "session_low": float(low),
                    "level_available_time": avail,
                    "tradable_level": True,
                }
                frozen_rows += [
                    {**common, "liquidity_family": "late_premarket_extreme", "level_type": "late_premarket_high", "liquidity_side": "high", "level_price": float(high), "source_bar_time": high_t},
                    {**common, "liquidity_family": "late_premarket_extreme", "level_type": "late_premarket_low", "liquidity_side": "low", "level_price": float(low), "source_bar_time": low_t},
                ]
    frozen = pd.DataFrame(frozen_rows)
    running = pd.DataFrame(running_rows)
    return frozen, running


def build_visible_swing_catalog(
    bars_ny: pd.DataFrame,
    days: Sequence,
    *,
    config: StructureSemanticConfig = StructureSemanticConfig(),
) -> pd.DataFrame:
    """Build causal low-timeframe pivots with continuous structural visibility.

    The catalogue intentionally keeps tiny pivots.  ``visibility_score`` and
    expanding causal percentile describe how obvious a pivot was relative to
    already-known local structure; they do not gate entry in R12.
    """
    rows: list[pd.DataFrame] = []
    for day in days:
        day_1m = slice_ny_day(bars_ny, pd.Timestamp(day).date(), PREMARKET_START, TRADE_END)
        if day_1m.empty:
            continue
        for tf in config.execution_timeframes:
            frame = aggregate_closed_bars(day_1m, int(tf))
            if frame.empty:
                continue
            piv = confirmed_pivots_with_excursion(frame, left=config.pivot_left, right=config.pivot_right)
            if piv.empty:
                continue
            ranges = pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")
            med = ranges.shift(1).rolling(config.range_context_bars, min_periods=max(5, config.range_context_bars // 4)).median()
            idx = pd.DatetimeIndex(frame.index)
            med_by_time = pd.Series(med.to_numpy(float), index=idx)
            vals = []
            for r in piv.to_dict("records"):
                ppos = int(r["pivot_pos"])
                conf = pd.Timestamp(r["confirmation_available_time"])
                # Context is based on bars strictly before the pivot bar.  If the
                # rolling value is unavailable, fall back to all completed bars
                # before the pivot, still causal.
                m = float(med.iloc[ppos]) if ppos < len(med) and np.isfinite(med.iloc[ppos]) else np.nan
                if not np.isfinite(m) or m <= EPS:
                    prior = ranges.iloc[:ppos]
                    m = float(prior.median()) if len(prior) and np.isfinite(prior.median()) else np.nan
                exc = float(r.get("two_sided_excursion_abs", np.nan))
                prom = float(r.get("local_prominence_abs", np.nan))
                exc_mult = exc / m if np.isfinite(exc) and np.isfinite(m) and m > EPS else np.nan
                prom_mult = prom / m if np.isfinite(prom) and np.isfinite(m) and m > EPS else np.nan
                # Continuous score only; no hard "this is an ITH" claim.
                score = (
                    np.log1p(max(exc_mult, 0.0)) + 0.5 * np.log1p(max(prom_mult, 0.0))
                    if np.isfinite(exc_mult) and np.isfinite(prom_mult) else np.nan
                )
                vals.append({**r, "ny_date": str(pd.Timestamp(day).date()), "execution_tf": f"{int(tf)}m", "execution_tf_minutes": int(tf), "prior_median_range_abs": m, "two_sided_excursion_vs_prior_range": exc_mult, "local_prominence_vs_prior_range": prom_mult, "visibility_score": score})
            p = pd.DataFrame(vals).sort_values(["confirmation_available_time", "pivot_time"], kind="mergesort")
            # Expanding causal percentile among same-side pivots confirmed so far.
            percentiles = []
            history = {"high": [], "low": []}
            for r in p.to_dict("records"):
                side = str(r["pivot_side"]); score = float(r["visibility_score"]) if np.isfinite(r["visibility_score"]) else np.nan
                hist = history[side]
                if np.isfinite(score):
                    arr = np.asarray(hist + [score], dtype=float)
                    pct = float((arr <= score).mean())
                    hist.append(score)
                else:
                    pct = np.nan
                percentiles.append(pct)
            p["causal_visibility_percentile"] = percentiles
            rows.append(p)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False).sort_values(["confirmation_available_time", "execution_tf", "pivot_side", "pivot_time"], kind="mergesort").reset_index(drop=True)


def _fvg_rows(frame: pd.DataFrame, *, is_long: bool) -> list[dict[str, object]]:
    if frame.empty:
        return []
    h = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    l = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    idx = pd.DatetimeIndex(frame.index)
    av = pd.DatetimeIndex(pd.to_datetime(frame["available_time"]))
    out: list[dict[str, object]] = []
    for k in range(2, len(frame)):
        valid = (np.isfinite(l[k]) and np.isfinite(h[k-2]) and l[k] > h[k-2]) if is_long else (np.isfinite(h[k]) and np.isfinite(l[k-2]) and h[k] < l[k-2])
        if not valid:
            continue
        near = float(l[k] if is_long else h[k])
        far = float(h[k-2] if is_long else l[k-2])
        out.append({"fvg_first_pos": k-2, "fvg_middle_pos": k-1, "fvg_third_pos": k, "fvg_first_bar_start": idx[k-2], "fvg_middle_bar_start": idx[k-1], "fvg_third_bar_start": idx[k], "fvg_available_time": av[k], "fvg_near_edge_entry": near, "fvg_far_edge": far, "fvg_size_abs": abs(near-far)})
    return out


def _nearest_active_internal_target(
    swing_catalog_day_tf: pd.DataFrame,
    *,
    is_long: bool,
    signal_time: pd.Timestamp,
    entry_price: float,
    exec_frame: pd.DataFrame,
) -> tuple[float, pd.Timestamp | pd.NaT, str]:
    if swing_catalog_day_tf.empty:
        return np.nan, pd.NaT, ""
    side = "high" if is_long else "low"
    p = swing_catalog_day_tf.loc[
        (swing_catalog_day_tf["pivot_side"] == side)
        & (pd.to_datetime(swing_catalog_day_tf["confirmation_available_time"]) <= signal_time)
    ].copy()
    if p.empty:
        return np.nan, pd.NaT, ""
    if is_long:
        p = p.loc[p["pivot_price"].astype(float) > entry_price]
        p = p.sort_values("pivot_price", ascending=True)
    else:
        p = p.loc[p["pivot_price"].astype(float) < entry_price]
        p = p.sort_values("pivot_price", ascending=False)
    if p.empty:
        return np.nan, pd.NaT, ""
    # Reject pivots already crossed after their confirmation and before signal.
    for r in p.to_dict("records"):
        conf = pd.Timestamp(r["confirmation_available_time"])
        path = exec_frame.loc[(pd.to_datetime(exec_frame["available_time"]) > conf) & (pd.to_datetime(exec_frame["available_time"]) <= signal_time)]
        price = float(r["pivot_price"])
        if is_long:
            broken = bool((pd.to_numeric(path["high"], errors="coerce") > price).any()) if not path.empty else False
        else:
            broken = bool((pd.to_numeric(path["low"], errors="coerce") < price).any()) if not path.empty else False
        if not broken:
            return price, pd.Timestamp(r["confirmation_available_time"]), "nearest_active_internal_swing"
    return np.nan, pd.NaT, ""



def _prepare_internal_target_index(
    exec_frame: pd.DataFrame,
    swings: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Precompute causal active-swing target state for fast exact queries.

    This preserves ``_nearest_active_internal_target`` semantics: a target pivot
    is usable once confirmed, and becomes consumed only after a later
    execution bar (strictly after confirmation) crosses its price.
    """
    if exec_frame.empty or swings.empty:
        return {"high": {}, "low": {}}
    av = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
    av_ns = _as_ns(av)
    hi = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
    out: dict[str, dict[str, object]] = {}
    for side in ("high", "low"):
        rows = swings.loc[swings["pivot_side"].eq(side)].to_dict("records")
        prices: list[float] = []
        conf_ns_list: list[int] = []
        break_ns_list: list[int] = []
        conf_ts_list: list[pd.Timestamp] = []
        for r in rows:
            price = float(r["pivot_price"])
            conf = pd.Timestamp(r["confirmation_available_time"])
            conf_ns = int(conf.value)
            # Legacy target logic checks available_time > confirmation_time.
            start = int(np.searchsorted(av_ns, conf_ns, side="right"))
            break_ns = np.iinfo(np.int64).max
            if start < len(exec_frame):
                mask = hi[start:] > price if side == "high" else lo[start:] < price
                hits = np.flatnonzero(mask)
                if len(hits):
                    break_ns = int(av_ns[start + int(hits[0])])
            prices.append(price)
            conf_ns_list.append(conf_ns)
            break_ns_list.append(break_ns)
            conf_ts_list.append(conf)
        out[side] = {
            "prices": np.asarray(prices, dtype=float),
            "conf_ns": np.asarray(conf_ns_list, dtype=np.int64),
            "break_ns": np.asarray(break_ns_list, dtype=np.int64),
            "conf_ts": conf_ts_list,
        }
    return out


def _query_internal_target_index(
    index: dict[str, dict[str, object]],
    *,
    is_long: bool,
    signal_time: pd.Timestamp,
    entry_price: float,
) -> tuple[float, pd.Timestamp | pd.NaT, str]:
    side = "high" if is_long else "low"
    data = index.get(side) or {}
    prices = data.get("prices")
    if prices is None or len(prices) == 0:
        return np.nan, pd.NaT, ""
    prices = np.asarray(prices, dtype=float)
    conf_ns = np.asarray(data["conf_ns"], dtype=np.int64)
    break_ns = np.asarray(data["break_ns"], dtype=np.int64)
    sig_ns = int(signal_time.value)
    active = (conf_ns <= sig_ns) & (break_ns > sig_ns)
    active &= prices > float(entry_price) if is_long else prices < float(entry_price)
    idxs = np.flatnonzero(active)
    if not len(idxs):
        return np.nan, pd.NaT, ""
    vals = prices[idxs]
    best_value = float(np.nanmin(vals) if is_long else np.nanmax(vals))
    # Preserve catalog order for equal-price ties, matching the legacy loop's
    # first surviving candidate after price ordering.
    best_candidates = idxs[np.flatnonzero(np.isclose(vals, best_value, rtol=0.0, atol=EPS))]
    best_i = int(best_candidates[0])
    return best_value, data["conf_ts"][best_i], "nearest_active_internal_swing"

def _first_valid_break_pos_by_pivot(
    exec_frame: pd.DataFrame,
    swings: pd.DataFrame,
    *,
    pivot_side: str,
) -> dict[str, int]:
    """Return each pivot's first causal wick-break execution position.

    This is sweep-independent.  The old R12 implementation rediscovered the
    same answer inside every ``sweep x bar x pivot`` loop and repeatedly sliced
    the whole execution frame.  Precomputing it preserves semantics while
    removing the dominant O(sweeps * bars * pivots * bars) path.
    """
    if exec_frame.empty or swings.empty:
        return {}
    av = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
    av_ns = _as_ns(av)
    hi = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
    out: dict[str, int] = {}
    for r in swings.loc[swings["pivot_side"].eq(pivot_side)].to_dict("records"):
        conf = pd.Timestamp(r["confirmation_available_time"])
        start = int(np.searchsorted(av_ns, int(conf.value), side="left"))
        if start >= len(exec_frame):
            continue
        price = float(r["pivot_price"])
        mask = hi[start:] > price if pivot_side == "high" else lo[start:] < price
        hits = np.flatnonzero(mask)
        if not len(hits):
            continue
        pid = f"{r['pivot_time']}|{price:.8f}|{int(r['execution_tf_minutes'])}|{pivot_side}"
        out[pid] = start + int(hits[0])
    return out


def _terminal_path_from_sweep(
    one_av: pd.DatetimeIndex,
    one_h: np.ndarray,
    one_l: np.ndarray,
    *,
    sweep_time: pd.Timestamp,
    is_long: bool,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Causal terminal extreme arrays from one sweep onward.

    Returns ``(one_start, terminal_price, terminal_index, terminal_version)``.
    Equal extrema do not advance terminal time/version, matching the previous
    strict ``<`` / ``>`` update behavior exactly.
    """
    one_ns = _as_ns(one_av)
    one_start = int(np.searchsorted(one_ns, int(sweep_time.value), side="left"))
    if one_start >= len(one_av):
        return one_start, np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src = one_l[one_start:] if is_long else one_h[one_start:]
    n = len(src)
    price = np.full(n, np.nan, dtype=float)
    idx = np.full(n, -1, dtype=np.int64)
    version = np.zeros(n, dtype=np.int64)
    best = np.nan
    best_i = -1
    ver = 0
    for i, px in enumerate(src):
        changed = np.isfinite(px) and (
            (not np.isfinite(best)) or (px < best if is_long else px > best)
        )
        if changed:
            best = float(px)
            best_i = i
            ver += 1
        price[i] = best
        idx[i] = best_i
        version[i] = ver
    return one_start, price, idx, version


def build_structure_break_fvg_atlas(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    swing_catalog: pd.DataFrame,
    *,
    config: StructureSemanticConfig = StructureSemanticConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate multiple causal MSS candidates and their associated FVG train.

    Semantics are unchanged from R12, but the implementation is algorithmically
    different for long-history / golden-replay speed:

    * each swing's first causal wick break is precomputed once per day/timeframe;
    * terminal extreme/version is accumulated once per sweep on the 1m path;
    * directional FVGs are built once per day/timeframe/direction;
    * only actual first-break positions are visited (not every bar x every pivot).

    One sweep may still emit multiple structure-break candidates and multiple
    FVG-train entries.  No visibility or entry-distance filter is introduced.
    """
    if sweeps.empty or swing_catalog.empty:
        return pd.DataFrame(), pd.DataFrame()
    break_rows: list[dict[str, object]] = []
    fvg_rows_out: list[dict[str, object]] = []
    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, PREMARKET_START, TRADE_END)
        if day_1m.empty:
            continue
        one_idx = pd.DatetimeIndex(day_1m.index)
        one_av = one_idx + pd.Timedelta(minutes=1)
        one_ns = _as_ns(one_av)
        one_h = pd.to_numeric(day_1m["high"], errors="coerce").to_numpy(float)
        one_l = pd.to_numeric(day_1m["low"], errors="coerce").to_numpy(float)

        for tf in config.execution_timeframes:
            exec_frame = aggregate_closed_bars(day_1m, int(tf))
            if exec_frame.empty:
                continue
            idx = pd.DatetimeIndex(exec_frame.index)
            av = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
            av_ns = _as_ns(av)
            o = pd.to_numeric(exec_frame["open"], errors="coerce").to_numpy(float)
            h = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
            l = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
            c = pd.to_numeric(exec_frame["close"], errors="coerce").to_numpy(float)

            tf_swings = swing_catalog.loc[
                (swing_catalog["ny_date"].astype(str) == str(day_text))
                & (swing_catalog["execution_tf_minutes"].astype(int) == int(tf))
            ].copy()
            if tf_swings.empty:
                continue

            # Sweep-independent caches.
            first_break_high = _first_valid_break_pos_by_pivot(exec_frame, tf_swings, pivot_side="high")
            first_break_low = _first_valid_break_pos_by_pivot(exec_frame, tf_swings, pivot_side="low")
            fvg_long = _fvg_rows(exec_frame, is_long=True)
            fvg_short = _fvg_rows(exec_frame, is_long=False)
            # Precompute exact causal active-target state once.  Real R12
            # golden sessions can emit thousands of break x FVG combinations;
            # scanning/sorting the swing DataFrame for each row dominated Stage 4.
            internal_target_index = _prepare_internal_target_index(exec_frame, tf_swings)

            # Avoid repeated string/datetime conversions inside the sweep loop.
            swing_records_by_side: dict[str, list[dict[str, object]]] = {"high": [], "low": []}
            for r in tf_swings.to_dict("records"):
                rr = dict(r)
                rr["_pivot_time_ts"] = pd.Timestamp(rr["pivot_time"])
                rr["_confirm_ts"] = pd.Timestamp(rr["confirmation_available_time"])
                side = str(rr["pivot_side"])
                rr["_pivot_id"] = f"{rr['pivot_time']}|{float(rr['pivot_price']):.8f}|{int(tf)}|{side}"
                swing_records_by_side[side].append(rr)

            for sweep in day_sweeps.to_dict("records"):
                if not bool(sweep.get("setup_eligible_at_sweep", True)):
                    continue
                sweep_time = pd.Timestamp(sweep["sweep_time"])
                is_long = str(sweep["trade_side"]) == "LONG"
                ref_side = "high" if is_long else "low"
                first_pos = int(np.searchsorted(av_ns, int(sweep_time.value), side="right"))
                if first_pos >= len(exec_frame):
                    continue

                one_start, term_price_path, term_idx_path, term_ver_path = _terminal_path_from_sweep(
                    one_av, one_h, one_l, sweep_time=sweep_time, is_long=is_long
                )
                if not len(term_price_path):
                    continue

                first_break_map = first_break_high if is_long else first_break_low
                lookback_start = sweep_time - pd.Timedelta(minutes=int(config.structure_lookback_minutes))

                # Only pivots whose FIRST valid causal break is after this sweep
                # can be active MSS barriers for this episode.
                by_break_pos: dict[int, list[dict[str, object]]] = {}
                for r in swing_records_by_side[ref_side]:
                    if r["_pivot_time_ts"] < lookback_start:
                        continue
                    bpos = first_break_map.get(str(r["_pivot_id"]))
                    if bpos is None or int(bpos) < first_pos:
                        continue
                    by_break_pos.setdefault(int(bpos), []).append(r)
                if not by_break_pos:
                    continue

                fvg_all = fvg_long if is_long else fvg_short
                for pos in sorted(by_break_pos):
                    now = pd.Timestamp(av[pos])
                    if now > _day_anchor(day, 16, 30):
                        break

                    # Terminal state available by this execution-bar close.
                    end_j = int(np.searchsorted(one_ns, int(now.value), side="right") - 1)
                    rel = end_j - one_start
                    if rel < 0:
                        continue
                    rel = min(rel, len(term_price_path) - 1)
                    terminal_price = float(term_price_path[rel])
                    ti = int(term_idx_path[rel])
                    if not np.isfinite(terminal_price) or ti < 0:
                        continue
                    terminal_time = pd.Timestamp(one_av[one_start + ti])
                    terminal_version = int(term_ver_path[rel])

                    newly: list[dict[str, object]] = []
                    for r in by_break_pos[pos]:
                        # Same availability check as old loop's p-selection.
                        if r["_confirm_ts"] > now:
                            continue
                        price = float(r["pivot_price"])
                        if is_long and not (price > terminal_price):
                            continue
                        if (not is_long) and not (price < terminal_price):
                            continue
                        wick_break = bool(h[pos] > price) if is_long else bool(l[pos] < price)
                        if not wick_break:  # defensive: should hold by cache construction
                            continue
                        close_break = bool(c[pos] > price) if is_long else bool(c[pos] < price)
                        rr = dict(r)
                        rr["pivot_id"] = str(r["_pivot_id"])
                        rr["wick_break"] = wick_break
                        rr["close_break"] = close_break
                        newly.append(rr)
                    if not newly:
                        continue

                    latest_id = max(newly, key=lambda x: pd.Timestamp(x["pivot_time"]))["pivot_id"]
                    score_candidates = [x for x in newly if np.isfinite(float(x.get("visibility_score", np.nan)))]
                    best_score_id = max(score_candidates, key=lambda x: float(x["visibility_score"]))["pivot_id"] if score_candidates else latest_id
                    barrier_id = (max(newly, key=lambda x: float(x["pivot_price"])) if is_long else min(newly, key=lambda x: float(x["pivot_price"]))) ["pivot_id"]

                    # Candidate FVGs for this break are filtered from a cached
                    # directional list, not rebuilt from the whole day.
                    train_for_break = [
                        f for f in fvg_all
                        if int(f["fvg_middle_pos"]) <= pos
                        and int(f["fvg_third_pos"]) <= pos + 1
                        and pd.Timestamp(f["fvg_available_time"]) >= terminal_time
                    ]
                    break_middle = [f for f in train_for_break if int(f["fvg_middle_pos"]) == pos]
                    break_cap_entry = float(break_middle[-1]["fvg_near_edge_entry"]) if break_middle else np.nan

                    for r in newly:
                        pid = str(r["pivot_id"])
                        ref_price = float(r["pivot_price"])
                        relation = "post_terminal" if pd.Timestamp(r["pivot_time"]) >= terminal_time else ("post_sweep_pre_terminal" if pd.Timestamp(r["pivot_time"]) >= sweep_time else "pre_sweep")
                        break_row = {
                            **sweep,
                            "execution_tf": f"{int(tf)}m", "execution_tf_minutes": int(tf),
                            "terminal_version": terminal_version,
                            "terminal_extreme_time": terminal_time,
                            "terminal_extreme_price": terminal_price,
                            "mss_reference_time": pd.Timestamp(r["pivot_time"]),
                            "mss_reference_price": ref_price,
                            "mss_reference_available_time": pd.Timestamp(r["confirmation_available_time"]),
                            "mss_reference_relation": relation,
                            "visibility_score": r.get("visibility_score", np.nan),
                            "causal_visibility_percentile": r.get("causal_visibility_percentile", np.nan),
                            "two_sided_excursion_vs_prior_range": r.get("two_sided_excursion_vs_prior_range", np.nan),
                            "local_prominence_vs_prior_range": r.get("local_prominence_vs_prior_range", np.nan),
                            "reference_is_latest_newly_broken": pid == latest_id,
                            "reference_is_highest_visibility_newly_broken": pid == best_score_id,
                            "reference_is_outermost_barrier_newly_broken": pid == barrier_id,
                            "break_bar_start": pd.Timestamp(idx[pos]),
                            "break_available_time": now,
                            "break_wick_cross": bool(r["wick_break"]),
                            "break_close_cross": bool(r["close_break"]),
                            "break_open": float(o[pos]), "break_high": float(h[pos]), "break_low": float(l[pos]), "break_close": float(c[pos]),
                        }
                        tpos = int(np.searchsorted(av_ns, int(terminal_time.value), side="left"))
                        tpos = max(0, min(tpos, pos))
                        signed = (c[tpos:pos+1] - o[tpos:pos+1]) * (1.0 if is_long else -1.0)
                        break_row["terminal_to_break_minutes"] = float((now - terminal_time).total_seconds() / 60.0)
                        break_row["directional_bar_fraction"] = float((signed > 0).mean()) if len(signed) else np.nan
                        break_row["path_net_distance_abs"] = abs(float(c[pos]) - terminal_price)
                        travel = float(np.nansum(np.abs(np.diff(np.r_[terminal_price, c[tpos:pos+1]]))))
                        break_row["path_efficiency"] = float(break_row["path_net_distance_abs"] / travel) if travel > EPS else np.nan
                        break_row["break_overshoot_abs"] = float(c[pos] - ref_price) if is_long else float(ref_price - c[pos])
                        break_rows.append(break_row)

                        for seq, f in enumerate(train_for_break, start=1):
                            entry = float(f["fvg_near_edge_entry"])
                            swing_cap = ref_price + config.absolute_entry_buffer if is_long else ref_price - config.absolute_entry_buffer
                            swing_pass = entry <= swing_cap + EPS if is_long else entry >= swing_cap - EPS
                            break_cap = break_cap_entry + config.absolute_entry_buffer if is_long and np.isfinite(break_cap_entry) else (break_cap_entry - config.absolute_entry_buffer if (not is_long and np.isfinite(break_cap_entry)) else np.nan)
                            break_pass = (entry <= break_cap + EPS if is_long else entry >= break_cap - EPS) if np.isfinite(break_cap) else False
                            signal_time = max(now, pd.Timestamp(f["fvg_available_time"]))
                            internal_target, internal_avail, internal_src = _query_internal_target_index(
                                internal_target_index, is_long=is_long, signal_time=signal_time, entry_price=entry
                            )
                            fvg_rows_out.append({
                                **break_row, **f,
                                "fvg_train_sequence": int(seq),
                                "fvg_middle_relation_to_break": "break_bar_middle" if int(f["fvg_middle_pos"]) == pos else "pre_break_middle",
                                "entry_distance_from_broken_swing": float(entry-ref_price) if is_long else float(ref_price-entry),
                                "swing_buffer_cap_price": float(swing_cap),
                                "swing_buffer_cap_pass": bool(swing_pass),
                                "break_middle_fvg_cap_base_entry": break_cap_entry,
                                "break_middle_fvg_buffer_cap_price": break_cap,
                                "break_middle_fvg_buffer_cap_pass": bool(break_pass),
                                "signal_time": signal_time,
                                "stop_price": terminal_price,
                                "nearest_internal_target_price": internal_target,
                                "nearest_internal_target_available_time": internal_avail,
                                "nearest_internal_target_source": internal_src,
                            })
    return pd.DataFrame(break_rows), pd.DataFrame(fvg_rows_out)


def build_r13_primary_break_fvg_compact(
    bars_ny: pd.DataFrame,
    sweeps: pd.DataFrame,
    swing_catalog: pd.DataFrame,
    *,
    config: StructureSemanticConfig = StructureSemanticConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memory-bounded R13 Stage-4 semantic scan.

    R12's wide atlas intentionally emits every newly-broken pivot x every
    eligible FVG in a displacement train.  That is useful for a one-day golden
    replay, but on multi-year data the Cartesian expansion can reach millions
    of wide Python dictionaries and exhaust RAM before R13 gets a chance to
    consolidate them.

    R13's actual replay semantics are much smaller and were already fixed
    *before* seeing PnL: it uses the outermost newly-broken barrier, then keeps
    the first break in each causal visibility tier for each terminal version.
    For each retained narrative, only the pre-declared FVG execution choices
    are needed (first train, last pre/on-break, break-middle, closest to the
    broken swing).  This function performs exactly that causal consolidation
    *during* the scan, rather than materialising the discarded Cartesian rows.

    Returns
    -------
    primary_breaks:
        Exact R13 outermost/tiered primary MSS narratives.
    compact_fvgs:
        Union of FVG rows required to reproduce R13's fixed entry selectors.
        No PnL-based filtering is used.
    audit:
        Per-day/timeframe counts including how many wide rows were avoided.
    """
    if sweeps.empty or swing_catalog.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    primary_rows: list[dict[str, object]] = []
    compact_fvg_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for day_text, day_sweeps in sweeps.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_1m = slice_ny_day(bars_ny, day, PREMARKET_START, TRADE_END)
        if day_1m.empty:
            continue
        one_idx = pd.DatetimeIndex(day_1m.index)
        one_av = one_idx + pd.Timedelta(minutes=1)
        one_ns = _as_ns(one_av)
        one_h = pd.to_numeric(day_1m["high"], errors="coerce").to_numpy(float)
        one_l = pd.to_numeric(day_1m["low"], errors="coerce").to_numpy(float)

        for tf in config.execution_timeframes:
            exec_frame = aggregate_closed_bars(day_1m, int(tf))
            if exec_frame.empty:
                continue
            idx = pd.DatetimeIndex(exec_frame.index)
            av = pd.DatetimeIndex(pd.to_datetime(exec_frame["available_time"]))
            av_ns = _as_ns(av)
            o = pd.to_numeric(exec_frame["open"], errors="coerce").to_numpy(float)
            h = pd.to_numeric(exec_frame["high"], errors="coerce").to_numpy(float)
            l = pd.to_numeric(exec_frame["low"], errors="coerce").to_numpy(float)
            c = pd.to_numeric(exec_frame["close"], errors="coerce").to_numpy(float)

            tf_swings = swing_catalog.loc[
                (swing_catalog["ny_date"].astype(str) == str(day_text))
                & (swing_catalog["execution_tf_minutes"].astype(int) == int(tf))
            ].copy()
            if tf_swings.empty:
                continue

            first_break_high = _first_valid_break_pos_by_pivot(exec_frame, tf_swings, pivot_side="high")
            first_break_low = _first_valid_break_pos_by_pivot(exec_frame, tf_swings, pivot_side="low")
            fvg_long = _fvg_rows(exec_frame, is_long=True)
            fvg_short = _fvg_rows(exec_frame, is_long=False)
            internal_target_index = _prepare_internal_target_index(exec_frame, tf_swings)

            swing_records_by_side: dict[str, list[dict[str, object]]] = {"high": [], "low": []}
            for r in tf_swings.to_dict("records"):
                rr = dict(r)
                rr["_pivot_time_ts"] = pd.Timestamp(rr["pivot_time"])
                rr["_confirm_ts"] = pd.Timestamp(rr["confirmation_available_time"])
                side = str(rr["pivot_side"])
                rr["_pivot_id"] = f"{rr['pivot_time']}|{float(rr['pivot_price']):.8f}|{int(tf)}|{side}"
                swing_records_by_side[side].append(rr)

            raw_break_rows = 0
            raw_fvg_rows = 0
            emitted_primary = 0
            emitted_compact_fvg = 0

            for sweep in day_sweeps.to_dict("records"):
                if not bool(sweep.get("setup_eligible_at_sweep", True)):
                    continue
                sweep_time = pd.Timestamp(sweep["sweep_time"])
                is_long = str(sweep["trade_side"]) == "LONG"
                ref_side = "high" if is_long else "low"
                first_pos = int(np.searchsorted(av_ns, int(sweep_time.value), side="right"))
                if first_pos >= len(exec_frame):
                    continue

                one_start, term_price_path, term_idx_path, term_ver_path = _terminal_path_from_sweep(
                    one_av, one_h, one_l, sweep_time=sweep_time, is_long=is_long
                )
                if not len(term_price_path):
                    continue

                first_break_map = first_break_high if is_long else first_break_low
                lookback_start = sweep_time - pd.Timedelta(minutes=int(config.structure_lookback_minutes))
                by_break_pos: dict[int, list[dict[str, object]]] = {}
                for r in swing_records_by_side[ref_side]:
                    if r["_pivot_time_ts"] < lookback_start:
                        continue
                    bpos = first_break_map.get(str(r["_pivot_id"]))
                    if bpos is None or int(bpos) < first_pos:
                        continue
                    by_break_pos.setdefault(int(bpos), []).append(r)
                if not by_break_pos:
                    continue

                fvg_all = fvg_long if is_long else fvg_short
                # Exact equivalent of select_tiered_primary_narratives(): one
                # first outermost-barrier break per terminal-version / causal
                # visibility tier for this event/timeframe.
                seen_terminal_tiers: set[tuple[int, str]] = set()

                for pos in sorted(by_break_pos):
                    now = pd.Timestamp(av[pos])
                    if now > _day_anchor(day, 16, 30):
                        break
                    end_j = int(np.searchsorted(one_ns, int(now.value), side="right") - 1)
                    rel = end_j - one_start
                    if rel < 0:
                        continue
                    rel = min(rel, len(term_price_path) - 1)
                    terminal_price = float(term_price_path[rel])
                    ti = int(term_idx_path[rel])
                    if not np.isfinite(terminal_price) or ti < 0:
                        continue
                    terminal_time = pd.Timestamp(one_av[one_start + ti])
                    terminal_version = int(term_ver_path[rel])

                    newly: list[dict[str, object]] = []
                    for r in by_break_pos[pos]:
                        if r["_confirm_ts"] > now:
                            continue
                        price = float(r["pivot_price"])
                        if is_long and not (price > terminal_price):
                            continue
                        if (not is_long) and not (price < terminal_price):
                            continue
                        wick_break = bool(h[pos] > price) if is_long else bool(l[pos] < price)
                        if not wick_break:
                            continue
                        close_break = bool(c[pos] > price) if is_long else bool(c[pos] < price)
                        rr = dict(r)
                        rr["pivot_id"] = str(r["_pivot_id"])
                        rr["wick_break"] = wick_break
                        rr["close_break"] = close_break
                        newly.append(rr)
                    if not newly:
                        continue

                    raw_break_rows += len(newly)
                    # R13's primary path uses only the outermost newly-broken
                    # barrier.  Keep exactly that row instead of all pivots.
                    r = max(newly, key=lambda x: float(x["pivot_price"])) if is_long else min(newly, key=lambda x: float(x["pivot_price"]))
                    ref_price = float(r["pivot_price"])
                    pct = float(r.get("causal_visibility_percentile", np.nan)) if np.isfinite(r.get("causal_visibility_percentile", np.nan)) else np.nan
                    if not np.isfinite(pct):
                        tier = "unknown"
                    elif pct < 0.50:
                        tier = "micro_lt_p50"
                    elif pct < 0.80:
                        tier = "visible_p50_p80"
                    else:
                        tier = "strong_ge_p80"
                    tier_key = (terminal_version, tier)

                    train_for_break = [
                        f for f in fvg_all
                        if int(f["fvg_middle_pos"]) <= pos
                        and int(f["fvg_third_pos"]) <= pos + 1
                        and pd.Timestamp(f["fvg_available_time"]) >= terminal_time
                    ]
                    # Count the R12-wide Cartesian rows that would have been
                    # materialised.  This is diagnostics only.
                    raw_fvg_rows += len(newly) * len(train_for_break)

                    if tier_key in seen_terminal_tiers:
                        continue
                    seen_terminal_tiers.add(tier_key)

                    relation = "post_terminal" if pd.Timestamp(r["pivot_time"]) >= terminal_time else ("post_sweep_pre_terminal" if pd.Timestamp(r["pivot_time"]) >= sweep_time else "pre_sweep")
                    break_row = {
                        **sweep,
                        "execution_tf": f"{int(tf)}m", "execution_tf_minutes": int(tf),
                        "terminal_version": terminal_version,
                        "terminal_extreme_time": terminal_time,
                        "terminal_extreme_price": terminal_price,
                        "mss_reference_time": pd.Timestamp(r["pivot_time"]),
                        "mss_reference_price": ref_price,
                        "mss_reference_available_time": pd.Timestamp(r["confirmation_available_time"]),
                        "mss_reference_relation": relation,
                        "visibility_score": r.get("visibility_score", np.nan),
                        "causal_visibility_percentile": r.get("causal_visibility_percentile", np.nan),
                        "two_sided_excursion_vs_prior_range": r.get("two_sided_excursion_vs_prior_range", np.nan),
                        "local_prominence_vs_prior_range": r.get("local_prominence_vs_prior_range", np.nan),
                        "reference_is_latest_newly_broken": False,
                        "reference_is_highest_visibility_newly_broken": False,
                        "reference_is_outermost_barrier_newly_broken": True,
                        "break_bar_start": pd.Timestamp(idx[pos]),
                        "break_available_time": now,
                        "break_wick_cross": bool(r["wick_break"]),
                        "break_close_cross": bool(r["close_break"]),
                        "break_open": float(o[pos]), "break_high": float(h[pos]), "break_low": float(l[pos]), "break_close": float(c[pos]),
                        "reference_model_r13": "outermost_barrier_tiered_primary",
                        "structure_visibility_tier_r13": tier,
                    }
                    tpos = int(np.searchsorted(av_ns, int(terminal_time.value), side="left"))
                    tpos = max(0, min(tpos, pos))
                    signed = (c[tpos:pos+1] - o[tpos:pos+1]) * (1.0 if is_long else -1.0)
                    break_row["terminal_to_break_minutes"] = float((now - terminal_time).total_seconds() / 60.0)
                    break_row["directional_bar_fraction"] = float((signed > 0).mean()) if len(signed) else np.nan
                    break_row["path_net_distance_abs"] = abs(float(c[pos]) - terminal_price)
                    travel = float(np.nansum(np.abs(np.diff(np.r_[terminal_price, c[tpos:pos+1]]))))
                    break_row["path_efficiency"] = float(break_row["path_net_distance_abs"] / travel) if travel > EPS else np.nan
                    break_row["break_overshoot_abs"] = float(c[pos] - ref_price) if is_long else float(ref_price - c[pos])
                    break_row["narrative_attempt_sequence_r13"] = int(emitted_primary + 1)  # overwritten below per event/tf after concat
                    primary_rows.append(break_row)
                    emitted_primary += 1

                    if not train_for_break:
                        continue
                    break_middle = [(j, f) for j, f in enumerate(train_for_break, start=1) if int(f["fvg_middle_pos"]) == pos]
                    break_cap_entry = float(break_middle[-1][1]["fvg_near_edge_entry"]) if break_middle else np.nan
                    closest_j, closest_f = min(
                        enumerate(train_for_break, start=1),
                        key=lambda jf: abs(float(jf[1]["fvg_near_edge_entry"]) - ref_price),
                    )
                    selected: dict[int, dict[str, object]] = {
                        1: train_for_break[0],
                        len(train_for_break): train_for_break[-1],
                        int(closest_j): closest_f,
                    }
                    if break_middle:
                        selected[int(break_middle[-1][0])] = break_middle[-1][1]

                    for seq in sorted(selected):
                        f = selected[seq]
                        entry = float(f["fvg_near_edge_entry"])
                        swing_cap = ref_price + config.absolute_entry_buffer if is_long else ref_price - config.absolute_entry_buffer
                        swing_pass = entry <= swing_cap + EPS if is_long else entry >= swing_cap - EPS
                        break_cap = break_cap_entry + config.absolute_entry_buffer if is_long and np.isfinite(break_cap_entry) else (break_cap_entry - config.absolute_entry_buffer if (not is_long and np.isfinite(break_cap_entry)) else np.nan)
                        break_pass = (entry <= break_cap + EPS if is_long else entry >= break_cap - EPS) if np.isfinite(break_cap) else False
                        signal_time = max(now, pd.Timestamp(f["fvg_available_time"]))
                        internal_target, internal_avail, internal_src = _query_internal_target_index(
                            internal_target_index, is_long=is_long, signal_time=signal_time, entry_price=entry
                        )
                        fvg_break_context = {k: v for k, v in break_row.items() if k != "reference_model_r13"}
                        compact_fvg_rows.append({
                            **fvg_break_context, **f,
                            "fvg_train_sequence": int(seq),
                            "fvg_middle_relation_to_break": "break_bar_middle" if int(f["fvg_middle_pos"]) == pos else "pre_break_middle",
                            "entry_distance_from_broken_swing": float(entry-ref_price) if is_long else float(ref_price-entry),
                            "swing_buffer_cap_price": float(swing_cap),
                            "swing_buffer_cap_pass": bool(swing_pass),
                            "break_middle_fvg_cap_base_entry": break_cap_entry,
                            "break_middle_fvg_buffer_cap_price": break_cap,
                            "break_middle_fvg_buffer_cap_pass": bool(break_pass),
                            "signal_time": signal_time,
                            "stop_price": terminal_price,
                            "nearest_internal_target_price": internal_target,
                            "nearest_internal_target_available_time": internal_avail,
                            "nearest_internal_target_source": internal_src,
                        })
                        emitted_compact_fvg += 1

            audit_rows.append({
                "ny_date": str(day_text),
                "execution_tf": f"{int(tf)}m",
                "physical_sweeps": int(len(day_sweeps)),
                "causal_swings": int(len(tf_swings)),
                "r12_wide_break_rows_equivalent": int(raw_break_rows),
                "r12_wide_fvg_rows_equivalent": int(raw_fvg_rows),
                "r13_primary_narratives": int(emitted_primary),
                "r13_compact_fvg_rows": int(emitted_compact_fvg),
            })

    primary = pd.DataFrame(primary_rows)
    if not primary.empty:
        primary = primary.sort_values(["event_id", "execution_tf", "break_available_time", "mss_reference_price"], kind="mergesort").reset_index(drop=True)
        primary["narrative_attempt_sequence_r13"] = primary.groupby(["event_id", "execution_tf"], sort=False).cumcount() + 1
    compact = pd.DataFrame(compact_fvg_rows)
    audit = pd.DataFrame(audit_rows)
    return primary, compact, audit

def expand_entry_target_variants(
    breaks: pd.DataFrame,
    fvgs: pd.DataFrame,
    *,
    config: StructureSemanticConfig = StructureSemanticConfig(),
) -> pd.DataFrame:
    """Create non-optimised R12 entry variants for direct comparison."""
    rows: list[dict[str, object]] = []
    if not fvgs.empty:
        for r in fvgs.to_dict("records"):
            for model, valid in (
                ("fvg_train_uncapped", True),
                ("fvg_swing_plusminus_0p10_cap", bool(r.get("swing_buffer_cap_pass", False))),
                ("fvg_break_middle_plusminus_0p10_cap", bool(r.get("break_middle_fvg_buffer_cap_pass", False))),
            ):
                if not valid:
                    continue
                base = {**r, "entry_model": model, "entry_order_type": "limit", "entry_price": float(r["fvg_near_edge_entry"]), "entry_available_time": pd.Timestamp(r["signal_time"])}
                # Existing target stays only if still profitable/available.  A
                # nearest active internal swing is an explicit alternative when
                # the original external narrative has already been spent.
                targets = []
                existing_target = r.get("target_price", np.nan)
                touched = r.get("external_target_first_touch_time", r.get("opposite_target_first_touch_time", pd.NaT))
                touched_before_signal = (not pd.isna(touched)) and pd.Timestamp(touched) <= pd.Timestamp(base["entry_available_time"])
                if np.isfinite(float(existing_target)) and not touched_before_signal:
                    targets.append(("existing_sweep_target", existing_target))
                if np.isfinite(float(r.get("nearest_internal_target_price", np.nan))):
                    targets.append(("nearest_internal_structure_target", float(r["nearest_internal_target_price"])))
                for target_model, target in targets:
                    if not np.isfinite(float(target)):
                        continue
                    is_long = str(r["trade_side"]) == "LONG"; entry = float(base["entry_price"])
                    reward = float(target)-entry if is_long else entry-float(target)
                    risk = entry-float(r["stop_price"]) if is_long else float(r["stop_price"])-entry
                    if reward <= EPS or risk <= EPS:
                        continue
                    rows.append({**base, "target_model_r12": target_model, "target_price": float(target), "risk_abs": float(risk), "planned_reward_abs": float(reward), "planned_rr": float(reward/risk)})
    # Direct next-open market entry after a close break.  One per break/target,
    # independent of FVG availability.
    if not breaks.empty:
        for r in breaks.loc[breaks["break_close_cross"].fillna(False)].to_dict("records"):
            # We don't have the next open stored in the break row.  The research
            # script attaches it from raw bars causally before replay.
            rows.append({**r, "entry_model": "close_break_next_open_market", "entry_order_type": "market_next_open", "entry_price": np.nan, "entry_available_time": pd.Timestamp(r["break_available_time"]), "stop_price": float(r.get("terminal_extreme_price", np.nan)), "target_model_r12": "existing_sweep_target", "target_price": float(r.get("target_price", np.nan))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["attempt_id_r12"] = [f"R12|{i:09d}" for i in range(len(out))]
    return out


def build_causal_sweep_events_for_levels(
    bars_ny: pd.DataFrame,
    levels: pd.DataFrame,
) -> pd.DataFrame:
    """Detect the first wick raid *after the level itself is available*.

    Unlike the old R02 sweep builder, a spent external target does not delete
    the raid from the research universe.  R12 keeps the sweep and later tests a
    nearer internal target.  This is required for cases such as 2026-08-05
    10:40: the prior upper external liquidity was already consumed, but a new
    long MSS may still be worth studying with a shorter internal objective.
    """
    if levels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day_text, day_levels in levels.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        trade = slice_ny_day(bars_ny, day, dtime(8, 30), TRADE_END)
        if trade.empty:
            continue
        av = pd.DatetimeIndex(trade.index) + pd.Timedelta(minutes=1)
        h = pd.to_numeric(trade["high"], errors="coerce").to_numpy(float)
        l = pd.to_numeric(trade["low"], errors="coerce").to_numpy(float)
        for level in day_levels.to_dict("records"):
            if not bool(level.get("tradable_level", True)):
                continue
            level_av = pd.Timestamp(level.get("level_available_time", _day_anchor(day, 8, 30)))
            start = int(np.searchsorted(_as_ns(av), int(level_av.value), side="left"))
            if start >= len(trade):
                continue
            side = str(level["liquidity_side"]); price = float(level["level_price"])
            mask = h[start:] > price if side == "high" else l[start:] < price
            hits = np.flatnonzero(mask)
            if not len(hits):
                continue
            pos = start + int(hits[0]); sweep_time = pd.Timestamp(av[pos]); bar_start = pd.Timestamp(trade.index[pos])
            is_long = side == "low"; trade_side = "LONG" if is_long else "SHORT"
            # Prefer an explicitly paired target, otherwise use the relevant
            # session/premarket opposite extreme if available.
            target = level.get("paired_target_price", np.nan)
            if not np.isfinite(float(target)) if target is not None else True:
                target = level.get("session_high" if is_long else "session_low", np.nan)
            if not np.isfinite(float(target)) if target is not None else True:
                target = level.get("premarket_high" if is_long else "premarket_low", np.nan)
            target = float(target) if target is not None and np.isfinite(float(target)) else np.nan
            target_touch = pd.NaT
            if np.isfinite(target):
                before = trade.iloc[: pos + 1]
                if is_long:
                    touched = pd.to_numeric(before["high"], errors="coerce") >= target
                else:
                    touched = pd.to_numeric(before["low"], errors="coerce") <= target
                if touched.any():
                    target_touch = pd.Timestamp(before.index[np.flatnonzero(touched.to_numpy(bool))[0]]) + pd.Timedelta(minutes=1)
            rows.append({
                **level,
                "event_id": f"{day_text}|{level.get('liquidity_family',level.get('level_type','level'))}|{side}|{price:.8f}|{level_av}",
                "trade_side": trade_side,
                "sweep_bar_start": bar_start,
                "sweep_time": sweep_time,
                "sweep_price_extreme_initial": float(l[pos] if is_long else h[pos]),
                "sweep_distance_pct": abs(float((l[pos] if is_long else h[pos])) / price - 1.0) if abs(price) > EPS else np.nan,
                "target_price": target,
                "external_target_fresh_at_sweep": bool(pd.isna(target_touch)) if np.isfinite(target) else False,
                "external_target_first_touch_time": target_touch,
                "setup_eligible_at_sweep": True,
                "setup_rejection_reason": "",
            })
    return pd.DataFrame(rows).sort_values(["sweep_time", "liquidity_family", "level_price"], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()
