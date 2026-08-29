#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry-archetype and stop-survival primitives for SOXL ICT R16.

R16 starts from R15 daily liquidity-path events and asks a narrower execution
question: after a real liquidity raid, which *causal* entry style gets the
trader into the reversal with the lowest immediate-stop probability while
still capturing 50/75/100% of the predeclared dealing range?

The helpers here deliberately keep entry generation separate from future-path
labelling.  Reclaim/MSS/FVG/OB signals are produced only from information that
was available at the signal time.  Future bars are touched only by lifecycle
replay and outcome summaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Iterable

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, NY_TZ, aggregate_closed_bars, confirmed_pivots, slice_ny_day


@dataclass(frozen=True)
class EntryArchetypeConfig:
    trade_end: dtime = dtime(16, 30)
    round_trip_cost: float = 0.0011
    approach_lookback_minutes: int = 60
    hybrid_fvg_lookback_minutes: int = 30
    hybrid_fvg_forward_minutes: int = 10
    visible_swing_percentile: float = 0.50


def _day_anchor(day, t: dtime) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=t.hour, minutes=t.minute)


def _as_ts(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize(NY_TZ)
    return t


def _safe_float(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _milestone_price(row: dict[str, object] | pd.Series, frac: float) -> float:
    lo = _safe_float(row.get("lower_price"))
    hi = _safe_float(row.get("upper_price"))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return np.nan
    is_long = str(row.get("trade_side", "")) == "LONG"
    return float(lo + (hi - lo) * frac) if is_long else float(hi - (hi - lo) * frac)


def _side_terminal(high: np.ndarray, low: np.ndarray, *, is_long: bool) -> float:
    if len(high) == 0:
        return np.nan
    return float(np.nanmin(low) if is_long else np.nanmax(high))


def _fvg_catalog(frame: pd.DataFrame, *, is_long: bool) -> pd.DataFrame:
    """Build all causal three-bar FVGs for one closed-bar execution frame."""
    if frame.empty or len(frame) < 3:
        return pd.DataFrame()
    hi = pd.to_numeric(frame["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(frame["low"], errors="coerce").to_numpy(float)
    idx = pd.DatetimeIndex(frame.index)
    av = pd.to_datetime(frame["available_time"])
    rows: list[dict[str, object]] = []
    for k in range(2, len(frame)):
        if is_long:
            if not (lo[k] > hi[k - 2] + EPS):
                continue
            near, far = float(lo[k]), float(hi[k - 2])
        else:
            if not (hi[k] < lo[k - 2] - EPS):
                continue
            near, far = float(hi[k]), float(lo[k - 2])
        rows.append({
            "fvg_first_bar_start": pd.Timestamp(idx[k - 2]),
            "fvg_middle_bar_start": pd.Timestamp(idx[k - 1]),
            "fvg_third_bar_start": pd.Timestamp(idx[k]),
            "fvg_available_time": pd.Timestamp(av.iloc[k] if isinstance(av, pd.Series) else av[k]),
            "fvg_near_edge": near,
            "fvg_far_edge": far,
            "fvg_ce": float((near + far) / 2.0),
            "fvg_size_abs": abs(near - far),
        })
    return pd.DataFrame(rows)


def attach_approach_compression_features(
    bars_ny: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Attach causal pre-raid approach/compression descriptors to R15 paths.

    The feature deliberately avoids a hard "three waves" gate.  It records the
    number of successively closer confirmed micro swings and whether the last
    three such pivots contract toward the liquidity level.  The right-hand bar
    used to confirm each pivot must have closed before the raid.
    """
    if paths.empty:
        return paths.copy()
    rows: list[dict[str, object]] = []
    day_cache: dict[str, pd.DataFrame] = {}
    for r in paths.to_dict("records"):
        rec = dict(r)
        if str(r.get("first_raid_side")) not in {"high", "low"} or pd.isna(r.get("first_raid_time")):
            rows.append(rec)
            continue
        day_text = str(r["ny_date"])
        day = day_cache.get(day_text)
        if day is None:
            d = pd.Timestamp(day_text).date()
            day = slice_ny_day(bars_ny, d, dtime(4, 0), config.trade_end).copy()
            day["available_time"] = day.index + pd.Timedelta(minutes=1)
            day_cache[day_text] = day
        raid = _as_ts(r["first_raid_time"])
        start = raid - pd.Timedelta(minutes=int(config.approach_lookback_minutes))
        ctx = day.loc[(pd.to_datetime(day["available_time"]) <= raid) & (day.index >= start)].copy()
        level = _safe_float(r.get("source_level_price"))
        if ctx.empty or not np.isfinite(level):
            rows.append(rec)
            continue
        close = pd.to_numeric(ctx["close"], errors="coerce").to_numpy(float)
        if len(close) >= 2:
            toward_sign = 1.0 if str(r["first_raid_side"]) == "high" else -1.0
            net = toward_sign * (close[-1] - close[0])
            travel = float(np.nansum(np.abs(np.diff(close))))
            rec["approach_efficiency"] = float(max(0.0, net) / travel) if travel > EPS else np.nan
            d0 = abs(level - close[0]); d1 = abs(level - close[-1])
            rec["approach_distance_contraction_ratio"] = float(d1 / d0) if d0 > EPS else np.nan
        else:
            rec["approach_efficiency"] = np.nan
            rec["approach_distance_contraction_ratio"] = np.nan

        # Realized-range contraction: latest 15m versus the preceding context.
        rr = pd.to_numeric(ctx["high"], errors="coerce") - pd.to_numeric(ctx["low"], errors="coerce")
        recent = rr.iloc[-15:] if len(rr) >= 15 else rr
        prior = rr.iloc[:-15] if len(rr) > 15 else pd.Series(dtype=float)
        recent_med = float(recent.median()) if len(recent) else np.nan
        prior_med = float(prior.median()) if len(prior) else np.nan
        rec["approach_recent_range_vs_prior"] = recent_med / prior_med if np.isfinite(prior_med) and prior_med > EPS else np.nan

        piv = confirmed_pivots(ctx, left=1, right=1)
        if not piv.empty:
            piv = piv.loc[pd.to_datetime(piv["confirmation_available_time"]) <= raid].copy()
        wanted = "low" if str(r["first_raid_side"]) == "high" else "high"
        p = piv.loc[piv["pivot_side"].eq(wanted)].sort_values("pivot_time", kind="mergesort") if not piv.empty else pd.DataFrame()
        vals = pd.to_numeric(p.get("pivot_price"), errors="coerce").dropna().to_numpy(float) if not p.empty else np.array([], dtype=float)
        # Count the monotonic tail moving toward the source liquidity.
        count = 0
        if len(vals):
            count = 1
            for j in range(len(vals) - 1, 0, -1):
                toward = vals[j] > vals[j - 1] + EPS if wanted == "low" else vals[j] < vals[j - 1] - EPS
                if not toward:
                    break
                count += 1
        rec["approach_monotonic_swing_count"] = int(count)
        if len(vals) >= 3:
            dist = np.abs(level - vals[-3:])
            rec["approach_three_swing_contraction"] = bool(dist[2] < dist[1] - EPS and dist[1] < dist[0] - EPS)
            rec["approach_three_swing_distance_ratio"] = float(dist[2] / dist[0]) if dist[0] > EPS else np.nan
        else:
            rec["approach_three_swing_contraction"] = False
            rec["approach_three_swing_distance_ratio"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def select_first_mss_narratives(
    primary: pd.DataFrame,
    *,
    min_visibility: float | None = None,
) -> pd.DataFrame:
    """Select one causal MSS narrative per physical path event/timeframe.

    This is a research selector, not a profitability filter: the earliest
    narrative is chosen by availability time.  ``min_visibility`` only creates
    an explicit comparison variant (e.g. first visible versus first any MSS).
    """
    if primary.empty:
        return pd.DataFrame()
    q = primary.copy()
    if min_visibility is not None:
        v = pd.to_numeric(q.get("causal_visibility_percentile"), errors="coerce")
        q = q.loc[v >= float(min_visibility)].copy()
    if q.empty:
        return q
    q["_break"] = pd.to_datetime(q["break_available_time"])
    q["_ref"] = pd.to_datetime(q["mss_reference_available_time"])
    keys = ["event_id", "execution_tf"]
    q = q.sort_values(keys + ["_break", "_ref", "mss_reference_time"], kind="mergesort")
    return q.drop_duplicates(keys, keep="first").drop(columns=["_break", "_ref"]).reset_index(drop=True)


def _attach_common_entry_fields(rec: dict[str, object], path: dict[str, object], *, archetype: str, family: str) -> dict[str, object]:
    out = {**path, **rec}
    out["entry_archetype"] = archetype
    out["entry_family"] = family
    out["path_event_id"] = str(path.get("path_event_id", rec.get("event_id", "")))
    out["event_id"] = out["path_event_id"]
    return out


def build_reclaim_entry_candidates(
    bars_ny: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Generate non-MSS reclaim entries: next-open market and level retest."""
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    day_cache: dict[str, pd.DataFrame] = {}
    for path in paths.to_dict("records"):
        side0 = str(path.get("first_raid_side", ""))
        if side0 not in {"high", "low"} or pd.isna(path.get("first_raid_time")):
            continue
        day_text = str(path["ny_date"]); d = pd.Timestamp(day_text).date()
        day = day_cache.get(day_text)
        if day is None:
            day = slice_ny_day(bars_ny, d, dtime(8, 30), config.trade_end).copy()
            day["available_time"] = day.index + pd.Timedelta(minutes=1)
            day_cache[day_text] = day
        raid = _as_ts(path["first_raid_time"]); level = _safe_float(path.get("source_level_price"))
        if day.empty or not np.isfinite(level):
            continue
        av = pd.to_datetime(day["available_time"])
        after = day.loc[av >= raid].copy()
        if after.empty:
            continue
        is_long = side0 == "low"
        closes = pd.to_numeric(after["close"], errors="coerce")
        reclaim_mask = closes >= level - EPS if is_long else closes <= level + EPS
        if not bool(reclaim_mask.any()):
            continue
        reclaim_bar_start = pd.Timestamp(reclaim_mask.idxmax())
        signal = reclaim_bar_start + pd.Timedelta(minutes=1)
        hist = day.loc[(day.index + pd.Timedelta(minutes=1) >= raid) & (day.index <= reclaim_bar_start)]
        stop = _side_terminal(
            pd.to_numeric(hist["high"], errors="coerce").to_numpy(float),
            pd.to_numeric(hist["low"], errors="coerce").to_numpy(float),
            is_long=is_long,
        )
        if not np.isfinite(stop):
            continue
        ns = pd.DatetimeIndex(day.index).as_unit("ns").asi8
        pos = int(np.searchsorted(ns, int(signal.value), side="left"))
        if pos < len(day):
            market_price = float(pd.to_numeric(day["open"], errors="coerce").iloc[pos])
            base = {
                "trade_side": "LONG" if is_long else "SHORT",
                "execution_tf": "1m",
                "entry_order_type": "market_next_open",
                "entry_available_time": signal,
                "entry_price": market_price,
                "stop_price": stop,
                "reclaim_signal_time": signal,
                "reclaim_bar_start": reclaim_bar_start,
                "mss_reference_price": np.nan,
                "causal_visibility_percentile": np.nan,
            }
            rows.append(_attach_common_entry_fields(base, path, archetype="raid_reclaim_next_open_market", family="reclaim"))
        # Limit order rests back at the raided level after a confirmed reclaim.
        base = {
            "trade_side": "LONG" if is_long else "SHORT",
            "execution_tf": "1m",
            "entry_order_type": "limit",
            "entry_available_time": signal,
            "entry_price": level,
            "stop_price": stop,
            "reclaim_signal_time": signal,
            "reclaim_bar_start": reclaim_bar_start,
            "mss_reference_price": np.nan,
            "causal_visibility_percentile": np.nan,
        }
        rows.append(_attach_common_entry_fields(base, path, archetype="raid_reclaim_level_retest_limit", family="reclaim"))
    return pd.DataFrame(rows)


def _narrative_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    for c in ("break_available_time", "mss_reference_time", "terminal_extreme_time"):
        if c in q:
            q[c] = pd.to_datetime(q[c])
    return q


def build_mss_fvg_entry_candidates(
    r15_entries: pd.DataFrame,
    selected_mss: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Reuse R15's causal FVG choices for one selected MSS per sweep/TF."""
    if r15_entries.empty or selected_mss.empty:
        return pd.DataFrame()
    wanted_models = {
        "first_train_near": "first_fvg_near",
        "first_train_ce": "first_fvg_ce",
        "break_middle_near": "break_fvg_near",
        "break_middle_ce": "break_fvg_ce",
        "closest_to_broken_swing_near": "closest_fvg_near",
        "last_pre_or_on_break_near": "last_pre_or_on_break_fvg_near",
    }
    e = _narrative_key_frame(r15_entries)
    s = _narrative_key_frame(selected_mss)
    keys = ["event_id", "execution_tf", "break_available_time", "mss_reference_time"]
    for k in keys:
        if k not in e or k not in s:
            return pd.DataFrame()
    slim = s[keys].drop_duplicates()
    q = e.merge(slim.assign(_selected=True), on=keys, how="inner")
    q = q.loc[q["entry_model_r13"].isin(wanted_models)].copy()
    if q.empty:
        return q
    q["entry_family"] = "mss_fvg"
    q["entry_archetype"] = [f"mss_{label}_{wanted_models[str(x)]}" for x in q["entry_model_r13"]]
    q["path_event_id"] = q["event_id"].astype(str)
    return q.sort_values(["event_id", "execution_tf", "entry_archetype", "entry_available_time"], kind="mergesort").drop_duplicates(
        ["event_id", "execution_tf", "entry_archetype"], keep="first"
    ).reset_index(drop=True)


def build_mss_market_entry_candidates(
    bars_ny: pd.DataFrame,
    selected_mss: pd.DataFrame,
    *,
    label: str,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Close-confirmed MSS -> next 1m/2m bar open market entry."""
    if selected_mss.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    frame_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for r in selected_mss.to_dict("records"):
        if not bool(r.get("break_close_cross", False)):
            continue
        tf = int(r.get("execution_tf_minutes", str(r.get("execution_tf", "1m")).replace("m", "")))
        day_text = str(r["ny_date"]); key = (day_text, tf)
        frame = frame_cache.get(key)
        if frame is None:
            day = slice_ny_day(bars_ny, pd.Timestamp(day_text).date(), dtime(8, 30), config.trade_end)
            frame = aggregate_closed_bars(day, tf)
            frame_cache[key] = frame
        signal = _as_ts(r["break_available_time"])
        starts = pd.DatetimeIndex(frame.index).as_unit("ns").asi8
        pos = int(np.searchsorted(starts, int(signal.value), side="left"))
        if pos >= len(frame):
            continue
        row = dict(r)
        row.update({
            "entry_family": "mss_market",
            "entry_archetype": f"mss_{label}_close_break_next_open_market",
            "entry_order_type": "market_next_open",
            "entry_available_time": signal,
            "entry_price": float(pd.to_numeric(frame["open"], errors="coerce").iloc[pos]),
            "stop_price": _safe_float(r.get("terminal_extreme_price")),
            "path_event_id": str(r["event_id"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _find_order_block(frame: pd.DataFrame, *, terminal_time: pd.Timestamp, break_bar_start: pd.Timestamp, is_long: bool) -> dict[str, object] | None:
    if frame.empty:
        return None
    q = frame.loc[(pd.to_datetime(frame["available_time"]) >= terminal_time) & (frame.index <= break_bar_start)].copy()
    if q.empty:
        return None
    op = pd.to_numeric(q["open"], errors="coerce")
    cl = pd.to_numeric(q["close"], errors="coerce")
    mask = cl < op - EPS if is_long else cl > op + EPS
    q = q.loc[mask]
    if q.empty:
        return None
    t = pd.Timestamp(q.index[-1]); rr = q.iloc[-1]
    return {
        "ob_bar_start": t,
        "ob_available_time": pd.Timestamp(rr["available_time"]),
        "ob_open": float(rr["open"]),
        "ob_high": float(rr["high"]),
        "ob_low": float(rr["low"]),
        "ob_close": float(rr["close"]),
        "ob_mid": float((float(rr["high"]) + float(rr["low"])) / 2.0),
    }


def build_order_block_entry_candidates(
    bars_ny: pd.DataFrame,
    selected_mss: pd.DataFrame,
    *,
    label: str,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Quantitative OB proxy tied to the selected MSS displacement leg.

    The OB proxy is the last opposing closed candle between the known terminal
    extreme and the structure-break bar.  It is intentionally labelled a proxy
    rather than presented as a canonical discretionary ICT definition.
    """
    if selected_mss.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for r in selected_mss.to_dict("records"):
        tf = int(r.get("execution_tf_minutes", str(r.get("execution_tf", "1m")).replace("m", "")))
        day_text = str(r["ny_date"]); key = (day_text, tf)
        frame = cache.get(key)
        if frame is None:
            day = slice_ny_day(bars_ny, pd.Timestamp(day_text).date(), dtime(8, 30), config.trade_end)
            frame = aggregate_closed_bars(day, tf); cache[key] = frame
        is_long = str(r.get("trade_side")) == "LONG"
        ob = _find_order_block(frame, terminal_time=_as_ts(r["terminal_extreme_time"]), break_bar_start=_as_ts(r["break_bar_start"]), is_long=is_long)
        if ob is None:
            continue
        for suffix, price in (("ob_open", ob["ob_open"]), ("ob_mid", ob["ob_mid"])):
            row = {**r, **ob}
            row.update({
                "entry_family": "order_block_proxy",
                "entry_archetype": f"mss_{label}_{suffix}_limit",
                "entry_order_type": "limit",
                "entry_available_time": _as_ts(r["break_available_time"]),
                "entry_price": float(price),
                "stop_price": _safe_float(r.get("terminal_extreme_price")),
                "path_event_id": str(r["event_id"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _catalog_for_day_tf(bars_ny: pd.DataFrame, day_text: str, tf: int, trade_end: dtime) -> pd.DataFrame:
    day = slice_ny_day(bars_ny, pd.Timestamp(day_text).date(), dtime(8, 30), trade_end)
    return aggregate_closed_bars(day, tf)


def build_ob_fvg_overlap_candidates(
    bars_ny: pd.DataFrame,
    selected_mss: pd.DataFrame,
    *,
    label: str,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Limit at the midpoint of an OB-proxy x displacement-FVG overlap."""
    if selected_mss.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for r in selected_mss.to_dict("records"):
        tf = int(r.get("execution_tf_minutes", str(r.get("execution_tf", "1m")).replace("m", "")))
        day_text = str(r["ny_date"]); key = (day_text, tf)
        data = cache.get(key)
        if data is None:
            frame = _catalog_for_day_tf(bars_ny, day_text, tf, config.trade_end)
            data = (frame, _fvg_catalog(frame, is_long=True), _fvg_catalog(frame, is_long=False)); cache[key] = data
        frame, bull, bear = data
        is_long = str(r.get("trade_side")) == "LONG"
        ob = _find_order_block(frame, terminal_time=_as_ts(r["terminal_extreme_time"]), break_bar_start=_as_ts(r["break_bar_start"]), is_long=is_long)
        if ob is None:
            continue
        cat = bull if is_long else bear
        if cat.empty:
            continue
        terminal = _as_ts(r["terminal_extreme_time"]); signal = _as_ts(r["break_available_time"])
        q = cat.loc[(pd.to_datetime(cat["fvg_available_time"]) >= terminal) & (pd.to_datetime(cat["fvg_available_time"]) <= signal + pd.Timedelta(minutes=tf))].copy()
        if q.empty:
            continue
        # Prefer the FVG whose middle candle is the actual break bar, otherwise
        # the latest one already causally known around the break.
        exact = q.loc[pd.to_datetime(q["fvg_middle_bar_start"]) == _as_ts(r["break_bar_start"])]
        f = (exact.iloc[-1] if not exact.empty else q.sort_values("fvg_available_time", kind="mergesort").iloc[-1]).to_dict()
        f_lo = min(float(f["fvg_near_edge"]), float(f["fvg_far_edge"])); f_hi = max(float(f["fvg_near_edge"]), float(f["fvg_far_edge"]))
        ov_lo = max(float(ob["ob_low"]), f_lo); ov_hi = min(float(ob["ob_high"]), f_hi)
        if ov_hi < ov_lo - EPS:
            continue
        row = {**r, **ob, **{f"overlap_{k}": v for k, v in f.items()}}
        row.update({
            "entry_family": "ob_fvg_overlap",
            "entry_archetype": f"mss_{label}_ob_fvg_overlap_mid_limit",
            "entry_order_type": "limit",
            "entry_available_time": max(signal, _as_ts(f["fvg_available_time"])),
            "entry_price": float((ov_lo + ov_hi) / 2.0),
            "ob_fvg_overlap_low": float(ov_lo),
            "ob_fvg_overlap_high": float(ov_hi),
            "stop_price": _safe_float(r.get("terminal_extreme_price")),
            "path_event_id": str(r["event_id"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_hybrid_2m_structure_1m_fvg_candidates(
    bars_ny: pd.DataFrame,
    selected_2m: pd.DataFrame,
    *,
    label: str,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Confirm structure on 2m, execute from a causally-known 1m FVG."""
    if selected_2m.empty:
        return pd.DataFrame()
    q2 = selected_2m.loc[selected_2m["execution_tf"].astype(str).eq("2m")].copy()
    if q2.empty:
        return q2
    rows: list[dict[str, object]] = []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for r in q2.to_dict("records"):
        day_text = str(r["ny_date"]); data = cache.get(day_text)
        if data is None:
            frame = _catalog_for_day_tf(bars_ny, day_text, 1, config.trade_end)
            data = (_fvg_catalog(frame, is_long=True), _fvg_catalog(frame, is_long=False)); cache[day_text] = data
        bull, bear = data; is_long = str(r.get("trade_side")) == "LONG"; cat = bull if is_long else bear
        if cat.empty:
            continue
        terminal = _as_ts(r["terminal_extreme_time"]); signal = _as_ts(r["break_available_time"])
        lo_time = signal - pd.Timedelta(minutes=int(config.hybrid_fvg_lookback_minutes))
        hi_time = signal + pd.Timedelta(minutes=int(config.hybrid_fvg_forward_minutes))
        q = cat.loc[
            (pd.to_datetime(cat["fvg_available_time"]) >= max(terminal, lo_time))
            & (pd.to_datetime(cat["fvg_available_time"]) <= hi_time)
        ].copy()
        if q.empty:
            continue
        known = q.loc[pd.to_datetime(q["fvg_available_time"]) <= signal]
        if not known.empty:
            f = known.sort_values("fvg_available_time", kind="mergesort").iloc[-1]
            entry_available = signal
            timing = "known_at_2m_break"
        else:
            f = q.sort_values("fvg_available_time", kind="mergesort").iloc[0]
            entry_available = _as_ts(f["fvg_available_time"])
            timing = "first_after_2m_break"
        for suffix, price in (("near", float(f["fvg_near_edge"])), ("ce", float(f["fvg_ce"]))):
            row = {**r, **{f"hybrid_{k}": v for k, v in f.to_dict().items()}}
            row.update({
                "entry_family": "hybrid_2m_structure_1m_fvg",
                "entry_archetype": f"mss_{label}_2m_structure_1m_fvg_{suffix}_limit",
                "entry_order_type": "limit",
                "entry_available_time": entry_available,
                "entry_price": price,
                "stop_price": _safe_float(r.get("terminal_extreme_price")),
                "hybrid_fvg_timing": timing,
                "execution_tf": "2m->1m",
                "path_event_id": str(r["event_id"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def attach_path_metadata(entries: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical R15 path metadata without losing rows after concat.

    R16 concatenates heterogeneous archetype frames before this helper runs.
    Pandas therefore creates a union schema: if *one* archetype already has a
    column such as ``range_model``, every other archetype receives that column
    filled with NaN.  The old implementation only attached columns that were
    globally absent from ``entries`` and consequently left all MSS/FVG rows
    with NaN path metadata.

    This helper now merges a temporary canonical path copy for every metadata
    field, validates any already-populated values, and fills only row-level
    missing values.  Existing non-null entry metadata always wins.
    """
    if entries.empty:
        return entries.copy()
    if "path_event_id" not in entries.columns:
        raise KeyError("entries missing path_event_id")
    if "path_event_id" not in paths.columns:
        raise KeyError("paths missing path_event_id")

    pcols = [
        "path_event_id", "ny_date", "range_model", "first_raid_side", "first_raid_time", "first_reclaim_time",
        "trade_side", "source_level_price", "target_price", "lower_price", "upper_price", "range_width_abs",
        "first_raid_penetration_frac_range", "reclaim_minutes", "traversal_complete", "max_progress_fraction",
        "approach_efficiency", "approach_distance_contraction_ratio", "approach_recent_range_vs_prior",
        "approach_monotonic_swing_count", "approach_three_swing_contraction", "approach_three_swing_distance_ratio",
    ]
    p = paths[[c for c in pcols if c in paths.columns]].copy()
    dup = p["path_event_id"].duplicated(keep=False)
    if bool(dup.any()):
        # A path_event_id must identify one unique daily path.  Silently using
        # drop_duplicates here could attach the wrong range to an entry.
        check_cols = [c for c in p.columns if c != "path_event_id"]
        conflict = p.loc[dup].groupby("path_event_id", sort=False, dropna=False)[check_cols].nunique(dropna=False).max(axis=1)
        bad = conflict.loc[conflict > 1]
        if not bad.empty:
            raise ValueError(f"conflicting path metadata for {len(bad)} path_event_id values")
        p = p.drop_duplicates("path_event_id", keep="first")

    q = entries.copy()
    meta_cols = [c for c in p.columns if c != "path_event_id"]
    rename = {c: f"__pathmeta_{c}" for c in meta_cols}
    q = q.merge(p.rename(columns=rename), on="path_event_id", how="left", validate="many_to_one")

    for col in meta_cols:
        shadow = rename[col]
        if col not in q.columns:
            q[col] = q[shadow]
        else:
            existing = q[col]
            canonical = q[shadow]
            both = existing.notna() & canonical.notna()
            if bool(both.any()):
                if col.endswith("_time") or col in {"ny_date"}:
                    left = existing.loc[both].astype(str)
                    right = canonical.loc[both].astype(str)
                elif pd.api.types.is_numeric_dtype(existing) or pd.api.types.is_numeric_dtype(canonical):
                    left = pd.to_numeric(existing.loc[both], errors="coerce")
                    right = pd.to_numeric(canonical.loc[both], errors="coerce")
                    comparable = left.notna() & right.notna()
                    mismatch = pd.Series(False, index=left.index)
                    if bool(comparable.any()):
                        mismatch.loc[comparable] = ~np.isclose(left.loc[comparable], right.loc[comparable], rtol=1e-10, atol=1e-10, equal_nan=True)
                    if bool(mismatch.any()):
                        raise ValueError(f"entry/path metadata mismatch for column={col}")
                    left = right = None
                else:
                    left = existing.loc[both].astype(str)
                    right = canonical.loc[both].astype(str)
                if left is not None and bool((left != right).any()):
                    raise ValueError(f"entry/path metadata mismatch for column={col}")
            q[col] = existing.where(existing.notna(), canonical)
        q = q.drop(columns=[shadow])

    if "event_id" not in q:
        q["event_id"] = q["path_event_id"].astype(str)
    return q


def attach_causal_entry_state(
    bars_ny: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Attach only information observable by each entry signal."""
    if entries.empty:
        return entries.copy()
    rows: list[dict[str, object]] = []
    cache: dict[str, pd.DataFrame] = {}
    for r in entries.to_dict("records"):
        rec = dict(r); day_text = str(r["ny_date"]); d = pd.Timestamp(day_text).date()
        day = cache.get(day_text)
        if day is None:
            day = slice_ny_day(bars_ny, d, dtime(8, 30), config.trade_end).copy(); day["available_time"] = day.index + pd.Timedelta(minutes=1); cache[day_text] = day
        signal = _as_ts(r["entry_available_time"]); raid = _as_ts(r["first_raid_time"]); source = _safe_float(r.get("source_level_price"))
        hist = day.loc[(pd.to_datetime(day["available_time"]) >= raid) & (pd.to_datetime(day["available_time"]) <= signal)]
        is_long = str(r.get("trade_side")) == "LONG"
        if not hist.empty and np.isfinite(source):
            hi = pd.to_numeric(hist["high"], errors="coerce").to_numpy(float); lo = pd.to_numeric(hist["low"], errors="coerce").to_numpy(float); cl = pd.to_numeric(hist["close"], errors="coerce").to_numpy(float)
            if is_long:
                pen = max(0.0, source - float(np.nanmin(lo)))
                cross = lo < source - EPS
                reclaim_now = bool(cl[-1] >= source - EPS)
            else:
                pen = max(0.0, float(np.nanmax(hi)) - source)
                cross = hi > source + EPS
                reclaim_now = bool(cl[-1] <= source + EPS)
            # Count *episodes* entering the swept side, not bars outside.
            prev = np.r_[False, cross[:-1]]
            rec["raid_count_so_far_at_entry"] = int(np.sum(cross & ~prev))
            width = _safe_float(r.get("range_width_abs"))
            rec["penetration_so_far_frac_range"] = pen / width if np.isfinite(width) and width > EPS else np.nan
            rec["source_reclaimed_at_entry"] = reclaim_now
        else:
            rec["raid_count_so_far_at_entry"] = np.nan; rec["penetration_so_far_frac_range"] = np.nan; rec["source_reclaimed_at_entry"] = False
        entry = _safe_float(r.get("entry_price")); stop = _safe_float(r.get("stop_price")); width = _safe_float(r.get("range_width_abs"))
        risk_abs = entry - stop if is_long else stop - entry
        rec["initial_risk_abs"] = risk_abs
        rec["initial_risk_frac_range"] = risk_abs / width if np.isfinite(risk_abs) and np.isfinite(width) and width > EPS else np.nan
        lo_p = _safe_float(r.get("lower_price")); hi_p = _safe_float(r.get("upper_price"))
        if np.isfinite(entry) and np.isfinite(lo_p) and np.isfinite(hi_p) and hi_p > lo_p:
            rec["entry_progress_fraction"] = (entry - lo_p) / (hi_p - lo_p) if is_long else (hi_p - entry) / (hi_p - lo_p)
        else:
            rec["entry_progress_fraction"] = np.nan
        rec["signal_minutes_from_raid"] = float((signal - raid).total_seconds() / 60.0)
        rows.append(rec)
    return pd.DataFrame(rows)


def _first_true(mask: np.ndarray) -> int | None:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else None


def _simulate_target_fast(
    *,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    idx: pd.DatetimeIndex,
    fill_pos: int,
    entry: float,
    stop: float,
    target: float,
    is_long: bool,
    cost: float,
) -> dict[str, object]:
    gross_stop = (stop - entry) / entry if is_long else (entry - stop) / entry
    target_ahead = target > entry + EPS if is_long else target < entry - EPS
    if not target_ahead:
        return {"ahead": False, "hit": False, "hit_time": pd.NaT, "minutes": np.nan, "net": np.nan, "reason": "already_passed_at_fill"}
    h = hi[fill_pos:]; l = lo[fill_pos:]
    stop_rel = _first_true(l <= stop + EPS) if is_long else _first_true(h >= stop - EPS)
    target_rel = _first_true(h >= target - EPS) if is_long else _first_true(l <= target + EPS)
    # Stop wins same-minute ambiguity.
    if stop_rel is not None and (target_rel is None or stop_rel <= target_rel):
        return {"ahead": True, "hit": False, "hit_time": pd.NaT, "minutes": np.nan, "net": float(gross_stop - cost), "reason": "stop"}
    if target_rel is not None:
        j = fill_pos + target_rel
        gross = (target - entry) / entry if is_long else (entry - target) / entry
        mins = float((idx[j] - idx[fill_pos]).total_seconds() / 60.0)
        return {"ahead": True, "hit": True, "hit_time": idx[j], "minutes": mins, "net": float(gross - cost), "reason": "milestone"}
    close_px = float(cl[-1])
    gross = (close_px - entry) / entry if is_long else (entry - close_px) / entry
    return {"ahead": True, "hit": False, "hit_time": pd.NaT, "minutes": np.nan, "net": float(gross - cost), "reason": "session_close"}


def replay_entry_survival(
    bars_ny: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    config: EntryArchetypeConfig = EntryArchetypeConfig(),
) -> pd.DataFrame:
    """Replay candidates with bounded NumPy first-hit scans.

    Limit orders are cancelled if the known terminal stop is invalidated or the
    opposite boundary is reached before the order fills.  Same-minute
    fill+stop is allowed to fill and then conservatively resolves to stop.
    """
    if entries.empty:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    cache: dict[str, dict[str, object]] = {}
    for r in entries.to_dict("records"):
        rec = dict(r); day_text = str(r["ny_date"]); d = pd.Timestamp(day_text).date()
        data = cache.get(day_text)
        if data is None:
            day = slice_ny_day(bars_ny, d, dtime(8, 30), config.trade_end)
            idx = pd.DatetimeIndex(day.index).as_unit("ns")
            data = {
                "idx": idx, "ns": idx.asi8,
                "open": pd.to_numeric(day["open"], errors="coerce").to_numpy(float),
                "high": pd.to_numeric(day["high"], errors="coerce").to_numpy(float),
                "low": pd.to_numeric(day["low"], errors="coerce").to_numpy(float),
                "close": pd.to_numeric(day["close"], errors="coerce").to_numpy(float),
            }; cache[day_text] = data
        idx = data["idx"]; ns = data["ns"]; op = data["open"]; hi = data["high"]; lo = data["low"]; cl = data["close"]
        is_long = str(r.get("trade_side")) == "LONG"; entry = _safe_float(r.get("entry_price")); stop = _safe_float(r.get("stop_price"))
        signal = _as_ts(r["entry_available_time"])
        rec.update({"filled": False, "fill_time": pd.NaT, "fill_wait_minutes": np.nan, "stop_hit": False, "stop_time": pd.NaT, "stop_minutes_after_fill": np.nan, "mfe_r": np.nan, "mae_r": np.nan, "unfilled_reason": ""})
        risk = entry - stop if is_long else stop - entry
        if not (np.isfinite(entry) and np.isfinite(stop) and risk > EPS):
            rec["unfilled_reason"] = "invalid_risk"; out.append(rec); continue
        start = int(np.searchsorted(ns, int(signal.value), side="left"))
        if start >= len(idx):
            rec["unfilled_reason"] = "signal_after_session"; out.append(rec); continue
        order_type = str(r.get("entry_order_type", "limit")); fill_pos: int | None = None
        target100 = _milestone_price(r, 1.0)
        if order_type == "market_next_open":
            fill_pos = start
            rec["entry_price_replay"] = float(op[fill_pos]); entry = float(op[fill_pos]); risk = entry - stop if is_long else stop - entry
            if risk <= EPS:
                rec["unfilled_reason"] = "invalid_market_risk"; out.append(rec); continue
        else:
            h = hi[start:]; l = lo[start:]
            fill_rel = _first_true((l <= entry + EPS) & (h >= entry - EPS))
            stop_rel = _first_true(l <= stop + EPS) if is_long else _first_true(h >= stop - EPS)
            target_rel = None
            if np.isfinite(target100):
                target_rel = _first_true(h >= target100 - EPS) if is_long else _first_true(l <= target100 + EPS)
            if fill_rel is None:
                rec["unfilled_reason"] = "stop_before_fill" if stop_rel is not None and (target_rel is None or stop_rel <= target_rel) else ("opposite_before_fill" if target_rel is not None else "session_end_unfilled")
                out.append(rec); continue
            first_block = min([x for x in (stop_rel, target_rel) if x is not None], default=None)
            if first_block is not None and first_block < fill_rel:
                rec["unfilled_reason"] = "stop_before_fill" if stop_rel is not None and stop_rel == first_block else "opposite_before_fill"
                out.append(rec); continue
            fill_pos = start + fill_rel
        rec["filled"] = True; rec["fill_time"] = idx[fill_pos]; rec["fill_wait_minutes"] = float((idx[fill_pos] - signal).total_seconds() / 60.0)
        h = hi[fill_pos:]; l = lo[fill_pos:]
        stop_rel = _first_true(l <= stop + EPS) if is_long else _first_true(h >= stop - EPS)
        stop_pos = None if stop_rel is None else fill_pos + stop_rel
        end_rel = stop_rel if stop_rel is not None else len(h) - 1
        hs = h[: end_rel + 1]; ls = l[: end_rel + 1]
        max_fav = float(np.nanmax(hs) - entry) if is_long else float(entry - np.nanmin(ls))
        max_adv = float(entry - np.nanmin(ls)) if is_long else float(np.nanmax(hs) - entry)
        if stop_pos is not None:
            rec["stop_hit"] = True; rec["stop_time"] = idx[stop_pos]; rec["stop_minutes_after_fill"] = float((idx[stop_pos] - idx[fill_pos]).total_seconds() / 60.0)
        rec["mfe_r"] = float(max(0.0, max_fav) / risk); rec["mae_r"] = float(-max(0.0, max_adv) / risk)
        for mins in (1, 3, 5, 10):
            rec[f"stop_within_{mins}m"] = bool(stop_pos is not None and (idx[stop_pos] - idx[fill_pos]) <= pd.Timedelta(minutes=mins))
        for frac in (0.25, 0.50, 0.75, 1.00):
            pct = int(frac * 100); target = _milestone_price(r, frac)
            sim = _simulate_target_fast(hi=hi, lo=lo, cl=cl, idx=idx, fill_pos=fill_pos, entry=entry, stop=stop, target=target, is_long=is_long, cost=float(config.round_trip_cost))
            rec[f"milestone_{pct}_ahead_at_fill"] = bool(sim["ahead"])
            rec[f"milestone_{pct}_before_stop"] = bool(sim["hit"])
            rec[f"milestone_{pct}_hit_time_from_entry"] = sim["hit_time"]
            rec[f"milestone_{pct}_minutes_after_fill"] = sim["minutes"]
            rec[f"net_return_exit_{pct}"] = sim["net"]
            rec[f"exit_reason_{pct}"] = sim["reason"]
            if bool(sim["ahead"]):
                reward = target - entry if is_long else entry - target
                rec[f"rr_to_{pct}"] = float(reward / risk) if reward > EPS else np.nan
            else:
                rec[f"rr_to_{pct}"] = np.nan
        if bool(rec.get("stop_within_5m", False)):
            rec["survival_outcome"] = "immediate_stop_le_5m"
        elif bool(rec.get("milestone_50_before_stop", False)):
            rec["survival_outcome"] = "reached_50_before_stop"
        elif bool(rec.get("stop_hit", False)):
            rec["survival_outcome"] = "later_stop_before_50"
        else:
            rec["survival_outcome"] = "session_survivor_below_50"
        out.append(rec)
    return pd.DataFrame(out)

def _profit_factor(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    pos = float(x.clip(lower=0).sum()); neg = float(-x.clip(upper=0).sum())
    if neg > EPS:
        return pos / neg
    return np.inf if pos > 0 else np.nan


def summarize_entry_archetypes(replayed: pd.DataFrame, *, group_extra: Iterable[str] = ()) -> pd.DataFrame:
    if replayed.empty:
        return pd.DataFrame()
    group_cols = [c for c in [*group_extra, "range_model", "entry_archetype", "execution_tf"] if c in replayed.columns]
    rows: list[dict[str, object]] = []
    for key, g in replayed.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        meta = dict(zip(group_cols, key)); filled = g.loc[g["filled"].fillna(False).astype(bool)].copy()
        row = {**meta, "candidates": len(g), "filled": len(filled), "fill_rate": len(filled) / len(g) if len(g) else np.nan}
        if len(filled):
            row.update({
                "stop_rate": float(filled["stop_hit"].fillna(False).mean()),
                "immediate_stop_1m_rate": float(filled["stop_within_1m"].fillna(False).mean()),
                "immediate_stop_3m_rate": float(filled["stop_within_3m"].fillna(False).mean()),
                "immediate_stop_5m_rate": float(filled["stop_within_5m"].fillna(False).mean()),
                "immediate_stop_10m_rate": float(filled["stop_within_10m"].fillna(False).mean()),
                "median_mfe_r": float(pd.to_numeric(filled["mfe_r"], errors="coerce").median()),
                "median_mae_r": float(pd.to_numeric(filled["mae_r"], errors="coerce").median()),
                "median_initial_risk_frac_range": float(pd.to_numeric(filled.get("initial_risk_frac_range"), errors="coerce").median()),
            })
            for pct in (25, 50, 75, 100):
                ahead = filled.loc[filled[f"milestone_{pct}_ahead_at_fill"].fillna(False).astype(bool)]
                row[f"milestone_{pct}_eligible"] = len(ahead)
                row[f"milestone_{pct}_before_stop_rate"] = float(ahead[f"milestone_{pct}_before_stop"].fillna(False).mean()) if len(ahead) else np.nan
                row[f"profit_factor_exit_{pct}"] = _profit_factor(ahead[f"net_return_exit_{pct}"]) if len(ahead) else np.nan
                row[f"mean_net_return_exit_{pct}"] = float(pd.to_numeric(ahead[f"net_return_exit_{pct}"], errors="coerce").mean()) if len(ahead) else np.nan
                row[f"median_rr_to_{pct}"] = float(pd.to_numeric(ahead[f"rr_to_{pct}"], errors="coerce").median()) if len(ahead) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_stop_survival_features(replayed: pd.DataFrame) -> pd.DataFrame:
    """Compare causal entry-time features of immediate stops vs survivors."""
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[replayed["filled"].fillna(False).astype(bool)].copy()
    if q.empty:
        return pd.DataFrame()
    features = [
        "approach_efficiency", "approach_distance_contraction_ratio", "approach_recent_range_vs_prior",
        "approach_monotonic_swing_count", "approach_three_swing_distance_ratio",
        "penetration_so_far_frac_range", "raid_count_so_far_at_entry", "signal_minutes_from_raid",
        "initial_risk_frac_range", "entry_progress_fraction", "causal_visibility_percentile",
        "terminal_to_break_minutes", "directional_bar_fraction", "path_efficiency", "break_overshoot_frac_range",
    ]
    rows: list[dict[str, object]] = []
    for (arch, outcome), g in q.groupby(["entry_archetype", "survival_outcome"], sort=True, dropna=False):
        row = {"entry_archetype": arch, "survival_outcome": outcome, "n": len(g)}
        for f in features:
            if f in g:
                row[f"median_{f}"] = float(pd.to_numeric(g[f], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_compression_bins(replayed: pd.DataFrame) -> pd.DataFrame:
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[replayed["filled"].fillna(False).astype(bool)].copy()
    if q.empty or "approach_monotonic_swing_count" not in q:
        return pd.DataFrame()
    cnt = pd.to_numeric(q["approach_monotonic_swing_count"], errors="coerce").fillna(0)
    q["approach_swing_bucket"] = pd.cut(cnt, [-np.inf, 0.5, 1.5, 2.5, np.inf], labels=["0", "1", "2", "3_plus"])
    q["three_wave_contraction"] = q.get("approach_three_swing_contraction", False).fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for (bucket, contraction), g in q.groupby(["approach_swing_bucket", "three_wave_contraction"], observed=True, sort=True):
        eligible = g.loc[g["milestone_50_ahead_at_fill"].fillna(False).astype(bool)]
        rows.append({
            "approach_swing_bucket": str(bucket), "three_wave_contraction": bool(contraction), "filled": len(g),
            "immediate_stop_5m_rate": float(g["stop_within_5m"].fillna(False).mean()),
            "milestone_50_before_stop_rate": float(eligible["milestone_50_before_stop"].fillna(False).mean()) if len(eligible) else np.nan,
            "milestone_75_before_stop_rate": float(g.loc[g["milestone_75_ahead_at_fill"].fillna(False).astype(bool), "milestone_75_before_stop"].fillna(False).mean()) if bool(g["milestone_75_ahead_at_fill"].fillna(False).any()) else np.nan,
        })
    return pd.DataFrame(rows)


def period_label(value) -> str:
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    if t < pd.Timestamp("2025-01-01"):
        return "discovery_2023h2_2024"
    if t < pd.Timestamp("2026-01-01"):
        return "forward_2025"
    return "late_2026"


def summarize_fixed_feature_bins(replayed: pd.DataFrame) -> pd.DataFrame:
    """Fixed, predeclared bins for finding obvious immediate-stop conditions.

    No quantiles are fit to PnL.  The bins are semantic scales so any pattern
    can later be checked unchanged in 2025/2026 rather than tuned in-sample.
    """
    if replayed.empty:
        return pd.DataFrame()
    q = replayed.loc[replayed["filled"].fillna(False).astype(bool)].copy()
    if q.empty:
        return pd.DataFrame()
    specs: list[tuple[str, list[float], list[str]]] = [
        ("causal_visibility_percentile", [-np.inf, 0.5, 0.8, np.inf], ["micro_lt_p50", "visible_p50_p80", "strong_ge_p80"]),
        ("initial_risk_frac_range", [-np.inf, 0.10, 0.25, 0.50, np.inf], ["lt_0p10", "0p10_0p25", "0p25_0p50", "ge_0p50"]),
        ("signal_minutes_from_raid", [-np.inf, 5, 15, 30, 60, np.inf], ["le_5m", "5_15m", "15_30m", "30_60m", "gt_60m"]),
        ("entry_progress_fraction", [-np.inf, 0.25, 0.50, 0.75, np.inf], ["lt_25pct", "25_50pct", "50_75pct", "ge_75pct"]),
        ("penetration_so_far_frac_range", [-np.inf, 0.05, 0.15, 0.30, np.inf], ["shallow_lt_5pct", "5_15pct", "15_30pct", "deep_ge_30pct"]),
        ("approach_efficiency", [-np.inf, 0.25, 0.50, 0.75, np.inf], ["lt_0p25", "0p25_0p50", "0p50_0p75", "ge_0p75"]),
    ]
    rows: list[dict[str, object]] = []
    for feature, edges, labels in specs:
        if feature not in q:
            continue
        vals = pd.to_numeric(q[feature], errors="coerce")
        bins = pd.cut(vals, edges, labels=labels)
        work = q.assign(_bin=bins)
        for (arch, bucket), g in work.dropna(subset=["_bin"]).groupby(["entry_archetype", "_bin"], observed=True, sort=True):
            e50 = g.loc[g["milestone_50_ahead_at_fill"].fillna(False).astype(bool)]
            rows.append({
                "feature": feature,
                "bucket": str(bucket),
                "entry_archetype": arch,
                "n": len(g),
                "immediate_stop_5m_rate": float(g["stop_within_5m"].fillna(False).mean()),
                "stop_rate": float(g["stop_hit"].fillna(False).mean()),
                "milestone_50_before_stop_rate": float(e50["milestone_50_before_stop"].fillna(False).mean()) if len(e50) else np.nan,
                "profit_factor_exit_50": _profit_factor(e50["net_return_exit_50"]) if len(e50) else np.nan,
                "mean_net_return_exit_50": float(pd.to_numeric(e50["net_return_exit_50"], errors="coerce").mean()) if len(e50) else np.nan,
            })
    # Raid count is discrete and deserves exact buckets.
    if "raid_count_so_far_at_entry" in q:
        cnt = pd.to_numeric(q["raid_count_so_far_at_entry"], errors="coerce").fillna(0).clip(lower=0)
        bucket = np.where(cnt <= 1, "1", np.where(cnt == 2, "2", np.where(cnt == 3, "3", "4_plus")))
        work = q.assign(_bin=bucket)
        for (arch, b), g in work.groupby(["entry_archetype", "_bin"], sort=True):
            e50 = g.loc[g["milestone_50_ahead_at_fill"].fillna(False).astype(bool)]
            rows.append({"feature":"raid_count_so_far_at_entry","bucket":str(b),"entry_archetype":arch,"n":len(g),
                         "immediate_stop_5m_rate":float(g["stop_within_5m"].fillna(False).mean()),"stop_rate":float(g["stop_hit"].fillna(False).mean()),
                         "milestone_50_before_stop_rate":float(e50["milestone_50_before_stop"].fillna(False).mean()) if len(e50) else np.nan,
                         "profit_factor_exit_50":_profit_factor(e50["net_return_exit_50"]) if len(e50) else np.nan,
                         "mean_net_return_exit_50":float(pd.to_numeric(e50["net_return_exit_50"],errors="coerce").mean()) if len(e50) else np.nan})
    return pd.DataFrame(rows)
