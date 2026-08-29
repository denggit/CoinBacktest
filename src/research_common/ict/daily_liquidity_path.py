#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily liquidity-range path atlas primitives for SOXL ICT research.

This module deliberately starts *before* entry filters.  For each valid trading
session it freezes several causal premarket range definitions, labels what price
did after the first raid of either boundary, and only then attaches causal
MSS/FVG execution candidates.  Future path information is used exclusively as
an outcome label; it is never available to candidate generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, NY_TZ, aggregate_closed_bars, slice_ny_day
from .premarket_mss_fvg_v2 import confirmed_pivots_with_excursion


@dataclass(frozen=True)
class DailyLiquidityPathConfig:
    early_start: dtime = dtime(4, 0)
    early_anchor: dtime = dtime(8, 30)
    rth_anchor: dtime = dtime(9, 30)
    trade_end: dtime = dtime(16, 30)
    prominent_pivot_left: int = 1
    prominent_pivot_right: int = 1
    round_trip_cost: float = 0.0011


def _anchor(day, t: dtime) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(NY_TZ) + pd.Timedelta(hours=t.hour, minutes=t.minute)


def _range_extremes(frame: pd.DataFrame) -> tuple[float, pd.Timestamp, float, pd.Timestamp]:
    if frame.empty:
        return np.nan, pd.NaT, np.nan, pd.NaT
    h = pd.to_numeric(frame["high"], errors="coerce")
    l = pd.to_numeric(frame["low"], errors="coerce")
    if h.dropna().empty or l.dropna().empty:
        return np.nan, pd.NaT, np.nan, pd.NaT
    ht = pd.Timestamp(h.idxmax()); lt = pd.Timestamp(l.idxmin())
    return float(h.loc[ht]), ht, float(l.loc[lt]), lt


def _prominent_15m_pair(frame_1m: pd.DataFrame, anchor_time: pd.Timestamp, cfg: DailyLiquidityPathConfig) -> dict[str, object] | None:
    if frame_1m.empty:
        return None
    tf = aggregate_closed_bars(frame_1m, 15)
    if tf.empty:
        return None
    piv = confirmed_pivots_with_excursion(tf, left=cfg.prominent_pivot_left, right=cfg.prominent_pivot_right)
    if piv.empty:
        return None
    piv = piv.loc[pd.to_datetime(piv["confirmation_available_time"]) <= anchor_time].copy()
    if piv.empty:
        return None
    ranges = pd.to_numeric(tf["high"], errors="coerce") - pd.to_numeric(tf["low"], errors="coerce")
    med = float(ranges.loc[tf.index < anchor_time].median()) if bool((tf.index < anchor_time).any()) else np.nan
    if not np.isfinite(med) or med <= EPS:
        med = float(ranges.median()) if np.isfinite(ranges.median()) else np.nan
    piv["prominence_score"] = pd.to_numeric(piv["two_sided_excursion_abs"], errors="coerce") / med if np.isfinite(med) and med > EPS else np.nan
    hi = piv.loc[piv["pivot_side"].eq("high")].sort_values(["prominence_score", "pivot_time"], ascending=[False, False], kind="mergesort")
    lo = piv.loc[piv["pivot_side"].eq("low")].sort_values(["prominence_score", "pivot_time"], ascending=[False, False], kind="mergesort")
    if hi.empty or lo.empty:
        return None
    hr = hi.iloc[0]; lr = lo.iloc[0]
    upper = float(hr["pivot_price"]); lower = float(lr["pivot_price"])
    if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
        return None
    return {
        "upper_price": upper,
        "upper_source_time": pd.Timestamp(hr["pivot_time"]),
        "upper_confirmation_time": pd.Timestamp(hr["confirmation_available_time"]),
        "upper_prominence_score": float(hr.get("prominence_score", np.nan)),
        "lower_price": lower,
        "lower_source_time": pd.Timestamp(lr["pivot_time"]),
        "lower_confirmation_time": pd.Timestamp(lr["confirmation_available_time"]),
        "lower_prominence_score": float(lr.get("prominence_score", np.nan)),
    }


def build_daily_range_definitions(
    bars_ny: pd.DataFrame,
    days: Sequence,
    *,
    config: DailyLiquidityPathConfig = DailyLiquidityPathConfig(),
) -> pd.DataFrame:
    """Freeze several causal range interpretations for every session.

    The study compares models; it does not pick the one with best PnL in this
    function.  08:30 models can trade from 08:30; 09:30 models are only visible
    after 09:30.
    """
    rows: list[dict[str, object]] = []
    for day in days:
        day_text = str(pd.Timestamp(day).date())
        early = slice_ny_day(bars_ny, pd.Timestamp(day).date(), config.early_start, config.early_anchor)
        full = slice_ny_day(bars_ny, pd.Timestamp(day).date(), config.early_start, config.rth_anchor)
        for model, frame, anchor_t in (
            ("early_extreme_0400_0830", early, config.early_anchor),
            ("full_premarket_extreme_0400_0930", full, config.rth_anchor),
        ):
            upper, ut, lower, lt = _range_extremes(frame)
            if np.isfinite(upper) and np.isfinite(lower) and upper > lower:
                anchor_ts = _anchor(day, anchor_t)
                rows.append({
                    "ny_date": day_text, "range_model": model,
                    "range_available_time": anchor_ts, "path_start_time": anchor_ts,
                    "upper_price": upper, "lower_price": lower,
                    "upper_source_time": ut, "lower_source_time": lt,
                    "upper_confirmation_time": anchor_ts, "lower_confirmation_time": anchor_ts,
                    "upper_prominence_score": np.nan, "lower_prominence_score": np.nan,
                    "range_width_abs": upper-lower,
                })
        for model, frame, anchor_t in (
            ("prominent_15m_pair_0830", early, config.early_anchor),
            ("prominent_15m_pair_0930", full, config.rth_anchor),
        ):
            anchor_ts = _anchor(day, anchor_t)
            pair = _prominent_15m_pair(frame, anchor_ts, config)
            if pair is None:
                continue
            rows.append({
                "ny_date": day_text, "range_model": model,
                "range_available_time": anchor_ts, "path_start_time": anchor_ts,
                **pair,
                "range_width_abs": float(pair["upper_price"])-float(pair["lower_price"]),
            })
    return pd.DataFrame(rows)


def _first_cross_positions(high: np.ndarray, low: np.ndarray, upper: float, lower: float) -> tuple[int | None, int | None]:
    hp = np.flatnonzero(high > upper + EPS)
    lp = np.flatnonzero(low < lower - EPS)
    return (int(hp[0]) if len(hp) else None, int(lp[0]) if len(lp) else None)


def _first_milestone_pos(high: np.ndarray, low: np.ndarray, *, is_long: bool, threshold: float, start: int) -> int | None:
    arr = high[start:] >= threshold - EPS if is_long else low[start:] <= threshold + EPS
    pos = np.flatnonzero(arr)
    return int(start + pos[0]) if len(pos) else None


def _crossing_count(series: np.ndarray, level: float, *, above: bool) -> int:
    if len(series) < 2:
        return 0
    state = series > level + EPS if above else series < level - EPS
    return int(np.sum(state[1:] & ~state[:-1]))


def build_daily_path_outcomes(
    bars_ny: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    config: DailyLiquidityPathConfig = DailyLiquidityPathConfig(),
) -> pd.DataFrame:
    """Label the post-anchor daily path for every range model.

    These are *outcome labels*.  They may use later bars because no trading
    decision is made by this function.  Entry research must join them only after
    its causal candidate has already been generated.
    """
    if ranges.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for r in ranges.to_dict("records"):
        day = pd.Timestamp(r["ny_date"]).date()
        start = pd.Timestamp(r["path_start_time"])
        end = _anchor(day, config.trade_end)
        daybars = slice_ny_day(bars_ny, day, start.timetz().replace(tzinfo=None), config.trade_end)
        if daybars.empty:
            continue
        h = pd.to_numeric(daybars["high"], errors="coerce").to_numpy(float)
        l = pd.to_numeric(daybars["low"], errors="coerce").to_numpy(float)
        c = pd.to_numeric(daybars["close"], errors="coerce").to_numpy(float)
        idx = pd.DatetimeIndex(daybars.index)
        upper = float(r["upper_price"]); lower = float(r["lower_price"]); width = upper-lower
        hp, lp = _first_cross_positions(h, l, upper, lower)
        same_bar = hp is not None and lp is not None and hp == lp
        if hp is None and lp is None:
            first_side = "none"; first_pos = None
        elif same_bar:
            first_side = "both_same_bar"; first_pos = hp
        elif hp is None or (lp is not None and lp < hp):
            first_side = "low"; first_pos = lp
        else:
            first_side = "high"; first_pos = hp

        base = dict(r)
        base.update({
            "first_raid_side": first_side,
            "first_raid_bar_start": pd.NaT if first_pos is None else idx[first_pos],
            "first_raid_time": pd.NaT if first_pos is None else idx[first_pos] + pd.Timedelta(minutes=1),
            "first_raid_same_bar_both_sides": bool(same_bar),
            "range_end_time": end,
        })
        if first_pos is None:
            rows.append({**base, "trade_side": "", "target_price": np.nan, "traversal_complete": False,
                         "path_archetype": "no_boundary_raid", "max_progress_fraction": 0.0})
            continue
        if same_bar:
            rows.append({**base, "trade_side": "", "target_price": np.nan, "traversal_complete": True,
                         "path_archetype": "same_bar_double_raid_ambiguous", "max_progress_fraction": 1.0})
            continue

        is_long = first_side == "low"
        trade_side = "LONG" if is_long else "SHORT"
        source_level = lower if is_long else upper
        target = upper if is_long else lower
        target_pos = _first_milestone_pos(h, l, is_long=is_long, threshold=target, start=first_pos)
        traversal = target_pos is not None
        stop_end = target_pos if target_pos is not None else len(daybars)-1
        seg_h = h[first_pos:stop_end+1]; seg_l = l[first_pos:stop_end+1]; seg_c = c[first_pos:stop_end+1]
        raid_pen = max(0.0, source_level - l[first_pos]) if is_long else max(0.0, h[first_pos]-source_level)
        max_pen = max(0.0, source_level - np.nanmin(seg_l)) if is_long else max(0.0, np.nanmax(seg_h)-source_level)
        if is_long:
            progress = (np.nanmax(seg_h)-lower) / width if width > EPS else np.nan
            outside = seg_c < lower - EPS
            reclaim = np.flatnonzero(seg_c >= lower - EPS)
            rer = _crossing_count(seg_l, lower, above=False)
        else:
            progress = (upper-np.nanmin(seg_l)) / width if width > EPS else np.nan
            outside = seg_c > upper + EPS
            reclaim = np.flatnonzero(seg_c <= upper + EPS)
            rer = _crossing_count(seg_h, upper, above=True)
        reclaim_rel = int(reclaim[0]) if len(reclaim) else None
        reclaim_pos = first_pos + reclaim_rel if reclaim_rel is not None else None
        progress_clip = float(max(0.0, progress)) if np.isfinite(progress) else np.nan

        milestones: dict[str, object] = {}
        for frac in (0.25, 0.50, 0.75, 1.00):
            level = lower + width*frac if is_long else upper-width*frac
            p = _first_milestone_pos(h, l, is_long=is_long, threshold=level, start=first_pos)
            milestones[f"milestone_{int(frac*100)}_time"] = pd.NaT if p is None else idx[p] + pd.Timedelta(minutes=1)
            milestones[f"milestone_{int(frac*100)}_minutes"] = np.nan if p is None else float((idx[p]-idx[first_pos]).total_seconds()/60.0)

        if traversal:
            archetype = "first_raid_to_opposite_boundary"
        elif progress_clip >= 0.75:
            archetype = "partial_reversal_75_100"
        elif progress_clip >= 0.50:
            archetype = "partial_reversal_50_75"
        elif reclaim_pos is not None:
            archetype = "reclaim_but_less_than_half_range"
        else:
            archetype = "same_side_acceptance_or_continuation"
        rows.append({
            **base, **milestones,
            "trade_side": trade_side, "source_level_price": source_level, "target_price": target,
            "traversal_complete": bool(traversal),
            "opposite_hit_time": pd.NaT if target_pos is None else idx[target_pos] + pd.Timedelta(minutes=1),
            "opposite_hit_minutes_from_raid": np.nan if target_pos is None else float((idx[target_pos]-idx[first_pos]).total_seconds()/60.0),
            "first_raid_penetration_abs": float(raid_pen),
            "first_raid_penetration_frac_range": float(raid_pen/width) if width > EPS else np.nan,
            "max_same_side_penetration_abs": float(max_pen),
            "max_same_side_penetration_frac_range": float(max_pen/width) if width > EPS else np.nan,
            "first_reclaim_time": pd.NaT if reclaim_pos is None else idx[reclaim_pos] + pd.Timedelta(minutes=1),
            "reclaim_minutes": np.nan if reclaim_pos is None else float((idx[reclaim_pos]-idx[first_pos]).total_seconds()/60.0),
            "outside_close_count_before_target_or_close": int(np.sum(outside)),
            "same_side_raid_count": int(rer),
            "max_progress_fraction": progress_clip,
            "path_archetype": archetype,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["path_event_id"] = [f"PATH|{d}|{m}|{i:06d}" for i, (d, m) in enumerate(zip(out["ny_date"], out["range_model"]))]
    return out


def path_events_to_sweep_events(paths: pd.DataFrame) -> pd.DataFrame:
    """Convert first-raids into causal sweep rows for the MSS/FVG engine."""
    if paths.empty:
        return pd.DataFrame()
    q = paths.loc[paths["first_raid_side"].isin(["high", "low"])].copy()
    if q.empty:
        return q
    q["event_id"] = q["path_event_id"].astype(str)
    q["sweep_time"] = pd.to_datetime(q["first_raid_time"])
    q["liquidity_side"] = q["first_raid_side"].astype(str)
    q["level_price"] = np.where(q["first_raid_side"].eq("low"), pd.to_numeric(q["lower_price"], errors="coerce"), pd.to_numeric(q["upper_price"], errors="coerce"))
    q["liquidity_family"] = q["range_model"].astype(str)
    q["level_type"] = q["range_model"].astype(str) + "_" + q["first_raid_side"].astype(str)
    q["setup_eligible_at_sweep"] = True
    return q


def summarize_path_outcomes(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows = []
    for model, g in paths.groupby("range_model", sort=True):
        valid = g.loc[g["first_raid_side"].isin(["high", "low"])].copy()
        trav = valid.loc[valid["traversal_complete"].fillna(False).astype(bool)]
        rows.append({
            "range_model": model,
            "days": int(g["ny_date"].nunique()),
            "days_with_first_raid": int(valid["ny_date"].nunique()),
            "first_raid_day_rate": float(valid["ny_date"].nunique()/max(1,g["ny_date"].nunique())),
            "traversal_days": int(trav["ny_date"].nunique()),
            "traversal_rate_given_raid": float(len(trav)/len(valid)) if len(valid) else np.nan,
            "traversals_per_session": float(len(trav)/max(1,g["ny_date"].nunique())),
            "median_minutes_raid_to_opposite": float(pd.to_numeric(trav["opposite_hit_minutes_from_raid"], errors="coerce").median()) if len(trav) else np.nan,
            "median_max_progress_fraction_failed": float(pd.to_numeric(valid.loc[~valid["traversal_complete"].fillna(False).astype(bool), "max_progress_fraction"], errors="coerce").median()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def period_label(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    if t < pd.Timestamp("2025-01-01"):
        return "discovery_2023h2_2024"
    if t < pd.Timestamp("2026-01-01"):
        return "forward_2025"
    return "late_2026"


def summarize_path_by_period(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    q = paths.copy(); q["period"] = [period_label(x) for x in q["ny_date"]]
    rows=[]
    for (model, period), g in q.groupby(["range_model","period"], sort=True):
        valid=g.loc[g["first_raid_side"].isin(["high","low"])]
        rows.append({"range_model":model,"period":period,"days":int(g["ny_date"].nunique()),"raid_days":int(valid["ny_date"].nunique()),"traversal_days":int(valid.loc[valid["traversal_complete"].fillna(False).astype(bool),"ny_date"].nunique()),"traversal_rate_given_raid":float(valid["traversal_complete"].fillna(False).mean()) if len(valid) else np.nan})
    return pd.DataFrame(rows)


def replay_entry_candidates(
    bars_ny: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    cost: float = 0.0011,
    trade_end: dtime = dtime(16, 30),
) -> pd.DataFrame:
    """Conservative 1m lifecycle replay for path-study entry candidates.

    Limit orders are cancelled if target or stop is reached before fill.
    Same-minute stop+target after fill is resolved to stop.  This helper is
    intentionally account-agnostic: R15 studies entry-path mechanics, not a
    portfolio with overlapping positions.
    """
    if entries.empty:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    cache: dict[str, dict[str, object]] = {}
    for r in entries.to_dict("records"):
        day_text = str(r["ny_date"])
        data = cache.get(day_text)
        if data is None:
            day = slice_ny_day(bars_ny, pd.Timestamp(day_text).date(), dtime(8, 30), trade_end)
            idx = pd.DatetimeIndex(day.index).as_unit("ns")
            data = {
                "idx": idx, "ns": idx.asi8,
                "open": pd.to_numeric(day["open"], errors="coerce").to_numpy(float),
                "high": pd.to_numeric(day["high"], errors="coerce").to_numpy(float),
                "low": pd.to_numeric(day["low"], errors="coerce").to_numpy(float),
                "close": pd.to_numeric(day["close"], errors="coerce").to_numpy(float),
            }
            cache[day_text] = data
        idx = data["idx"]; ns = data["ns"]; op = data["open"]; hi = data["high"]; lo = data["low"]; cl = data["close"]
        is_long = str(r.get("trade_side", "")) == "LONG"
        entry = float(r.get("entry_price", np.nan)); stop = float(r.get("stop_price", np.nan)); target = float(r.get("target_price", np.nan))
        signal = pd.Timestamp(r.get("entry_available_time", r.get("signal_time")))
        rec = dict(r)
        rec.update({"filled": False, "fill_time": pd.NaT, "exit_time": pd.NaT, "exit_reason": "invalid", "gross_return": np.nan, "net_return": np.nan, "mfe_r": np.nan, "mae_r": np.nan})
        if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target)):
            out.append(rec); continue
        risk = entry-stop if is_long else stop-entry
        reward = target-entry if is_long else entry-target
        if risk <= EPS or reward <= EPS:
            rec["exit_reason"] = "invalid_risk_reward"; out.append(rec); continue
        pos = int(np.searchsorted(ns, int(signal.value), side="left"))
        if pos >= len(idx):
            rec["exit_reason"] = "signal_after_session"; out.append(rec); continue
        order_type = str(r.get("entry_order_type", "limit"))
        fill_pos: int | None = None
        if order_type == "market_next_open":
            # entry_time/price were already attached from the causal execution frame.
            fill_pos = pos
        else:
            for j in range(pos, len(idx)):
                target_pre = hi[j] >= target-EPS if is_long else lo[j] <= target+EPS
                stop_pre = lo[j] <= stop+EPS if is_long else hi[j] >= stop-EPS
                fill = lo[j] <= entry+EPS <= hi[j]
                if fill:
                    fill_pos = j; break
                if target_pre:
                    rec["exit_reason"] = "target_before_fill"; break
                if stop_pre:
                    rec["exit_reason"] = "stop_before_fill"; break
            if fill_pos is None:
                if rec["exit_reason"] == "invalid": rec["exit_reason"] = "session_end_unfilled"
                out.append(rec); continue
        rec["filled"] = True; rec["fill_time"] = idx[fill_pos]
        max_fav = 0.0; max_adv = 0.0; exit_pos = None; exit_price = np.nan; reason = "session_close"
        for j in range(fill_pos, len(idx)):
            if is_long:
                max_fav = max(max_fav, hi[j]-entry); max_adv = max(max_adv, entry-lo[j])
                hit_s = lo[j] <= stop+EPS; hit_t = hi[j] >= target-EPS
            else:
                max_fav = max(max_fav, entry-lo[j]); max_adv = max(max_adv, hi[j]-entry)
                hit_s = hi[j] >= stop-EPS; hit_t = lo[j] <= target+EPS
            if hit_s:  # conservative if both hit in same minute
                exit_pos=j; exit_price=stop; reason="stop"; break
            if hit_t:
                exit_pos=j; exit_price=target; reason="target"; break
        if exit_pos is None:
            exit_pos=len(idx)-1; exit_price=float(cl[-1]); reason="session_close"
        gross = (exit_price-entry)/entry if is_long else (entry-exit_price)/entry
        rec["exit_time"] = idx[exit_pos]
        rec["exit_reason"] = reason
        rec["gross_return"] = float(gross)
        rec["net_return"] = float(gross-cost)
        rec["mfe_r"] = float(max_fav/risk)
        rec["mae_r"] = float(-max_adv/risk)
        out.append(rec)
    return pd.DataFrame(out)


def summarize_entry_capture(replayed: pd.DataFrame) -> pd.DataFrame:
    if replayed.empty:
        return pd.DataFrame()
    rows=[]
    group_cols=[c for c in ["range_model","execution_tf","structure_visibility_tier_r13","entry_model_r13"] if c in replayed.columns]
    for key,g in replayed.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple): key=(key,)
        meta=dict(zip(group_cols,key))
        filled=g.loc[g["filled"].fillna(False).astype(bool)].copy()
        wins=filled.loc[pd.to_numeric(filled["net_return"], errors="coerce")>0]
        pos=pd.to_numeric(filled["net_return"], errors="coerce").clip(lower=0).sum()
        neg=-pd.to_numeric(filled["net_return"], errors="coerce").clip(upper=0).sum()
        traversal_days=int(g.loc[g["traversal_complete"].fillna(False).astype(bool),"ny_date"].nunique()) if "traversal_complete" in g else 0
        captured_days=int(filled.loc[(filled["traversal_complete"].fillna(False).astype(bool)) & (filled["exit_reason"].eq("target")),"ny_date"].nunique()) if "traversal_complete" in filled else 0
        rows.append({**meta,
            "candidates":len(g),"filled":len(filled),"fill_rate":len(filled)/len(g) if len(g) else np.nan,
            "win_rate":len(wins)/len(filled) if len(filled) else np.nan,
            "profit_factor":float(pos/neg) if neg>EPS else (np.inf if pos>0 else np.nan),
            "mean_net_return":float(pd.to_numeric(filled["net_return"], errors="coerce").mean()) if len(filled) else np.nan,
            "target_hit_rate":float(filled["exit_reason"].eq("target").mean()) if len(filled) else np.nan,
            "target_before_fill_rate":float(g["exit_reason"].eq("target_before_fill").mean()),
            "traversal_days":traversal_days,"captured_traversal_days":captured_days,
            "capture_rate_of_traversal_days":captured_days/traversal_days if traversal_days else np.nan,
            "median_mfe_r":float(pd.to_numeric(filled["mfe_r"], errors="coerce").median()) if len(filled) else np.nan,
            "median_mae_r":float(pd.to_numeric(filled["mae_r"], errors="coerce").median()) if len(filled) else np.nan,
        })
    return pd.DataFrame(rows)
