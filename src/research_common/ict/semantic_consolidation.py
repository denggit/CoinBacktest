#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 causal semantic-consolidation primitives for SOXL ICT research.

R13 does not try to manufacture a single strict ICT setup.  It consolidates the
wide R12 atlas into comparable, causal narratives while preserving the raw
candidates:

* liquidity consumption is a state/feature vector rather than binary touched;
* a shallow raid can remain a conservative target later in the same session;
* structure references are compared by causal selection models (latest,
  highest-visibility, outermost barrier), not collapsed to one universal rule;
* FVG trains are reduced to a handful of pre-declared execution choices without
  a hard Swing +/- $0.10 gate;
* target state and entry-distance diagnostics are attached before replay, never
  from future PnL.

The heuristic liquidity-state labels are descriptive research labels only.  The
continuous state features are the source of truth for later validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, NY_TZ, TRADE_END, aggregate_closed_bars, slice_ny_day


@dataclass(frozen=True)
class LiquidityConsumptionConfig:
    context_bars: int = 20
    shallow_penetration_vs_range: float = 0.50
    deep_penetration_vs_range: float = 2.00
    accepted_consecutive_closes: int = 2
    quick_reclaim_minutes: int = 3


def _as_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.as_unit("ns").asi8


def _max_consecutive_true(x: np.ndarray) -> np.ndarray:
    """Cumulative maximum run length of True, vector output."""
    out = np.zeros(len(x), dtype=np.int32)
    run = 0
    best = 0
    for i, flag in enumerate(np.asarray(x, dtype=bool)):
        if flag:
            run += 1
            best = max(best, run)
        else:
            run = 0
        out[i] = best
    return out


def _state_label(
    *,
    breached: bool,
    penetration_vs_range: float,
    max_consecutive_outside_closes: int,
    reclaim_minutes: float,
    config: LiquidityConsumptionConfig,
) -> str:
    if not breached:
        return "fresh"
    shallow = np.isfinite(penetration_vs_range) and penetration_vs_range <= float(config.shallow_penetration_vs_range)
    quick = np.isfinite(reclaim_minutes) and reclaim_minutes <= float(config.quick_reclaim_minutes)
    if shallow and max_consecutive_outside_closes <= 1 and quick:
        return "shallow_probe_equal_like"
    accepted = max_consecutive_outside_closes >= int(config.accepted_consecutive_closes)
    deep = np.isfinite(penetration_vs_range) and penetration_vs_range >= float(config.deep_penetration_vs_range)
    if accepted or deep:
        return "accepted_or_deep_consumed"
    return "partial_consumed"


def build_liquidity_consumption_query_index(
    bars_ny: pd.DataFrame,
    levels: pd.DataFrame,
    *,
    config: LiquidityConsumptionConfig = LiquidityConsumptionConfig(),
) -> dict[tuple[str, str, float], dict[str, object]]:
    """Precompute same-session causal consumption paths for each level.

    A session level never persists into another date here.  If a raid extreme
    later becomes a genuine swing, that swing is represented by the separate
    swing hierarchy, not by carrying this session label forward.
    """
    if levels.empty:
        return {}
    out: dict[tuple[str, str, float], dict[str, object]] = {}
    for day_text, day_levels in levels.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        day_bars = slice_ny_day(bars_ny, day, dtime(4, 0), TRADE_END)
        if day_bars.empty:
            continue
        idx = pd.DatetimeIndex(day_bars.index)
        av = idx + pd.Timedelta(minutes=1)
        av_ns = _as_ns(av)
        high = pd.to_numeric(day_bars["high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(day_bars["low"], errors="coerce").to_numpy(float)
        close = pd.to_numeric(day_bars["close"], errors="coerce").to_numpy(float)
        ranges = high - low

        # Deduplicate price-identical levels by day/side while keeping the
        # earliest availability.  State at a price is market state, independent
        # of how many labels point to it.
        tmp = day_levels.copy()
        tmp["_avail"] = pd.to_datetime(tmp.get("level_available_time"), errors="coerce")
        for (side, price), g in tmp.groupby(["liquidity_side", "level_price"], sort=False):
            side = str(side); price = float(price)
            avail_candidates = pd.to_datetime(g["_avail"], errors="coerce").dropna()
            level_av = pd.Timestamp(avail_candidates.min()) if len(avail_candidates) else pd.Timestamp.combine(day, dtime(8,30)).tz_localize(NY_TZ)
            start = int(np.searchsorted(av_ns, int(level_av.value), side="left"))
            if start >= len(day_bars):
                continue

            if side == "high":
                penetration = np.maximum(high - price, 0.0)
                breached = high > price
                outside_close = close > price
                reclaimed_close = close <= price
            else:
                penetration = np.maximum(price - low, 0.0)
                breached = low < price
                outside_close = close < price
                reclaimed_close = close >= price

            # Before availability state is irrelevant; make all cumulative
            # arrays start from the causal availability point.
            p = penetration[start:]
            b = breached[start:]
            oc = outside_close[start:]
            rc = reclaimed_close[start:]
            max_pen = np.maximum.accumulate(np.where(np.isfinite(p), p, 0.0))
            outside_count = np.cumsum(oc.astype(np.int32))
            max_run = _max_consecutive_true(oc)

            first_hit_rel = int(np.flatnonzero(b)[0]) if np.any(b) else -1
            first_hit_abs = start + first_hit_rel if first_hit_rel >= 0 else -1
            first_reclaim_abs = -1
            if first_hit_rel >= 0:
                reclaim_hits = np.flatnonzero(rc[first_hit_rel:])
                if len(reclaim_hits):
                    first_reclaim_abs = start + first_hit_rel + int(reclaim_hits[0])

            # Prior local volatility is frozen at the first raid.  This makes
            # penetration comparable across SOXL price/volatility regimes.
            prior_med = np.nan
            if first_hit_abs >= 0:
                lo_ctx = max(0, first_hit_abs - int(config.context_bars))
                prior = ranges[lo_ctx:first_hit_abs]
                prior = prior[np.isfinite(prior) & (prior > EPS)]
                if len(prior):
                    prior_med = float(np.median(prior))

            out[(str(day_text), side, round(price, 8))] = {
                "available_ns": av_ns[start:],
                "max_penetration_abs": max_pen,
                "outside_close_count": outside_count,
                "max_consecutive_outside_closes": max_run,
                "first_breach_time": pd.Timestamp(av[first_hit_abs]) if first_hit_abs >= 0 else pd.NaT,
                "first_reclaim_time": pd.Timestamp(av[first_reclaim_abs]) if first_reclaim_abs >= 0 else pd.NaT,
                "prior_median_1m_range_abs": prior_med,
                "level_price": price,
                "liquidity_side": side,
                "level_available_time": level_av,
            }
    return out


def query_liquidity_consumption_state(
    index: dict[tuple[str, str, float], dict[str, object]],
    *,
    day_text: str,
    side: str,
    price: float,
    query_time: pd.Timestamp,
    config: LiquidityConsumptionConfig = LiquidityConsumptionConfig(),
) -> dict[str, object]:
    key = (str(day_text), str(side), round(float(price), 8))
    data = index.get(key)
    if data is None:
        return {
            "liquidity_state": "unknown",
            "liquidity_breached": False,
            "liquidity_max_penetration_abs": np.nan,
            "liquidity_max_penetration_pct": np.nan,
            "liquidity_penetration_vs_prior_range": np.nan,
            "liquidity_outside_close_count": np.nan,
            "liquidity_max_consecutive_outside_closes": np.nan,
            "liquidity_first_breach_time": pd.NaT,
            "liquidity_first_reclaim_time": pd.NaT,
            "liquidity_reclaim_minutes": np.nan,
            "liquidity_prior_median_1m_range_abs": np.nan,
        }
    av_ns = np.asarray(data["available_ns"], dtype=np.int64)
    qpos = int(np.searchsorted(av_ns, int(pd.Timestamp(query_time).value), side="right") - 1)
    if qpos < 0:
        return {
            "liquidity_state": "fresh",
            "liquidity_breached": False,
            "liquidity_max_penetration_abs": 0.0,
            "liquidity_max_penetration_pct": 0.0,
            "liquidity_penetration_vs_prior_range": 0.0,
            "liquidity_outside_close_count": 0,
            "liquidity_max_consecutive_outside_closes": 0,
            "liquidity_first_breach_time": pd.NaT,
            "liquidity_first_reclaim_time": pd.NaT,
            "liquidity_reclaim_minutes": np.nan,
            "liquidity_prior_median_1m_range_abs": data.get("prior_median_1m_range_abs", np.nan),
        }
    pen = float(np.asarray(data["max_penetration_abs"])[qpos])
    price0 = float(data["level_price"])
    med = float(data.get("prior_median_1m_range_abs", np.nan))
    pen_mult = pen / med if np.isfinite(med) and med > EPS else np.nan
    first_breach = data.get("first_breach_time", pd.NaT)
    first_reclaim = data.get("first_reclaim_time", pd.NaT)
    # A reclaim that happens after the query is not yet known.
    if not pd.isna(first_reclaim) and pd.Timestamp(first_reclaim) > pd.Timestamp(query_time):
        first_reclaim = pd.NaT
    reclaim_minutes = (
        (pd.Timestamp(first_reclaim) - pd.Timestamp(first_breach)).total_seconds() / 60.0
        if not pd.isna(first_breach) and not pd.isna(first_reclaim) else np.nan
    )
    breached_now = bool(pen > EPS and not pd.isna(first_breach) and pd.Timestamp(first_breach) <= pd.Timestamp(query_time))
    run = int(np.asarray(data["max_consecutive_outside_closes"])[qpos])
    label = _state_label(
        breached=breached_now,
        penetration_vs_range=pen_mult,
        max_consecutive_outside_closes=run,
        reclaim_minutes=reclaim_minutes,
        config=config,
    )
    return {
        "liquidity_state": label,
        "liquidity_breached": breached_now,
        "liquidity_max_penetration_abs": pen,
        "liquidity_max_penetration_pct": pen / abs(price0) if abs(price0) > EPS else np.nan,
        "liquidity_penetration_vs_prior_range": pen_mult,
        "liquidity_outside_close_count": int(np.asarray(data["outside_close_count"])[qpos]),
        "liquidity_max_consecutive_outside_closes": run,
        "liquidity_first_breach_time": first_breach if breached_now else pd.NaT,
        "liquidity_first_reclaim_time": first_reclaim,
        "liquidity_reclaim_minutes": reclaim_minutes,
        "liquidity_prior_median_1m_range_abs": med,
    }


def attach_consumption_state_to_fvg_rows(
    fvgs: pd.DataFrame,
    *,
    state_index: dict[tuple[str, str, float], dict[str, object]],
    config: LiquidityConsumptionConfig = LiquidityConsumptionConfig(),
) -> pd.DataFrame:
    """Attach source-liquidity and opposite-target state at each FVG signal."""
    if fvgs.empty:
        return fvgs.copy()
    rows = []
    for r in fvgs.to_dict("records"):
        day_text = str(r["ny_date"])
        signal = pd.Timestamp(r["signal_time"])
        trade_side = str(r["trade_side"])
        source_side = "low" if trade_side == "LONG" else "high"
        target_side = "high" if trade_side == "LONG" else "low"
        source = query_liquidity_consumption_state(
            state_index, day_text=day_text, side=source_side, price=float(r["level_price"]), query_time=signal, config=config
        )
        target_price = r.get("target_price", np.nan)
        if target_price is not None and np.isfinite(float(target_price)):
            target = query_liquidity_consumption_state(
                state_index, day_text=day_text, side=target_side, price=float(target_price), query_time=signal, config=config
            )
        else:
            target = {"liquidity_state": "missing"}
        row = dict(r)
        for k, v in source.items():
            row[f"source_{k}"] = v
        for k, v in target.items():
            row[f"target_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def select_reference_narratives(breaks: pd.DataFrame) -> pd.DataFrame:
    """Reduce R12 break atlas to pre-declared causal reference models.

    The raw break atlas remains available.  R13 only selects one reference per
    event/timeframe/break-bar for each model so performance comparisons are not
    inflated by hundreds of equivalent pivots.
    """
    if breaks.empty:
        return pd.DataFrame()
    specs = [
        ("latest_newly_broken", "reference_is_latest_newly_broken"),
        ("highest_visibility_newly_broken", "reference_is_highest_visibility_newly_broken"),
        ("outermost_barrier_newly_broken", "reference_is_outermost_barrier_newly_broken"),
    ]
    rows = []
    for model, flag in specs:
        if flag not in breaks:
            continue
        q = breaks.loc[breaks[flag].fillna(False).astype(bool)].copy()
        if q.empty:
            continue
        q["reference_model_r13"] = model
        rows.append(q)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    # The same pivot can satisfy multiple selector rules.  Keep the model labels
    # but never duplicate within one model/break.
    dedup = ["event_id", "execution_tf", "break_available_time", "reference_model_r13"]
    return out.sort_values(dedup + ["mss_reference_time"], kind="mergesort").drop_duplicates(dedup, keep="last").reset_index(drop=True)


def _fvg_ce(row: pd.Series | dict[str, object]) -> float:
    return float((float(row["fvg_near_edge_entry"]) + float(row["fvg_far_edge"])) / 2.0)


def consolidate_fvg_entry_choices(
    narratives: pd.DataFrame,
    fvgs_with_state: pd.DataFrame,
) -> pd.DataFrame:
    """Choose the fixed R13 FVG execution models without Python group loops.

    Semantics are identical to the original implementation: first train near/CE,
    last pre-or-on-break near, latest break-middle near/CE, and closest-to-broken-
    swing near.  The implementation performs global stable sorts/drop-duplicates
    instead of sorting one tiny DataFrame for every narrative.  This matters for
    long-history SOXL where >100k narrative groups are common.
    """
    if narratives.empty or fvgs_with_state.empty:
        return pd.DataFrame()
    key_cols = ["event_id", "execution_tf", "break_available_time", "mss_reference_time", "mss_reference_price"]
    ncols = key_cols + ["reference_model_r13"]
    n = narratives[ncols].drop_duplicates().copy()
    merged = fvgs_with_state.merge(n, on=key_cols, how="inner", validate="many_to_many")
    if merged.empty:
        return pd.DataFrame()

    group_cols = ["event_id", "execution_tf", "break_available_time", "reference_model_r13"]
    order_cols = ["fvg_available_time", "fvg_third_pos", "fvg_train_sequence"]
    ordered = merged.sort_values(group_cols + order_cols, kind="mergesort").reset_index(drop=True)

    selected: list[pd.DataFrame] = []

    def emit(frame: pd.DataFrame, model: str, price_mode: str) -> None:
        if frame.empty:
            return
        q = frame.copy()
        q["entry_model_r13"] = model
        q["entry_order_type"] = "limit"
        if price_mode == "near":
            q["entry_price"] = pd.to_numeric(q["fvg_near_edge_entry"], errors="coerce")
            if "swing_buffer_cap_pass" in q:
                q["legacy_swing_0p10_cap_pass_r13"] = q["swing_buffer_cap_pass"].astype(bool)
            else:
                q["legacy_swing_0p10_cap_pass_r13"] = False
        else:
            q["entry_price"] = (
                pd.to_numeric(q["fvg_near_edge_entry"], errors="coerce")
                + pd.to_numeric(q["fvg_far_edge"], errors="coerce")
            ) / 2.0
            q["legacy_swing_0p10_cap_pass_r13"] = np.nan
        q["entry_available_time"] = pd.to_datetime(q["signal_time"])
        ref = pd.to_numeric(q["mss_reference_price"], errors="coerce")
        ent = pd.to_numeric(q["entry_price"], errors="coerce")
        is_long = q["trade_side"].astype(str).eq("LONG")
        q["entry_distance_from_broken_swing_r13"] = np.where(is_long, ent - ref, ref - ent)
        q["entry_distance_abs_r13"] = pd.to_numeric(q["entry_distance_from_broken_swing_r13"], errors="coerce").abs()
        selected.append(q)

    first = ordered.drop_duplicates(group_cols, keep="first")
    last = ordered.drop_duplicates(group_cols, keep="last")
    emit(first, "first_train_near", "near")
    emit(first, "first_train_ce", "ce")
    emit(last, "last_pre_or_on_break_near", "near")

    bm = ordered.loc[ordered["fvg_middle_relation_to_break"].astype(str).eq("break_bar_middle")]
    if not bm.empty:
        bm_last = bm.drop_duplicates(group_cols, keep="last")
        emit(bm_last, "break_middle_near", "near")
        emit(bm_last, "break_middle_ce", "ce")

    dist = pd.to_numeric(ordered["entry_distance_from_broken_swing"], errors="coerce").abs()
    closest_pool = ordered.loc[dist.notna()].copy()
    if not closest_pool.empty:
        closest_pool["_distance_abs_r13_tmp"] = pd.to_numeric(closest_pool["entry_distance_from_broken_swing"], errors="coerce").abs()
        closest_pool = closest_pool.sort_values(group_cols + ["_distance_abs_r13_tmp"] + order_cols, kind="mergesort")
        closest = closest_pool.drop_duplicates(group_cols, keep="first").drop(columns=["_distance_abs_r13_tmp"])
        emit(closest, "closest_to_broken_swing_near", "near")

    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True, sort=False)
    # Preserve the original model order inside each group so downstream audit
    # output remains deterministic even though selection is vectorised.
    model_order = {
        "first_train_near": 0,
        "first_train_ce": 1,
        "last_pre_or_on_break_near": 2,
        "break_middle_near": 3,
        "break_middle_ce": 4,
        "closest_to_broken_swing_near": 5,
    }
    out["_model_order_r13"] = out["entry_model_r13"].map(model_order).fillna(99).astype(int)
    out = out.sort_values(group_cols + ["_model_order_r13"], kind="mergesort").drop(columns=["_model_order_r13"]).reset_index(drop=True)
    out["attempt_id_r13"] = [f"R13|FVG|{i:09d}" for i in range(len(out))]
    return out

def expand_target_state_variants(entries: pd.DataFrame) -> pd.DataFrame:
    """Vectorised target expansion with the same R13 target semantics."""
    if entries.empty:
        return pd.DataFrame()
    base = entries.copy().reset_index(drop=True)
    base["_row_order_r13"] = np.arange(len(base), dtype=np.int64)
    entry = pd.to_numeric(base["entry_price"], errors="coerce")
    stop = pd.to_numeric(base["stop_price"], errors="coerce")
    is_long = base["trade_side"].astype(str).eq("LONG")
    risk_all = pd.Series(np.where(is_long, entry - stop, stop - entry), index=base.index, dtype=float)
    frames: list[pd.DataFrame] = []

    def emit(model: str, source_col: str, order: int, extra_mask: pd.Series | None = None) -> None:
        if source_col not in base:
            return
        target_all = pd.to_numeric(base[source_col], errors="coerce")
        mask = target_all.notna() & np.isfinite(target_all.to_numpy(float, na_value=np.nan)) & risk_all.gt(EPS)
        if extra_mask is not None:
            mask &= extra_mask.fillna(False).astype(bool)
        if not bool(mask.any()):
            return
        q = base.loc[mask].copy()
        target = target_all.loc[mask].astype(float)
        ent = entry.loc[mask].astype(float)
        lng = is_long.loc[mask]
        reward = pd.Series(np.where(lng, target - ent, ent - target), index=q.index, dtype=float)
        valid = reward.gt(EPS) & np.isfinite(reward.to_numpy(float, na_value=np.nan))
        if not bool(valid.any()):
            return
        q = q.loc[valid].copy()
        target = target.loc[valid]
        reward = reward.loc[valid]
        risk = risk_all.loc[q.index].astype(float)
        q["target_model_r13"] = model
        q["target_price_r13"] = target.to_numpy(float)
        q["risk_abs_r13"] = risk.to_numpy(float)
        q["planned_reward_abs_r13"] = reward.to_numpy(float)
        q["planned_rr_r13"] = reward.to_numpy(float) / risk.to_numpy(float)
        q["_target_order_r13"] = int(order)
        frames.append(q)

    emit("external_any_state", "target_price", 0)
    state_ok = ~base.get("target_liquidity_state", pd.Series("unknown", index=base.index)).astype(str).eq("accepted_or_deep_consumed")
    emit("external_if_not_fully_consumed", "target_price", 1, state_ok)
    emit("local_equilibrium_50", "local_equilibrium_50", 2)
    emit("local_opposite_15m_swing", "local_opposite_15m_price", 3)
    emit("nearest_internal_structure", "nearest_internal_target_price", 4)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=False, sort=False)
    out = out.sort_values(["_row_order_r13", "_target_order_r13"], kind="mergesort").drop(columns=["_row_order_r13", "_target_order_r13"]).reset_index(drop=True)
    out["attempt_id_r13"] = [f"R13|TGT|{i:09d}" for i in range(len(out))]
    return out

def add_market_next_open_choices(narratives: pd.DataFrame) -> pd.DataFrame:
    """Create close-break market choices with structural stop attached."""
    if narratives.empty:
        return pd.DataFrame()
    q = narratives.loc[narratives["break_close_cross"].fillna(False).astype(bool)].copy()
    if q.empty:
        return q
    q["entry_model_r13"] = "close_break_next_open_market"
    q["entry_order_type"] = "market_next_open"
    q["entry_price"] = np.nan
    q["entry_available_time"] = pd.to_datetime(q["break_available_time"])
    q["signal_time"] = pd.to_datetime(q["break_available_time"])
    q["stop_price"] = pd.to_numeric(q["terminal_extreme_price"], errors="coerce")
    q["attempt_id_r13"] = [f"R13|MKT|{i:09d}" for i in range(len(q))]
    return q


def attach_market_next_open_from_bars(bars_ny: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach next execution-bar open using vectorised per-day/timeframe search."""
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    out["market_next_open_price"] = np.nan
    out["market_next_open_time"] = pd.Series(pd.NaT, index=out.index, dtype=f"datetime64[ns, {NY_TZ}]")
    grouped = out.groupby(["ny_date", "execution_tf_minutes"], sort=False).groups
    for (day_text, tf), idxs in grouped.items():
        day = pd.Timestamp(day_text).date()
        one = slice_ny_day(bars_ny, day, dtime(4, 0), TRADE_END)
        frame = aggregate_closed_bars(one, int(tf))
        if frame.empty:
            continue
        starts = pd.DatetimeIndex(frame.index).as_unit("ns")
        starts_ns = starts.asi8
        idx_arr = np.asarray(list(idxs), dtype=np.int64)
        t = pd.DatetimeIndex(pd.to_datetime(out.loc[idx_arr, "break_available_time"])).as_unit("ns")
        pos = np.searchsorted(starts_ns, t.asi8, side="left")
        valid = pos < len(frame)
        if not bool(np.any(valid)):
            continue
        good_idx = idx_arr[valid]
        good_pos = pos[valid]
        out.loc[good_idx, "market_next_open_price"] = pd.to_numeric(frame.iloc[good_pos]["open"], errors="coerce").to_numpy(float)
        out.loc[good_idx, "market_next_open_time"] = pd.DatetimeIndex(frame.index[good_pos]).to_numpy()
        out.loc[good_idx, "entry_price"] = out.loc[good_idx, "market_next_open_price"].to_numpy(float)
        out.loc[good_idx, "entry_available_time"] = pd.DatetimeIndex(frame.index[good_pos]).to_numpy()
    return out

def expand_market_target_state_variants(
    market_rows: pd.DataFrame,
    *,
    state_index: dict[tuple[str, str, float], dict[str, object]],
    config: LiquidityConsumptionConfig = LiquidityConsumptionConfig(),
) -> pd.DataFrame:
    """Attach target state and target choices to market-next-open entries."""
    if market_rows.empty:
        return pd.DataFrame()
    rows = []
    for r in market_rows.to_dict("records"):
        entry = r.get("entry_price", np.nan)
        stop = r.get("stop_price", r.get("terminal_extreme_price", np.nan))
        if entry is None or stop is None or not np.isfinite(float(entry)) or not np.isfinite(float(stop)):
            continue
        rec = dict(r)
        rec["stop_price"] = float(stop)
        signal = pd.Timestamp(rec.get("entry_available_time", rec["break_available_time"]))
        target = rec.get("target_price", np.nan)
        target_state = {"liquidity_state": "missing"}
        if target is not None and np.isfinite(float(target)):
            target_side = "high" if str(rec["trade_side"]) == "LONG" else "low"
            target_state = query_liquidity_consumption_state(
                state_index,
                day_text=str(rec["ny_date"]),
                side=target_side,
                price=float(target),
                query_time=signal,
                config=config,
            )
        for k, v in target_state.items():
            rec[f"target_{k}"] = v
        rec["nearest_internal_target_price"] = rec.get("nearest_internal_target_price", np.nan)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    base = pd.DataFrame(rows)
    # Market breaks do not have the per-FVG internal-target query.  Keep only
    # external targets in R13 market-entry comparison; internal market targets
    # can be studied later without contaminating this execution test.
    out_rows = []
    for r in base.to_dict("records"):
        ext = r.get("target_price", np.nan)
        if ext is None or not np.isfinite(float(ext)):
            continue
        state = str(r.get("target_liquidity_state", "unknown"))
        for tmodel in ["external_any_state"] + (["external_if_not_fully_consumed"] if state != "accepted_or_deep_consumed" else []):
            is_long = str(r["trade_side"]) == "LONG"
            entry = float(r["entry_price"]); stop = float(r["stop_price"]); target = float(ext)
            risk = entry-stop if is_long else stop-entry
            reward = target-entry if is_long else entry-target
            if risk <= EPS or reward <= EPS:
                continue
            out_rows.append({**r,"target_model_r13":tmodel,"target_price_r13":target,"risk_abs_r13":risk,"planned_reward_abs_r13":reward,"planned_rr_r13":reward/risk})
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["attempt_id_r13"] = [f"R13|MKT-TGT|{i:09d}" for i in range(len(out))]
    return out


def select_progressive_structure_narratives(narratives: pd.DataFrame) -> pd.DataFrame:
    """Keep causal structure progression while allowing re-entry after re-sweeps.

    Within one sweep/timeframe/reference model, a new attempt is retained when:
    * a deeper terminal extreme created a new terminal version; or
    * the newly broken barrier extends beyond the prior retained barrier.

    This removes thousands of redundant weaker pivots without imposing a PnL
    or visibility threshold, and preserves the user's "micro attempt can fail,
    later stronger MSS can still trade" requirement.
    """
    if narratives.empty:
        return narratives.copy()
    rows = []
    group_cols = ["event_id", "execution_tf", "reference_model_r13"]
    for _, g in narratives.groupby(group_cols, sort=False, dropna=False):
        g = g.sort_values(["break_available_time", "terminal_version", "mss_reference_price"], kind="mergesort")
        last_version = None
        last_barrier = np.nan
        for r in g.to_dict("records"):
            ver = int(r.get("terminal_version", 0))
            px = float(r["mss_reference_price"])
            is_long = str(r["trade_side"]) == "LONG"
            new_version = last_version is None or ver != last_version
            advances = (not np.isfinite(last_barrier)) or (px > last_barrier + EPS if is_long else px < last_barrier - EPS)
            if new_version or advances:
                rows.append(r)
                last_version = ver
                last_barrier = px
    out = pd.DataFrame(rows)
    if not out.empty:
        out["narrative_attempt_sequence_r13"] = out.groupby(group_cols, sort=False).cumcount() + 1
    return out


def select_tiered_primary_narratives(narratives: pd.DataFrame) -> pd.DataFrame:
    """Causal low-frequency narrative view for long-history replay.

    Uses the outermost newly-broken barrier at each break, then keeps the first
    break in each causal visibility tier for each terminal version.  This keeps
    micro and clearer follow-up MSS attempts without replaying every tiny pivot.
    Tiers are descriptive percentile buckets, not PnL-tuned thresholds.
    """
    if narratives.empty:
        return pd.DataFrame()
    q = narratives.loc[narratives["reference_model_r13"].eq("outermost_barrier_newly_broken")].copy()
    if q.empty:
        return q
    pct = pd.to_numeric(q["causal_visibility_percentile"], errors="coerce")
    q["structure_visibility_tier_r13"] = np.select(
        [pct < 0.50, pct < 0.80],
        ["micro_lt_p50", "visible_p50_p80"],
        default="strong_ge_p80",
    )
    q.loc[pct.isna(), "structure_visibility_tier_r13"] = "unknown"
    keys = ["event_id", "execution_tf", "terminal_version", "structure_visibility_tier_r13"]
    q = q.sort_values(keys + ["break_available_time", "mss_reference_price"], kind="mergesort")
    q = q.drop_duplicates(keys, keep="first").reset_index(drop=True)
    q["narrative_attempt_sequence_r13"] = q.groupby(["event_id", "execution_tf"], sort=False).cumcount() + 1
    q["reference_model_r13"] = "outermost_barrier_tiered_primary"
    return q
