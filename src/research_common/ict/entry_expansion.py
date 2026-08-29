#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal primitives for SOXL ICT R11 entry-opportunity expansion research.

The module deliberately separates *liquidity discovery*, *target selection* and
*entry execution*.  No R11 feature is allowed to filter the existing R09 broad
Sweep -> MSS -> FVG universe.

New research components:

1. Causally confirmed intraday 15m swing highs/lows that can become new
   liquidity after the premarket range has already been partially/fully
   consumed.
2. Local intraday dealing-range targets: equilibrium (50%) and the fresh
   opposite 15m swing.
3. Alternative retracement entries on the same frozen MSS setup: FVG near
   edge, FVG consequent-encroachment (50%), and two explicit quantitative
   order-block proxies.  The order-block variants are research proxies rather
   than a claim that one mechanical candle formula exhausts ICT discretion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import EPS, TRADE_END, aggregate_closed_bars, slice_ny_day
from .premarket_mss_fvg_v2 import confirmed_pivots_with_excursion


@dataclass(frozen=True)
class EntryExpansionConfig:
    intraday_timeframe_minutes: int = 15
    intraday_pivot_left: int = 1
    intraday_pivot_right: int = 1
    intraday_start: dtime = dtime(8, 30)
    intraday_end: dtime = dtime(16, 30)
    # Descriptive only. No candidate is gated by this threshold in R11.
    obvious_excursion_multiple: float = 1.0
    entry_models: tuple[str, ...] = (
        "fvg_near_edge",
        "fvg_ce_50",
        "order_block_open_proxy",
        "order_block_midpoint_proxy",
    )
    target_models: tuple[str, ...] = (
        "local_equilibrium_50",
        "local_opposite_15m_swing",
    )


def _as_ny_ts(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _premarket_context(premarket_levels: pd.DataFrame) -> pd.DataFrame:
    if premarket_levels.empty:
        return pd.DataFrame()
    cols = [
        "ny_date", "premarket_high", "premarket_low", "premarket_range",
        "premarket_range_pct", "premarket_close", "premarket_median_15m_range",
    ]
    cols = [c for c in cols if c in premarket_levels.columns]
    return premarket_levels[cols].drop_duplicates("ny_date").set_index("ny_date")


def _first_strict_cross_time(
    trade_1m: pd.DataFrame,
    *,
    side: str,
    price: float,
    start_available_time: pd.Timestamp,
    end_available_time: pd.Timestamp | None = None,
) -> pd.Timestamp | pd.NaT:
    if trade_1m.empty or not np.isfinite(price):
        return pd.NaT
    idx = pd.DatetimeIndex(trade_1m.index)
    avail = idx + pd.Timedelta(minutes=1)
    mask = avail > pd.Timestamp(start_available_time)
    if end_available_time is not None:
        mask &= avail <= pd.Timestamp(end_available_time)
    if not mask.any():
        return pd.NaT
    sub = trade_1m.loc[mask]
    if sub.empty:
        return pd.NaT
    touched = (
        pd.to_numeric(sub["high"], errors="coerce") > float(price)
        if side == "high"
        else pd.to_numeric(sub["low"], errors="coerce") < float(price)
    )
    hits = sub.loc[touched]
    if hits.empty:
        return pd.NaT
    return pd.Timestamp(hits.index[0]) + pd.Timedelta(minutes=1)


def _premarket_consumption_times(trade_1m: pd.DataFrame, pm_high: float, pm_low: float) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    if trade_1m.empty:
        return pd.NaT, pd.NaT
    idx = pd.DatetimeIndex(trade_1m.index)
    highs = pd.to_numeric(trade_1m["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(trade_1m["low"], errors="coerce").to_numpy(float)
    hi_pos = np.flatnonzero(highs > float(pm_high))
    lo_pos = np.flatnonzero(lows < float(pm_low))
    hi_time = pd.Timestamp(idx[int(hi_pos[0])]) + pd.Timedelta(minutes=1) if hi_pos.size else pd.NaT
    lo_time = pd.Timestamp(idx[int(lo_pos[0])]) + pd.Timedelta(minutes=1) if lo_pos.size else pd.NaT
    return hi_time, lo_time


def _consumption_state_at(hi_time, lo_time, at: pd.Timestamp) -> str:
    hi = pd.notna(hi_time) and pd.Timestamp(hi_time) <= at
    lo = pd.notna(lo_time) and pd.Timestamp(lo_time) <= at
    if hi and lo:
        return "both_premarket_sides_consumed"
    if hi:
        return "premarket_high_consumed_only"
    if lo:
        return "premarket_low_consumed_only"
    return "premarket_both_fresh"


def build_intraday_15m_swing_catalog(
    bars_ny: pd.DataFrame,
    days: Sequence,
    premarket_levels: pd.DataFrame,
    *,
    config: EntryExpansionConfig = EntryExpansionConfig(),
) -> pd.DataFrame:
    """Build every causal intraday 15m swing; strength is descriptive only.

    The rolling median used to scale swing excursion uses only completed 15m
    bars available by that pivot's confirmation time. This prevents a hidden
    full-day volatility lookahead.
    """
    ctx = _premarket_context(premarket_levels)
    if ctx.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    tf = int(config.intraday_timeframe_minutes)

    for day in days:
        day_text = str(pd.Timestamp(day).date())
        if day_text not in ctx.index:
            continue
        trade = slice_ny_day(bars_ny, pd.Timestamp(day).date(), config.intraday_start, config.intraday_end)
        if trade.empty:
            continue
        frame = aggregate_closed_bars(trade, tf)
        if frame.empty:
            continue
        piv = confirmed_pivots_with_excursion(
            frame, left=int(config.intraday_pivot_left), right=int(config.intraday_pivot_right)
        )
        if piv.empty:
            continue
        c = ctx.loc[day_text]
        pm_high = float(c["premarket_high"])
        pm_low = float(c["premarket_low"])
        pm_range = float(c["premarket_range"])
        pm_close = float(c["premarket_close"])
        pm_median = float(c.get("premarket_median_15m_range", np.nan))
        pm_hi_swept, pm_lo_swept = _premarket_consumption_times(trade, pm_high, pm_low)
        frame_ranges = pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")
        frame_avail = pd.to_datetime(frame["available_time"])

        for rec in piv.to_dict("records"):
            available = pd.Timestamp(rec["confirmation_available_time"])
            if available < pd.Timestamp(trade.index.min()) + pd.Timedelta(minutes=tf):
                continue
            # Only completed bars with available_time <= pivot confirmation.
            known_ranges = frame_ranges.loc[frame_avail <= available]
            known_ranges = pd.to_numeric(known_ranges, errors="coerce")
            known_ranges = known_ranges[np.isfinite(known_ranges)]
            rolling_median = float(known_ranges.median()) if len(known_ranges) else np.nan
            excursion = float(rec.get("two_sided_excursion_abs", np.nan))
            prominence = float(rec.get("local_prominence_abs", np.nan))
            excursion_mult = excursion / rolling_median if np.isfinite(excursion) and np.isfinite(rolling_median) and rolling_median > EPS else np.nan
            prominence_mult = prominence / rolling_median if np.isfinite(prominence) and np.isfinite(rolling_median) and rolling_median > EPS else np.nan
            source = pd.Timestamp(rec["pivot_time"])
            side = str(rec["pivot_side"])
            price = float(rec["pivot_price"])
            rows.append({
                "ny_date": day_text,
                "liquidity_side": side,
                "level_type": "intraday_15m_swing",
                "liquidity_family": "intraday_15m_swing",
                "level_price": price,
                "source_bar_time": source,
                "level_available_time": available,
                "pivot_side": side,
                "pivot_time": source,
                "pivot_price": price,
                "pivot_pos": int(rec["pivot_pos"]),
                "confirmation_available_time": available,
                "local_prominence_abs": prominence,
                "left_excursion_abs": float(rec.get("left_excursion_abs", np.nan)),
                "right_excursion_abs": float(rec.get("right_excursion_abs", np.nan)),
                "two_sided_excursion_abs": excursion,
                "intraday_known_median_15m_range": rolling_median,
                "intraday_excursion_vs_known_median_15m_range": excursion_mult,
                "intraday_prominence_vs_known_median_15m_range": prominence_mult,
                "intraday_obvious_ge_1x_known_median": bool(np.isfinite(excursion_mult) and excursion_mult >= float(config.obvious_excursion_multiple)),
                "liquidity_strength": "intraday_15m_causal_unfiltered",
                "tradable_level": True,
                "rejection_reason": "",
                "premarket_high": pm_high,
                "premarket_low": pm_low,
                "premarket_range": pm_range,
                "premarket_range_pct": float(c.get("premarket_range_pct", np.nan)),
                "premarket_close": pm_close,
                "premarket_median_15m_range": pm_median,
                "premarket_high_first_consumed_time": pm_hi_swept,
                "premarket_low_first_consumed_time": pm_lo_swept,
                "premarket_consumption_state_at_level_confirmation": _consumption_state_at(pm_hi_swept, pm_lo_swept, available),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["ny_date", "level_available_time", "liquidity_side", "level_price"], kind="mergesort").reset_index(drop=True)


def _latest_opposite_intraday_pivot(
    catalog_day: pd.DataFrame,
    trade_1m: pd.DataFrame,
    *,
    liquidity_side: str,
    sweep_time: pd.Timestamp,
    swept_price: float,
) -> Mapping[str, object] | None:
    opposite = "low" if liquidity_side == "high" else "high"
    candidates = catalog_day.loc[
        (catalog_day["liquidity_side"].astype(str) == opposite)
        & (pd.to_datetime(catalog_day["level_available_time"]) <= sweep_time)
    ].copy()
    if candidates.empty:
        return None
    # Spatially coherent local range.
    px = pd.to_numeric(candidates["level_price"], errors="coerce")
    if liquidity_side == "low":
        candidates = candidates.loc[px > float(swept_price)].copy()
    else:
        candidates = candidates.loc[px < float(swept_price)].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(["level_available_time", "source_bar_time"], ascending=[False, False], kind="mergesort")
    # Prefer the latest still-unconsumed opposing swing. This is evaluated only
    # with 1m bars available before the current sweep.
    for rec in candidates.to_dict("records"):
        consumed = _first_strict_cross_time(
            trade_1m,
            side=str(rec["liquidity_side"]),
            price=float(rec["level_price"]),
            start_available_time=pd.Timestamp(rec["level_available_time"]),
            end_available_time=sweep_time,
        )
        if pd.isna(consumed):
            return rec
    return None


def build_intraday_15m_sweep_events(
    bars_ny: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    config: EntryExpansionConfig = EntryExpansionConfig(),
) -> pd.DataFrame:
    """Detect first post-confirmation sweep of each intraday 15m swing.

    A physical sweep is emitted once, with local equilibrium/opposite-swing
    target metadata. Target variants are expanded separately so opportunity
    counts are not inflated by target experiments.
    """
    if catalog.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for day_text, day_cat in catalog.groupby("ny_date", sort=True):
        day = pd.Timestamp(day_text).date()
        trade = slice_ny_day(bars_ny, day, config.intraday_start, config.intraday_end)
        if trade.empty:
            continue
        idx = pd.DatetimeIndex(trade.index)
        highs = pd.to_numeric(trade["high"], errors="coerce").to_numpy(float)
        lows = pd.to_numeric(trade["low"], errors="coerce").to_numpy(float)
        available = idx + pd.Timedelta(minutes=1)
        day_records = day_cat.to_dict("records")

        for level in day_records:
            level_avail = pd.Timestamp(level["level_available_time"])
            side = str(level["liquidity_side"])
            price = float(level["level_price"])
            start_pos = int(np.searchsorted(available.as_unit("ns").asi8, int(level_avail.value), side="right"))
            if start_pos >= len(trade):
                continue
            mask = highs[start_pos:] > price if side == "high" else lows[start_pos:] < price
            hits = np.flatnonzero(mask)
            if hits.size == 0:
                continue
            pos = start_pos + int(hits[0])
            bar_start = pd.Timestamp(idx[pos])
            sweep_time = bar_start + pd.Timedelta(minutes=1)
            extreme = float(highs[pos] if side == "high" else lows[pos])
            trade_side = "SHORT" if side == "high" else "LONG"

            opposite = _latest_opposite_intraday_pivot(
                day_cat, trade,
                liquidity_side=side,
                sweep_time=sweep_time,
                swept_price=price,
            )
            if opposite is None:
                local_opposite_price = np.nan
                local_mid = np.nan
                opposite_available = pd.NaT
                local_range_valid = False
            else:
                local_opposite_price = float(opposite["level_price"])
                local_mid = float((price + local_opposite_price) / 2.0)
                opposite_available = pd.Timestamp(opposite["level_available_time"])
                local_range_valid = bool(
                    (trade_side == "LONG" and local_opposite_price > price)
                    or (trade_side == "SHORT" and local_opposite_price < price)
                )

            pm_hi = level.get("premarket_high_first_consumed_time", pd.NaT)
            pm_lo = level.get("premarket_low_first_consumed_time", pd.NaT)
            rows.append({
                **level,
                "event_id": f"{day_text}|{side}|intraday_15m_swing|{price:.8f}|{level_avail.isoformat()}",
                "trade_side": trade_side,
                "sweep_bar_start": bar_start,
                "sweep_time": sweep_time,
                "sweep_price_extreme_initial": extreme,
                "sweep_distance_pct": abs(extreme / price - 1.0) if abs(price) > EPS else np.nan,
                "sweep_minute_of_session": int((bar_start.hour * 60 + bar_start.minute) - (8 * 60 + 30)),
                "sweep_hour_ny": int(bar_start.hour),
                "premarket_consumption_state_at_sweep": _consumption_state_at(pm_hi, pm_lo, sweep_time),
                "local_opposite_15m_price": local_opposite_price,
                "local_opposite_15m_available_time": opposite_available,
                "local_equilibrium_50": local_mid,
                "local_dealing_range_valid": local_range_valid,
                "setup_eligible_at_sweep": local_range_valid,
                "setup_rejection_reason": "" if local_range_valid else "no_fresh_opposite_intraday_15m_swing",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Same minute may cross multiple nearby intraday pivots. Keep the deepest
    # physical sweep, not one pseudo-trade per crossed level.
    deduped: list[dict[str, object]] = []
    keys = ["ny_date", "trade_side", "sweep_time", "liquidity_family"]
    for _, g in out.groupby(keys, sort=True, dropna=False):
        px = pd.to_numeric(g["level_price"], errors="coerce")
        side = str(g["trade_side"].iloc[0])
        choose = px.idxmin() if side == "LONG" else px.idxmax()
        rec = dict(out.loc[choose])
        rec["same_family_levels_swept"] = int(len(g))
        rec["same_family_swept_prices"] = ",".join(f"{float(x):.8f}" for x in sorted(px.dropna().tolist()))
        deduped.append(rec)
    return pd.DataFrame(deduped).sort_values(["sweep_time", "trade_side"], kind="mergesort").reset_index(drop=True)


def expand_intraday_target_models(
    sweeps: pd.DataFrame,
    *,
    config: EntryExpansionConfig = EntryExpansionConfig(),
) -> pd.DataFrame:
    """Create target research variants without multiplying physical sweeps."""
    if sweeps.empty:
        return sweeps.copy()
    rows: list[dict[str, object]] = []
    for rec in sweeps.to_dict("records"):
        side = str(rec["trade_side"])
        candidates = {
            "local_equilibrium_50": rec.get("local_equilibrium_50", np.nan),
            "local_opposite_15m_swing": rec.get("local_opposite_15m_price", np.nan),
        }
        for model in config.target_models:
            target = float(candidates.get(model, np.nan))
            if not np.isfinite(target):
                continue
            # At sweep time the exact FVG entry is not known yet. Require only
            # that the target lies in the intended reversal direction relative
            # to the swept liquidity level; attempt builder later validates
            # reward relative to the actual entry.
            level = float(rec["level_price"])
            directional = (side == "LONG" and target > level) or (side == "SHORT" and target < level)
            if not directional:
                continue
            row = dict(rec)
            row["target_model"] = str(model)
            row["target_price"] = target
            row["opposite_target_fresh_at_sweep"] = True
            row["opposite_target_first_touch_time"] = pd.NaT
            row["same_bar_target_sweep_ambiguity"] = False
            row["setup_eligible_at_sweep"] = bool(rec.get("setup_eligible_at_sweep", True))
            row["event_id"] = f"{rec['event_id']}|target={model}"
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_execution_day(bars_ny: pd.DataFrame, day, tf: int) -> pd.DataFrame:
    one = slice_ny_day(bars_ny, pd.Timestamp(day).date(), dtime(4, 0), TRADE_END)
    return aggregate_closed_bars(one, int(tf)) if not one.empty else pd.DataFrame()


def attach_order_block_proxy_features(
    attempts: pd.DataFrame,
    bars_ny: pd.DataFrame,
) -> pd.DataFrame:
    """Attach latest opposite-close candle inside terminal->MSS leg.

    This is an intentionally explicit *quantitative proxy* for research.  It is
    not asserted to be a complete canonical ICT order-block definition.
    """
    if attempts.empty:
        return attempts.copy()
    out = attempts.copy()
    cols = {
        "ob_proxy_available": False,
        "ob_proxy_open": np.nan,
        "ob_proxy_high": np.nan,
        "ob_proxy_low": np.nan,
        "ob_proxy_close": np.nan,
        "ob_proxy_open_entry": np.nan,
        "ob_proxy_midpoint_entry": np.nan,
        "ob_proxy_mitigated_before_signal": False,
        "ob_proxy_definition": "latest opposite-close execution bar inside terminal_to_mss leg",
    }
    for k, v in cols.items():
        out[k] = v
    # Keep tz-aware timestamps as object here because rows can carry different
    # DST offsets across a multi-year New York sample. This avoids pandas
    # incompatible-dtype warnings while preserving exact aware Timestamps.
    out["ob_proxy_bar_start"] = pd.Series([pd.NaT] * len(out), index=out.index, dtype="object")
    out["ob_proxy_available_time"] = pd.Series([pd.NaT] * len(out), index=out.index, dtype="object")

    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for i, row in out.iterrows():
        day_text = str(row["ny_date"])
        tf = int(row["execution_tf_minutes"])
        key = (day_text, tf)
        frame = cache.get(key)
        if frame is None:
            frame = _aggregate_execution_day(bars_ny, pd.Timestamp(day_text).date(), tf)
            cache[key] = frame
        if frame.empty:
            continue
        terminal = pd.Timestamp(row["episode_terminal_extreme_time"])
        mss_time = pd.Timestamp(row["mss_time"])
        signal = pd.Timestamp(row["signal_time"])
        available = pd.to_datetime(frame["available_time"])
        leg = frame.loc[(available >= terminal) & (available <= mss_time)].copy()
        if leg.empty:
            continue
        is_long = str(row["trade_side"]) == "LONG"
        opposite_close = (
            pd.to_numeric(leg["close"], errors="coerce") < pd.to_numeric(leg["open"], errors="coerce")
            if is_long
            else pd.to_numeric(leg["close"], errors="coerce") > pd.to_numeric(leg["open"], errors="coerce")
        )
        candidates = leg.loc[opposite_close]
        if candidates.empty:
            continue
        # Latest known opposite-close candle prior to MSS confirmation.
        ob = candidates.iloc[-1]
        ob_start = pd.Timestamp(candidates.index[-1])
        ob_available = pd.Timestamp(ob["available_time"])
        op = float(ob["open"]); hi = float(ob["high"]); lo = float(ob["low"]); cl = float(ob["close"])
        open_entry = op
        midpoint_entry = float((hi + lo) / 2.0)

        # Diagnostic: did price revisit the proxy range after the candle closed
        # but before the final signal was available? Never used as a gate.
        after = frame.loc[(pd.to_datetime(frame["available_time"]) > ob_available) & (pd.to_datetime(frame["available_time"]) <= signal)]
        if after.empty:
            mitigated = False
        elif is_long:
            mitigated = bool((pd.to_numeric(after["low"], errors="coerce") <= hi).any())
        else:
            mitigated = bool((pd.to_numeric(after["high"], errors="coerce") >= lo).any())

        out.at[i, "ob_proxy_available"] = True
        out.at[i, "ob_proxy_bar_start"] = ob_start
        out.at[i, "ob_proxy_available_time"] = ob_available
        out.at[i, "ob_proxy_open"] = op
        out.at[i, "ob_proxy_high"] = hi
        out.at[i, "ob_proxy_low"] = lo
        out.at[i, "ob_proxy_close"] = cl
        out.at[i, "ob_proxy_open_entry"] = open_entry
        out.at[i, "ob_proxy_midpoint_entry"] = midpoint_entry
        out.at[i, "ob_proxy_mitigated_before_signal"] = mitigated
    return out


def expand_entry_models(
    attempts: pd.DataFrame,
    bars_ny: pd.DataFrame,
    *,
    config: EntryExpansionConfig = EntryExpansionConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand causal entry-price variants on a frozen signal universe."""
    if attempts.empty:
        return attempts.copy(), pd.DataFrame()
    work = attach_order_block_proxy_features(attempts, bars_ny)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for rec in work.to_dict("records"):
        near = float(rec["fvg_near_edge_entry"])
        far = float(rec["fvg_far_edge"])
        entries = {
            "fvg_near_edge": near,
            "fvg_ce_50": float((near + far) / 2.0),
            "order_block_open_proxy": float(rec.get("ob_proxy_open_entry", np.nan)),
            "order_block_midpoint_proxy": float(rec.get("ob_proxy_midpoint_entry", np.nan)),
        }
        is_long = str(rec["trade_side"]) == "LONG"
        stop = float(rec["stop_price"])
        target = float(rec["target_price"])
        signal_close = float(rec.get("signal_close", np.nan))
        for model in config.entry_models:
            entry = float(entries.get(model, np.nan))
            valid = np.isfinite(entry)
            reason = ""
            if not valid:
                reason = "entry_price_unavailable"
            elif is_long and not (stop < entry < target):
                valid = False; reason = "long_entry_not_between_stop_and_target"
            elif (not is_long) and not (target < entry < stop):
                valid = False; reason = "short_entry_not_between_target_and_stop"
            elif model.startswith("order_block") and np.isfinite(signal_close) and is_long and entry > signal_close + EPS:
                valid = False; reason = "buy_limit_above_signal_market"
            elif model.startswith("order_block") and np.isfinite(signal_close) and (not is_long) and entry < signal_close - EPS:
                valid = False; reason = "sell_limit_below_signal_market"

            audit_rows.append({
                "attempt_id": rec["attempt_id"], "entry_model": model,
                "entry_price": entry, "valid_entry_variant": bool(valid),
                "rejection_reason": reason,
                "ob_proxy_available": bool(rec.get("ob_proxy_available", False)),
                "ob_proxy_mitigated_before_signal": bool(rec.get("ob_proxy_mitigated_before_signal", False)),
            })
            if not valid:
                continue
            risk = (entry - stop) if is_long else (stop - entry)
            reward = (target - entry) if is_long else (entry - target)
            if risk <= EPS or reward <= EPS:
                continue
            row = dict(rec)
            row["entry_model"] = model
            row["entry_model_price"] = entry
            # replay_attempts intentionally consumes this legacy field.
            row["fvg_near_edge_entry"] = entry
            row["risk_abs"] = float(risk)
            row["risk_pct"] = float(risk / abs(entry)) if abs(entry) > EPS else np.nan
            row["planned_reward_abs"] = float(reward)
            row["planned_rr"] = float(reward / risk)
            row["attempt_id"] = f"{rec['attempt_id']}|entry={model}"
            rows.append(row)
    expanded = pd.DataFrame(rows)
    if not expanded.empty:
        expanded = expanded.sort_values(["signal_time", "attempt_id"], kind="mergesort").reset_index(drop=True)
    return expanded, pd.DataFrame(audit_rows)


def entry_expansion_causal_audit(
    intraday_catalog: pd.DataFrame,
    intraday_sweeps: pd.DataFrame,
    expanded_attempts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    def add(name: str, ok: pd.Series | bool, detail: str) -> None:
        if isinstance(ok, bool):
            bad = 0 if ok else 1
        else:
            bad = int((~ok.fillna(False)).sum())
        rows.append({"check": name, "passed": bad == 0, "violations": bad, "detail": detail})

    if not intraday_catalog.empty:
        add(
            "intraday_level_available_after_source",
            pd.to_datetime(intraday_catalog["level_available_time"]) > pd.to_datetime(intraday_catalog["source_bar_time"]),
            "15m swing liquidity can only exist after causal right-side confirmation",
        )
    if not intraday_sweeps.empty:
        add(
            "intraday_sweep_after_level_confirmation",
            pd.to_datetime(intraday_sweeps["sweep_time"]) > pd.to_datetime(intraday_sweeps["level_available_time"]),
            "intraday liquidity cannot be swept before it is causally known",
        )
        valid_local = intraday_sweeps["local_dealing_range_valid"].fillna(False).astype(bool)
        if valid_local.any():
            add(
                "local_opposite_swing_known_by_sweep",
                pd.to_datetime(intraday_sweeps.loc[valid_local, "local_opposite_15m_available_time"]) <= pd.to_datetime(intraday_sweeps.loc[valid_local, "sweep_time"]),
                "local dealing-range opposite swing must already be confirmed at sweep",
            )
    if not expanded_attempts.empty:
        if "ob_proxy_available_time" in expanded_attempts.columns:
            mask = expanded_attempts["entry_model"].astype(str).str.startswith("order_block")
            if mask.any():
                add(
                    "order_block_proxy_known_by_signal",
                    pd.to_datetime(expanded_attempts.loc[mask, "ob_proxy_available_time"]) <= pd.to_datetime(expanded_attempts.loc[mask, "signal_time"]),
                    "order-block proxy candle must be closed before the order signal",
                )
        add(
            "expanded_entry_positive_risk",
            pd.to_numeric(expanded_attempts["risk_abs"], errors="coerce") > 0,
            "every entry variant must keep the original structural stop beyond entry",
        )
        add(
            "expanded_entry_positive_reward",
            pd.to_numeric(expanded_attempts["planned_reward_abs"], errors="coerce") > 0,
            "every target/entry variant must have positive directional reward",
        )
    return pd.DataFrame(rows)
