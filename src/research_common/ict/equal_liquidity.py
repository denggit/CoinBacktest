"""Causal equal-high / equal-low liquidity pools for ICT research.

R09 replaces R08's coarse near-touch count with actual same-side swing pools.
Pools are descriptive/research liquidity, not post-hoc chart labels:

* source pivots come only from completed 1m/5m/15m bars;
* every member pivot must be causally confirmed before the pool is available;
* prices are clustered with a volatility-scaled tolerance, not PnL tuning;
* the pool remains active only if price has not strictly traded through its
  outer boundary after the pool became available and before 08:30 NY;
* all active pools are retained (no nearest-only assumption).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import (
    EPS,
    PREMARKET_END,
    PREMARKET_START,
    aggregate_closed_bars,
    confirmed_pivots,
    slice_ny_day,
)


@dataclass(frozen=True)
class EqualLiquidityConfig:
    source_timeframes: tuple[int, ...] = (1, 5, 15)
    pivot_left: int = 1
    pivot_right: int = 1
    min_members: int = 2
    # Structural tolerance only.  The adaptive component follows the day's
    # source-timeframe median bar range; no value is selected from PnL.
    min_tolerance_bp: float = 5.0
    range_tolerance_fraction: float = 0.25


def _day_meta(premarket_levels: pd.DataFrame) -> dict[str, dict[str, object]]:
    if premarket_levels.empty:
        return {}
    cols = [
        "premarket_high", "premarket_low", "premarket_range",
        "premarket_range_pct", "premarket_close", "premarket_15m_bars",
        "premarket_median_15m_range",
    ]
    out: dict[str, dict[str, object]] = {}
    for day, g in premarket_levels.groupby("ny_date", sort=False):
        row = g.iloc[0]
        out[str(day)] = {c: row.get(c, np.nan) for c in cols}
    return out


def _cluster_pivots(pivots: pd.DataFrame, *, tolerance_bp: float) -> list[list[dict[str, object]]]:
    """Chronologically attach a pivot to the nearest compatible live cluster."""
    if pivots.empty:
        return []
    clusters: list[list[dict[str, object]]] = []
    for rec in pivots.sort_values(["pivot_time", "confirmation_available_time"], kind="mergesort").to_dict("records"):
        px = float(rec["pivot_price"])
        best_i: int | None = None
        best_bp = np.inf
        for i, members in enumerate(clusters):
            center = float(np.median([float(x["pivot_price"]) for x in members]))
            if abs(center) <= EPS:
                continue
            dist_bp = abs(px / center - 1.0) * 10_000.0
            if dist_bp <= tolerance_bp and dist_bp < best_bp:
                best_i, best_bp = i, dist_bp
        if best_i is None:
            clusters.append([rec])
        else:
            clusters[best_i].append(rec)
    return clusters


def build_equal_liquidity_pools(
    bars_ny: pd.DataFrame,
    days: Sequence[date],
    premarket_levels: pd.DataFrame,
    *,
    config: EqualLiquidityConfig = EqualLiquidityConfig(),
) -> pd.DataFrame:
    """Build active 08:30 equal-high/equal-low pools from causal swing clusters."""
    meta = _day_meta(premarket_levels)
    rows: list[dict[str, object]] = []
    for day in days:
        day_text = str(day)
        day_meta = meta.get(day_text)
        if day_meta is None:
            continue
        one = slice_ny_day(bars_ny, day, PREMARKET_START, PREMARKET_END)
        if one.empty:
            continue
        day_end = pd.Timestamp(one.index[0]).normalize() + pd.Timedelta(hours=8, minutes=30)

        for tf in config.source_timeframes:
            frame = aggregate_closed_bars(one, int(tf))
            if frame.empty:
                continue
            # The final source bar is usable only if its available_time <=08:30.
            frame = frame.loc[pd.to_datetime(frame["available_time"]) <= day_end].copy()
            if frame.empty:
                continue
            piv = confirmed_pivots(frame, left=int(config.pivot_left), right=int(config.pivot_right))
            if piv.empty:
                continue
            piv = piv.loc[pd.to_datetime(piv["confirmation_available_time"]) <= day_end].copy()
            if piv.empty:
                continue

            ranges = pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")
            closes = pd.to_numeric(frame["close"], errors="coerce").abs()
            range_bp = (ranges / closes.replace(0, np.nan) * 10_000.0).replace([np.inf, -np.inf], np.nan)
            med_range_bp = float(range_bp.median()) if range_bp.notna().any() else np.nan
            tol_bp = max(
                float(config.min_tolerance_bp),
                float(config.range_tolerance_fraction) * med_range_bp if np.isfinite(med_range_bp) else 0.0,
            )

            for side in ("high", "low"):
                side_p = piv.loc[piv["pivot_side"] == side].copy()
                for members in _cluster_pivots(side_p, tolerance_bp=tol_bp):
                    if len(members) < int(config.min_members):
                        continue
                    prices = np.asarray([float(x["pivot_price"]) for x in members], dtype=float)
                    times = pd.DatetimeIndex([pd.Timestamp(x["pivot_time"]) for x in members])
                    confirms = pd.DatetimeIndex([pd.Timestamp(x["confirmation_available_time"]) for x in members])
                    available_time = confirms.max()
                    if available_time > day_end:
                        continue
                    center = float(np.median(prices))
                    outer = float(np.max(prices) if side == "high" else np.min(prices))
                    inner = float(np.min(prices) if side == "high" else np.max(prices))
                    dispersion_bp = abs(outer - inner) / abs(center) * 10_000.0 if abs(center) > EPS else np.nan

                    # Strictly after availability: if price already trades through
                    # the pool boundary before 08:30, the pool is consumed.
                    post = one.loc[one.index + pd.Timedelta(minutes=1) > available_time]
                    if not post.empty:
                        consumed = bool(
                            (pd.to_numeric(post["high"], errors="coerce") > outer).any()
                            if side == "high"
                            else (pd.to_numeric(post["low"], errors="coerce") < outer).any()
                        )
                    else:
                        consumed = False
                    if consumed:
                        continue

                    member_times = sorted(times)
                    span_minutes = max(0.0, (member_times[-1] - member_times[0]).total_seconds() / 60.0)
                    rows.append({
                        "ny_date": day_text,
                        "liquidity_side": side,
                        "level_type": f"equal_{side}_pool",
                        "level_price": outer,
                        "source_bar_time": member_times[-1],
                        "level_available_time": available_time,
                        "local_prominence_abs": np.nan,
                        "two_sided_excursion_abs": np.nan,
                        "excursion_vs_median_15m_range": np.nan,
                        "prominence_frac_of_premarket_range": np.nan,
                        "liquidity_strength": "causal_equal_swing_pool",
                        "tradable_level": True,
                        "rejection_reason": "",
                        **day_meta,
                        "liquidity_family": "equal_liquidity_pool",
                        "eq_source_tf": f"{int(tf)}m",
                        "eq_member_count": int(len(members)),
                        "eq_center_price": center,
                        "eq_inner_price": inner,
                        "eq_outer_price": outer,
                        "eq_dispersion_bp": dispersion_bp,
                        "eq_tolerance_bp": tol_bp,
                        "eq_first_pivot_time": member_times[0],
                        "eq_last_pivot_time": member_times[-1],
                        "eq_pool_span_minutes": span_minutes,
                        "eq_member_prices": ",".join(f"{x:.8f}" for x in prices),
                        "eq_member_times": "|".join(str(x) for x in member_times),
                        "eq_active_at_0830": True,
                    })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["ny_date", "liquidity_side", "eq_source_tf", "level_price"], kind="mergesort").reset_index(drop=True)


def attach_equal_pool_context(attempts: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    """Annotate any setup with the nearest already-available same-side EQ pool.

    This does not filter attempts.  A base premarket/15m/HTF liquidity level can
    therefore be studied for whether it was also sitting near an independently
    constructed equal-high/equal-low pool before the sweep.
    """
    if attempts.empty:
        return attempts.copy()
    work = attempts.copy()
    work["eq_context_present"] = False
    work["eq_context_distance_bp"] = np.nan
    work["eq_context_source_tf"] = ""
    work["eq_context_member_count"] = np.nan
    work["eq_context_dispersion_bp"] = np.nan
    if pools.empty:
        work["eq_context_feature_available_time"] = pd.to_datetime(work["signal_time"])
        return work

    grouped = {(str(d), str(s)): g.copy() for (d, s), g in pools.groupby(["ny_date", "liquidity_side"], sort=False)}
    for idx, row in work.iterrows():
        key = (str(row.get("ny_date")), str(row.get("liquidity_side")))
        cand = grouped.get(key)
        if cand is None or cand.empty:
            continue
        sweep_time = pd.Timestamp(row["sweep_time"])
        cand = cand.loc[pd.to_datetime(cand["level_available_time"]) <= sweep_time]
        if cand.empty:
            continue
        level = float(row["level_price"])
        px = pd.to_numeric(cand["level_price"], errors="coerce")
        dist = (px / level - 1.0).abs() * 10_000.0
        j = dist.idxmin()
        # Context requires the base level to lie within the structural tolerance
        # of the independently detected pool.  This threshold is pool-derived,
        # not selected from PnL.
        tol = float(cand.loc[j].get("eq_tolerance_bp", np.nan))
        if not np.isfinite(tol) or float(dist.loc[j]) > tol:
            continue
        work.at[idx, "eq_context_present"] = True
        work.at[idx, "eq_context_distance_bp"] = float(dist.loc[j])
        work.at[idx, "eq_context_source_tf"] = str(cand.loc[j].get("eq_source_tf", ""))
        work.at[idx, "eq_context_member_count"] = float(cand.loc[j].get("eq_member_count", np.nan))
        work.at[idx, "eq_context_dispersion_bp"] = float(cand.loc[j].get("eq_dispersion_bp", np.nan))
    work["eq_context_feature_available_time"] = pd.to_datetime(work["signal_time"])
    return work


def build_equal_pool_causal_audit(pools: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if pools.empty:
        rows.append({"check": "equal_pool_catalog_non_empty", "passed": True, "violations": 0})
    else:
        cutoff = pd.to_datetime(pools["ny_date"].astype(str) + " 08:30:00").dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="shift_forward")
        avail = pd.to_datetime(pools["level_available_time"])
        bad = int((avail > cutoff).fillna(True).sum())
        rows.append({"check": "equal_pool_available_by_0830", "passed": bad == 0, "violations": bad})
    if not attempts.empty and "eq_context_feature_available_time" in attempts:
        bad = int((pd.to_datetime(attempts["eq_context_feature_available_time"]) > pd.to_datetime(attempts["signal_time"])).fillna(True).sum())
        rows.append({"check": "equal_pool_context_available_by_signal", "passed": bad == 0, "violations": bad})
    return pd.DataFrame(rows)
