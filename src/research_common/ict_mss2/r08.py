#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R08 full-trend ICT structure atlas.

This module deliberately separates three layers that visual discretionary work
can blur together:

1. Classical ICT/Larry-Williams recursive swing hierarchy on one chart
   timeframe: ST -> IT -> LT.
2. A *completed* large structural leg between opposite LT anchors.  The leg is
   not trend-qualified unless its internal IT highs/lows progress monotonically
   in the expected direction and a subsequent break of the last opposing IT
   level confirms that the leg has ended.
3. Trend-qualified historical liquidity.  Short-term swings are construction
   inputs only; they never enter this key-liquidity table.  For a completed
   bearish leg we retain its LTH + internal ITHs as future buy-side liquidity;
   for a completed bullish leg we retain its LTL + internal ITLs as future
   sell-side liquidity.  A candidate is activated only after the *whole leg*
   and its reversal BOS are causally known, and it is discarded if price has
   already consumed it by that activation timestamp.

The 3% / 5% / 7% scales are CoinBacktest research sensitivities, not claimed ICT
canonical thresholds.  They quantify how large the completed historical trend
was without redefining the underlying recursive swing labels.

ICT 2022 Episode 12 also describes imbalance-rebalance swings that may be
mentally promoted to intermediate-term status.  R08 does NOT silently merge
that discretionary extension into the classical recursive hierarchy; it is
reserved for a separately auditable extension after the classical chart labels
pass manual review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R08Config:
    structure_timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
    )
    trend_scales: tuple[float, ...] = (0.03, 0.05, 0.07)
    min_it_swings_per_side: int = 2

    def validate(self) -> "R08Config":
        if not self.structure_timeframes:
            raise ValueError("structure_timeframes cannot be empty")
        if any(int(m) <= 0 for _, m in self.structure_timeframes):
            raise ValueError("structure timeframe minutes must be positive")
        scales = tuple(float(x) for x in self.trend_scales)
        if scales != tuple(sorted(set(scales))) or any(x <= 0 for x in scales):
            raise ValueError("trend_scales must be sorted unique positive fractions")
        if int(self.min_it_swings_per_side) < 1:
            raise ValueError("min_it_swings_per_side must be >= 1")
        return self


def _strict_st_mask(values: np.ndarray, side: str) -> np.ndarray:
    """Classical three-swing ST mask: center must exceed both adjacent bars."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    finite = np.isfinite(x[:-2]) & np.isfinite(x[1:-1]) & np.isfinite(x[2:])
    if side == "high":
        ok = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    elif side == "low":
        ok = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
    else:
        raise ValueError("side must be high/low")
    out[1:-1] = finite & ok
    return out


def _is_extreme(center: float, left: float, right: float, side: str) -> bool:
    if not all(np.isfinite(v) for v in (center, left, right)):
        return False
    if side == "high":
        return bool(center > left and center > right)
    if side == "low":
        return bool(center < left and center < right)
    raise ValueError("side must be high/low")


def build_classical_ict_hierarchy(
    bars_1m: pd.DataFrame,
    *,
    timeframe: str,
    minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strict classical ST -> IT -> LT recursively and causally.

    Returns
    -------
    hierarchy:
        One row per ST swing with causal IT/LT availability times.
    aggregated:
        Complete left-labelled OHLC bars used to construct the hierarchy.
    """
    htf = aggregate_bars(bars_1m, int(minutes))
    if htf.empty:
        return pd.DataFrame(), htf
    delta = pd.Timedelta(minutes=int(minutes))
    parts: list[pd.DataFrame] = []
    for side, col in (("high", "high"), ("low", "low")):
        values = pd.to_numeric(htf[col], errors="coerce").to_numpy(dtype=float)
        st_mask = _strict_st_mask(values, side)
        pos = np.flatnonzero(st_mask)
        if len(pos) == 0:
            continue
        # ST relation requires the right neighbour bar to close.
        st_available = htf.index[pos + 1] + delta
        part = pd.DataFrame({
            "swing_id": [f"R08_{timeframe}_{side.upper()}_{int(p):08d}" for p in pos],
            "pivot_side": side,
            "source_timeframe": str(timeframe),
            "source_timeframe_min": int(minutes),
            "pivot_pos_htf": pos.astype(np.int64),
            "pivot_time": htf.index[pos],
            "pivot_bar_end_time": htf.index[pos] + delta,
            "level_price": values[pos],
            "st_available_time": pd.to_datetime(st_available),
            "it_available_time": pd.NaT,
            "lt_available_time": pd.NaT,
            "is_st": np.ones(len(pos), dtype=np.int8),
            "is_it": np.zeros(len(pos), dtype=np.int8),
            "is_lt": np.zeros(len(pos), dtype=np.int8),
        })
        price = part["level_price"].to_numpy(dtype=float)
        it_members: list[int] = []
        it_avail: dict[int, pd.Timestamp] = {}
        for j in range(1, len(part) - 1):
            if not _is_extreme(price[j], price[j - 1], price[j + 1], side):
                continue
            available = max(
                pd.Timestamp(part.iloc[j]["st_available_time"]),
                pd.Timestamp(part.iloc[j + 1]["st_available_time"]),
            )
            part.at[j, "is_it"] = 1
            part.at[j, "it_available_time"] = available
            it_members.append(j)
            it_avail[j] = available
        for k in range(1, len(it_members) - 1):
            lj, cj, rj = it_members[k - 1], it_members[k], it_members[k + 1]
            if not _is_extreme(price[cj], price[lj], price[rj], side):
                continue
            available = max(it_avail[cj], it_avail[rj])
            part.at[cj, "is_lt"] = 1
            part.at[cj, "lt_available_time"] = available
        part["classical_swing_class"] = np.select(
            [part["is_lt"].eq(1), part["is_it"].eq(1)], ["LT", "IT"], default="ST"
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(), htf
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["pivot_time", "pivot_side", "swing_id"], kind="stable").reset_index(drop=True)
    return out, htf


def build_multi_timeframe_hierarchy(
    bars_1m: pd.DataFrame,
    *,
    config: R08Config | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    cfg = (config or R08Config()).validate()
    pieces: list[pd.DataFrame] = []
    bar_map: dict[int, pd.DataFrame] = {}
    reporter = ProgressReporter("[r08-hierarchy]", total=len(cfg.structure_timeframes), every=1, enabled=progress)
    for i, (tf, minutes) in enumerate(cfg.structure_timeframes, start=1):
        part, htf = build_classical_ict_hierarchy(bars_1m, timeframe=tf, minutes=minutes)
        if not part.empty:
            pieces.append(part)
        bar_map[int(minutes)] = htf
        reporter.update(i)
    reporter.close()
    hierarchy = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return hierarchy, bar_map


def _compress_lt_anchors(lt: pd.DataFrame) -> pd.DataFrame:
    """Ex-post alternating LT extremes for whole-leg construction.

    Consecutive same-side LT candidates are compressed to the more extreme one.
    This compression is *not* usable live.  A completed leg is activated only
    after the opposite endpoint and reversal BOS are known, so no future label
    leaks into an earlier decision.
    """
    if lt.empty:
        return lt.copy()
    x = lt.sort_values(["pivot_time", "swing_id"], kind="stable")
    kept: list[pd.Series] = []
    for _, row in x.iterrows():
        if not kept:
            kept.append(row)
            continue
        if str(row["pivot_side"]) != str(kept[-1]["pivot_side"]):
            kept.append(row)
            continue
        side = str(row["pivot_side"])
        new_px = float(row["level_price"])
        old_px = float(kept[-1]["level_price"])
        more_extreme = new_px > old_px if side == "high" else new_px < old_px
        if more_extreme:
            kept[-1] = row
    return pd.DataFrame(kept).reset_index(drop=True)


def _strict_monotonic(values: Iterable[float], direction: int) -> bool:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return True
    d = np.diff(arr)
    return bool(np.all(d > 0)) if direction > 0 else bool(np.all(d < 0))


def _sequence_string(part: pd.DataFrame) -> str:
    if part.empty:
        return ""
    return "|".join(
        f"{pd.Timestamp(t).isoformat()}@{float(p):.8g}"
        for t, p in zip(part["pivot_time"], pd.to_numeric(part["level_price"], errors="coerce"))
    )


def _first_close_break(
    htf: pd.DataFrame,
    *,
    after_time: pd.Timestamp,
    side: str,
    level: float,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    if htf.empty or not np.isfinite(level):
        return pd.NaT, np.nan
    idx_ns = htf.index.to_numpy(dtype="datetime64[ns]")
    start = int(np.searchsorted(idx_ns, np.datetime64(pd.Timestamp(after_time)), side="right"))
    if start >= len(htf):
        return pd.NaT, np.nan
    close = pd.to_numeric(htf["close"], errors="coerce").to_numpy(dtype=float)
    segment = close[start:]
    if side == "above":
        hits = np.flatnonzero(segment > float(level))
    elif side == "below":
        hits = np.flatnonzero(segment < float(level))
    else:
        raise ValueError("side must be above/below")
    if len(hits) == 0:
        return pd.NaT, np.nan
    pos = start + int(hits[0])
    available = pd.Timestamp(htf.iloc[pos]["bar_end_time"])
    return available, float(close[pos])


def build_completed_trend_legs(
    hierarchy: pd.DataFrame,
    bar_map: dict[int, pd.DataFrame],
    *,
    config: R08Config | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Build full LT-to-LT legs and require post-terminal IT-BOS confirmation."""
    cfg = (config or R08Config()).validate()
    if hierarchy.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups = list(hierarchy.groupby("source_timeframe_min", sort=True))
    reporter = ProgressReporter("[r08-completed-legs]", total=len(groups), every=1, enabled=progress)
    for gi, (minutes, part) in enumerate(groups, start=1):
        htf = bar_map[int(minutes)]
        lt = part.loc[part["is_lt"].eq(1)].copy()
        anchors = _compress_lt_anchors(lt)
        its = part.loc[part["is_it"].eq(1)].sort_values(["pivot_time", "swing_id"], kind="stable")
        leg_no = 0
        for j in range(len(anchors) - 1):
            start = anchors.iloc[j]
            end = anchors.iloc[j + 1]
            if str(start["pivot_side"]) == str(end["pivot_side"]):
                continue
            bullish = str(start["pivot_side"]) == "low" and str(end["pivot_side"]) == "high"
            bearish = str(start["pivot_side"]) == "high" and str(end["pivot_side"]) == "low"
            if not (bullish or bearish):
                continue
            direction = 1 if bullish else -1
            start_time = pd.Timestamp(start["pivot_time"])
            end_time = pd.Timestamp(end["pivot_time"])
            if end_time <= start_time:
                continue
            inside = its.loc[its["pivot_time"].between(start_time, end_time, inclusive="both")].copy()
            highs = inside.loc[inside["pivot_side"].eq("high")].sort_values("pivot_time", kind="stable")
            lows = inside.loc[inside["pivot_side"].eq("low")].sort_values("pivot_time", kind="stable")
            high_prices = pd.to_numeric(highs["level_price"], errors="coerce").to_numpy(dtype=float)
            low_prices = pd.to_numeric(lows["level_price"], errors="coerce").to_numpy(dtype=float)
            high_monotonic = _strict_monotonic(high_prices, direction)
            low_monotonic = _strict_monotonic(low_prices, direction)
            sequence_depth_ok = len(highs) >= int(cfg.min_it_swings_per_side) and len(lows) >= int(cfg.min_it_swings_per_side)
            sequence_integrity = bool(high_monotonic and low_monotonic and sequence_depth_ok)

            start_px = float(start["level_price"]); end_px = float(end["level_price"])
            move = (end_px / start_px - 1.0) * direction if abs(start_px) > EPS else np.nan
            duration_minutes = (end_time - start_time).total_seconds() / 60.0

            # A completed bullish leg reverses only after close < latest ITL;
            # bearish mirror closes > latest ITH.  This is a deliberate,
            # auditable quantization of the user's "large BOS confirms reversal"
            # requirement.  ST breaks never complete a large trend leg.
            ref = lows.iloc[-1] if bullish and len(lows) else (highs.iloc[-1] if bearish and len(highs) else None)
            if ref is None:
                bos_available = pd.NaT; bos_close = np.nan; ref_id = ""; ref_px = np.nan; ref_time = pd.NaT
            else:
                ref_id = str(ref["swing_id"]); ref_px = float(ref["level_price"]); ref_time = pd.Timestamp(ref["pivot_time"])
                bos_available, bos_close = _first_close_break(
                    htf,
                    after_time=end_time,
                    side="below" if bullish else "above",
                    level=ref_px,
                )
            start_avail = pd.Timestamp(start["lt_available_time"]) if pd.notna(start["lt_available_time"]) else pd.NaT
            end_avail = pd.Timestamp(end["lt_available_time"]) if pd.notna(end["lt_available_time"]) else pd.NaT
            ref_avail = pd.Timestamp(ref["it_available_time"]) if ref is not None and pd.notna(ref["it_available_time"]) else pd.NaT
            known_times = [x for x in (start_avail, end_avail, ref_avail, bos_available) if pd.notna(x)]
            leg_available = max(known_times) if len(known_times) == 4 else pd.NaT
            bos_confirmed = int(pd.notna(bos_available))
            leg_no += 1
            row: dict[str, object] = {
                "trend_leg_id": f"R08_{int(minutes)}m_LEG_{leg_no:07d}",
                "source_timeframe": str(start["source_timeframe"]),
                "source_timeframe_min": int(minutes),
                "trend_direction": "bullish" if bullish else "bearish",
                "trade_direction_sign": direction,
                "origin_swing_id": str(start["swing_id"]),
                "origin_class": "LTL" if bullish else "LTH",
                "origin_time": start_time,
                "origin_price": start_px,
                "terminal_swing_id": str(end["swing_id"]),
                "terminal_class": "LTH" if bullish else "LTL",
                "terminal_time": end_time,
                "terminal_price": end_px,
                "trend_move_pct": float(move) if np.isfinite(move) else np.nan,
                "trend_duration_minutes": float(duration_minutes),
                "it_high_count": int(len(highs)),
                "it_low_count": int(len(lows)),
                "ith_sequence": _sequence_string(highs),
                "itl_sequence": _sequence_string(lows),
                "ith_monotonic_flag": int(high_monotonic),
                "itl_monotonic_flag": int(low_monotonic),
                "sequence_depth_ok_flag": int(sequence_depth_ok),
                "directional_integrity_flag": int(sequence_integrity),
                "reversal_bos_structure": "ITL" if bullish else "ITH",
                "reversal_bos_reference_swing_id": ref_id,
                "reversal_bos_reference_time": ref_time,
                "reversal_bos_reference_price": ref_px,
                "reversal_bos_available_time": bos_available,
                "reversal_bos_close_price": bos_close,
                "reversal_bos_confirmed_flag": bos_confirmed,
                "leg_available_time": leg_available,
            }
            for scale in cfg.trend_scales:
                name = f"scale_ge_{int(round(scale * 100)):02d}pct_flag"
                row[name] = int(np.isfinite(move) and move >= float(scale))
            row["trend_qualified_ge3_flag"] = int(
                sequence_integrity and bos_confirmed and np.isfinite(move) and move >= min(cfg.trend_scales)
            )
            rows.append(row)
        reporter.update(gi)
    reporter.close()
    return pd.DataFrame(rows)


def build_bos_events(
    hierarchy: pd.DataFrame,
    bar_map: dict[int, pd.DataFrame],
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """First close-through for every classical IT/LT level after it is known."""
    if hierarchy.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    candidates = hierarchy.loc[hierarchy["is_it"].eq(1)].copy()
    reporter = ProgressReporter("[r08-bos]", total=len(candidates), every=max(1, len(candidates)//100), enabled=progress)
    done = 0
    for minutes, part in candidates.groupby("source_timeframe_min", sort=False):
        htf = bar_map[int(minutes)]
        idx_ns = htf.index.to_numpy(dtype="datetime64[ns]")
        close = pd.to_numeric(htf["close"], errors="coerce").to_numpy(dtype=float)
        tree = SegmentThresholdIndex(close)
        avail_ns = pd.to_datetime(htf["bar_end_time"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        for r in part.itertuples(index=False):
            side = str(r.pivot_side)
            level = float(r.level_price)
            known = pd.Timestamp(r.lt_available_time) if int(r.is_lt) == 1 and pd.notna(r.lt_available_time) else pd.Timestamp(r.it_available_time)
            start = int(np.searchsorted(avail_ns, np.datetime64(known), side="right"))
            if side == "high":
                pos = tree.first_geq(start, len(htf)-1, np.nextafter(level, np.inf))
                bos_dir = 1
            else:
                pos = tree.first_leq(start, len(htf)-1, np.nextafter(level, -np.inf))
                bos_dir = -1
            if pos >= 0:
                bos_bar_time = pd.Timestamp(htf.index[pos]); bos_available = pd.Timestamp(htf.iloc[pos]["bar_end_time"]); bos_close = float(close[pos])
            else:
                bos_bar_time = pd.NaT; bos_available = pd.NaT; bos_close = np.nan
            rows.append({
                "swing_id": str(r.swing_id),
                "source_timeframe": str(r.source_timeframe),
                "source_timeframe_min": int(minutes),
                "structure_class": "LT" if int(r.is_lt) == 1 else "IT",
                "pivot_side": side,
                "pivot_time": pd.Timestamp(r.pivot_time),
                "level_price": level,
                "structure_available_time": known,
                "bos_direction": bos_dir,
                "bos_bar_time": bos_bar_time,
                "bos_available_time": bos_available,
                "bos_close_price": bos_close,
            })
            done += 1; reporter.update(done)
    reporter.close()
    return pd.DataFrame(rows)


def build_trend_qualified_liquidity(
    bars_1m: pd.DataFrame,
    hierarchy: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    config: R08Config | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Build classified IT/LT liquidity from completed clean >=3% trend legs.

    R08.1 fixes the R08 projection bug by explicitly classifying every retained
    swing relative to the completed trend timeframe:

    * ``native``: swing timeframe == trend timeframe.  This is the canonical
      ICT full-trend liquidity set.
    * ``nested_lower_tf``: a causally-confirmed lower-timeframe IT/LT swing
      inside a completed higher-timeframe trend.  Kept as a separate research
      taxonomy because it may contain edge, but never mixed with native levels.
    * ``invalid_higher_tf_projection``: a higher-timeframe swing projected into
      a lower-timeframe trend.  This was an R08 construction artifact and is
      explicitly rejected from future key-liquidity use.
    """
    cfg = (config or R08Config()).validate()
    if hierarchy.empty or legs.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    high_tree = SegmentThresholdIndex(pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float))
    low_tree = SegmentThresholdIndex(pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float))
    bar_avail = (bars.index + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]")
    h_by_id = hierarchy.set_index("swing_id", drop=False)
    rows: list[dict[str, object]] = []
    qualified = legs.loc[legs["trend_qualified_ge3_flag"].eq(1)].copy()
    reporter = ProgressReporter("[r08-key-liquidity]", total=len(qualified), every=max(1, len(qualified)//100), enabled=progress)
    for i, leg in enumerate(qualified.itertuples(index=False), start=1):
        start_t = pd.Timestamp(leg.origin_time); end_t = pd.Timestamp(leg.terminal_time); activation = pd.Timestamp(leg.leg_available_time)
        direction = str(leg.trend_direction)
        desired_side = "low" if direction == "bullish" else "high"
        members = hierarchy.loc[
            hierarchy["pivot_time"].between(start_t, end_t, inclusive="both")
            & hierarchy["pivot_side"].eq(desired_side)
            & hierarchy["is_it"].eq(1)
        ].copy()
        origin_id = str(leg.origin_swing_id)
        if origin_id in h_by_id.index and origin_id not in set(members["swing_id"].astype(str)):
            members = pd.concat([h_by_id.loc[[origin_id]].copy(), members], ignore_index=True)
        members = members.drop_duplicates("swing_id").sort_values("pivot_time", kind="stable")
        trend_tf_min = int(leg.source_timeframe_min)
        for m in members.itertuples(index=False):
            swing_tf_min = int(m.source_timeframe_min)
            if swing_tf_min == trend_tf_min:
                projection_scope = "native"
            elif swing_tf_min < trend_tf_min:
                projection_scope = "nested_lower_tf"
            else:
                projection_scope = "invalid_higher_tf_projection"
            level = float(m.level_price)
            pivot_t = pd.Timestamp(m.pivot_time)
            own_known = (
                pd.Timestamp(m.lt_available_time)
                if str(m.swing_id) == origin_id and pd.notna(m.lt_available_time)
                else pd.Timestamp(m.it_available_time)
            )
            level_activation = max(activation, own_known)
            activation_pos = int(np.searchsorted(bar_avail, np.datetime64(level_activation), side="right")) - 1
            future_start = activation_pos + 1
            own_known_pos = int(np.searchsorted(bar_avail, np.datetime64(own_known), side="right"))
            if desired_side == "high":
                consumed_pos = high_tree.first_geq(max(0, own_known_pos), activation_pos, level) if activation_pos >= own_known_pos else -1
                future_pos = high_tree.first_geq(max(0, future_start), len(bars)-1, level)
            else:
                consumed_pos = low_tree.first_leq(max(0, own_known_pos), activation_pos, level) if activation_pos >= own_known_pos else -1
                future_pos = low_tree.first_leq(max(0, future_start), len(bars)-1, level)
            active = int(consumed_pos < 0)
            future_sweep_time = pd.Timestamp(bars.index[future_pos]) if future_pos >= 0 else pd.NaT
            future_sweep_available = pd.Timestamp(bars.index[future_pos] + pd.Timedelta(minutes=1)) if future_pos >= 0 else pd.NaT
            rows.append({
                "trend_leg_id": str(leg.trend_leg_id),
                "source_timeframe": str(leg.source_timeframe),
                "source_timeframe_min": trend_tf_min,
                "trend_direction": direction,
                "trend_move_pct": float(leg.trend_move_pct),
                "trend_origin_time": start_t,
                "trend_terminal_time": end_t,
                "trend_available_time": activation,
                "liquidity_side": "SSL" if desired_side == "low" else "BSL",
                "swing_id": str(m.swing_id),
                "swing_source_timeframe": str(m.source_timeframe),
                "swing_source_timeframe_min": swing_tf_min,
                "projection_scope": projection_scope,
                "canonical_key_liquidity_flag": int(projection_scope == "native"),
                "nested_lower_tf_flag": int(projection_scope == "nested_lower_tf"),
                "invalid_projection_flag": int(projection_scope == "invalid_higher_tf_projection"),
                "swing_is_lt": int(m.is_lt),
                "swing_role": ("LTL" if desired_side == "low" else "LTH") if str(m.swing_id) == origin_id else ("ITL" if desired_side == "low" else "ITH"),
                "pivot_time": pivot_t,
                "level_price": level,
                "own_it_available_time": pd.Timestamp(m.it_available_time) if pd.notna(m.it_available_time) else pd.NaT,
                "own_lt_available_time": pd.Timestamp(m.lt_available_time) if pd.notna(m.lt_available_time) else pd.NaT,
                "liquidity_activation_time": level_activation,
                "consumed_before_activation_flag": int(consumed_pos >= 0),
                "active_at_activation_flag": active,
                "first_sweep_after_activation_time": future_sweep_time if active else pd.NaT,
                "first_sweep_after_activation_available_time": future_sweep_available if active else pd.NaT,
                **{f"scale_ge_{int(round(s*100)):02d}pct_flag": int(float(leg.trend_move_pct) >= s) for s in cfg.trend_scales},
            })
        reporter.update(i)
    reporter.close()
    return pd.DataFrame(rows)


def build_projection_impact_atlas(
    bars_1m: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    research_start: pd.Timestamp,
    research_end: pd.Timestamp,
    horizons_minutes: tuple[int, ...] = (60, 180, 360, 720, 1440),
) -> pd.DataFrame:
    """Causal sweep-only diagnostic used to compare projection taxonomies.

    One physical swing is counted once per projection scope, using its earliest
    valid activation. Entry is the next 1m bar open after the first sweep.  This
    is intentionally an *impact diagnostic*, not the promoted trading strategy.
    """
    if liquidity.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    idx = bars.index
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    x = liquidity.loc[
        liquidity["active_at_activation_flag"].eq(1)
        & pd.to_datetime(liquidity["first_sweep_after_activation_available_time"], errors="coerce").notna()
    ].copy()
    if x.empty:
        return pd.DataFrame()
    x["liquidity_activation_time"] = pd.to_datetime(x["liquidity_activation_time"], errors="coerce")
    x["first_sweep_after_activation_time"] = pd.to_datetime(x["first_sweep_after_activation_time"], errors="coerce")
    x["first_sweep_after_activation_available_time"] = pd.to_datetime(x["first_sweep_after_activation_available_time"], errors="coerce")
    x = x.sort_values(["projection_scope", "swing_id", "liquidity_activation_time", "trend_move_pct"], ascending=[True, True, True, False], kind="stable")
    x = x.drop_duplicates(["projection_scope", "swing_id"], keep="first")
    x = x.loc[x["first_sweep_after_activation_time"].between(pd.Timestamp(research_start), pd.Timestamp(research_end), inclusive="both")].copy()
    rows: list[dict[str, object]] = []
    idx_ns = idx.to_numpy(dtype="datetime64[ns]")
    for r in x.itertuples(index=False):
        entry_time = pd.Timestamp(r.first_sweep_after_activation_available_time)
        ep = int(np.searchsorted(idx_ns, np.datetime64(entry_time), side="left"))
        if ep < 0 or ep >= len(idx) or not np.isfinite(opens[ep]) or opens[ep] <= 0:
            continue
        direction = 1.0 if str(r.liquidity_side) == "SSL" else -1.0
        row = {
            "projection_scope": str(r.projection_scope),
            "swing_id": str(r.swing_id),
            "trend_leg_id": str(r.trend_leg_id),
            "trend_timeframe": str(r.source_timeframe),
            "swing_timeframe": str(r.swing_source_timeframe),
            "liquidity_side": str(r.liquidity_side),
            "swing_role": str(r.swing_role),
            "trend_move_pct": float(r.trend_move_pct),
            "sweep_time": pd.Timestamp(r.first_sweep_after_activation_time),
            "entry_time": pd.Timestamp(idx[ep]),
            "entry_price": float(opens[ep]),
            "trade_direction": int(direction),
        }
        for h in horizons_minutes:
            xp = ep + int(h) - 1
            col = f"gross_return_{int(h)}m"
            row[col] = (float(closes[xp]) / float(opens[ep]) - 1.0) * direction if xp < len(idx) and np.isfinite(closes[xp]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_projection_impact(
    impact: pd.DataFrame,
    *,
    horizons_minutes: tuple[int, ...] = (60, 180, 360, 720, 1440),
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    roundtrip_cost_1x: float = 0.0011,
) -> pd.DataFrame:
    if impact.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []

    def emit(p: pd.DataFrame, scope: object, side: object, slice_type: str, slice_value: object) -> None:
        for h in horizons_minutes:
            gross = pd.to_numeric(p[f"gross_return_{int(h)}m"], errors="coerce").dropna()
            for mult in cost_multipliers:
                net = gross - float(roundtrip_cost_1x) * float(mult)
                gp = float(net.loc[net > 0].sum())
                gl = float(-net.loc[net < 0].sum())
                rows.append({
                    "projection_scope": scope,
                    "liquidity_side": side,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "horizon_minutes": int(h),
                    "cost_multiplier": float(mult),
                    "trades": int(len(net)),
                    "win_rate": float((net > 0).mean()) if len(net) else np.nan,
                    "mean_gross_return": float(gross.mean()) if len(gross) else np.nan,
                    "mean_net_return": float(net.mean()) if len(net) else np.nan,
                    "profit_factor": gp / gl if gl > EPS else np.nan,
                    "expectancy_positive_flag": int(float(net.mean()) > 0) if len(net) else 0,
                })

    for (scope, side), p in impact.groupby(["projection_scope", "liquidity_side"], dropna=False, sort=True):
        emit(p, scope, side, "overall", "all")
        if "trend_timeframe" in p.columns:
            for tf, q in p.groupby("trend_timeframe", dropna=False, sort=True):
                emit(q, scope, side, "trend_timeframe", tf)
        if "trend_move_pct" in p.columns:
            move = pd.to_numeric(p["trend_move_pct"], errors="coerce")
            for threshold in (0.03, 0.05, 0.07):
                q = p.loc[move.ge(threshold)]
                emit(q, scope, side, "trend_scale_ge", f"{int(threshold*100)}pct")
    return pd.DataFrame(rows)

def summarize_hierarchy(hierarchy: pd.DataFrame) -> pd.DataFrame:
    if hierarchy.empty:
        return pd.DataFrame()
    x = hierarchy.copy()
    return (
        x.groupby(["source_timeframe", "pivot_side"], dropna=False)
        .agg(st_swings=("swing_id", "size"), it_swings=("is_it", "sum"), lt_swings=("is_lt", "sum"))
        .reset_index()
    )


def summarize_trend_scales(legs: pd.DataFrame, *, config: R08Config | None = None) -> pd.DataFrame:
    cfg = (config or R08Config()).validate()
    if legs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (tf, direction), p in legs.groupby(["source_timeframe", "trend_direction"], sort=True):
        for s in cfg.trend_scales:
            flag = f"scale_ge_{int(round(s*100)):02d}pct_flag"
            q = p.loc[p[flag].eq(1)]
            clean = q.loc[q["directional_integrity_flag"].eq(1) & q["reversal_bos_confirmed_flag"].eq(1)]
            rows.append({
                "source_timeframe": tf,
                "trend_direction": direction,
                "trend_scale_pct": s * 100.0,
                "lt_to_lt_candidate_legs": int(len(q)),
                "clean_completed_legs": int(len(clean)),
                "clean_share": float(len(clean) / len(q)) if len(q) else np.nan,
                "median_move_pct": float(pd.to_numeric(clean["trend_move_pct"], errors="coerce").median()) if len(clean) else np.nan,
                "median_duration_hours": float(pd.to_numeric(clean["trend_duration_minutes"], errors="coerce").median() / 60.0) if len(clean) else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_key_liquidity(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["projection_scope", "source_timeframe", "swing_source_timeframe", "liquidity_side", "swing_role"] if c in levels.columns]
    return (
        levels.groupby(group_cols, dropna=False)
        .agg(
            context_rows=("swing_id", "size"),
            unique_physical_levels=("swing_id", "nunique"),
            active_context_rows=("active_at_activation_flag", "sum"),
            unique_active_levels=("swing_id", lambda s: int(s[levels.loc[s.index, "active_at_activation_flag"].eq(1)].nunique())),
            swept_later_context_rows=("first_sweep_after_activation_time", lambda s: int(pd.to_datetime(s, errors="coerce").notna().sum())),
            median_trend_move_pct=("trend_move_pct", "median"),
        )
        .reset_index()
    )


def r08_causal_audit(hierarchy: pd.DataFrame, legs: pd.DataFrame, liquidity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if hierarchy.empty:
        rows.append({"check": "hierarchy_nonempty", "violations": 1})
    else:
        it = hierarchy.loc[hierarchy["is_it"].eq(1)]
        lt = hierarchy.loc[hierarchy["is_lt"].eq(1)]
        rows.extend([
            {"check": "it_available_before_own_st", "violations": int((pd.to_datetime(it["it_available_time"], errors="coerce") < pd.to_datetime(it["st_available_time"], errors="coerce")).sum())},
            {"check": "lt_available_before_it", "violations": int((pd.to_datetime(lt["lt_available_time"], errors="coerce") < pd.to_datetime(lt["it_available_time"], errors="coerce")).sum())},
        ])
    if not legs.empty:
        a = pd.to_datetime(legs["leg_available_time"], errors="coerce")
        end = pd.to_datetime(legs["terminal_time"], errors="coerce")
        bos = pd.to_datetime(legs["reversal_bos_available_time"], errors="coerce")
        rows.extend([
            {"check": "leg_available_before_terminal", "violations": int((a.notna() & end.notna() & (a <= end)).sum())},
            {"check": "leg_available_before_bos", "violations": int((a.notna() & bos.notna() & (a < bos)).sum())},
        ])
    if not liquidity.empty:
        own = pd.to_datetime(liquidity["own_it_available_time"], errors="coerce")
        act = pd.to_datetime(liquidity["liquidity_activation_time"], errors="coerce")
        sweep = pd.to_datetime(liquidity["first_sweep_after_activation_available_time"], errors="coerce")
        rows.extend([
            {"check": "liquidity_activation_before_own_it", "violations": int((own.notna() & act.notna() & (act < own)).sum())},
            {"check": "future_sweep_not_after_activation", "violations": int((sweep.notna() & act.notna() & (sweep <= act)).sum())},
            {"check": "st_only_liquidity_leaked", "violations": int((~liquidity["swing_role"].astype(str).isin(["ITH", "ITL", "LTH", "LTL"])).sum())},
            {"check": "consumed_level_marked_active", "violations": int((liquidity["consumed_before_activation_flag"].eq(1) & liquidity["active_at_activation_flag"].eq(1)).sum())},
        ])
        if "projection_scope" in liquidity.columns:
            rows.extend([
                {"check": "native_scope_timeframe_mismatch", "violations": int((liquidity["projection_scope"].eq("native") & liquidity["source_timeframe_min"].ne(liquidity["swing_source_timeframe_min"])).sum())},
                {"check": "nested_scope_not_lower_tf", "violations": int((liquidity["projection_scope"].eq("nested_lower_tf") & liquidity["swing_source_timeframe_min"].ge(liquidity["source_timeframe_min"])).sum())},
                {"check": "invalid_scope_not_higher_tf", "violations": int((liquidity["projection_scope"].eq("invalid_higher_tf_projection") & liquidity["swing_source_timeframe_min"].le(liquidity["source_timeframe_min"])).sum())},
            ])
    return pd.DataFrame(rows)
