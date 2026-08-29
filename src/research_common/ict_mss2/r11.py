#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R11.1 continuous visible-liquidity path atlas.

R11.1 corrects the original R11 day-open framing for ETH.  ETH is a 24/7
market: 00:00 is only a reporting boundary and never a liquidity reset,
activation boundary, target-freeze boundary, or strategy rule.

The atlas therefore runs continuously:
* every causally confirmed classical IT/LT swing (15m/30m/1H/4H) becomes
  active at its real ``it_available_time``;
* it remains active until its first causal consumption;
* newly confirmed liquidity can become active at any minute, including later
  in the same calendar day;
* each root sweep freezes the opposite-side liquidity that is active *at that
  sweep time*; paths may cross midnight without reset;
* calendar date is retained for reporting only.

STH/STL remain construction-only and never enter the trading liquidity map.
Completed-trend/native/nested/3-5-7% fields remain descriptive labels only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex


@dataclass(frozen=True)
class R11Config:
    region_cluster_bps: float = 10.0
    region_cluster_sensitivities_bps: tuple[float, ...] = (5.0, 10.0, 20.0)
    horizons_minutes: tuple[int, ...] = (15, 30, 60, 180, 360, 720, 1440)
    opposite_horizons_minutes: tuple[int, ...] = (1440, 2880)
    landmark_max_minutes: int = 180
    sequence_horizon_minutes: int = 2880

    def validate(self) -> "R11Config":
        if self.region_cluster_bps <= 0:
            raise ValueError("region_cluster_bps must be positive")
        if any(x <= 0 for x in self.region_cluster_sensitivities_bps):
            raise ValueError("cluster sensitivities must be positive")
        if any(int(x) <= 0 for x in self.horizons_minutes):
            raise ValueError("horizons must be positive")
        if any(int(x) <= 0 for x in self.opposite_horizons_minutes):
            raise ValueError("opposite horizons must be positive")
        if self.landmark_max_minutes <= 0 or self.sequence_horizon_minutes <= 0:
            raise ValueError("landmark/sequence horizons must be positive")
        return self


def _first_consumption_positions(bars: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Attach first touch after the level is actually causally available.

    ``it_available_time`` is a wall-clock availability timestamp.  If a level
    becomes available at 01:00, the 00:59-01:00 bar cannot retroactively consume
    it.  The first eligible 1m bar therefore has ``bar_start >= 01:00``.
    """
    if levels.empty:
        return levels.copy()
    b = normalize_1m_bars(bars)
    high_tree = SegmentThresholdIndex(pd.to_numeric(b["high"], errors="coerce").to_numpy(float))
    low_tree = SegmentThresholdIndex(pd.to_numeric(b["low"], errors="coerce").to_numpy(float))
    bar_starts = b.index.to_numpy(dtype="datetime64[ns]")
    rows: list[dict[str, object]] = []
    for r in levels.itertuples(index=False):
        known = pd.Timestamp(r.it_available_time)
        start = int(np.searchsorted(bar_starts, np.datetime64(known), side="left"))
        px = float(r.level_price)
        if start >= len(b):
            pos = -1
        elif str(r.pivot_side) == "high":
            pos = high_tree.first_geq(start, len(b) - 1, px)
        else:
            pos = low_tree.first_leq(start, len(b) - 1, px)
        d = r._asdict()
        d["liquidity_side"] = "BSL" if str(r.pivot_side) == "high" else "SSL"
        d["it_activation_time"] = known
        d["first_consumption_pos_1m"] = int(pos)
        d["first_consumption_time"] = pd.Timestamp(b.index[pos]) if pos >= 0 else pd.NaT
        d["first_consumption_available_time"] = (
            pd.Timestamp(b.index[pos] + pd.Timedelta(minutes=1)) if pos >= 0 else pd.NaT
        )
        rows.append(d)
    return pd.DataFrame(rows)


def build_visible_it_lt_liquidity(
    bars_1m: pd.DataFrame,
    hierarchy: pd.DataFrame,
    trend_liquidity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """All causally confirmed physical IT/LT levels; no trend-quality filter."""
    if hierarchy.empty:
        return pd.DataFrame()
    h = hierarchy.loc[hierarchy["is_it"].eq(1)].copy()
    h["it_available_time"] = pd.to_datetime(h["it_available_time"], errors="coerce")
    h["lt_available_time"] = pd.to_datetime(h["lt_available_time"], errors="coerce")
    h = h.dropna(subset=["it_available_time", "level_price", "pivot_time"]).drop_duplicates("swing_id")
    h = _first_consumption_positions(bars_1m, h)

    # Completed-trend context is descriptive only and collapsed to physical
    # swing grain, preventing context joins from multiplying liquidity levels.
    if trend_liquidity is not None and not trend_liquidity.empty:
        q = trend_liquidity.copy()
        scope = q.get("projection_scope", pd.Series(index=q.index, dtype=str))
        q = q.loc[~scope.eq("invalid_higher_tf_projection")]
        if not q.empty:
            agg = q.groupby("swing_id", sort=False).agg(
                qualified_context_count=("trend_leg_id", "nunique"),
                max_completed_trend_tf_min=("source_timeframe_min", "max"),
                max_completed_trend_move_pct=("trend_move_pct", "max"),
                native_context_flag=("canonical_key_liquidity_flag", "max"),
                nested_context_flag=("nested_lower_tf_flag", "max"),
                trend_scale_ge3_flag=("scale_ge_03pct_flag", "max"),
                trend_scale_ge5_flag=("scale_ge_05pct_flag", "max"),
                trend_scale_ge7_flag=("scale_ge_07pct_flag", "max"),
            ).reset_index()
            h = h.merge(agg, on="swing_id", how="left", validate="one_to_one")
    defaults = {
        "qualified_context_count": 0,
        "max_completed_trend_tf_min": np.nan,
        "max_completed_trend_move_pct": np.nan,
        "native_context_flag": 0,
        "nested_context_flag": 0,
        "trend_scale_ge3_flag": 0,
        "trend_scale_ge5_flag": 0,
        "trend_scale_ge7_flag": 0,
    }
    for c, v in defaults.items():
        if c not in h.columns:
            h[c] = v
        elif isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            h[c] = h[c].fillna(v)
    return h.sort_values(["it_activation_time", "pivot_time", "swing_id"], kind="stable").reset_index(drop=True)


def _role_at_time(frame: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    lt_ready = (
        frame["is_lt"].eq(1)
        & pd.to_datetime(frame["lt_available_time"], errors="coerce").notna()
        & pd.to_datetime(frame["lt_available_time"], errors="coerce").le(when)
    )
    return pd.Series(
        np.where(
            lt_ready,
            np.where(frame["pivot_side"].eq("low"), "LTL", "LTH"),
            np.where(frame["pivot_side"].eq("low"), "ITL", "ITH"),
        ),
        index=frame.index,
    )


def _cluster_prices(part: pd.DataFrame, tolerance_bps: float) -> list[pd.DataFrame]:
    """Single-link nearby active physical levels into display/research zones."""
    if part.empty:
        return []
    x = part.sort_values(["level_price", "swing_id"], kind="stable")
    groups: list[list[int]] = []
    current: list[int] = []
    prev = np.nan
    for idx, row in x.iterrows():
        px = float(row["level_price"])
        if not current:
            current = [idx]
            prev = px
            continue
        tol = abs(prev) * float(tolerance_bps) / 10000.0
        if abs(px - prev) <= tol + 1e-12:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
        prev = px
    if current:
        groups.append(current)
    return [x.loc[g].copy() for g in groups]


def _region_record(c: pd.DataFrame, *, when: pd.Timestamp, side: str, prefix: str, ordinal: int) -> dict[str, object]:
    roles = _role_at_time(c, when)
    lo = float(pd.to_numeric(c["level_price"], errors="coerce").min())
    hi = float(pd.to_numeric(c["level_price"], errors="coerce").max())
    center = float(pd.to_numeric(c["level_price"], errors="coerce").median())
    piv = pd.to_datetime(c["pivot_time"], errors="coerce")
    return {
        "region_id": f"{prefix}_{side}_{ordinal:03d}",
        "liquidity_side": side,
        "zone_low": lo,
        "zone_high": hi,
        "zone_center": center,
        "member_count": int(c["swing_id"].nunique()),
        "it_count": int(roles.astype(str).str.startswith("IT").sum()),
        "lt_count": int(roles.astype(str).str.startswith("LT").sum()),
        "max_swing_tf_min": int(pd.to_numeric(c["source_timeframe_min"], errors="coerce").max()),
        "qualified_context_any": int(pd.to_numeric(c["qualified_context_count"], errors="coerce").fillna(0).gt(0).any()),
        "native_context_any": int(pd.to_numeric(c["native_context_flag"], errors="coerce").fillna(0).max()),
        "nested_context_any": int(pd.to_numeric(c["nested_context_flag"], errors="coerce").fillna(0).max()),
        "trend_ge5_any": int(pd.to_numeric(c["trend_scale_ge5_flag"], errors="coerce").fillna(0).max()),
        "trend_ge7_any": int(pd.to_numeric(c["trend_scale_ge7_flag"], errors="coerce").fillna(0).max()),
        "oldest_age_days": float(max(0.0, (when - piv.min()).total_seconds() / 86400.0)),
        "swing_ids": "|".join(sorted(c["swing_id"].astype(str).unique())),
        "roles": "|".join(sorted(roles.astype(str).unique())),
        "source_timeframes": "|".join(sorted(c["source_timeframe"].astype(str).unique())),
        "region_available_time": pd.to_datetime(c["it_activation_time"], errors="coerce").max(),
    }


def _active_levels(visible: pd.DataFrame, when: pd.Timestamp, *, side: str | None = None, strictly_unconsumed: bool = False) -> pd.DataFrame:
    """Levels visible just before/at ``when`` with no calendar-session reset."""
    if visible.empty:
        return visible.copy()
    act = pd.to_datetime(visible["it_activation_time"], errors="coerce")
    cons = pd.to_datetime(visible["first_consumption_time"], errors="coerce")
    if strictly_unconsumed:
        alive = cons.isna() | cons.gt(when)
    else:
        alive = cons.isna() | cons.ge(when)
    mask = act.le(when) & alive
    if side is not None:
        mask &= visible["liquidity_side"].eq(side)
    return visible.loc[mask].copy()




class _ActiveTargetIndex:
    """Price-sorted physical-liquidity index for fast event-time target lookup."""

    def __init__(self, visible: pd.DataFrame, side: str):
        q = visible.loc[visible["liquidity_side"].eq(side)].copy()
        q = q.sort_values(["level_price", "swing_id"], kind="stable").reset_index(drop=True)
        self.frame = q
        self.side = side
        self.price = pd.to_numeric(q.get("level_price"), errors="coerce").to_numpy(float)
        self.activation = pd.to_datetime(q.get("it_activation_time"), errors="coerce").to_numpy(dtype="datetime64[ns]")
        self.consumption = pd.to_datetime(q.get("first_consumption_time"), errors="coerce").to_numpy(dtype="datetime64[ns]")

    def _is_active(self, i: int, when64: np.datetime64) -> bool:
        if i < 0 or i >= len(self.price):
            return False
        if np.isnat(self.activation[i]) or self.activation[i] > when64:
            return False
        return np.isnat(self.consumption[i]) or self.consumption[i] > when64

    def nearest_region(self, *, when: pd.Timestamp, reference_price: float, tolerance_bps: float) -> dict[str, object] | None:
        if len(self.price) == 0:
            return None
        t64 = np.datetime64(pd.Timestamp(when))
        if self.side == "BSL":
            j = int(np.searchsorted(self.price, float(reference_price), side="right"))
            while j < len(self.price) and not self._is_active(j, t64):
                j += 1
            if j >= len(self.price):
                return None
            members = [j]
            prev = float(self.price[j])
            k = j + 1
            while k < len(self.price):
                if not self._is_active(k, t64):
                    k += 1
                    continue
                px = float(self.price[k])
                if abs(px - prev) <= abs(prev) * float(tolerance_bps) / 10000.0 + 1e-12:
                    members.append(k); prev = px; k += 1
                else:
                    break
        else:
            j = int(np.searchsorted(self.price, float(reference_price), side="left")) - 1
            while j >= 0 and not self._is_active(j, t64):
                j -= 1
            if j < 0:
                return None
            members = [j]
            prev = float(self.price[j])
            k = j - 1
            while k >= 0:
                if not self._is_active(k, t64):
                    k -= 1
                    continue
                px = float(self.price[k])
                if abs(prev - px) <= abs(prev) * float(tolerance_bps) / 10000.0 + 1e-12:
                    members.append(k); prev = px; k -= 1
                else:
                    break
        c = self.frame.iloc[sorted(members)].copy()
        return _region_record(c, when=pd.Timestamp(when), side=self.side, prefix=f"R11T_{pd.Timestamp(when):%Y%m%d%H%M}", ordinal=1)


def _nearest_active_opposite_region(
    visible: pd.DataFrame,
    *, when: pd.Timestamp, root_side: str, reference_price: float,
    tolerance_bps: float,
) -> dict[str, object] | None:
    """Freeze nearest still-unconsumed opposite region at the root sweep time."""
    opposite = "BSL" if root_side == "SSL" else "SSL"
    q = _active_levels(visible, when, side=opposite, strictly_unconsumed=True)
    if q.empty:
        return None
    if opposite == "BSL":
        q = q.loc[pd.to_numeric(q["level_price"], errors="coerce") > float(reference_price)]
    else:
        q = q.loc[pd.to_numeric(q["level_price"], errors="coerce") < float(reference_price)]
    if q.empty:
        return None
    clusters = _cluster_prices(q, tolerance_bps)
    regs = [_region_record(c, when=when, side=opposite, prefix=f"R11T_{when:%Y%m%d%H%M}", ordinal=i + 1) for i, c in enumerate(clusters)]
    if opposite == "BSL":
        regs = [r for r in regs if float(r["zone_low"]) > float(reference_price)]
        regs.sort(key=lambda r: float(r["zone_low"]))
    else:
        regs = [r for r in regs if float(r["zone_high"]) < float(reference_price)]
        regs.sort(key=lambda r: float(r["zone_high"]), reverse=True)
    return regs[0] if regs else None


def build_continuous_sweep_events(
    bars_1m: pd.DataFrame,
    visible: pd.DataFrame,
    *, research_start: pd.Timestamp,
    research_end: pd.Timestamp,
    tolerance_bps: float = 10.0,
) -> pd.DataFrame:
    """One continuous root event per ``sweep_time × side``.

    Physical IT/LT levels first consumed on the same 1m bar are one market
    event.  Nearby consumed levels are clustered only to describe how many
    visible zones were traversed on that bar.  No day-open snapshot is used.
    """
    if visible.empty:
        return pd.DataFrame()
    start = pd.Timestamp(research_start)
    end = pd.Timestamp(research_end)
    q = visible.copy()
    q["first_consumption_time"] = pd.to_datetime(q["first_consumption_time"], errors="coerce")
    q = q.loc[q["first_consumption_time"].notna() & q["first_consumption_time"].between(start, end, inclusive="both")]
    if q.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (when, side), p in q.groupby(["first_consumption_time", "liquidity_side"], sort=True):
        when = pd.Timestamp(when)
        clusters = _cluster_prices(p, tolerance_bps)
        regs = [_region_record(c, when=when, side=str(side), prefix=f"R11S_{when:%Y%m%d%H%M}", ordinal=i + 1) for i, c in enumerate(clusters)]
        # Union boundary = all same-side liquidity physically consumed on this bar.
        union_low = min(float(r["zone_low"]) for r in regs)
        union_high = max(float(r["zone_high"]) for r in regs)
        row = {
            "sweep_event_id": f"R11_SWEEP_{when:%Y%m%d%H%M}_{side}",
            "sweep_time": when,
            "report_date": when.date().isoformat(),  # reporting only
            "liquidity_side": str(side),
            "zone_low": union_low,
            "zone_high": union_high,
            "swept_region_count": len(regs),
            "swept_level_count": int(p["swing_id"].nunique()),
            "max_swing_tf_min": int(pd.to_numeric(p["source_timeframe_min"], errors="coerce").max()),
            "lt_count": int(sum(int(r["lt_count"]) for r in regs)),
            "qualified_context_any": int(pd.to_numeric(p["qualified_context_count"], errors="coerce").fillna(0).gt(0).any()),
            "native_context_any": int(pd.to_numeric(p["native_context_flag"], errors="coerce").fillna(0).max()),
            "nested_context_any": int(pd.to_numeric(p["nested_context_flag"], errors="coerce").fillna(0).max()),
            "trend_ge5_any": int(pd.to_numeric(p["trend_scale_ge5_flag"], errors="coerce").fillna(0).max()),
            "trend_ge7_any": int(pd.to_numeric(p["trend_scale_ge7_flag"], errors="coerce").fillna(0).max()),
            "swing_ids": "|".join(sorted(p["swing_id"].astype(str).unique())),
            "swept_regions": ";".join(f"{r['zone_low']:.8g}-{r['zone_high']:.8g}" for r in regs),
            "max_member_activation_time": pd.to_datetime(p["it_activation_time"], errors="coerce").max(),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["sweep_time", "liquidity_side"], kind="stable").reset_index(drop=True)
    if not out.empty:
        counts = out.groupby("sweep_time")["liquidity_side"].transform("nunique")
        out["same_bar_two_sided_flag"] = counts.gt(1).astype(int)
    return out


def _first_region_sweep(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    r: pd.Series | dict,
    start_pos: int,
    end_pos: int,
) -> int:
    if start_pos > end_pos:
        return -1
    if str(r["liquidity_side"]) == "BSL":
        return int(high_tree.first_geq(start_pos, end_pos, float(r["zone_high"])))
    return int(low_tree.first_leq(start_pos, end_pos, float(r["zone_low"])))


def _first_reclaim(close: np.ndarray, *, sweep_pos: int, side: str, zone_low: float, zone_high: float, max_minutes: int) -> int:
    end = min(len(close) - 1, sweep_pos + int(max_minutes))
    seg = close[sweep_pos + 1 : end + 1]
    if side == "SSL":
        hit = np.flatnonzero(seg > float(zone_high) + 1e-12)
    else:
        hit = np.flatnonzero(seg < float(zone_low) - 1e-12)
    return int(sweep_pos + 1 + hit[0]) if len(hit) else -1


def _first_post_sweep_st_mss(
    htf: pd.DataFrame,
    *, sweep_time: pd.Timestamp,
    side: str,
    minutes: int,
    max_minutes: int,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    if htf.empty:
        return pd.NaT, np.nan
    delta = pd.Timedelta(minutes=minutes)
    lo = pd.Timestamp(sweep_time)
    hi = lo + pd.Timedelta(minutes=max_minutes)
    q = htf.loc[(htf.index >= lo.floor(f"{minutes}min")) & (htf.index <= hi)]
    if len(q) < 5:
        return pd.NaT, np.nan
    if side == "SSL":
        vals = pd.to_numeric(q["high"], errors="coerce").to_numpy(float)
        close = pd.to_numeric(q["close"], errors="coerce").to_numpy(float)
        piv = np.flatnonzero((vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:])) + 1
        cmp = lambda c, p: c > p
    else:
        vals = pd.to_numeric(q["low"], errors="coerce").to_numpy(float)
        close = pd.to_numeric(q["close"], errors="coerce").to_numpy(float)
        piv = np.flatnonzero((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])) + 1
        cmp = lambda c, p: c < p
    for p in piv:
        pivot_time = q.index[p]
        if pivot_time < sweep_time:
            continue
        available = q.index[p + 1] + delta
        j0 = int(q.index.searchsorted(available, side="left"))
        for j in range(j0, len(q)):
            if cmp(float(close[j]), float(vals[p])):
                return pd.Timestamp(q.index[j] + delta), float(vals[p])
    return pd.NaT, np.nan


def _first_directional_fvg(
    b: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    *, sweep_pos: int,
    side: str,
    max_minutes: int,
) -> tuple[pd.Timestamp | pd.NaT, float, float]:
    end = min(len(b) - 1, sweep_pos + int(max_minutes))
    start = max(2, sweep_pos + 1)
    for i in range(start, end + 1):
        if side == "SSL" and low[i] > high[i - 2] + 1e-12:
            return pd.Timestamp(b.index[i] + pd.Timedelta(minutes=1)), float(high[i - 2]), float(low[i])
        if side == "BSL" and high[i] < low[i - 2] - 1e-12:
            return pd.Timestamp(b.index[i] + pd.Timedelta(minutes=1)), float(high[i]), float(low[i - 2])
    return pd.NaT, np.nan, np.nan


def _compress_side_sequence(frame: pd.DataFrame) -> list[str]:
    seq: list[str] = []
    for side in frame["liquidity_side"].astype(str):
        if not seq or seq[-1] != side:
            seq.append(side)
    return seq


def build_continuous_path_atlas(
    bars_1m: pd.DataFrame,
    visible: pd.DataFrame,
    sweep_events: pd.DataFrame,
    *, config: R11Config | None = None,
) -> pd.DataFrame:
    """One root path per unique sweep minute, continuous across midnight."""
    cfg = (config or R11Config()).validate()
    if sweep_events.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    high_arr = pd.to_numeric(b["high"], errors="coerce").to_numpy(float)
    low_arr = pd.to_numeric(b["low"], errors="coerce").to_numpy(float)
    close_arr = pd.to_numeric(b["close"], errors="coerce").to_numpy(float)
    high_tree = SegmentThresholdIndex(high_arr)
    low_tree = SegmentThresholdIndex(low_arr)
    htf_map = {m: aggregate_bars(b, m) for m in (1, 2, 5)}
    events = sweep_events.sort_values(["sweep_time", "liquidity_side"], kind="stable").reset_index(drop=True)
    event_times = pd.to_datetime(events["sweep_time"], errors="coerce")
    event_times_np = event_times.to_numpy(dtype="datetime64[ns]")
    unique_times = pd.Index(event_times.dropna().unique()).sort_values()
    target_index = {
        "SSL": _ActiveTargetIndex(visible, "SSL"),
        "BSL": _ActiveTargetIndex(visible, "BSL"),
    }
    rows: list[dict[str, object]] = []

    for root_time in unique_times:
        root_time = pd.Timestamp(root_time)
        same = events.loc[event_times.eq(root_time)]
        pos = int(b.index.searchsorted(root_time, side="left"))
        if pos >= len(b) or b.index[pos] != root_time:
            continue
        row: dict[str, object] = {
            "path_id": f"R11_PATH_{root_time:%Y%m%d%H%M}",
            "root_sweep_time": root_time,
            "report_date": root_time.date().isoformat(),  # reporting only; never logic
            "root_bar_open": float(b.iloc[pos]["open"]),
            "root_bar_high": float(b.iloc[pos]["high"]),
            "root_bar_low": float(b.iloc[pos]["low"]),
            "root_bar_close": float(b.iloc[pos]["close"]),
            "same_bar_two_sided_flag": int(same["liquidity_side"].nunique() > 1),
            "root_swept_side_count": int(same["liquidity_side"].nunique()),
            "root_swept_level_count": int(pd.to_numeric(same["swept_level_count"], errors="coerce").sum()),
        }
        if same["liquidity_side"].nunique() != 1:
            row["root_sweep_side"] = "BOTH"
            row["path_archetype"] = "same_bar_two_sided"
            rows.append(row)
            continue

        root = same.iloc[0]
        side = str(root["liquidity_side"])
        row.update({
            "root_sweep_side": side,
            "root_zone_low": float(root["zone_low"]),
            "root_zone_high": float(root["zone_high"]),
            "root_swept_region_count": int(root["swept_region_count"]),
            "root_max_tf_min": int(root["max_swing_tf_min"]),
            "root_lt_count": int(root["lt_count"]),
            "root_qualified_context_any": int(root["qualified_context_any"]),
            "root_native_context_any": int(root["native_context_any"]),
            "root_nested_context_any": int(root["nested_context_any"]),
            "root_trend_ge5_any": int(root["trend_ge5_any"]),
            "root_trend_ge7_any": int(root["trend_ge7_any"]),
        })

        # Freeze opposite target using only levels active at the exact root time;
        # newly confirmed liquidity later is allowed in later paths, never here.
        opposite = "BSL" if side == "SSL" else "SSL"
        target = target_index[opposite].nearest_region(
            when=root_time,
            reference_price=float(b.iloc[pos]["close"]),
            tolerance_bps=cfg.region_cluster_bps,
        )
        if target is not None:
            for k, v in target.items():
                row[f"opposite_target_{k}"] = v
            for hz in cfg.opposite_horizons_minutes:
                end_pos = min(len(b) - 1, pos + int(hz))
                pos2 = _first_region_sweep(high_tree, low_tree, target, pos + 1, end_pos)
                row[f"opposite_target_hit_{hz}m_flag"] = int(pos2 >= 0)
                row[f"opposite_target_hit_{hz}m_time"] = pd.Timestamp(b.index[pos2]) if pos2 >= 0 else pd.NaT
                row[f"opposite_target_hit_{hz}m_delay_min"] = (
                    float((b.index[pos2] - root_time).total_seconds() / 60.0) if pos2 >= 0 else np.nan
                )
        else:
            row["opposite_target_region_id"] = None

        # Continuous future sweep sequence, no reset at 00:00.
        root64 = np.datetime64(root_time)
        hi64 = np.datetime64(root_time + pd.Timedelta(minutes=cfg.sequence_horizon_minutes))
        j0 = int(np.searchsorted(event_times_np, root64, side="right"))
        j1 = int(np.searchsorted(event_times_np, hi64, side="right"))
        future = events.iloc[j0:j1]
        seq = _compress_side_sequence(future)
        row["future_side_sequence_48h"] = ">".join(seq)
        row["future_sweep_event_count_48h"] = int(len(future))
        opp_future = future.loc[future["liquidity_side"].eq(opposite)]
        row["first_opposite_sweep_time_48h"] = pd.Timestamp(opp_future.iloc[0]["sweep_time"]) if len(opp_future) else pd.NaT
        row["first_opposite_sweep_delay_min_48h"] = (
            float((pd.Timestamp(opp_future.iloc[0]["sweep_time"]) - root_time).total_seconds() / 60.0)
            if len(opp_future) else np.nan
        )

        reclaim = _first_reclaim(
            close_arr,
            sweep_pos=pos,
            side=side,
            zone_low=float(root["zone_low"]),
            zone_high=float(root["zone_high"]),
            max_minutes=cfg.landmark_max_minutes,
        )
        row["reclaim_available_time"] = pd.Timestamp(b.index[reclaim] + pd.Timedelta(minutes=1)) if reclaim >= 0 else pd.NaT
        for m in (1, 2, 5):
            t, lvl = _first_post_sweep_st_mss(
                htf_map[m], sweep_time=root_time, side=side, minutes=m, max_minutes=cfg.landmark_max_minutes
            )
            row[f"post_sweep_st_mss_{m}m_available_time"] = t
            row[f"post_sweep_st_mss_{m}m_level"] = lvl
        fvg_t, fvg_lo, fvg_hi = _first_directional_fvg(
            b, high_arr, low_arr, sweep_pos=pos, side=side, max_minutes=cfg.landmark_max_minutes
        )
        row["first_directional_fvg_available_time"] = fvg_t
        row["first_directional_fvg_low"] = fvg_lo
        row["first_directional_fvg_high"] = fvg_hi

        entry_pos = min(len(b) - 1, pos + 1)
        entry = float(b.iloc[entry_pos]["open"])
        direction = 1 if side == "SSL" else -1
        row["sweep_next_open_time"] = pd.Timestamp(b.index[entry_pos])
        row["sweep_next_open_price"] = entry
        for hz in cfg.horizons_minutes:
            end_pos = min(len(b) - 1, entry_pos + int(hz) - 1)
            seg = b.iloc[entry_pos : end_pos + 1]
            if seg.empty:
                continue
            row[f"ret_{hz}m"] = direction * (float(seg.iloc[-1]["close"]) / entry - 1.0)
            row[f"mfe_{hz}m"] = (
                float(seg["high"].max()) / entry - 1.0
                if direction > 0
                else entry / float(seg["low"].min()) - 1.0
            )
            row[f"mae_{hz}m"] = (
                float(seg["low"].min()) / entry - 1.0
                if direction > 0
                else 1.0 - float(seg["high"].max()) / entry
            )

        target24 = int(row.get("opposite_target_hit_1440m_flag", 0) or 0)
        if target is None:
            archetype = f"{side.lower()}_no_visible_opposite_target"
        elif target24:
            archetype = f"{side.lower()}_to_frozen_{opposite.lower()}_within_24h"
        else:
            archetype = f"{side.lower()}_no_frozen_{opposite.lower()}_hit_24h"
        row["path_archetype"] = archetype
        rows.append(row)

    return pd.DataFrame(rows).sort_values("root_sweep_time", kind="stable").reset_index(drop=True)


def build_event_time_liquidity_snapshot(
    bars_1m: pd.DataFrame,
    visible: pd.DataFrame,
    root_times: list[pd.Timestamp] | pd.Series | pd.Index,
    *, tolerance_bps: float = 10.0,
) -> pd.DataFrame:
    """Small manual-review snapshot: all active regions at selected root times."""
    if visible.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    rows: list[dict[str, object]] = []
    for when in pd.to_datetime(pd.Index(root_times), errors="coerce").dropna().unique():
        when = pd.Timestamp(when)
        pos = int(b.index.searchsorted(when, side="left"))
        if pos >= len(b):
            continue
        ref = float(b.iloc[pos]["close"])
        q = _active_levels(visible, when, strictly_unconsumed=False)
        for side, p in q.groupby("liquidity_side", sort=True):
            regs = [_region_record(c, when=when, side=str(side), prefix=f"R11M_{when:%Y%m%d%H%M}", ordinal=i + 1) for i, c in enumerate(_cluster_prices(p, tolerance_bps))]
            if side == "BSL":
                regs = [r for r in regs if float(r["zone_low"]) > ref]
                regs.sort(key=lambda r: float(r["zone_low"]))
            else:
                regs = [r for r in regs if float(r["zone_high"]) < ref]
                regs.sort(key=lambda r: float(r["zone_high"]), reverse=True)
            for rank, r in enumerate(regs, start=1):
                r = dict(r)
                r["root_sweep_time"] = when
                r["reference_close"] = ref
                r["nearest_rank_at_root"] = rank
                rows.append(r)
    return pd.DataFrame(rows)


def summarize_path_archetypes(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, p in paths.groupby("path_archetype", dropna=False, sort=True):
        rows.append({
            "path_archetype": key,
            "root_events": len(p),
            "share": len(p) / len(paths),
            "opposite_24h_hit_rate": pd.to_numeric(p.get("opposite_target_hit_1440m_flag"), errors="coerce").mean(),
            "mean_ret_60m": pd.to_numeric(p.get("ret_60m"), errors="coerce").mean(),
            "mean_ret_360m": pd.to_numeric(p.get("ret_360m"), errors="coerce").mean(),
            "mean_ret_1440m": pd.to_numeric(p.get("ret_1440m"), errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def summarize_first_sweep(paths: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible name: summarize continuous root sweeps, not day-open first sweeps."""
    if paths.empty or "root_sweep_side" not in paths:
        return pd.DataFrame()
    q = paths.loc[paths["root_sweep_side"].isin(["SSL", "BSL"])].copy()
    if q.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["root_sweep_side", "root_max_tf_min", "root_swept_region_count"]
    for key, p in q.groupby(group_cols, dropna=False, sort=True):
        rows.append({
            **dict(zip(group_cols, key)),
            "root_events": len(p),
            "opposite_24h_hit_rate": pd.to_numeric(p.get("opposite_target_hit_1440m_flag"), errors="coerce").mean(),
            "opposite_48h_hit_rate": pd.to_numeric(p.get("opposite_target_hit_2880m_flag"), errors="coerce").mean(),
            "mean_ret_60m": pd.to_numeric(p.get("ret_60m"), errors="coerce").mean(),
            "mean_ret_360m": pd.to_numeric(p.get("ret_360m"), errors="coerce").mean(),
            "mean_ret_1440m": pd.to_numeric(p.get("ret_1440m"), errors="coerce").mean(),
            "median_mfe_1440m": pd.to_numeric(p.get("mfe_1440m"), errors="coerce").median(),
            "median_mae_1440m": pd.to_numeric(p.get("mae_1440m"), errors="coerce").median(),
        })
    return pd.DataFrame(rows)


def r11_causal_audit(visible: pd.DataFrame, sweep_events: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not visible.empty:
        cons = pd.to_datetime(visible["first_consumption_time"], errors="coerce")
        act = pd.to_datetime(visible["it_activation_time"], errors="coerce")
        rows.append({
            "check": "consumption_bar_not_before_it_activation",
            "violations": int((cons.notna() & act.notna() & cons.lt(act)).sum()),
        })
    if not sweep_events.empty:
        rows.append({
            "check": "sweep_members_known_by_root_time",
            "violations": int((pd.to_datetime(sweep_events["max_member_activation_time"], errors="coerce") > pd.to_datetime(sweep_events["sweep_time"], errors="coerce")).sum()),
        })
    if not paths.empty:
        root = pd.to_datetime(paths.get("root_sweep_time"), errors="coerce")
        reclaim = pd.to_datetime(paths.get("reclaim_available_time"), errors="coerce")
        rows.append({
            "check": "reclaim_not_before_root_sweep",
            "violations": int((reclaim.notna() & root.notna() & reclaim.le(root)).sum()),
        })
        if "opposite_target_region_available_time" in paths.columns:
            target_av = pd.to_datetime(paths["opposite_target_region_available_time"], errors="coerce")
            target_violations = int((target_av.notna() & root.notna() & target_av.gt(root)).sum())
        else:
            target_violations = 0
        rows.append({
            "check": "opposite_target_known_at_root_sweep",
            "violations": target_violations,
        })
        for m in (1, 2, 5):
            x = pd.to_datetime(paths.get(f"post_sweep_st_mss_{m}m_available_time"), errors="coerce")
            rows.append({
                "check": f"post_sweep_mss_{m}m_not_before_root_sweep",
                "violations": int((x.notna() & root.notna() & x.le(root)).sum()),
            })
    return pd.DataFrame(rows)
