#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R12 completed-trend swing sweep -> opposite-liquidity path atlas.

The study begins only with physical ICT IT/LT swings that became valid after a
completed R08.1 trend context was causally known while the swing was still
unconsumed.  Native and nested-lower-TF contexts are retained as labels; ST
swings and invalid higher-TF projections never enter the universe.

For each physical first sweep, R12 freezes the nearest still-unconsumed
opposite completed-trend liquidity and the nearest deeper same-side liquidity.
The path begins on the next 1m bar and is classified by first passage.  This is
a path study, not an entry/SL/TP strategy.
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
class R12Config:
    region_cluster_bps: float = 10.0
    path_horizon_minutes: int = 30 * 24 * 60
    landmark_max_minutes: int = 6 * 60

    def validate(self) -> "R12Config":
        if self.region_cluster_bps <= 0:
            raise ValueError("region_cluster_bps must be positive")
        if self.path_horizon_minutes <= 0:
            raise ValueError("path_horizon_minutes must be positive")
        if self.landmark_max_minutes <= 0:
            raise ValueError("landmark_max_minutes must be positive")
        return self


def _parse_times(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def prepare_completed_trend_contexts(native: pd.DataFrame, nested: pd.DataFrame) -> pd.DataFrame:
    parts = [x for x in (native, nested) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts, ignore_index=True, sort=False)
    if "projection_scope" in q.columns:
        q = q.loc[~q["projection_scope"].astype(str).eq("invalid_higher_tf_projection")]
    q = q.loc[pd.to_numeric(q["active_at_activation_flag"], errors="coerce").fillna(0).eq(1)].copy()
    q = _parse_times(q, [
        "pivot_time", "own_it_available_time", "own_lt_available_time",
        "liquidity_activation_time", "trend_origin_time", "trend_terminal_time",
        "trend_available_time",
    ])
    q = q.dropna(subset=["swing_id", "liquidity_activation_time", "level_price", "pivot_time"])
    q = q.sort_values(["swing_id", "liquidity_activation_time", "trend_leg_id"], kind="stable")
    q = q.drop_duplicates(["swing_id", "trend_leg_id", "projection_scope"], keep="first")
    return q.reset_index(drop=True)


def _first_consumption_after_activation(bars_1m: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()
    b = normalize_1m_bars(bars_1m)
    high_tree = SegmentThresholdIndex(pd.to_numeric(b["high"], errors="coerce").to_numpy(float))
    low_tree = SegmentThresholdIndex(pd.to_numeric(b["low"], errors="coerce").to_numpy(float))
    starts = b.index.to_numpy(dtype="datetime64[ns]")
    rows: list[dict[str, object]] = []
    for r in levels.itertuples(index=False):
        activation = pd.Timestamp(r.qualification_activation_time)
        start = int(np.searchsorted(starts, np.datetime64(activation), side="left"))
        px = float(r.level_price)
        if start >= len(b):
            pos = -1
        elif str(r.pivot_side) == "high":
            pos = int(high_tree.first_geq(start, len(b) - 1, px))
        else:
            pos = int(low_tree.first_leq(start, len(b) - 1, px))
        d = r._asdict()
        d["liquidity_side"] = "BSL" if str(r.pivot_side) == "high" else "SSL"
        d["first_consumption_pos_1m"] = pos
        d["first_consumption_time"] = pd.Timestamp(b.index[pos]) if pos >= 0 else pd.NaT
        d["first_consumption_available_time"] = pd.Timestamp(b.index[pos] + pd.Timedelta(minutes=1)) if pos >= 0 else pd.NaT
        rows.append(d)
    return pd.DataFrame(rows)


def build_completed_trend_physical_liquidity(bars_1m: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    """One row per physical swing; later trend contexts never inflate the level count."""
    if contexts.empty:
        return pd.DataFrame()
    q = contexts.sort_values(["swing_id", "liquidity_activation_time"], kind="stable")
    rows: list[dict[str, object]] = []
    for swing_id, p in q.groupby("swing_id", sort=False):
        first = p.iloc[0]
        prices = pd.to_numeric(p["level_price"], errors="coerce").dropna().unique()
        sides = p["liquidity_side"].astype(str).dropna().unique()
        if len(prices) != 1 or len(sides) != 1:
            raise ValueError(f"inconsistent physical swing context for {swing_id}")
        own_it = pd.to_datetime(p["own_it_available_time"], errors="coerce")
        own_lt = pd.to_datetime(p["own_lt_available_time"], errors="coerce")
        rows.append({
            "swing_id": str(swing_id),
            "pivot_side": "high" if sides[0] == "BSL" else "low",
            "level_price": float(prices[0]),
            "pivot_time": pd.Timestamp(first["pivot_time"]),
            "swing_source_timeframe": str(first.get("swing_source_timeframe", "")),
            "swing_source_timeframe_min": int(first.get("swing_source_timeframe_min", 0)),
            "own_it_available_time": own_it.min() if own_it.notna().any() else pd.NaT,
            "own_lt_available_time": own_lt.min() if own_lt.notna().any() else pd.NaT,
            "qualification_activation_time": pd.to_datetime(p["liquidity_activation_time"]).min(),
            "first_qualifying_projection_scope": str(first.get("projection_scope", "")),
            "first_qualifying_trend_leg_id": str(first.get("trend_leg_id", "")),
        })
    return _first_consumption_after_activation(bars_1m, pd.DataFrame(rows)).sort_values(
        ["qualification_activation_time", "pivot_time", "swing_id"], kind="stable"
    ).reset_index(drop=True)


class _ContextLookup:
    def __init__(self, contexts: pd.DataFrame):
        self.by_swing: dict[str, pd.DataFrame] = {}
        if contexts.empty:
            return
        for sid, p in contexts.groupby("swing_id", sort=False):
            self.by_swing[str(sid)] = p.sort_values("liquidity_activation_time", kind="stable").reset_index(drop=True)

    def summarize(self, swing_ids: Iterable[str], when: pd.Timestamp) -> dict[str, object]:
        pieces: list[pd.DataFrame] = []
        for sid in swing_ids:
            p = self.by_swing.get(str(sid))
            if p is None:
                continue
            x = p.loc[p["liquidity_activation_time"].le(when)]
            if not x.empty:
                pieces.append(x)
        if not pieces:
            return {
                "known_context_count": 0, "known_trend_leg_count": 0,
                "max_known_trend_tf_min": np.nan, "max_known_trend_move_pct": np.nan,
                "native_context_any": 0, "nested_context_any": 0,
                "trend_ge5_any": 0, "trend_ge7_any": 0,
            }
        q = pd.concat(pieces, ignore_index=True, sort=False)
        return {
            "known_context_count": int(len(q)),
            "known_trend_leg_count": int(q["trend_leg_id"].astype(str).nunique()),
            "max_known_trend_tf_min": float(pd.to_numeric(q["source_timeframe_min"], errors="coerce").max()),
            "max_known_trend_move_pct": float(pd.to_numeric(q["trend_move_pct"], errors="coerce").max()),
            "native_context_any": int(q["projection_scope"].astype(str).eq("native").any()),
            "nested_context_any": int(q["projection_scope"].astype(str).eq("nested_lower_tf").any()),
            "trend_ge5_any": int(pd.to_numeric(q["scale_ge_05pct_flag"], errors="coerce").fillna(0).gt(0).any()),
            "trend_ge7_any": int(pd.to_numeric(q["scale_ge_07pct_flag"], errors="coerce").fillna(0).gt(0).any()),
        }


def _role_at_time(row: pd.Series, when: pd.Timestamp) -> str:
    lt = pd.to_datetime(pd.Series([row.get("own_lt_available_time")]), errors="coerce").iloc[0]
    is_lt = pd.notna(lt) and pd.Timestamp(lt) <= when
    if str(row["liquidity_side"]) == "SSL":
        return "LTL" if is_lt else "ITL"
    return "LTH" if is_lt else "ITH"


def _cluster_indices(price: np.ndarray, idx: list[int], tolerance_bps: float) -> list[list[int]]:
    if not idx:
        return []
    ordered = sorted(idx, key=lambda i: (float(price[i]), i))
    groups: list[list[int]] = []
    cur = [ordered[0]]; prev = float(price[ordered[0]])
    for i in ordered[1:]:
        px = float(price[i]); tol = abs(prev) * float(tolerance_bps) / 10000.0
        if abs(px - prev) <= tol + EPS:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
        prev = px
    groups.append(cur)
    return groups


class _PhysicalLiquidityIndex:
    def __init__(self, physical: pd.DataFrame, side: str):
        q = physical.loc[physical["liquidity_side"].eq(side)].sort_values(["level_price", "swing_id"], kind="stable").reset_index(drop=True)
        self.frame = q; self.side = side
        self.price = pd.to_numeric(q["level_price"], errors="coerce").to_numpy(float)
        self.activation = pd.to_datetime(q["qualification_activation_time"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        self.consumption = pd.to_datetime(q["first_consumption_time"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    def _active(self, i: int, when64: np.datetime64, root64: np.datetime64) -> bool:
        if i < 0 or i >= len(self.price):
            return False
        if np.isnat(self.activation[i]) or self.activation[i] > when64:
            return False
        return np.isnat(self.consumption[i]) or self.consumption[i] > root64

    def nearest_regions(self, *, when: pd.Timestamp, root_time: pd.Timestamp, reference_price: float, tolerance_bps: float, max_regions: int) -> list[pd.DataFrame]:
        """Return only the nearest active price clusters, stopping as soon as enough are found.

        This deliberately avoids materializing/clustering the entire active side
        on every root event.  With thousands of historical swings, the old
        gather-all-then-truncate path was the dominant R12 complexity hotspot.
        """
        if len(self.price) == 0 or max_regions <= 0:
            return []
        when64 = np.datetime64(when); root64 = np.datetime64(root_time)
        groups: list[list[int]] = []
        current: list[int] = []
        prev_active_price = np.nan

        def consume(i: int) -> bool:
            nonlocal current, prev_active_price
            if not self._active(i, when64, root64):
                return False
            px = float(self.price[i])
            if not current:
                current = [i]; prev_active_price = px; return False
            tol = abs(prev_active_price) * float(tolerance_bps) / 10000.0
            if abs(px - prev_active_price) <= tol + EPS:
                current.append(i); prev_active_price = px; return False
            groups.append(current)
            current = [i]; prev_active_price = px
            return len(groups) >= max_regions

        if self.side == "BSL":
            i = int(np.searchsorted(self.price, reference_price, side="right"))
            while i < len(self.price) and len(groups) < max_regions:
                if consume(i):
                    break
                i += 1
        else:
            i = int(np.searchsorted(self.price, reference_price, side="left")) - 1
            while i >= 0 and len(groups) < max_regions:
                if consume(i):
                    break
                i -= 1
        if current and len(groups) < max_regions:
            groups.append(current)
        return [self.frame.iloc[sorted(g)].copy() for g in groups[:max_regions]]


def _region_record(p: pd.DataFrame, *, side: str, when: pd.Timestamp, ctx: _ContextLookup) -> dict[str, object]:
    px = pd.to_numeric(p["level_price"], errors="coerce")
    lo = float(px.min()); hi = float(px.max())
    ids = list(p["swing_id"].astype(str))
    roles = [_role_at_time(r, when) for _, r in p.iterrows()]
    piv = pd.to_datetime(p["pivot_time"], errors="coerce")
    return {
        "liquidity_side": side, "zone_low": lo, "zone_high": hi,
        "touch_price": lo if side == "BSL" else hi,
        "full_sweep_price": hi if side == "BSL" else lo,
        "member_count": int(len(set(ids))),
        "lt_member_count": int(sum(x.startswith("LT") for x in roles)),
        "max_swing_tf_min": int(pd.to_numeric(p["swing_source_timeframe_min"], errors="coerce").max()),
        "min_swing_tf_min": int(pd.to_numeric(p["swing_source_timeframe_min"], errors="coerce").min()),
        "oldest_age_days": float(max(0.0, (when - piv.min()).total_seconds()/86400.0)),
        "newest_age_days": float(max(0.0, (when - piv.max()).total_seconds()/86400.0)),
        "swing_ids": "|".join(sorted(set(ids))), "roles": "|".join(sorted(set(roles))),
        **ctx.summarize(ids, when),
    }


def build_root_sweep_events(bars_1m: pd.DataFrame, physical: pd.DataFrame, contexts: pd.DataFrame, *, research_start: pd.Timestamp, research_end: pd.Timestamp, config: R12Config | None = None) -> pd.DataFrame:
    cfg = (config or R12Config()).validate()
    if physical.empty:
        return pd.DataFrame()
    b = normalize_1m_bars(bars_1m)
    q = physical.copy(); q["first_consumption_time"] = pd.to_datetime(q["first_consumption_time"], errors="coerce")
    q = q.loc[q["first_consumption_time"].notna() & q["first_consumption_time"].between(research_start, research_end, inclusive="both")]
    ctx = _ContextLookup(contexts); rows = []
    for (when, side), p in q.groupby(["first_consumption_time", "liquidity_side"], sort=True):
        when = pd.Timestamp(when); available = when + pd.Timedelta(minutes=1)
        pos = int(b.index.searchsorted(when, side="left"))
        if pos >= len(b) or b.index[pos] != when:
            continue
        pp = p.reset_index(drop=True); prices = pd.to_numeric(pp["level_price"], errors="coerce").to_numpy(float)
        groups = _cluster_indices(prices, list(range(len(pp))), cfg.region_cluster_bps)
        regs = [_region_record(pp.iloc[g], side=str(side), when=available, ctx=ctx) for g in groups]
        lo = min(x["zone_low"] for x in regs); hi = max(x["zone_high"] for x in regs)
        bar = b.iloc[pos]; rng = float(bar["high"] - bar["low"])
        if side == "SSL":
            depth = max(0.0, (lo - float(bar["low"])) / max(abs(lo), EPS) * 10000.0)
            wick = max(0.0, min(float(bar["open"]), float(bar["close"])) - float(bar["low"]))
            close_loc = (float(bar["close"]) - float(bar["low"])) / rng if rng > EPS else np.nan
            reclaim = int(float(bar["close"]) > hi)
        else:
            depth = max(0.0, (float(bar["high"]) - hi) / max(abs(hi), EPS) * 10000.0)
            wick = max(0.0, float(bar["high"]) - max(float(bar["open"]), float(bar["close"])))
            close_loc = (float(bar["high"]) - float(bar["close"])) / rng if rng > EPS else np.nan
            reclaim = int(float(bar["close"]) < lo)
        own_lt = pd.to_datetime(pp["own_lt_available_time"], errors="coerce")
        piv = pd.to_datetime(pp["pivot_time"], errors="coerce")
        c = ctx.summarize(pp["swing_id"].astype(str), available)
        row = {
            "root_event_id": f"R12_{when:%Y%m%d%H%M}_{side}", "root_sweep_time": when,
            "root_sweep_available_time": available, "report_date": when.date().isoformat(), "root_side": str(side),
            "root_zone_low": float(lo), "root_zone_high": float(hi), "root_region_count": len(regs),
            "root_level_count": int(pp["swing_id"].nunique()), "root_swing_ids": "|".join(sorted(pp["swing_id"].astype(str).unique())),
            "root_max_swing_tf_min": int(pd.to_numeric(pp["swing_source_timeframe_min"], errors="coerce").max()),
            "root_min_swing_tf_min": int(pd.to_numeric(pp["swing_source_timeframe_min"], errors="coerce").min()),
            "root_lt_count": int((own_lt.notna() & own_lt.le(available)).sum()),
            "root_oldest_age_days": float(max(0.0, (available-piv.min()).total_seconds()/86400.0)),
            "root_newest_age_days": float(max(0.0, (available-piv.max()).total_seconds()/86400.0)),
            "root_bar_open": float(bar["open"]), "root_bar_high": float(bar["high"]), "root_bar_low": float(bar["low"]), "root_bar_close": float(bar["close"]),
            "root_bar_range_pct": rng/max(abs(float(bar["open"])), EPS), "root_sweep_depth_bps": depth,
            "root_rejection_wick_share": wick/rng if rng > EPS else np.nan, "root_reversal_close_location": close_loc,
            "root_same_bar_full_reclaim_flag": reclaim, **{f"root_{k}": v for k,v in c.items()},
        }
        prev = pos-1
        for mins in (5,15,60):
            j = max(0, prev-mins)
            row[f"pre_sweep_ret_{mins}m"] = (float(b.iloc[prev]["close"])/float(b.iloc[j]["close"])-1.0) if prev>=0 and j<prev and abs(float(b.iloc[j]["close"]))>EPS else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["root_sweep_time","root_side"], kind="stable").reset_index(drop=True)
    if not out.empty:
        out["same_bar_two_sided_root_flag"] = out.groupby("root_sweep_time")["root_side"].transform("nunique").gt(1).astype(int)
    return out


def _first_touch(high_tree, low_tree, side, price, start, end):
    if start > end: return -1
    return int(high_tree.first_geq(start,end,float(price))) if side=="BSL" else int(low_tree.first_leq(start,end,float(price)))


def _first_reclaim(close, root_pos, side, zone_low, zone_high, end):
    seg = close[root_pos+1:end+1]
    hit = np.flatnonzero(seg > zone_high + EPS) if side=="SSL" else np.flatnonzero(seg < zone_low - EPS)
    return int(root_pos+1+hit[0]) if len(hit) else -1


def _first_post_sweep_st_mss(htf, sweep_time, side, minutes, max_minutes):
    if htf.empty: return pd.NaT, np.nan
    delta = pd.Timedelta(minutes=minutes); hi = sweep_time + pd.Timedelta(minutes=max_minutes)
    q = htf.loc[(htf.index >= sweep_time.floor(f"{minutes}min")) & (htf.index <= hi)]
    if len(q)<5: return pd.NaT, np.nan
    if side=="SSL":
        vals=pd.to_numeric(q["high"],errors="coerce").to_numpy(float); close=pd.to_numeric(q["close"],errors="coerce").to_numpy(float)
        piv=np.flatnonzero((vals[1:-1]>vals[:-2])&(vals[1:-1]>vals[2:]))+1; cmp=lambda c,p:c>p
    else:
        vals=pd.to_numeric(q["low"],errors="coerce").to_numpy(float); close=pd.to_numeric(q["close"],errors="coerce").to_numpy(float)
        piv=np.flatnonzero((vals[1:-1]<vals[:-2])&(vals[1:-1]<vals[2:]))+1; cmp=lambda c,p:c<p
    for p in piv:
        if q.index[p] < sweep_time: continue
        av=q.index[p+1]+delta; j0=int(q.index.searchsorted(av,side="left"))
        for j in range(j0,len(q)):
            if cmp(float(close[j]),float(vals[p])): return pd.Timestamp(q.index[j]+delta),float(vals[p])
    return pd.NaT,np.nan


def _first_directional_fvg(b, high, low, root_pos, side, end):
    for i in range(max(2,root_pos+1),end+1):
        if side=="SSL" and low[i] > high[i-2] + EPS: return pd.Timestamp(b.index[i]+pd.Timedelta(minutes=1)),float(high[i-2]),float(low[i])
        if side=="BSL" and high[i] < low[i-2] - EPS: return pd.Timestamp(b.index[i]+pd.Timedelta(minutes=1)),float(high[i]),float(low[i-2])
    return pd.NaT,np.nan,np.nan


def _attach(prefix,row,reg):
    row[f"{prefix}_available_flag"] = int(reg is not None)
    if reg:
        for k,v in reg.items(): row[f"{prefix}_{k}"]=v


def build_opposite_liquidity_paths(bars_1m: pd.DataFrame, physical: pd.DataFrame, contexts: pd.DataFrame, root_events: pd.DataFrame, *, config: R12Config | None=None, progress=True) -> pd.DataFrame:
    cfg=(config or R12Config()).validate()
    if root_events.empty: return pd.DataFrame()
    b=normalize_1m_bars(bars_1m); high=pd.to_numeric(b["high"],errors="coerce").to_numpy(float); low=pd.to_numeric(b["low"],errors="coerce").to_numpy(float); close=pd.to_numeric(b["close"],errors="coerce").to_numpy(float)
    high_tree=SegmentThresholdIndex(high); low_tree=SegmentThresholdIndex(low); ctx=_ContextLookup(contexts)
    index={s:_PhysicalLiquidityIndex(physical,s) for s in ("SSL","BSL")}; htf={m:aggregate_bars(b,m) for m in (1,2,5)}
    rep=ProgressReporter("[r12-paths]",total=len(root_events),every=max(1,len(root_events)//100),enabled=progress); rows=[]
    for n,r in enumerate(root_events.itertuples(index=False),start=1):
        root_time=pd.Timestamp(r.root_sweep_time); av=pd.Timestamp(r.root_sweep_available_time); pos=int(b.index.searchsorted(root_time,side="left"))
        if pos>=len(b) or b.index[pos]!=root_time: rep.update(n); continue
        row=r._asdict(); side=str(r.root_side); opposite="BSL" if side=="SSL" else "SSL"; row["path_direction"]="long" if side=="SSL" else "short"; row["path_start_time"]=b.index[min(pos+1,len(b)-1)]; row["path_horizon_minutes"]=cfg.path_horizon_minutes
        if int(r.same_bar_two_sided_root_flag)==1:
            row["path_outcome"]="same_bar_two_sided_root_ambiguous"; rows.append(row); rep.update(n); continue
        opp_ref=float(r.root_bar_high) if opposite=="BSL" else float(r.root_bar_low)
        opp_frames=index[opposite].nearest_regions(when=av,root_time=root_time,reference_price=opp_ref,tolerance_bps=cfg.region_cluster_bps,max_regions=3)
        opp=[_region_record(x,side=opposite,when=av,ctx=ctx) for x in opp_frames]
        for i,x in enumerate(opp,start=1): _attach(f"opposite_{i}",row,x)
        if not opp: _attach("opposite_1",row,None)
        same_ref=float(r.root_bar_low) if side=="SSL" else float(r.root_bar_high)
        sf=index[side].nearest_regions(when=av,root_time=root_time,reference_price=same_ref,tolerance_bps=cfg.region_cluster_bps,max_regions=1)
        same=_region_record(sf[0],side=side,when=av,ctx=ctx) if sf else None; _attach("deeper_same_side",row,same)
        start=pos+1; end=min(len(b)-1,pos+cfg.path_horizon_minutes)
        opp_touch=_first_touch(high_tree,low_tree,opposite,opp[0]["touch_price"],start,end) if opp else -1
        opp_full=_first_touch(high_tree,low_tree,opposite,opp[0]["full_sweep_price"],start,end) if opp else -1
        same_touch=_first_touch(high_tree,low_tree,side,same["touch_price"],start,end) if same else -1
        row["opposite_1_touch_time"]=b.index[opp_touch] if opp_touch>=0 else pd.NaT; row["opposite_1_full_sweep_time"]=b.index[opp_full] if opp_full>=0 else pd.NaT; row["deeper_same_side_touch_time"]=b.index[same_touch] if same_touch>=0 else pd.NaT
        row["opposite_1_touch_delay_min"]=(b.index[opp_touch]-root_time).total_seconds()/60 if opp_touch>=0 else np.nan; row["deeper_same_side_touch_delay_min"]=(b.index[same_touch]-root_time).total_seconds()/60 if same_touch>=0 else np.nan
        if not opp: outcome="no_visible_opposite_liquidity"
        elif opp_touch>=0 and (same_touch<0 or opp_touch<same_touch): outcome="direct_opposite_delivery"
        elif opp_touch>=0 and same_touch>=0 and opp_touch==same_touch: outcome="same_bar_competing_barriers_ambiguous"
        elif same_touch>=0 and opp_touch>=0 and same_touch<opp_touch: outcome="cascade_then_opposite_delivery"
        elif same_touch>=0: outcome="same_side_continuation_no_opposite_hit"
        else: outcome="censored_no_barrier_hit"
        entry_pos=min(start,len(b)-1); entry=float(b.iloc[entry_pos]["open"]); row["next_open_time"]=b.index[entry_pos]; row["next_open_price"]=entry
        if opp:
            target=float(opp[0]["touch_price"]); dist=(target/entry-1) if side=="SSL" else (entry/target-1); row["opposite_1_distance_pct_from_next_open"]=dist; row["opposite_1_distance_bps_from_next_open"]=dist*10000
            stop_pos=min([x for x in (opp_touch,same_touch,end) if x>=0]); seg=b.iloc[entry_pos:stop_pos+1]
            best=float(seg["high"].max()) if side=="SSL" else float(seg["low"].min()); move=(best/entry-1) if side=="SSL" else (entry/best-1); progress_v=move/dist if dist>EPS else np.nan; row["max_target_progress_before_first_barrier"]=progress_v
            if outcome=="censored_no_barrier_hit" and pd.notna(progress_v):
                if progress_v>=.75: outcome="partial_reversal_ge75_no_barrier"
                elif progress_v>=.50: outcome="partial_reversal_ge50_no_barrier"
                elif progress_v>=.25: outcome="partial_reversal_ge25_no_barrier"
        else:
            row["opposite_1_distance_pct_from_next_open"]=np.nan; row["opposite_1_distance_bps_from_next_open"]=np.nan; row["max_target_progress_before_first_barrier"]=np.nan
        row["path_outcome"]=outcome; row["opposite_delivery_eventual_flag"]=int(opp_touch>=0); row["direct_opposite_delivery_flag"]=int(outcome=="direct_opposite_delivery"); row["same_side_first_flag"]=int(same_touch>=0 and (opp_touch<0 or same_touch<opp_touch))
        first_barrier=min([x for x in (opp_touch,same_touch,end) if x>=0]); seg=b.iloc[entry_pos:first_barrier+1]
        if side=="SSL": row["path_mfe_pct_before_first_barrier"]=float(seg["high"].max())/entry-1; row["path_mae_pct_before_first_barrier"]=float(seg["low"].min())/entry-1
        else: row["path_mfe_pct_before_first_barrier"]=entry/float(seg["low"].min())-1; row["path_mae_pct_before_first_barrier"]=1-float(seg["high"].max())/entry
        lend=min(end,pos+cfg.landmark_max_minutes); reclaim=_first_reclaim(close,pos,side,float(r.root_zone_low),float(r.root_zone_high),lend); row["reclaim_available_time"]=b.index[reclaim]+pd.Timedelta(minutes=1) if reclaim>=0 else pd.NaT; row["reclaim_delay_min"]=(b.index[reclaim]-root_time).total_seconds()/60+1 if reclaim>=0 else np.nan
        for m in (1,2,5):
            t,lvl=_first_post_sweep_st_mss(htf[m],root_time,side,m,cfg.landmark_max_minutes); row[f"post_sweep_st_mss_{m}m_available_time"]=t; row[f"post_sweep_st_mss_{m}m_level"]=lvl; row[f"post_sweep_st_mss_{m}m_delay_min"]=(t-root_time).total_seconds()/60 if pd.notna(t) else np.nan
        ft,fl,fh=_first_directional_fvg(b,high,low,pos,side,lend); row["first_directional_fvg_available_time"]=ft; row["first_directional_fvg_low"]=fl; row["first_directional_fvg_high"]=fh; row["first_directional_fvg_delay_min"]=(ft-root_time).total_seconds()/60 if pd.notna(ft) else np.nan
        for i,x in enumerate(opp[1:],start=2):
            tp=_first_touch(high_tree,low_tree,opposite,x["touch_price"],start,end); row[f"opposite_{i}_touch_time"]=b.index[tp] if tp>=0 else pd.NaT; row[f"opposite_{i}_hit_flag"]=int(tp>=0)
        rows.append(row); rep.update(n)
    rep.close(); return pd.DataFrame(rows).sort_values(["root_sweep_time","root_side"],kind="stable").reset_index(drop=True)


def summarize_path_outcomes(paths):
    if paths.empty:return pd.DataFrame()
    rows=[]
    for (side,outcome),p in paths.groupby(["root_side","path_outcome"],dropna=False,sort=True):
        rows.append({"root_side":side,"path_outcome":outcome,"events":len(p),"share_within_side":len(p)/max(1,int(paths["root_side"].eq(side).sum())),"median_opposite_distance_bps":pd.to_numeric(p.get("opposite_1_distance_bps_from_next_open"),errors="coerce").median(),"median_opposite_touch_delay_min":pd.to_numeric(p.get("opposite_1_touch_delay_min"),errors="coerce").median(),"median_deeper_same_side_delay_min":pd.to_numeric(p.get("deeper_same_side_touch_delay_min"),errors="coerce").median(),"median_path_mfe_pct":pd.to_numeric(p.get("path_mfe_pct_before_first_barrier"),errors="coerce").median(),"median_path_mae_pct":pd.to_numeric(p.get("path_mae_pct_before_first_barrier"),errors="coerce").median()})
    return pd.DataFrame(rows)


def summarize_root_taxonomy(paths):
    if paths.empty:return pd.DataFrame()
    q=paths.loc[paths["root_side"].isin(["SSL","BSL"])].copy(); q["root_age_bucket"]=pd.cut(pd.to_numeric(q["root_oldest_age_days"],errors="coerce"),[-np.inf,1,7,30,np.inf],labels=["lt1d","1_7d","7_30d","ge30d"],right=False).astype(str); q["delivery_success"]=q["path_outcome"].isin(["direct_opposite_delivery","cascade_then_opposite_delivery"]).astype(int); q["direct_success"]=q["path_outcome"].eq("direct_opposite_delivery").astype(int)
    specs=[["root_side","root_max_swing_tf_min"],["root_side","root_lt_count"],["root_side","root_native_context_any","root_nested_context_any"],["root_side","root_max_known_trend_tf_min"],["root_side","root_trend_ge5_any","root_trend_ge7_any"],["root_side","root_level_count"],["root_side","root_age_bucket"]]; rows=[]
    for cols in specs:
        for key,p in q.groupby(cols,dropna=False,sort=True):
            key=key if isinstance(key,tuple) else (key,); row={"grouping":"+".join(cols),**dict(zip(cols,key)),"events":len(p),"delivery_success_rate":p["delivery_success"].mean(),"direct_success_rate":p["direct_success"].mean(),"same_side_continuation_rate":p["path_outcome"].eq("same_side_continuation_no_opposite_hit").mean(),"median_target_distance_bps":pd.to_numeric(p["opposite_1_distance_bps_from_next_open"],errors="coerce").median(),"median_sweep_depth_bps":pd.to_numeric(p["root_sweep_depth_bps"],errors="coerce").median()}; rows.append(row)
    return pd.DataFrame(rows)


def summarize_success_failure_features(paths):
    if paths.empty:return pd.DataFrame()
    q=paths.loc[paths["path_outcome"].isin(["direct_opposite_delivery","cascade_then_opposite_delivery","same_side_continuation_no_opposite_hit"])].copy(); q["success_group"]=np.where(q["path_outcome"].isin(["direct_opposite_delivery","cascade_then_opposite_delivery"]),"opposite_delivery","same_side_failure")
    features=["root_level_count","root_region_count","root_max_swing_tf_min","root_lt_count","root_oldest_age_days","root_sweep_depth_bps","root_bar_range_pct","root_rejection_wick_share","root_reversal_close_location","root_max_known_trend_tf_min","root_max_known_trend_move_pct","pre_sweep_ret_5m","pre_sweep_ret_15m","pre_sweep_ret_60m","opposite_1_distance_bps_from_next_open"]; rows=[]
    for side,p0 in q.groupby("root_side",sort=True):
        gs={k:p for k,p in p0.groupby("success_group",sort=True)}
        for f in features:
            a=pd.to_numeric(gs.get("opposite_delivery",pd.DataFrame()).get(f),errors="coerce").dropna(); b=pd.to_numeric(gs.get("same_side_failure",pd.DataFrame()).get(f),errors="coerce").dropna(); pooled=pd.concat([a,b]); sd=pooled.std(ddof=0) if len(pooled) else np.nan; diff=a.mean()-b.mean() if len(a) and len(b) else np.nan
            rows.append({"root_side":side,"feature":f,"success_n":len(a),"failure_n":len(b),"success_mean":a.mean() if len(a) else np.nan,"failure_mean":b.mean() if len(b) else np.nan,"success_median":a.median() if len(a) else np.nan,"failure_median":b.median() if len(b) else np.nan,"mean_diff":diff,"standardized_mean_diff":diff/sd if pd.notna(sd) and sd>EPS else np.nan})
    return pd.DataFrame(rows)


def summarize_landmark_uplift(paths):
    if paths.empty:return pd.DataFrame()
    q=paths.loc[paths["root_side"].isin(["SSL","BSL"])].copy(); q["delivery_success"]=q["path_outcome"].isin(["direct_opposite_delivery","cascade_then_opposite_delivery"]).astype(int); lm={"reclaim":"reclaim_delay_min","mss_1m":"post_sweep_st_mss_1m_delay_min","mss_2m":"post_sweep_st_mss_2m_delay_min","mss_5m":"post_sweep_st_mss_5m_delay_min","directional_fvg":"first_directional_fvg_delay_min"}; rows=[]
    for side,p in q.groupby("root_side",sort=True):
        base=p["delivery_success"].mean()
        for name,col in lm.items():
            delay=pd.to_numeric(p.get(col),errors="coerce")
            for window in (15,30,60,180,360):
                mask=delay.notna()&delay.le(window); rate=p.loc[mask,"delivery_success"].mean() if mask.any() else np.nan; rows.append({"root_side":side,"landmark":name,"within_minutes":window,"root_events":len(p),"available_count":int(mask.sum()),"availability_rate":mask.mean(),"delivery_success_rate_when_available":rate,"unconditional_delivery_success_rate":base,"success_rate_uplift":rate-base if pd.notna(rate) else np.nan})
    return pd.DataFrame(rows)


def r12_causal_audit(contexts,physical,roots,paths):
    rows=[]
    if not physical.empty:
        a=pd.to_datetime(physical["qualification_activation_time"],errors="coerce"); c=pd.to_datetime(physical["first_consumption_time"],errors="coerce"); rows.append({"check":"physical_consumption_not_before_qualification","violations":int((c.notna()&c.lt(a)).sum())})
    if not roots.empty:
        r=pd.to_datetime(roots["root_sweep_time"],errors="coerce"); av=pd.to_datetime(roots["root_sweep_available_time"],errors="coerce"); rows.append({"check":"root_available_exactly_next_minute","violations":int(av.ne(r+pd.Timedelta(minutes=1)).sum())})
    if not paths.empty:
        rav=pd.to_datetime(paths["root_sweep_available_time"],errors="coerce")
        for col in ("reclaim_available_time","post_sweep_st_mss_1m_available_time","post_sweep_st_mss_2m_available_time","post_sweep_st_mss_5m_available_time","first_directional_fvg_available_time"):
            if col in paths:
                t=pd.to_datetime(paths[col],errors="coerce"); rows.append({"check":f"{col}_not_before_root_available","violations":int((t.notna()&t.lt(rav)).sum())})
    if not contexts.empty: rows.append({"check":"invalid_projection_absent","violations":int(contexts["projection_scope"].astype(str).eq("invalid_higher_tf_projection").sum())})
    return pd.DataFrame(rows)
