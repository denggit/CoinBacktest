#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R07 ICT family expansion helpers for ETH.

R07 stops treating the existing sell-side exhaustion Long family as the only
ICT-style opportunity.  It studies three complementary causal families:

1. Proper BSL reversal: buy-side liquidity must be followed by an actual
   bearish confirmation (reclaim / MSS / FVG retracement).  A sweep alone is
   never a short entry.
2. Liquidity expansion continuation: price closes *through* key liquidity,
   creates directional imbalance, then a resting limit order waits for an FVG
   retracement.  Both bullish and bearish continuations are supported.
3. FVG corridor scalp: after an already-confirmed directional state, a limit
   order waits inside the first directional FVG and targets a causally-existing
   opposite FVG or structural liquidity.  Market chasing is intentionally not
   supported for this small-range family.

All signal bars are left-labelled and become usable only at ``bar_end_time``.
Limit fills begin on the next eligible 1m bar.  Same-bar fill/stop ambiguity is
handled pessimistically by allowing the stop on the fill bar while targets can
only begin on the following bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import FenwickTree, SegmentThresholdIndex

from .core import EPS, _first_fvg_in_range, aggregate_bars, normalize_1m_bars
from .r02 import R02Config, attach_structural_exit_outcomes
from .r03 import r03_globalize_legacy_trade_ids


@dataclass(frozen=True)
class R07Config:
    execution_minutes: tuple[int, ...] = (1, 2, 5)
    acceptance_bars: int = 3
    fvg_after_acceptance_bars: int = 3
    limit_wait_minutes: int = 120
    corridor_limit_wait_minutes: int = 90
    stop_buffer_bps: float = 2.0
    limit_market_roundtrip_cost: float = 0.0008  # 0.03% maker entry + 0.05% taker exit
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    structural_exit_censor_minutes: int = 10_080

    def validate(self) -> "R07Config":
        if not self.execution_minutes or min(self.execution_minutes) <= 0:
            raise ValueError("execution_minutes must be positive")
        if min(self.acceptance_bars, self.fvg_after_acceptance_bars, self.limit_wait_minutes, self.corridor_limit_wait_minutes) <= 0:
            raise ValueError("R07 wait parameters must be positive")
        if self.stop_buffer_bps < 0:
            raise ValueError("stop buffer cannot be negative")
        if not 0 < self.limit_market_roundtrip_cost < 0.02:
            raise ValueError("limit/market roundtrip cost looks invalid")
        if any(x <= 0 for x in self.cost_scales):
            raise ValueError("cost scales must be positive")
        return self


def profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x.loc[x > 0].sum())
    gl = float(-x.loc[x < 0].sum())
    if gl <= EPS:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def _safe_mean(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan


def _safe_median(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(s.median()) if not s.empty else np.nan


def _first_stage_per_rule(stages: pd.DataFrame) -> pd.DataFrame:
    """Build broad causal quality cohorts without deleting weaker key-liquidity stages."""
    if stages.empty:
        return pd.DataFrame()
    s = stages.copy()
    pools = pd.to_numeric(s.get("ict_price_pools_cum"), errors="coerce").fillna(0)
    key = pd.to_numeric(s.get("ict_structural_key_pools_cum"), errors="coerce").fillna(0).ge(1)
    h4lt = (
        pd.to_numeric(s.get("ict_htf240_pools_cum"), errors="coerce").fillna(0).ge(1)
        | pd.to_numeric(s.get("ict_lt_pools_cum"), errors="coerce").fillna(0).ge(1)
    )
    rules = {
        "key_any": key,
        "n2_key": key & pools.ge(2),
        "n3_key": key & pools.ge(3),
        "h4_or_lt": h4lt,
        "n2_h4_or_lt": h4lt & pools.ge(2),
        "n3_h4_or_lt": h4lt & pools.ge(3),
    }
    rows: list[pd.DataFrame] = []
    order_cols = [c for c in ["episode_id", "episode_stage_no", "sweep_pos_1m"] if c in s.columns]
    s = s.sort_values(order_cols, kind="stable")
    for name, mask in rules.items():
        part = s.loc[mask].copy()
        if part.empty:
            continue
        part = part.drop_duplicates(["episode_id"], keep="first")
        part.insert(0, "quality_rule", name)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_reversal_confirmation_atlas(
    r02_features: pd.DataFrame,
    r02_labels: pd.DataFrame,
    hierarchy_stages: pd.DataFrame,
    refreshed_mss: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-score real reversal entries, especially BSL->Short, never sweep-only."""
    if r02_features.empty or r02_labels.empty or hierarchy_stages.empty:
        return pd.DataFrame()
    cohorts = _first_stage_per_rule(hierarchy_stages)
    if cohorts.empty:
        return pd.DataFrame()
    f = r02_features.copy()
    l = r02_labels.copy()
    # Old R02 reports restarted local trade IDs for each execution timeframe.
    # Repair them positionally only after proving feature/label row order matches.
    if len(f) != len(l):
        raise RuntimeError("R07 reversal atlas requires equal R02 feature/label row counts")
    f, l = r03_globalize_legacy_trade_ids(f, l)
    assert l is not None
    keys = ["trade_event_id", "stage_id", "episode_id"]
    for key in keys:
        if key not in f.columns or key not in l.columns:
            raise RuntimeError(f"R07 reversal atlas missing R02 tie-out key: {key}")
        if not np.array_equal(f[key].astype(str).to_numpy(), l[key].astype(str).to_numpy()):
            raise RuntimeError(f"R07 R02 feature/label row order differs on {key}; refusing unsafe join")
    if f["trade_event_id"].duplicated().any():
        raise RuntimeError("R07 R02 global trade IDs are still duplicated")
    keep_l = [c for c in l.columns if c.startswith("target_")]
    merged = pd.concat([f.reset_index(drop=True), l[keep_l].reset_index(drop=True)], axis=1)
    cohort_cols = [
        "quality_rule", "stage_id", "ict_price_pools_cum", "ict_it_plus_pools_cum",
        "ict_lt_pools_cum", "ict_htf240_pools_cum", "ict_multi_tf_pools_cum",
        "ict_structural_key_pools_cum", "ict_strongest_pool_rank_cum",
    ]
    cohort_cols = [c for c in cohort_cols if c in cohorts.columns]
    merged = merged.merge(cohorts[cohort_cols], on="stage_id", how="inner", validate="many_to_many")
    # Independent trade unit: first actual entry for an episode / TF / trigger / quality rule.
    merged = merged.sort_values(["entry_time", "episode_id", "execution_minutes", "trigger_type"], kind="stable")
    merged = merged.drop_duplicates(["quality_rule", "episode_id", "execution_minutes", "trigger_type"], keep="first")
    merged["family"] = np.where(
        pd.to_numeric(merged["trade_direction"], errors="coerce").lt(0),
        "bsl_reversal_short",
        "ssl_reversal_long",
    )
    if refreshed_mss is not None and not refreshed_mss.empty:
        # Refreshed post-sweep-ST MSS is a separate lawful ICT confirmation, not a sweep-only shortcut.
        r = refreshed_mss.copy()
        r = r.loc[r["reference_mode"].astype(str).eq("post_sweep_st")].copy()
        if not r.empty:
            r["family"] = np.where(pd.to_numeric(r["trade_direction"], errors="coerce").lt(0), "bsl_reversal_short", "ssl_reversal_long")
            r["target_htf240_net_return_cost2x"] = pd.to_numeric(r.get("target_htf240_net_return_cost2x"), errors="coerce")
            # Apply the same first-causal-stage quality cohorts so post-sweep MSS
            # can be judged within key-liquidity context instead of as a single
            # undifferentiated future-confirmation bucket.
            r = r.merge(cohorts[cohort_cols], on="stage_id", how="inner", validate="many_to_many")
            r = r.sort_values(["entry_time", "episode_id", "execution_minutes", "trigger_type"], kind="stable")
            r = r.drop_duplicates(["quality_rule", "episode_id", "execution_minutes", "trigger_type"], keep="first")
            r["source_table"] = "r033_refreshed_mss"
        merged["source_table"] = "r02_actual_entry"
        return pd.concat([merged, r], ignore_index=True, sort=False)
    merged["source_table"] = "r02_actual_entry"
    return merged.reset_index(drop=True)


def summarize_reversal_atlas(atlas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if atlas.empty:
        return pd.DataFrame(), pd.DataFrame()
    a = atlas.copy()
    a["year"] = pd.to_datetime(a.get("entry_time"), errors="coerce").dt.year
    rows, yearly = [], []
    group = ["family", "quality_rule", "execution_minutes", "trigger_type"]
    def calc(part: pd.DataFrame) -> dict[str, object]:
        col = "target_htf240_net_return_cost2x"
        if col not in part.columns:
            return {"trades": len(part), "pf_2x_htf240": np.nan, "mean_net_2x_htf240": np.nan, "win_rate_2x_htf240": np.nan}
        x = pd.to_numeric(part[col], errors="coerce").dropna()
        return {
            "trades": len(part), "resolved_htf240": len(x),
            "pf_2x_htf240": profit_factor(x),
            "mean_net_2x_htf240": float(x.mean()) if len(x) else np.nan,
            "win_rate_2x_htf240": float((x > 0).mean()) if len(x) else np.nan,
            "median_holding_min_htf240": _safe_median(part.get("target_htf240_holding_minutes")),
        }
    for key, part in a.groupby(group, dropna=False, sort=True):
        rows.append(dict(zip(group, key)) | calc(part))
    for key, part in a.groupby(group + ["year"], dropna=False, sort=True):
        yearly.append(dict(zip(group + ["year"], key)) | calc(part))
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def summarize_reversal_target_grid(atlas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-form target comparison for actual confirmed reversals.

    This prevents judging BSL reversal only by a distant 4H objective; small
    reversal trades get the same nearest-liquidity / R-target diagnostics.
    """
    if atlas.empty:
        return pd.DataFrame(), pd.DataFrame()
    a = atlas.copy()
    a["year"] = pd.to_datetime(a.get("entry_time"), errors="coerce").dt.year
    targets = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r1p0", "r2p0", "r3p0", "r5p0")
    group = ["family", "quality_rule", "execution_minutes", "trigger_type"]
    rows: list[dict[str, object]] = []
    years: list[dict[str, object]] = []
    def emit(part: pd.DataFrame, key: tuple, *, year: int | None = None) -> None:
        base = dict(zip(group, key))
        if year is not None:
            base["year"] = int(year)
        for target in targets:
            col = f"target_{target}_net_return_cost2x"
            if col not in part.columns:
                continue
            v = pd.to_numeric(part[col], errors="coerce").dropna()
            if v.empty:
                continue
            row = base | {
                "target": target, "resolved": int(len(v)), "pf_2x": profit_factor(v),
                "mean_net_2x": float(v.mean()), "win_rate_2x": float((v > 0).mean()),
            }
            (years if year is not None else rows).append(row)
    for key, part in a.groupby(group, dropna=False, sort=True):
        emit(part, key if isinstance(key, tuple) else (key,))
    for key, part in a.groupby(group + ["year"], dropna=False, sort=True):
        tup = key if isinstance(key, tuple) else (key,)
        emit(part, tup[:-1], year=int(tup[-1]))
    return pd.DataFrame(rows), pd.DataFrame(years)


def summarize_family_target_grid(trades: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare structural-liquidity and fixed-R objectives without target cherry-picking."""
    if trades.empty or outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = outcomes.copy()
    for c in ["family", "quality_rule", "trade_direction", "execution_minutes", "limit_variant", "stop_variant", "ob_overlap_flag", "entry_time"]:
        if c not in x.columns and c in trades.columns:
            x = x.merge(trades[["trade_event_id", c]], on="trade_event_id", how="left", validate="one_to_one")
    x["year"] = pd.to_datetime(x.get("entry_time"), errors="coerce").dt.year
    groups = [c for c in ["family", "quality_rule", "trade_direction", "execution_minutes", "limit_variant", "stop_variant", "ob_overlap_flag"] if c in x.columns]
    targets = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r1p0", "r2p0", "r3p0", "r5p0")
    rows: list[dict[str, object]] = []
    years: list[dict[str, object]] = []
    def collect(part: pd.DataFrame, base: dict[str, object], dest: list[dict[str, object]]) -> None:
        for target in targets:
            for scale, suffix in ((1, "base"), (2, "cost2x"), (3, "cost3x")):
                col = f"target_{target}_net_return_{suffix}"
                if col not in part.columns:
                    continue
                v = pd.to_numeric(part[col], errors="coerce").dropna()
                if v.empty:
                    continue
                dest.append(base | {
                    "target": target, "cost_scale": scale, "resolved": int(len(v)),
                    "pf": profit_factor(v), "mean_net": float(v.mean()), "win_rate": float((v > 0).mean()),
                })
    for key, part in x.groupby(groups, dropna=False, sort=True):
        tup = key if isinstance(key, tuple) else (key,)
        collect(part, dict(zip(groups, tup)), rows)
    for key, part in x.groupby(groups + ["year"], dropna=False, sort=True):
        tup = key if isinstance(key, tuple) else (key,)
        collect(part, dict(zip(groups + ["year"], tup)), years)
    return pd.DataFrame(rows), pd.DataFrame(years)


def build_fvg_lifecycle(
    primary_1m: pd.DataFrame,
    *,
    execution_minutes: Sequence[int] = (1, 2, 5),
    show_progress: bool = False,
) -> pd.DataFrame:
    """Causal FVG zones with full-rebalance lifecycle mapped to the 1m clock.

    FVG candidate detection is vectorized.  Only actual gaps enter the Python
    lifecycle loop, avoiding a multi-million-bar row loop on full-history 1m/2m/5m data.
    """
    bars1 = normalize_1m_bars(primary_1m)
    idx1 = pd.DatetimeIndex(bars1.index)
    low_idx = SegmentThresholdIndex(bars1["low"].to_numpy(dtype=float))
    high_idx = SegmentThresholdIndex(bars1["high"].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    for tf in execution_minutes:
        agg = aggregate_bars(bars1, int(tf))
        hi = agg["high"].to_numpy(dtype=float)
        lo = agg["low"].to_numpy(dtype=float)
        if len(agg) < 3:
            continue
        end = pd.DatetimeIndex(pd.to_datetime(agg["bar_end_time"], errors="coerce"))
        bull = lo[2:] > hi[:-2]
        bear = hi[2:] < lo[:-2]
        candidates = np.flatnonzero(bull | bear).astype(np.int64) + 2
        rep = ProgressReporter(
            f"[r07-fvg-{int(tf)}m]", total=len(candidates),
            every=max(1, len(candidates) // 100) if len(candidates) else 1, enabled=show_progress,
        )
        for n, pos in enumerate(candidates, 1):
            rep.update(n)
            pos = int(pos)
            if bool(lo[pos] > hi[pos - 2]):
                direction = 1
                lower, upper = float(hi[pos - 2]), float(lo[pos])
            else:
                direction = -1
                lower, upper = float(hi[pos]), float(lo[pos - 2])
            created_time = pd.Timestamp(end[pos])
            active_pos = int(idx1.searchsorted(created_time, side="left"))
            if active_pos >= len(bars1):
                continue
            if direction > 0:
                full_pos = low_idx.first_leq(active_pos, len(bars1) - 1, lower)
                target_boundary = upper
            else:
                full_pos = high_idx.first_geq(active_pos, len(bars1) - 1, upper)
                target_boundary = lower
            rows.append({
                "fvg_id": f"FVG_{int(tf)}M_{pos:09d}_{'BULL' if direction > 0 else 'BEAR'}",
                "source_tf_min": int(tf), "direction": int(direction), "fvg_exec_pos": pos,
                "created_time": created_time, "active_pos_1m": active_pos,
                "full_rebalance_pos_1m": int(full_pos), "lower": lower, "upper": upper,
                "proximal": upper if direction > 0 else lower, "ce": (lower + upper) / 2.0,
                "target_boundary": target_boundary,
                "width_bp": abs(upper / lower - 1.0) * 10_000.0 if lower > EPS else np.nan,
            })
        rep.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["active_pos_1m", "source_tf_min", "direction", "lower"], kind="stable"
    ).reset_index(drop=True)


class _DynamicFVGTargetBook:
    """Dynamic nearest active opposite-FVG target boundary."""
    def __init__(self, lifecycle: pd.DataFrame, *, direction: int):
        part = lifecycle.loc[pd.to_numeric(lifecycle["direction"], errors="coerce").eq(int(direction))].copy()
        self.prices = np.sort(pd.to_numeric(part.get("target_boundary"), errors="coerce").dropna().unique())
        self.rank = {float(p): i for i, p in enumerate(self.prices)}
        self.tree = FenwickTree(len(self.prices))
        self.add: dict[int, list[float]] = {}
        self.remove: dict[int, list[float]] = {}
        for r in part.itertuples(index=False):
            p = float(r.target_boundary)
            a = int(r.active_pos_1m)
            self.add.setdefault(a, []).append(p)
            f = int(r.full_rebalance_pos_1m)
            if f >= 0:
                self.remove.setdefault(f + 1, []).append(p)
        self.updates = sorted(set(self.add) | set(self.remove))
        self.ptr = 0
        self.current = -1

    def advance(self, pos: int) -> None:
        if pos < self.current:
            raise ValueError("FVG target book queries must be nondecreasing")
        while self.ptr < len(self.updates) and self.updates[self.ptr] <= pos:
            u = self.updates[self.ptr]
            for p in self.remove.get(u, []):
                self.tree.add(self.rank[p], -1)
            for p in self.add.get(u, []):
                self.tree.add(self.rank[p], +1)
            self.ptr += 1
        self.current = int(pos)

    def nearest(self, price: float, *, above: bool) -> float:
        n = len(self.prices)
        if n == 0:
            return np.nan
        if above:
            left = int(np.searchsorted(self.prices, price, side="right"))
            if left >= n or self.tree.range_sum(left, n) <= 0:
                return np.nan
            lo, hi = left, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.tree.range_sum(left, mid + 1) > 0:
                    hi = mid
                else:
                    lo = mid + 1
            return float(self.prices[lo])
        right = int(np.searchsorted(self.prices, price, side="left"))
        if right <= 0 or self.tree.range_sum(0, right) <= 0:
            return np.nan
        lo, hi = 0, right - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.tree.range_sum(mid, right) > 0:
                lo = mid
            else:
                hi = mid - 1
        return float(self.prices[lo])


def _bar_pos_at_or_after_available(exec_bars: pd.DataFrame, available: pd.Timestamp) -> int:
    end = pd.DatetimeIndex(pd.to_datetime(exec_bars["bar_end_time"], errors="coerce"))
    # A closed execution bar is usable only at its end. Search the first bar
    # whose close is at/after the already-known 1m sweep availability.
    return int(end.searchsorted(pd.Timestamp(available), side="left"))


def build_liquidity_expansion_continuations(
    primary_1m: pd.DataFrame,
    hierarchy_stages: pd.DataFrame,
    r02_stages: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    *,
    config: R07Config | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build bullish/bearish close-through-liquidity continuation limit entries."""
    cfg = (config or R07Config()).validate()
    bars1 = normalize_1m_bars(primary_1m)
    idx1 = pd.DatetimeIndex(bars1.index)
    low1 = bars1["low"].to_numpy(dtype=float)
    high1 = bars1["high"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low1)
    high_index = SegmentThresholdIndex(high1)
    base_stages = _first_stage_per_rule(hierarchy_stages)
    extra_candidates = [
        "sweep_available_time_1m", "min_consumed_level_price_cum",
        "max_consumed_level_price_cum", "episode_extreme_so_far", "liquidity_side",
    ]
    extra_cols = ["stage_id"] + [c for c in extra_candidates if c in r02_stages.columns and c not in base_stages.columns]
    stage_extra = r02_stages[extra_cols].copy()
    stages = base_stages.merge(stage_extra, on="stage_id", how="left", validate="many_to_one")
    if stages.empty:
        return pd.DataFrame(), pd.DataFrame()
    all_rows: list[dict[str, object]] = []
    total = len(stages) * len(cfg.execution_minutes)
    rep = ProgressReporter("[r07-continuation]", total=total, every=max(1, total // 100), enabled=show_progress)
    done = 0
    stop_buffer = cfg.stop_buffer_bps / 10_000.0
    for tf in cfg.execution_minutes:
        eb = aggregate_bars(bars1, int(tf))
        close = eb["close"].to_numpy(dtype=float)
        high = eb["high"].to_numpy(dtype=float)
        low = eb["low"].to_numpy(dtype=float)
        for row in stages.itertuples(index=False):
            done += 1; rep.update(done)
            available = pd.Timestamp(getattr(row, "sweep_available_time_1m"))
            start = _bar_pos_at_or_after_available(eb, available)
            if start < 0 or start >= len(eb):
                continue
            reversal_direction = int(getattr(row, "trade_direction", 0))
            if reversal_direction == 0:
                continue
            direction = -reversal_direction  # continuation through the swept liquidity
            boundary = float(row.max_consumed_level_price_cum) if direction > 0 else float(row.min_consumed_level_price_cum)
            if not np.isfinite(boundary):
                continue
            accept = -1
            for p in range(start, min(len(eb), start + cfg.acceptance_bars)):
                if (direction > 0 and close[p] > boundary) or (direction < 0 and close[p] < boundary):
                    accept = p; break
            if accept < 0:
                continue
            fvg_pos, lower, upper, proximal = _first_fvg_in_range(
                eb, direction, accept, min(len(eb) - 1, accept + cfg.fvg_after_acceptance_bars)
            )
            if fvg_pos < 0:
                continue
            signal_time = pd.Timestamp(eb["bar_end_time"].iloc[fvg_pos])
            order_start = int(idx1.searchsorted(signal_time, side="left"))
            if order_start >= len(bars1):
                continue
            # Structural thesis stop spans the actual expansion episode up to FVG confirmation.
            ep_start = int(getattr(row, "episode_start_pos_1m", max(0, int(getattr(row, "sweep_pos_1m", 0)))))
            signal_last = max(ep_start, order_start - 1)
            if direction > 0:
                extreme = float(np.nanmin(low1[max(0, ep_start):signal_last + 1]))
                structural_stop = extreme * (1.0 - stop_buffer)
                fvg_stop = float(lower) * (1.0 - stop_buffer)
            else:
                extreme = float(np.nanmax(high1[max(0, ep_start):signal_last + 1]))
                structural_stop = extreme * (1.0 + stop_buffer)
                fvg_stop = float(upper) * (1.0 + stop_buffer)
            if not np.isfinite(structural_stop):
                continue
            # Valid ICT-style OB context: an opposite candle immediately preceding the FVG/displacement overlaps the imbalance.
            ob_overlap = 0
            for q in range(max(0, fvg_pos - 3), fvg_pos):
                bullish_candle = float(eb["close"].iloc[q]) > float(eb["open"].iloc[q])
                opposite = (direction > 0 and not bullish_candle) or (direction < 0 and bullish_candle)
                if opposite and high[q] >= lower and low[q] <= upper:
                    ob_overlap = 1; break
            wait_end = min(len(bars1) - 1, order_start + cfg.limit_wait_minutes - 1)
            for limit_name, limit_price in (("proximal", float(proximal)), ("ce", float((lower + upper) / 2.0))):
                fill = low_index.first_leq(order_start, wait_end, limit_price) if direction > 0 else high_index.first_geq(order_start, wait_end, limit_price)
                if fill < 0 or not (low1[fill] <= limit_price <= high1[fill]):
                    continue
                for stop_name, stop in (("episode_structural", structural_stop), ("fvg_invalidation", fvg_stop)):
                    if direction > 0 and stop >= limit_price - EPS:
                        continue
                    if direction < 0 and stop <= limit_price + EPS:
                        continue
                    all_rows.append({
                        "family": "liquidity_expansion_continuation",
                        "quality_rule": row.quality_rule, "stage_id": row.stage_id, "episode_id": row.episode_id,
                        "trade_direction": direction, "execution_minutes": int(tf),
                        "signal_bar_time": eb.index[fvg_pos], "signal_available_time": signal_time,
                        "acceptance_exec_pos": accept, "fvg_exec_pos": fvg_pos, "fvg_lower": lower, "fvg_upper": upper,
                        "fvg_proximal": proximal, "fvg_ce": (lower + upper) / 2.0, "ob_overlap_flag": ob_overlap,
                        "entry_kind": "fvg_limit", "limit_variant": limit_name, "entry_pos_1m": int(fill),
                        "entry_time": bars1.index[fill], "entry_price": limit_price,
                        "stop_variant": stop_name, "stop_price": float(stop), "structural_extreme_pre_entry": extreme,
                        "breakout_boundary": boundary,
                    })
    rep.close()
    trades = pd.DataFrame(all_rows)
    if trades.empty:
        return trades, trades
    trades = trades.sort_values(["entry_pos_1m", "family", "episode_id", "execution_minutes", "limit_variant", "stop_variant"], kind="stable").reset_index(drop=True)
    trades.insert(0, "trade_event_id", [f"R07_CONT_{i+1:09d}" for i in range(len(trades))])
    r02cfg = R02Config(exit_censor_minutes=cfg.structural_exit_censor_minutes, stop_buffer_bps=cfg.stop_buffer_bps)
    outcomes = attach_structural_exit_outcomes(
        bars1, classified_lifecycle, trades,
        config=r02cfg, roundtrip_cost=cfg.limit_market_roundtrip_cost, show_progress=show_progress,
    )
    return trades, outcomes


def build_reversal_fvg_corridor_scalps(
    primary_1m: pd.DataFrame,
    r02_features: pd.DataFrame,
    hierarchy_stages: pd.DataFrame,
    fvg_lifecycle: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    *,
    config: R07Config | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """After a confirmed reclaim state, limit into the first FVG; never chase market."""
    cfg = (config or R07Config()).validate()
    bars1 = normalize_1m_bars(primary_1m)
    idx1 = pd.DatetimeIndex(bars1.index)
    low1 = bars1["low"].to_numpy(dtype=float)
    high1 = bars1["high"].to_numpy(dtype=float)
    close1 = bars1["close"].to_numpy(dtype=float)
    low_idx = SegmentThresholdIndex(low1); high_idx = SegmentThresholdIndex(high1)
    cohorts = _first_stage_per_rule(hierarchy_stages)
    ccols = [c for c in ["quality_rule", "stage_id"] if c in cohorts.columns]
    states = r02_features.loc[r02_features["trigger_type"].astype(str).eq("episode_reclaim")].copy()
    states = states.merge(cohorts[ccols], on="stage_id", how="inner", validate="many_to_many")
    states = states.sort_values(["signal_available_time", "episode_id", "execution_minutes"], kind="stable")
    states = states.drop_duplicates(["quality_rule", "episode_id", "execution_minutes"], keep="first")
    rows: list[dict[str, object]] = []
    exec_cache = {int(tf): aggregate_bars(bars1, int(tf)) for tf in sorted(pd.to_numeric(states["execution_minutes"], errors="coerce").dropna().astype(int).unique())}
    end_cache = {tf: pd.DatetimeIndex(pd.to_datetime(frame["bar_end_time"], errors="coerce")) for tf, frame in exec_cache.items()}
    rep = ProgressReporter("[r07-fvg-corridor]", total=len(states), every=max(1, len(states)//100), enabled=show_progress)
    for i, row in enumerate(states.itertuples(index=False), 1):
        rep.update(i)
        direction = int(row.trade_direction)
        tf = int(row.execution_minutes)
        eb = exec_cache.get(tf)
        if eb is None or eb.empty:
            continue
        end_times = end_cache[tf]
        signal_available = pd.Timestamp(row.signal_available_time)
        signal_exec = int(end_times.searchsorted(signal_available, side="left"))
        if signal_exec >= len(eb):
            continue
        fvg_pos, lower, upper, proximal = _first_fvg_in_range(eb, direction, signal_exec, min(len(eb)-1, signal_exec + 6))
        if fvg_pos < 0:
            continue
        fvg_time = pd.Timestamp(end_times[fvg_pos])
        start = int(idx1.searchsorted(fvg_time, side="left"))
        if start >= len(bars1):
            continue
        reclaim_stop = float(row.stop_price)
        local_stop = float(lower) * (1.0 - cfg.stop_buffer_bps / 10_000.0) if direction > 0 else float(upper) * (1.0 + cfg.stop_buffer_bps / 10_000.0)
        wait_end = min(len(bars1)-1, start + cfg.corridor_limit_wait_minutes - 1)
        for lv, lp in (("proximal", float(proximal)), ("ce", float((lower+upper)/2.0))):
            fill = low_idx.first_leq(start, wait_end, lp) if direction > 0 else high_idx.first_geq(start, wait_end, lp)
            if fill < 0 or not (low1[fill] <= lp <= high1[fill]):
                continue
            for stop_variant, stop in (("reclaim_structural", reclaim_stop), ("fvg_invalidation", local_stop)):
                if (direction > 0 and stop >= lp) or (direction < 0 and stop <= lp):
                    continue
                rows.append({
                    "family": "reversal_fvg_corridor_scalp", "quality_rule": row.quality_rule,
                    "base_trade_event_id": row.trade_event_id, "stage_id": row.stage_id, "episode_id": row.episode_id,
                    "trade_direction": direction, "execution_minutes": tf, "trigger_type": "reclaim_then_fvg_limit",
                    "signal_available_time": fvg_time, "entry_kind": "fvg_limit", "limit_variant": lv,
                    "entry_source": "post_reclaim_first_fvg",
                    "entry_pos_1m": int(fill), "entry_time": bars1.index[fill], "entry_price": lp,
                    "stop_variant": stop_variant, "stop_price": stop, "fvg_lower": lower, "fvg_upper": upper,
                    "fvg_target_price": np.nan,
                })
    rep.close()
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, trades, trades
    # Freeze the opposite-FVG target at FVG signal time, before the limit can fill.
    target_bear = _DynamicFVGTargetBook(fvg_lifecycle, direction=-1)
    target_bull = _DynamicFVGTargetBook(fvg_lifecycle, direction=1)
    target_prices = np.full(len(trades), np.nan, dtype=float)
    prefill_target_touched = np.zeros(len(trades), dtype=np.int8)
    order = np.argsort(pd.to_datetime(trades["signal_available_time"], errors="coerce").to_numpy(dtype="datetime64[ns]"), kind="stable")
    for j in order:
        signal_pos = int(idx1.searchsorted(pd.Timestamp(trades.iloc[j]["signal_available_time"]), side="left"))
        direction = int(trades.iloc[j]["trade_direction"])
        book = target_bear if direction > 0 else target_bull
        book.advance(signal_pos)
        # The small-corridor objective must still be ahead of the market when the
        # resting order is placed, not merely ahead of the future limit price.
        known_price = float(close1[max(0, signal_pos - 1)]) if signal_pos > 0 else float(close1[0])
        target = book.nearest(known_price, above=direction > 0)
        target_prices[j] = target
        if not np.isfinite(target):
            continue
        fill = int(trades.iloc[j]["entry_pos_1m"])
        if fill <= signal_pos:
            continue
        # If the frozen objective was already delivered on a completed bar
        # before the pullback filled, the setup is stale and must be cancelled.
        if direction > 0:
            touched = high_idx.first_geq(signal_pos, fill - 1, float(target))
        else:
            touched = low_idx.first_leq(signal_pos, fill - 1, float(target))
        if touched >= 0:
            prefill_target_touched[j] = 1
    trades["fvg_target_price"] = target_prices
    trades["target_touched_pre_fill_flag"] = prefill_target_touched
    trades = trades.loc[trades["target_touched_pre_fill_flag"].eq(0)].copy()
    trades = trades.sort_values(["entry_pos_1m", "episode_id", "execution_minutes", "entry_source", "limit_variant", "stop_variant"], kind="stable").reset_index(drop=True)
    trades.insert(0, "trade_event_id", [f"R07_CORRIDOR_{i+1:09d}" for i in range(len(trades))])
    outcomes = attach_structural_exit_outcomes(
        bars1, classified_lifecycle, trades,
        config=R02Config(exit_censor_minutes=cfg.structural_exit_censor_minutes, stop_buffer_bps=cfg.stop_buffer_bps),
        roundtrip_cost=cfg.limit_market_roundtrip_cost, show_progress=show_progress,
    )
    # Separate FVG-target competing-risk result.  This is the small-range version the user requested.
    out = trades[["trade_event_id", "fvg_target_price"]].copy()
    gross = np.full(len(trades), np.nan); net1=np.full(len(trades),np.nan); net2=np.full(len(trades),np.nan); net3=np.full(len(trades),np.nan)
    outcome_arr=np.full(len(trades),"no_target",dtype=object); hold=np.full(len(trades),np.nan)
    exit_pos_arr=np.full(len(trades),-1,dtype=np.int64)
    exit_time_arr=np.full(len(trades),np.datetime64("NaT"),dtype="datetime64[ns]")
    for j, r in enumerate(trades.itertuples(index=False)):
        target=float(r.fvg_target_price) if np.isfinite(r.fvg_target_price) else np.nan
        if not np.isfinite(target) or (r.trade_direction>0 and target<=r.entry_price) or (r.trade_direction<0 and target>=r.entry_price):
            continue
        e=int(r.entry_pos_1m); end=min(len(bars1)-1,e+cfg.structural_exit_censor_minutes-1)
        if r.trade_direction>0:
            sp=low_idx.first_leq(e,end,float(r.stop_price)); tp=high_idx.first_geq(e+1,end,target) if e+1<=end else -1
        else:
            sp=high_idx.first_geq(e,end,float(r.stop_price)); tp=low_idx.first_leq(e+1,end,target) if e+1<=end else -1
        if sp<0 and tp<0:
            outcome_arr[j]="censored"; continue
        if sp>=0 and (tp<0 or sp<=tp):
            x=-abs(float(r.stop_price)/float(r.entry_price)-1.0); p=sp; outcome_arr[j]="stop"
        else:
            x=int(r.trade_direction)*(target/float(r.entry_price)-1.0); p=tp; outcome_arr[j]="target"
        gross[j]=x; hold[j]=p-e+1
        exit_pos_arr[j]=int(p); exit_time_arr[j]=bars1.index[int(p)].to_datetime64()
        for arr,scale in ((net1,1.0),(net2,2.0),(net3,3.0)):
            arr[j]=x-cfg.limit_market_roundtrip_cost*scale
    out["fvg_target_outcome"]=outcome_arr
    out["fvg_target_exit_pos"]=exit_pos_arr; out["fvg_target_exit_time"]=exit_time_arr
    out["fvg_target_holding_minutes"]=hold; out["fvg_target_gross_return"]=gross
    out["fvg_target_net_return_base"]=net1; out["fvg_target_net_return_cost2x"]=net2; out["fvg_target_net_return_cost3x"]=net3
    return trades, outcomes, out


def summarize_family_outcomes(trades: pd.DataFrame, outcomes: pd.DataFrame, *, net_col: str = "target_htf240_net_return_cost2x") -> tuple[pd.DataFrame,pd.DataFrame]:
    if trades.empty or outcomes.empty:
        return pd.DataFrame(),pd.DataFrame()
    x=outcomes.copy()
    for c in ["family","quality_rule","trade_direction","execution_minutes","limit_variant","stop_variant","ob_overlap_flag","entry_time"]:
        if c not in x.columns and c in trades.columns:
            x=x.merge(trades[["trade_event_id",c]],on="trade_event_id",how="left",validate="one_to_one")
    x["year"]=pd.to_datetime(x.get("entry_time"),errors="coerce").dt.year
    groups=[c for c in ["family","quality_rule","trade_direction","execution_minutes","limit_variant","stop_variant","ob_overlap_flag"] if c in x.columns]
    def calc(p):
        v=pd.to_numeric(p.get(net_col),errors="coerce").dropna()
        return {"trades":len(p),"resolved":len(v),"pf_2x":profit_factor(v),"mean_net_2x":float(v.mean()) if len(v) else np.nan,"win_rate_2x":float((v>0).mean()) if len(v) else np.nan}
    rows=[]; yrs=[]
    for k,p in x.groupby(groups,dropna=False,sort=True): rows.append(dict(zip(groups,k if isinstance(k,tuple) else (k,)))|calc(p))
    for k,p in x.groupby(groups+["year"],dropna=False,sort=True): yrs.append(dict(zip(groups+["year"],k if isinstance(k,tuple) else (k,)))|calc(p))
    return pd.DataFrame(rows),pd.DataFrame(yrs)


def summarize_fvg_target_scalps(trades: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    if trades.empty or labels.empty:
        return pd.DataFrame(),pd.DataFrame()
    x=trades.merge(labels,on=["trade_event_id","fvg_target_price"],how="left",validate="one_to_one")
    x["year"]=pd.to_datetime(x["entry_time"],errors="coerce").dt.year
    groups=[c for c in ["quality_rule","trade_direction","execution_minutes","entry_source","limit_variant","stop_variant"] if c in x.columns]
    def calc(p):
        v=pd.to_numeric(p["fvg_target_net_return_cost2x"],errors="coerce").dropna()
        target=pd.to_numeric(p["fvg_target_price"],errors="coerce"); entry=pd.to_numeric(p["entry_price"],errors="coerce")
        d=pd.to_numeric(p["trade_direction"],errors="coerce")
        dist=(d*(target/entry-1.0))*100.0
        return {"trades":len(p),"resolved":len(v),"pf_2x":profit_factor(v),"mean_net_2x":float(v.mean()) if len(v) else np.nan,"win_rate_2x":float((v>0).mean()) if len(v) else np.nan,"median_target_distance_pct":_safe_median(dist),"median_holding_min":_safe_median(p["fvg_target_holding_minutes"])}
    rows=[]; yrs=[]
    for k,p in x.groupby(groups,dropna=False,sort=True): rows.append(dict(zip(groups,k))|calc(p))
    for k,p in x.groupby(groups+["year"],dropna=False,sort=True): yrs.append(dict(zip(groups+["year"],k))|calc(p))
    return pd.DataFrame(rows),pd.DataFrame(yrs)


def build_family_complementarity(
    r06_base: pd.DataFrame | None,
    new_trade_tables: Sequence[pd.DataFrame],
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Monthly activity + pairwise overlap; does not pretend unvalidated families are portfolio-ready."""
    rows=[]
    if r06_base is not None and not r06_base.empty:
        q=r06_base.copy(); q["family"]="r06_ssl_reversal_long"; rows.append(q[["family","entry_time"]])
    for t in new_trade_tables:
        if t is None or t.empty or "entry_time" not in t.columns:
            continue
        q=t.copy()
        if "family" not in q.columns: q["family"]="unknown"
        q["entry_time"] = pd.to_datetime(q["entry_time"], errors="coerce")
        q = q.dropna(subset=["entry_time"])
        # Sensitivity variants (quality rule, TF, CE/proximal, stop type) are
        # not independent market opportunities.  For complementarity only,
        # count one earliest opportunity per family/episode where possible.
        dedupe = [c for c in ["family", "episode_id"] if c in q.columns]
        if len(dedupe) >= 2:
            q = q.sort_values("entry_time", kind="stable").drop_duplicates(dedupe, keep="first")
        else:
            q = q.drop_duplicates(["family", "entry_time"], keep="first")
        rows.append(q[["family","entry_time"]])
    if not rows:
        return pd.DataFrame(),pd.DataFrame()
    allx=pd.concat(rows,ignore_index=True); allx["entry_time"]=pd.to_datetime(allx["entry_time"],errors="coerce"); allx=allx.dropna(subset=["entry_time"])
    allx["month"]=allx["entry_time"].dt.to_period("M").astype(str)
    m=allx.groupby(["month","family"],sort=True).size().rename("opportunities").reset_index()
    pivot=m.pivot(index="month",columns="family",values="opportunities").fillna(0).reset_index()
    fams=sorted(allx["family"].unique())
    sets={f:set(pd.to_datetime(allx.loc[allx["family"].eq(f),"entry_time"]).dt.floor("h")) for f in fams}
    ov=[]
    for i,a in enumerate(fams):
        for b in fams[i+1:]:
            inter=len(sets[a]&sets[b]); union=len(sets[a]|sets[b])
            ov.append({"family_a":a,"family_b":b,"same_hour_overlap":inter,"jaccard_same_hour":inter/union if union else np.nan,"a_hours":len(sets[a]),"b_hours":len(sets[b])})
    return pivot,pd.DataFrame(ov)


def r07_causal_audit(
    continuation_trades: pd.DataFrame,
    corridor_trades: pd.DataFrame,
    fvg_lifecycle: pd.DataFrame,
) -> pd.DataFrame:
    rows=[]
    def add(name,mask,total): rows.append({"check":name,"rows":int(total),"violations":int(np.asarray(mask,dtype=bool).sum())})
    if not continuation_trades.empty:
        sig=pd.to_datetime(continuation_trades["signal_available_time"],errors="coerce"); ent=pd.to_datetime(continuation_trades["entry_time"],errors="coerce")
        add("continuation_limit_entry_before_fvg_close",ent<sig,len(continuation_trades))
        add("continuation_market_entry_forbidden",~continuation_trades["entry_kind"].astype(str).eq("fvg_limit"),len(continuation_trades))
    if not corridor_trades.empty:
        sig=pd.to_datetime(corridor_trades["signal_available_time"],errors="coerce"); ent=pd.to_datetime(corridor_trades["entry_time"],errors="coerce")
        add("corridor_limit_entry_before_fvg_close",ent<sig,len(corridor_trades))
        add("corridor_market_entry_forbidden",~corridor_trades["entry_kind"].astype(str).eq("fvg_limit"),len(corridor_trades))
    if not fvg_lifecycle.empty:
        add("fvg_full_rebalance_before_activation",(pd.to_numeric(fvg_lifecycle["full_rebalance_pos_1m"],errors="coerce").ge(0)&(pd.to_numeric(fvg_lifecycle["full_rebalance_pos_1m"],errors="coerce")<pd.to_numeric(fvg_lifecycle["active_pos_1m"],errors="coerce"))),len(fvg_lifecycle))
    return pd.DataFrame(rows)
