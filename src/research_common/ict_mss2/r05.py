#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R05 entry timing, structural stop, and runner research helpers.

R05 deliberately separates three questions that should not share one stop:

1. Entry timing: does 1m/2m reclaim enter materially earlier than 5m without
   turning every micro swing into noise?
2. Initial invalidation: episode extreme vs final-sweep/reclaim-local structure.
3. Runner management: once the reversal is working, can a stop migrate only on
   *higher-quality* 2m/5m/15m structure (ITL/LTL) or on an unusually strong
   bullish displacement anchor, while preserving the 3%-5% right tail?

No 1m trailing-stop rule is implemented.  A new structural stop is only usable
from the first 1m bar whose start is at/after the confirming higher-timeframe
bar close.  Stops are monotone: they may move up, never down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import MSS2Config, aggregate_bars, build_execution_pivots, normalize_1m_bars
from src.research_common.ict_mss2.r04 import target_token
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex
from src.research_common.progress import ProgressReporter

EPS = 1e-12


@dataclass(frozen=True)
class R05Config:
    entry_minutes: tuple[int, ...] = (1, 2, 5)
    trail_minutes: tuple[int, ...] = (2, 5, 15)
    fixed_target_returns: tuple[float, ...] = (0.003, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05)
    max_horizon_minutes: int = 20_160  # 14d censor only
    market_roundtrip_cost: float = 0.0011
    stop_buffer_bps: float = 2.0
    shock_lookback_days: int = 7
    shock_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99)

    def validate(self) -> "R05Config":
        if any(int(x) <= 0 for x in self.entry_minutes + self.trail_minutes):
            raise ValueError("entry/trail minutes must be positive")
        if 1 in self.trail_minutes:
            raise ValueError("R05 intentionally forbids 1m trailing stops")
        if not self.fixed_target_returns or any(float(x) <= 0 for x in self.fixed_target_returns):
            raise ValueError("fixed targets must be positive")
        if int(self.max_horizon_minutes) <= 0:
            raise ValueError("max_horizon_minutes must be positive")
        if float(self.stop_buffer_bps) < 0:
            raise ValueError("stop_buffer_bps cannot be negative")
        if int(self.shock_lookback_days) <= 0:
            raise ValueError("shock_lookback_days must be positive")
        if any(not 0 < float(q) < 1 for q in self.shock_quantiles):
            raise ValueError("shock quantiles must be inside (0,1)")
        return self


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x.loc[x > 0].sum())
    gl = float(-x.loc[x < 0].sum())
    if gl <= EPS:
        return np.inf if gp > 0 else np.nan
    return gp / gl


def _hierarchy_extreme(center: float, left: float, right: float, side: str) -> bool:
    if not all(np.isfinite(v) for v in (center, left, right)):
        return False
    return bool(center < left and center < right) if side == "low" else bool(center > left and center > right)



EXCLUSIVE_OPPORTUNITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("under_0p3", -np.inf, 0.003),
    ("short_0p3_1p0", 0.003, 0.01),
    ("medium_1p0_3p0", 0.01, 0.03),
    ("swing_3p0_5p0", 0.03, 0.05),
    ("major_ge_5p0", 0.05, np.inf),
)


def build_exclusive_opportunity_buckets(
    opportunities: pd.DataFrame,
    primary_1m: pd.DataFrame,
    *,
    config: R05Config | None = None,
) -> pd.DataFrame:
    """Assign mutually-exclusive *future* opportunity buckets.

    Buckets are based on the maximum favorable excursion attainable *before*
    the frozen episode-extreme thesis stop (or the 14d research censor).  The
    stop-touch bar is excluded from MFE so a same-bar stop/target ambiguity is
    handled pessimistically.  These labels are descriptive only and must never
    be merged back into the causal entry feature table.

    This complements, rather than replaces, the nested target atlas.  The nested
    atlas answers upgrade probabilities (0.5 -> 1 -> 3 -> 5%), while these
    exclusive buckets let short/medium/swing/major paths be studied without a
    5% long-tail winner also contaminating the short-rebound cohort.
    """
    cfg = (config or R05Config()).validate()
    if opportunities.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    low_idx = SegmentThresholdIndex(low)
    n = len(bars)
    rows: list[dict[str, object]] = []
    for source in opportunities.itertuples(index=False):
        entry = float(source.entry_price)
        entry_pos = int(source.entry_pos_1m)
        stop = float(getattr(source, "stop_episode_extreme", np.nan))
        if not 0 <= entry_pos < n or not np.isfinite(entry) or entry <= 0 or not np.isfinite(stop) or not 0 < stop < entry:
            continue
        end = min(n - 1, entry_pos + int(cfg.max_horizon_minutes))
        full = bool(entry_pos + int(cfg.max_horizon_minutes) <= n - 1)
        stop_pos = int(low_idx.first_leq(entry_pos, end, stop))
        # Conservative stop-first semantics: do not count a high from the stop
        # touch bar as attainable MFE.
        path_end = (stop_pos - 1) if stop_pos >= 0 else end
        if path_end < entry_pos:
            max_high = entry
            peak_pos = entry_pos
        else:
            segment = high[entry_pos:path_end + 1]
            finite = np.isfinite(segment)
            if not finite.any():
                max_high = entry
                peak_pos = entry_pos
            else:
                safe = np.where(finite, segment, -np.inf)
                rel = int(np.argmax(safe))
                max_high = float(safe[rel])
                peak_pos = entry_pos + rel
        mfe = max(0.0, max_high / entry - 1.0) if np.isfinite(max_high) else np.nan
        resolved = bool(stop_pos >= 0 or full)
        if not resolved:
            bucket = "right_edge_incomplete"
        elif mfe < 0.003 - EPS:
            bucket = "under_0p3"
        elif mfe < 0.01 - EPS:
            bucket = "short_0p3_1p0"
        elif mfe < 0.03 - EPS:
            bucket = "medium_1p0_3p0"
        elif mfe < 0.05 - EPS:
            bucket = "swing_3p0_5p0"
        else:
            bucket = "major_ge_5p0"
        rows.append({
            "quality_rule": source.quality_rule,
            "episode_id": source.episode_id,
            "stage_id": getattr(source, "stage_id", ""),
            "trade_event_id": source.trade_event_id,
            "execution_minutes": int(source.execution_minutes),
            "entry_time": source.entry_time,
            "entry_price": entry,
            "thesis_stop_price": stop,
            "thesis_risk_return": (entry - stop) / entry,
            "opportunity_bucket": bucket,
            "bucket_resolved_flag": int(resolved),
            "thesis_stop_hit_flag": int(stop_pos >= 0),
            "thesis_stop_pos_1m": stop_pos,
            "max_favorable_return_before_thesis_stop": mfe,
            "minutes_to_peak_mfe": int(peak_pos - entry_pos) if np.isfinite(mfe) else np.nan,
            "right_edge_incomplete_flag": int(not resolved),
        })
    return pd.DataFrame(rows)


def _bucket_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    keys = [c for c in ("quality_rule", "trade_event_id", "execution_minutes") if c in left.columns and c in right.columns]
    if len(keys) < 3:
        raise ValueError("exclusive bucket join requires quality_rule, trade_event_id, execution_minutes")
    return keys


def attach_exclusive_bucket(rows: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    """Attach future bucket labels to a reporting table without many-to-many expansion."""
    if rows.empty or buckets.empty:
        return pd.DataFrame()
    keys = _bucket_join_keys(rows, buckets)
    bcols = keys + ["opportunity_bucket", "bucket_resolved_flag"]
    b = buckets.loc[:, bcols].drop_duplicates(keys, keep="first")
    if b.duplicated(keys).any():
        raise ValueError("duplicate exclusive bucket keys")
    out = rows.merge(b, on=keys, how="left", validate="many_to_one")
    return out


def summarize_exclusive_opportunity_buckets(buckets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if buckets.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = buckets.loc[buckets["bucket_resolved_flag"].eq(1)].copy()
    f["year"] = pd.to_datetime(f["entry_time"], errors="coerce").dt.year
    keys = ["quality_rule", "execution_minutes", "opportunity_bucket"]

    def _summ(part: pd.DataFrame) -> dict[str, object]:
        return {
            "opportunities": int(len(part)),
            "episodes": int(part["episode_id"].nunique()),
            "median_mfe_pct": float(pd.to_numeric(part["max_favorable_return_before_thesis_stop"], errors="coerce").median() * 100.0),
            "median_minutes_to_peak_mfe": float(pd.to_numeric(part["minutes_to_peak_mfe"], errors="coerce").median()),
            "median_thesis_risk_pct": float(pd.to_numeric(part["thesis_risk_return"], errors="coerce").median() * 100.0),
            "thesis_stop_hit_rate": float(pd.to_numeric(part["thesis_stop_hit_flag"], errors="coerce").mean()),
        }

    overall = [dict(zip(keys, key)) | _summ(part) for key, part in f.groupby(keys, dropna=False, sort=True)]
    ykeys = keys + ["year"]
    yearly = [dict(zip(ykeys, key)) | _summ(part) for key, part in f.groupby(ykeys, dropna=False, sort=True)]
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def summarize_initial_stop_by_bucket(outcomes: pd.DataFrame, buckets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = attach_exclusive_bucket(outcomes, buckets)
    if f.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = f.loc[f["bucket_resolved_flag"].eq(1)].copy()
    f["year"] = pd.to_datetime(f["entry_time"], errors="coerce").dt.year
    keys = ["quality_rule", "execution_minutes", "opportunity_bucket", "stop_variant", "target_return"]

    def _summ(part: pd.DataFrame) -> dict[str, object]:
        x = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        return {
            "trades": int(len(part)),
            "resolved": int(x.notna().sum()),
            "target_first_rate": float(pd.to_numeric(part["target_first_flag"], errors="coerce").mean()),
            "median_risk_pct": float(pd.to_numeric(part["risk_return"], errors="coerce").median() * 100.0),
            "cost2x_pf": _profit_factor(x.dropna()),
            "cost2x_expectancy": float(x.mean()) if x.notna().any() else np.nan,
        }

    overall = [dict(zip(keys, key)) | _summ(part) for key, part in f.groupby(keys, dropna=False, sort=True)]
    ykeys = keys + ["year"]
    yearly = [dict(zip(ykeys, key)) | _summ(part) for key, part in f.groupby(ykeys, dropna=False, sort=True)]
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def summarize_mae_by_bucket(mae: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    f = attach_exclusive_bucket(mae, buckets)
    if f.empty:
        return pd.DataFrame()
    f = f.loc[f["bucket_resolved_flag"].eq(1)].copy()
    rows = []
    keys = ["quality_rule", "execution_minutes", "opportunity_bucket", "stop_variant", "target_return"]
    for key, part in f.groupby(keys, dropna=False, sort=True):
        through = -pd.to_numeric(part["mae_through_target_bar"], errors="coerce") * 100.0
        before = -pd.to_numeric(part["mae_before_target_bar"], errors="coerce") * 100.0
        rows.append(dict(zip(keys, key)) | {
            "winner_samples": int(through.notna().sum()),
            "median_mae_through_target_pct": float(through.median()) if through.notna().any() else np.nan,
            "p75_mae_through_target_pct": float(through.quantile(0.75)) if through.notna().any() else np.nan,
            "p90_mae_through_target_pct": float(through.quantile(0.90)) if through.notna().any() else np.nan,
            "median_mae_before_target_bar_pct": float(before.median()) if before.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_trailing_by_bucket(trades: pd.DataFrame, buckets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = attach_exclusive_bucket(trades, buckets)
    if f.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = f.loc[f["bucket_resolved_flag"].eq(1)].copy()
    f["year"] = pd.to_datetime(f["entry_time"], errors="coerce").dt.year
    keys = ["quality_rule", "execution_minutes", "opportunity_bucket", "trailing_strategy", "initial_stop_variant"]

    def _summ(part: pd.DataFrame) -> dict[str, object]:
        x = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        return {
            "opportunities": int(len(part)),
            "resolved": int(x.notna().sum()),
            "median_updates": float(pd.to_numeric(part["trail_updates"], errors="coerce").median()),
            "median_holding_minutes": float(pd.to_numeric(part["holding_minutes"], errors="coerce").median()),
            "median_mfe_pct": float(pd.to_numeric(part["mfe_until_exit_or_censor"], errors="coerce").median() * 100.0),
            "median_capture_ratio": float(pd.to_numeric(part["capture_ratio_to_mfe"], errors="coerce").median()),
            "reached_3pct_rate": float(pd.to_numeric(part["reached_3pct_before_exit_flag"], errors="coerce").mean()),
            "reached_5pct_rate": float(pd.to_numeric(part["reached_5pct_before_exit_flag"], errors="coerce").mean()),
            "resolved_cost2x_pf": _profit_factor(x.dropna()),
            "resolved_cost2x_expectancy": float(x.mean()) if x.notna().any() else np.nan,
        }

    overall = [dict(zip(keys, key)) | _summ(part) for key, part in f.groupby(keys, dropna=False, sort=True)]
    ykeys = keys + ["year"]
    yearly = [dict(zip(ykeys, key)) | _summ(part) for key, part in f.groupby(ykeys, dropna=False, sort=True)]
    return pd.DataFrame(overall), pd.DataFrame(yearly)

def build_execution_swing_hierarchy(primary_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Build causal ST/IT/LT hierarchy on an execution timeframe.

    This is the same recursive swing-on-swing semantics used by R03.3, but it is
    applied directly to 2m/5m/15m bars for trailing-stop research.  The eventual
    IT/LT class is not backfilled: each row carries the first causal time at
    which that class is knowable.
    """
    bars = aggregate_bars(primary_1m, int(minutes))
    piv = build_execution_pivots(bars, int(minutes), MSS2Config())
    if piv.empty:
        return pd.DataFrame()
    p = piv.loc[:, ["pivot_side", "pivot_pos_htf", "pivot_time", "level_price", "initial_available_time"]].copy()
    p["pivot_time"] = pd.to_datetime(p["pivot_time"], errors="coerce")
    p["initial_available_time"] = pd.to_datetime(p["initial_available_time"], errors="coerce")
    p["ict_st_available_time"] = p["initial_available_time"]
    p["ict_it_available_time"] = pd.NaT
    p["ict_lt_available_time"] = pd.NaT

    for side, idx in p.groupby("pivot_side", sort=False).groups.items():
        part = p.loc[list(idx)].sort_values(["pivot_time", "pivot_pos_htf"], kind="stable")
        ids = part.index.to_numpy()
        price = pd.to_numeric(part["level_price"], errors="coerce").to_numpy(dtype=float)
        st_avail = pd.to_datetime(part["ict_st_available_time"], errors="coerce").tolist()
        it_members: list[int] = []
        it_avail: dict[int, pd.Timestamp] = {}
        for j in range(1, len(part) - 1):
            if not _hierarchy_extreme(price[j], price[j - 1], price[j + 1], str(side)):
                continue
            av = max(pd.Timestamp(st_avail[j]), pd.Timestamp(st_avail[j + 1]))
            center_idx = int(ids[j])
            p.at[center_idx, "ict_it_available_time"] = av
            it_members.append(j)
            it_avail[j] = av
        for k in range(1, len(it_members) - 1):
            l, c, r = it_members[k - 1], it_members[k], it_members[k + 1]
            if not _hierarchy_extreme(price[c], price[l], price[r], str(side)):
                continue
            av = max(it_avail[c], it_avail[r])
            p.at[int(ids[c]), "ict_lt_available_time"] = av

    p["source_timeframe_min"] = int(minutes)
    p["ict_class_eventual"] = np.select(
        [p["ict_lt_available_time"].notna(), p["ict_it_available_time"].notna()], ["LT", "IT"], default="ST"
    )
    return p.sort_values(["pivot_time", "pivot_side", "level_price"], kind="stable").reset_index(drop=True)


def build_quality_entry_universe(r02_trade_features: pd.DataFrame, hierarchy_stages: pd.DataFrame) -> pd.DataFrame:
    """Return first qualifying Long episode-reclaim trade for frozen quality rules.

    The stage rule is selected once per episode, independently of execution
    timeframe.  1m/2m/5m entries are then attached to that exact same causal
    stage, making timing/risk comparisons apples-to-apples.
    """
    if r02_trade_features.empty or hierarchy_stages.empty:
        return pd.DataFrame()
    t = r02_trade_features.copy()
    t = t.loc[
        pd.to_numeric(t.get("trade_direction"), errors="coerce").eq(1)
        & t.get("trigger_type", pd.Series("", index=t.index)).astype(str).eq("episode_reclaim")
        & pd.to_numeric(t.get("execution_minutes"), errors="coerce").isin([1, 2, 5])
    ].copy()
    if t.empty:
        return pd.DataFrame()
    t["stage_id"] = t["stage_id"].astype(str)
    s = hierarchy_stages.copy()
    s["stage_id"] = s["stage_id"].astype(str)
    s["sweep_bar_time_1m"] = pd.to_datetime(s["sweep_bar_time_1m"], errors="coerce")
    s = s.sort_values(["episode_id", "sweep_pos_1m", "stage_id"], kind="stable")

    n = pd.to_numeric(s.get("ict_price_pools_cum"), errors="coerce").fillna(0)
    h4 = pd.to_numeric(s.get("ict_htf240_pools_cum"), errors="coerce").fillna(0).ge(1)
    lt = pd.to_numeric(s.get("ict_lt_pools_cum"), errors="coerce").fillna(0).ge(1)
    itp = pd.to_numeric(s.get("ict_it_plus_pools_cum"), errors="coerce").fillna(0).ge(1)
    key = pd.to_numeric(s.get("ict_structural_key_pools_cum"), errors="coerce").fillna(0).ge(1)
    masks = {
        "n3_4h_or_lt": n.ge(3) & (h4 | lt),
        "n4_4h_or_lt": n.ge(4) & (h4 | lt),
        "n3_4h": n.ge(3) & h4,
        "n4_4h": n.ge(4) & h4,
        "n3_lt": n.ge(3) & lt,
        "n4_lt": n.ge(4) & lt,
        "n2_it_plus_key": n.ge(2) & itp & key,
    }

    stage_cols = [
        "stage_id", "episode_id", "sweep_pos_1m", "sweep_bar_time_1m", "episode_start_pos_1m",
        "episode_start_time_1m", "sweep_extreme_stage", "episode_extreme_so_far", "episode_elapsed_minutes",
        "ict_price_pools_cum", "ict_it_plus_pools_cum", "ict_lt_pools_cum", "ict_htf240_pools_cum",
        "ict_multi_tf_pools_cum", "ict_structural_key_pools_cum", "ict_strongest_pool_rank_cum",
    ]
    rows: list[pd.DataFrame] = []
    for rule, mask in masks.items():
        q = s.loc[mask].sort_values(["episode_id", "sweep_pos_1m", "stage_id"], kind="stable")
        if q.empty:
            continue
        first = q.drop_duplicates("episode_id", keep="first")[[c for c in stage_cols if c in q.columns]].copy()
        part = t.merge(first, on=["stage_id", "episode_id"], how="inner", validate="many_to_one", suffixes=("", "_stage"))
        if part.empty:
            continue
        part.insert(0, "quality_rule", rule)
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["signal_available_time"] = pd.to_datetime(out["signal_available_time"], errors="coerce")
    return out.sort_values(["quality_rule", "episode_id", "execution_minutes"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class _KnownLowLookup:
    """Compact causal IT/LT-low lookup used by the initial-stop atlas.

    The previous implementation rebuilt/copy-filtered/sorted a hierarchy
    DataFrame for every opportunity *and* every timeframe.  On a multi-year
    R05 run that turns a small as-of lookup into thousands of full DataFrame
    operations.  This index materializes the candidate rows once, then answers
    each query with a binary-search bound plus a short backward scan.

    Rows are sorted by (pivot_time, class_available_time), preserving the exact
    old semantics: if the same pivot has both an IT and a later-confirmed LT
    class, LT wins only after its own causal availability time.
    """

    pivot_ns: np.ndarray
    available_ns: np.ndarray
    level_price: np.ndarray
    class_name: np.ndarray

    @classmethod
    def from_hierarchy(cls, hierarchy: pd.DataFrame, classes: Sequence[str] = ("IT", "LT")) -> "_KnownLowLookup":
        if hierarchy.empty:
            return cls(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float), np.empty(0, dtype=object),
            )
        side = hierarchy.get("pivot_side", pd.Series("", index=hierarchy.index)).astype(str).eq("low")
        h = hierarchy.loc[side, [c for c in ("pivot_time", "level_price", "ict_it_available_time", "ict_lt_available_time") if c in hierarchy.columns]].copy()
        if h.empty or "pivot_time" not in h.columns or "level_price" not in h.columns:
            return cls(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float), np.empty(0, dtype=object),
            )

        # Pandas 3 preserves the source datetime resolution; ``asi8`` may then
        # be microseconds while ``Timestamp.value`` below is nanoseconds.  Keep
        # the binary-search arrays on an explicit ns axis.
        pivot_dt = pd.DatetimeIndex(pd.to_datetime(h["pivot_time"], errors="coerce")).as_unit("ns")
        price = pd.to_numeric(h["level_price"], errors="coerce").to_numpy(dtype=float)
        pivots: list[np.ndarray] = []
        avails: list[np.ndarray] = []
        levels: list[np.ndarray] = []
        names: list[np.ndarray] = []
        for klass, col in (("IT", "ict_it_available_time"), ("LT", "ict_lt_available_time")):
            if klass not in classes or col not in h.columns:
                continue
            avail_dt = pd.DatetimeIndex(pd.to_datetime(h[col], errors="coerce")).as_unit("ns")
            valid = (~pivot_dt.isna()) & (~avail_dt.isna()) & np.isfinite(price)
            if not np.any(valid):
                continue
            pivots.append(pivot_dt.asi8[valid])
            avails.append(avail_dt.asi8[valid])
            levels.append(price[valid])
            names.append(np.full(int(np.sum(valid)), klass, dtype=object))
        if not pivots:
            return cls(
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float), np.empty(0, dtype=object),
            )
        pivot_ns = np.concatenate(pivots).astype(np.int64, copy=False)
        available_ns = np.concatenate(avails).astype(np.int64, copy=False)
        level_price = np.concatenate(levels).astype(float, copy=False)
        class_name = np.concatenate(names)
        order = np.lexsort((available_ns, pivot_ns))
        return cls(pivot_ns[order], available_ns[order], level_price[order], class_name[order])

    def latest(
        self, *, at_time: pd.Timestamp, min_pivot_time: pd.Timestamp
    ) -> tuple[float, str, pd.Timestamp] | None:
        if self.pivot_ns.size == 0:
            return None
        at = pd.Timestamp(at_time)
        minimum = pd.Timestamp(min_pivot_time)
        if pd.isna(at) or pd.isna(minimum):
            return None
        at_ns = int(at.value)
        min_ns = int(minimum.value)
        hi = int(np.searchsorted(self.pivot_ns, at_ns, side="right") - 1)
        lo = int(np.searchsorted(self.pivot_ns, min_ns, side="left"))
        if hi < lo:
            return None
        # Search latest pivot first.  A row whose class is not yet available is
        # skipped; older already-confirmed rows remain valid candidates.
        for j in range(hi, lo - 1, -1):
            if int(self.available_ns[j]) <= at_ns:
                return (
                    float(self.level_price[j]),
                    str(self.class_name[j]),
                    pd.Timestamp(int(self.available_ns[j]), unit="ns"),
                )
        return None


def _latest_known_low(
    hierarchy: pd.DataFrame,
    *,
    at_time: pd.Timestamp,
    min_pivot_time: pd.Timestamp,
    classes: Sequence[str],
) -> tuple[float, str, pd.Timestamp] | None:
    """Compatibility helper; hot paths should prebuild ``_KnownLowLookup``."""
    return _KnownLowLookup.from_hierarchy(hierarchy, classes).latest(
        at_time=at_time, min_pivot_time=min_pivot_time
    )


def attach_initial_structural_stops(
    opportunities: pd.DataFrame,
    primary_1m: pd.DataFrame,
    *,
    hierarchy_by_tf: dict[int, pd.DataFrame] | None = None,
    config: R05Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Attach structural initial-stop candidates without using future bars."""
    cfg = (config or R05Config()).validate()
    if opportunities.empty:
        return opportunities.copy()
    bars = normalize_1m_bars(primary_1m)
    idx = bars.index
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    out = opportunities.copy()
    buf = float(cfg.stop_buffer_bps) / 10_000.0
    stop_cols: dict[str, np.ndarray] = {
        "stop_episode_extreme": np.full(len(out), np.nan),
        "stop_qualifying_stage_extreme": np.full(len(out), np.nan),
        "stop_reclaim_leg_extreme": np.full(len(out), np.nan),
        "stop_signal_bar_extreme": np.full(len(out), np.nan),
        "stop_itl_2m_at_entry": np.full(len(out), np.nan),
        "stop_itl_5m_at_entry": np.full(len(out), np.nan),
        "stop_itl_15m_at_entry": np.full(len(out), np.nan),
    }
    hier = hierarchy_by_tf or {}
    low_lookup = {int(tf): _KnownLowLookup.from_hierarchy(h, ("IT", "LT")) for tf, h in hier.items()}
    reporter = ProgressReporter(
        "[r05-stop-anchors]", total=len(out), every=max(1, len(out) // 100), enabled=show_progress
    )
    for i, row in enumerate(out.itertuples(index=False)):
        reporter.update(i + 1)
        entry = float(row.entry_price)
        entry_pos = int(row.entry_pos_1m)
        if not 0 <= entry_pos < len(bars) or not np.isfinite(entry) or entry <= 0:
            continue
        current_stop = float(getattr(row, "stop_price", np.nan))
        if np.isfinite(current_stop) and current_stop < entry:
            stop_cols["stop_episode_extreme"][i] = current_stop
        stage_ext = float(getattr(row, "sweep_extreme_stage", np.nan))
        if np.isfinite(stage_ext) and stage_ext < entry:
            stop_cols["stop_qualifying_stage_extreme"][i] = stage_ext * (1.0 - buf)
        sweep_pos = int(getattr(row, "sweep_pos_1m", -1))
        if 0 <= sweep_pos < entry_pos:
            extreme = float(np.nanmin(low[sweep_pos:entry_pos]))
            if np.isfinite(extreme) and extreme < entry:
                stop_cols["stop_reclaim_leg_extreme"][i] = extreme * (1.0 - buf)
        signal_start = pd.Timestamp(getattr(row, "signal_bar_time", pd.NaT))
        signal_end = pd.Timestamp(getattr(row, "signal_available_time", pd.NaT))
        if pd.notna(signal_start) and pd.notna(signal_end):
            a = int(idx.searchsorted(signal_start, side="left"))
            b = int(idx.searchsorted(signal_end, side="left"))
            if 0 <= a < b <= len(low):
                extreme = float(np.nanmin(low[a:b]))
                if np.isfinite(extreme) and extreme < entry:
                    stop_cols["stop_signal_bar_extreme"][i] = extreme * (1.0 - buf)
        min_pivot_time = pd.Timestamp(getattr(row, "episode_start_time_1m", signal_start))
        for tf in (2, 5, 15):
            lookup = low_lookup.get(tf)
            found = None if lookup is None else lookup.latest(
                at_time=pd.Timestamp(row.entry_time), min_pivot_time=min_pivot_time
            )
            if found is None:
                continue
            price, _, _ = found
            candidate = price * (1.0 - buf)
            if np.isfinite(candidate) and candidate < entry:
                stop_cols[f"stop_itl_{tf}m_at_entry"][i] = candidate
    entry_values = pd.to_numeric(out["entry_price"], errors="coerce").to_numpy(dtype=float)
    additions: dict[str, np.ndarray] = {}
    for c, values in stop_cols.items():
        additions[c] = values
        additions[c.replace("stop_", "risk_") + "_return"] = (entry_values - values) / entry_values
    # Materialize all stop/risk columns once to avoid DataFrame fragmentation.
    add_df = pd.DataFrame(additions, index=out.index)
    return pd.concat([out, add_df], axis=1)


def build_initial_stop_target_atlas(
    opportunities: pd.DataFrame,
    primary_1m: pd.DataFrame,
    *,
    config: R05Config | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare structural initial stops and compute conservative MAE-before-TP."""
    cfg = (config or R05Config()).validate()
    if opportunities.empty:
        return pd.DataFrame(), pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    high_idx = SegmentThresholdIndex(high)
    low_idx = SegmentThresholdIndex(low)
    range_idx = RangeMinMaxIndex(low)
    n = len(bars)
    stop_columns = [c for c in opportunities.columns if c.startswith("stop_") and not c.endswith("_price")]
    rows: list[dict[str, object]] = []
    mae_rows: list[dict[str, object]] = []
    reporter = ProgressReporter("[r05-initial-stops]", total=len(opportunities), every=max(1, len(opportunities) // 100), enabled=show_progress)
    for loop_i, source in enumerate(opportunities.itertuples(index=False), start=1):
        reporter.update(loop_i)
        entry = float(source.entry_price)
        entry_pos = int(source.entry_pos_1m)
        if not 0 <= entry_pos < n or not np.isfinite(entry) or entry <= 0:
            continue
        end = min(n - 1, entry_pos + int(cfg.max_horizon_minutes))
        full = int(entry_pos + int(cfg.max_horizon_minutes) <= n - 1)

        # Target first-touch locations and MAE-to-target depend on entry/path,
        # not on which structural stop variant is being compared.  Compute them
        # once per opportunity instead of repeating the same segment-tree query
        # for every stop candidate.
        target_cache: dict[float, tuple[int, float, float]] = {}
        for target in cfg.fixed_target_returns:
            target_f = float(target)
            tp = entry * (1.0 + target_f)
            tp_pos = int(high_idx.first_geq(entry_pos, end, tp))
            min_before = np.nan
            min_through = np.nan
            if tp_pos >= 0:
                min_through, _ = range_idx.query(entry_pos, tp_pos)
                min_before, _ = range_idx.query(entry_pos, max(entry_pos, tp_pos - 1))
            target_cache[target_f] = (tp_pos, float(min_before), float(min_through))

        for stop_col in stop_columns:
            stop = float(getattr(source, stop_col, np.nan))
            if not np.isfinite(stop) or not 0 < stop < entry:
                continue
            stop_pos = int(low_idx.first_leq(entry_pos, end, stop))
            risk = (entry - stop) / entry
            for target in cfg.fixed_target_returns:
                target_f = float(target)
                tp_pos, min_before, min_through = target_cache[target_f]
                target_first = tp_pos >= 0 and (stop_pos < 0 or tp_pos < stop_pos)
                stop_first = stop_pos >= 0 and (tp_pos < 0 or stop_pos <= tp_pos)
                gross = target_f if target_first else (stop / entry - 1.0 if stop_first else np.nan)
                net2 = gross - 2.0 * float(cfg.market_roundtrip_cost) if np.isfinite(gross) else np.nan
                rows.append({
                    "quality_rule": source.quality_rule,
                    "episode_id": source.episode_id,
                    "stage_id": source.stage_id,
                    "trade_event_id": source.trade_event_id,
                    "execution_minutes": int(source.execution_minutes),
                    "entry_time": source.entry_time,
                    "stop_variant": stop_col.replace("stop_", ""),
                    "target_return": target_f,
                    "risk_return": risk,
                    "target_first_flag": int(target_first),
                    "stop_first_flag": int(stop_first),
                    "censored_flag": int(not target_first and not stop_first and full),
                    "right_edge_incomplete_flag": int(not target_first and not stop_first and not full),
                    "gross_return": gross,
                    "net_return_cost2x": net2,
                })
                if target_first:
                    mae_rows.append({
                        "quality_rule": source.quality_rule,
                        "episode_id": source.episode_id,
                        "trade_event_id": source.trade_event_id,
                        "execution_minutes": int(source.execution_minutes),
                        "stop_variant": stop_col.replace("stop_", ""),
                        "target_return": target_f,
                        "minutes_to_target": int(tp_pos - entry_pos),
                        "mae_before_target_bar": min_before / entry - 1.0,
                        "mae_through_target_bar": min_through / entry - 1.0,
                    })
    return pd.DataFrame(rows), pd.DataFrame(mae_rows)


def summarize_initial_stop_atlas(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = outcomes.copy()
    f["year"] = pd.to_datetime(f["entry_time"], errors="coerce").dt.year
    def _summ(part: pd.DataFrame) -> dict[str, object]:
        x = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        return {
            "trades": int(len(part)), "resolved": int(x.notna().sum()),
            "target_first_rate": float(pd.to_numeric(part["target_first_flag"], errors="coerce").mean()),
            "median_risk_pct": float(pd.to_numeric(part["risk_return"], errors="coerce").median() * 100.0),
            "cost2x_pf": _profit_factor(x.dropna()),
            "cost2x_expectancy": float(x.mean()) if x.notna().any() else np.nan,
        }
    overall = []
    keys = ["quality_rule", "execution_minutes", "stop_variant", "target_return"]
    for key, part in f.groupby(keys, dropna=False, sort=True):
        overall.append(dict(zip(keys, key)) | _summ(part))
    yearly = []
    ykeys = keys + ["year"]
    for key, part in f.groupby(ykeys, dropna=False, sort=True):
        yearly.append(dict(zip(ykeys, key)) | _summ(part))
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def summarize_mae_before_target(mae: pd.DataFrame) -> pd.DataFrame:
    if mae.empty:
        return pd.DataFrame()
    rows = []
    keys = ["quality_rule", "execution_minutes", "stop_variant", "target_return"]
    for key, part in mae.groupby(keys, dropna=False, sort=True):
        x = -pd.to_numeric(part["mae_through_target_bar"], errors="coerce") * 100.0
        y = -pd.to_numeric(part["mae_before_target_bar"], errors="coerce") * 100.0
        rows.append(dict(zip(keys, key)) | {
            "winner_samples": int(x.notna().sum()),
            "median_mae_through_target_pct": float(x.median()) if x.notna().any() else np.nan,
            "p75_mae_through_target_pct": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            "p90_mae_through_target_pct": float(x.quantile(0.90)) if x.notna().any() else np.nan,
            "median_mae_before_target_bar_pct": float(y.median()) if y.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_trailing_events(primary_1m: pd.DataFrame, *, config: R05Config | None = None) -> pd.DataFrame:
    """Build 2m/5m/15m causal structural and displacement stop anchors."""
    cfg = (config or R05Config()).validate()
    bars1 = normalize_1m_bars(primary_1m)
    parts: list[pd.DataFrame] = []
    for tf in cfg.trail_minutes:
        agg = aggregate_bars(bars1, int(tf))
        h = build_execution_swing_hierarchy(bars1, int(tf))
        if not h.empty:
            lows = h.loc[h["pivot_side"].astype(str).eq("low")].copy()
            for klass, time_col in (("IT", "ict_it_available_time"), ("LT", "ict_lt_available_time")):
                q = lows.loc[lows[time_col].notna()].copy()
                if not q.empty:
                    q = pd.DataFrame({
                        "trail_tf_min": int(tf),
                        "event_type": f"{klass.lower()}l",
                        "activation_time": pd.to_datetime(q[time_col], errors="coerce"),
                        "anchor_time": pd.to_datetime(q["pivot_time"], errors="coerce"),
                        "anchor_price": pd.to_numeric(q["level_price"], errors="coerce"),
                        "bar_body_return": np.nan,
                        "bar_range_return": np.nan,
                        "bullish_fvg_flag": 0,
                        "shock_quantile": np.nan,
                    })
                    parts.append(q)
        if agg.empty:
            continue
        o = pd.to_numeric(agg["open"], errors="coerce")
        c = pd.to_numeric(agg["close"], errors="coerce")
        hi = pd.to_numeric(agg["high"], errors="coerce")
        lo = pd.to_numeric(agg["low"], errors="coerce")
        body_ret = (c - o) / o.replace(0, np.nan)
        range_ret = (hi - lo) / o.replace(0, np.nan)
        lookback = max(20, int(cfg.shock_lookback_days * 24 * 60 / int(tf)))
        minp = max(20, lookback // 4)
        bullish = body_ret.where(body_ret > 0)
        fvg = lo.gt(hi.shift(2)).fillna(False)
        for qv in cfg.shock_quantiles:
            rolling_q = bullish.shift(1).rolling(lookback, min_periods=minp).quantile(float(qv))
            mask = body_ret.gt(0) & body_ret.ge(rolling_q) & rolling_q.notna()
            if not mask.any():
                continue
            parts.append(pd.DataFrame({
                "trail_tf_min": int(tf),
                "event_type": f"bull_shock_q{int(round(float(qv)*100)):02d}",
                "activation_time": pd.to_datetime(agg.loc[mask, "bar_end_time"], errors="coerce").to_numpy(),
                "anchor_time": agg.index[mask].to_numpy(),
                "anchor_price": lo.loc[mask].to_numpy(dtype=float),
                "bar_body_return": body_ret.loc[mask].to_numpy(dtype=float),
                "bar_range_return": range_ret.loc[mask].to_numpy(dtype=float),
                "bullish_fvg_flag": fvg.loc[mask].astype(np.int8).to_numpy(),
                "shock_quantile": float(qv),
            }))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False).dropna(subset=["activation_time", "anchor_price"])
    return out.sort_values(["activation_time", "trail_tf_min", "event_type", "anchor_price"], kind="stable").reset_index(drop=True)


def _strategy_event_mask(events: pd.DataFrame, strategy: str) -> pd.Series:
    et = events["event_type"].astype(str)
    tf = pd.to_numeric(events["trail_tf_min"], errors="coerce")
    if strategy == "itl_2m":
        return tf.eq(2) & et.isin(["itl", "ltl"])
    if strategy == "itl_5m":
        return tf.eq(5) & et.isin(["itl", "ltl"])
    if strategy == "itl_15m":
        return tf.eq(15) & et.isin(["itl", "ltl"])
    if strategy == "ltl_5m":
        return tf.eq(5) & et.eq("ltl")
    if strategy == "ltl_15m":
        return tf.eq(15) & et.eq("ltl")
    if strategy == "shock95_2m":
        return tf.eq(2) & et.eq("bull_shock_q95")
    if strategy == "shock95_5m":
        return tf.eq(5) & et.eq("bull_shock_q95")
    if strategy == "shock95_fvg_5m":
        return tf.eq(5) & et.eq("bull_shock_q95") & pd.to_numeric(events["bullish_fvg_flag"], errors="coerce").eq(1)
    if strategy == "shock99_5m":
        return tf.eq(5) & et.eq("bull_shock_q99")
    if strategy == "shock99_fvg_5m":
        return tf.eq(5) & et.eq("bull_shock_q99") & pd.to_numeric(events["bullish_fvg_flag"], errors="coerce").eq(1)
    if strategy == "shock95_15m":
        return tf.eq(15) & et.eq("bull_shock_q95")
    if strategy == "shock95_fvg_15m":
        return tf.eq(15) & et.eq("bull_shock_q95") & pd.to_numeric(events["bullish_fvg_flag"], errors="coerce").eq(1)
    if strategy == "itl15_or_shock95_5m":
        return (tf.eq(15) & et.isin(["itl", "ltl"])) | (tf.eq(5) & et.eq("bull_shock_q95"))
    if strategy == "itl5_15_or_shock95_5m":
        return (tf.isin([5, 15]) & et.isin(["itl", "ltl"])) | (tf.eq(5) & et.eq("bull_shock_q95"))
    raise ValueError(f"unknown trailing strategy {strategy}")


def simulate_structural_trailing(
    opportunities: pd.DataFrame,
    primary_1m: pd.DataFrame,
    trailing_events: pd.DataFrame,
    *,
    initial_stop_column: str = "stop_episode_extreme",
    strategies: Sequence[str] = (
        "itl_2m", "itl_5m", "itl_15m", "ltl_5m", "ltl_15m",
        "shock95_2m", "shock95_5m", "shock95_fvg_5m", "shock99_5m", "shock99_fvg_5m",
        "shock95_15m", "shock95_fvg_15m", "itl15_or_shock95_5m", "itl5_15_or_shock95_5m",
    ),
    config: R05Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Event-driven monotone structural trailing on naked 1m K.

    A trailing event confirmed at T is first usable on the 1m bar starting T.
    Within the bar that creates the ITL/LTL/shock anchor, the *old* stop remains
    active.  This prevents same-bar hindsight.
    """
    cfg = (config or R05Config()).validate()
    if opportunities.empty or trailing_events.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(primary_1m)
    idx = bars.index
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    hi_range = RangeMinMaxIndex(high)
    buf = float(cfg.stop_buffer_bps) / 10_000.0
    events = trailing_events.copy()
    events["activation_time"] = pd.to_datetime(events["activation_time"], errors="coerce")
    events["activation_pos_1m"] = idx.searchsorted(pd.DatetimeIndex(events["activation_time"]), side="left")
    events = events.loc[pd.to_numeric(events["activation_pos_1m"], errors="coerce").lt(len(idx))].copy()
    # Pre-index each strategy once.  Do not repeatedly filter the full event
    # table inside every trade loop.  Searchsorted then restricts each trade to
    # only the events that can actually affect its 14-day path.
    event_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for strategy in strategies:
        part = events.loc[_strategy_event_mask(events, strategy)].sort_values(
            ["activation_pos_1m", "anchor_price"], ascending=[True, False], kind="stable"
        )
        event_arrays[strategy] = (
            pd.to_numeric(part["activation_pos_1m"], errors="coerce").to_numpy(dtype=np.int64),
            pd.to_numeric(part["anchor_price"], errors="coerce").to_numpy(dtype=float),
            part["event_type"].astype(str).to_numpy(dtype=object),
            pd.to_numeric(part["trail_tf_min"], errors="coerce").to_numpy(dtype=np.int16),
        )

    rows: list[dict[str, object]] = []
    reporter = ProgressReporter("[r05-trailing]", total=len(opportunities), every=max(1, len(opportunities) // 100), enabled=show_progress)

    for loop_i, source in enumerate(opportunities.itertuples(index=False), start=1):
        reporter.update(loop_i)
        entry_pos = int(source.entry_pos_1m)
        entry = float(source.entry_price)
        initial_stop = float(getattr(source, initial_stop_column, np.nan))
        if not 0 <= entry_pos < len(idx) or not np.isfinite(entry) or not np.isfinite(initial_stop) or not 0 < initial_stop < entry:
            continue
        end = min(len(idx) - 1, entry_pos + int(cfg.max_horizon_minutes))
        full = int(entry_pos + int(cfg.max_horizon_minutes) <= len(idx) - 1)
        for strategy in strategies:
            positions, anchors, event_types, event_tfs = event_arrays[strategy]
            left = int(np.searchsorted(positions, entry_pos, side="right"))
            right = int(np.searchsorted(positions, end, side="right"))
            active_stop = initial_stop
            cursor = entry_pos
            exit_pos = -1
            exit_price = np.nan
            updates = 0
            first_update_pos = -1
            last_event_type = "none"
            last_event_tf = -1
            for j in range(left, right):
                pos = int(positions[j])
                if pos <= cursor:
                    continue
                breach = low_index.first_leq(cursor, pos - 1, active_stop)
                if breach >= 0:
                    exit_pos = int(breach)
                    exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else float(active_stop)
                    break
                candidate = float(anchors[j]) * (1.0 - buf)
                if not np.isfinite(candidate) or candidate <= active_stop:
                    cursor = pos
                    continue
                # New stop is active from this 1m bar.  If the market has
                # already opened below it, a stop order exits at that open.
                active_stop = candidate
                updates += 1
                if first_update_pos < 0:
                    first_update_pos = pos
                last_event_type = str(event_types[j])
                last_event_tf = int(event_tfs[j])
                if np.isfinite(open_[pos]) and open_[pos] <= active_stop:
                    exit_pos = pos
                    exit_price = float(open_[pos])
                    break
                cursor = pos
            if exit_pos < 0:
                breach = low_index.first_leq(cursor, end, active_stop)
                if breach >= 0:
                    exit_pos = int(breach)
                    exit_price = float(min(open_[breach], active_stop)) if np.isfinite(open_[breach]) else float(active_stop)
            path_end = exit_pos if exit_pos >= 0 else end
            _, max_high = hi_range.query(entry_pos, path_end) if path_end >= entry_pos else (np.nan, np.nan)
            mfe = max_high / entry - 1.0 if np.isfinite(max_high) else np.nan
            gross = exit_price / entry - 1.0 if exit_pos >= 0 and np.isfinite(exit_price) else np.nan
            net2 = gross - 2.0 * float(cfg.market_roundtrip_cost) if np.isfinite(gross) else np.nan
            capture = gross / mfe if np.isfinite(gross) and np.isfinite(mfe) and mfe > EPS else np.nan
            rows.append({
                "quality_rule": source.quality_rule,
                "episode_id": source.episode_id,
                "trade_event_id": source.trade_event_id,
                "execution_minutes": int(source.execution_minutes),
                "entry_time": source.entry_time,
                "trailing_strategy": strategy,
                "initial_stop_variant": initial_stop_column.replace("stop_", ""),
                "initial_stop_price": initial_stop,
                "initial_risk_return": (entry - initial_stop) / entry,
                "trail_updates": updates,
                "first_update_minutes": int(first_update_pos - entry_pos) if first_update_pos >= 0 else np.nan,
                "last_trail_event_type": last_event_type,
                "last_trail_event_tf": last_event_tf,
                "final_stop_price": active_stop,
                "exit_flag": int(exit_pos >= 0),
                "exit_pos_1m": exit_pos,
                "exit_time": idx[exit_pos] if exit_pos >= 0 else pd.NaT,
                "holding_minutes": int(exit_pos - entry_pos) if exit_pos >= 0 else np.nan,
                "censored_flag": int(exit_pos < 0 and full),
                "right_edge_incomplete_flag": int(exit_pos < 0 and not full),
                "gross_return": gross,
                "net_return_cost2x": net2,
                "mfe_until_exit_or_censor": mfe,
                "capture_ratio_to_mfe": capture,
                "reached_3pct_before_exit_flag": int(np.isfinite(max_high) and max_high >= entry * 1.03),
                "reached_5pct_before_exit_flag": int(np.isfinite(max_high) and max_high >= entry * 1.05),
            })
    return pd.DataFrame(rows)


def summarize_trailing_results(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = trades.copy()
    f["year"] = pd.to_datetime(f["entry_time"], errors="coerce").dt.year
    keys = ["quality_rule", "execution_minutes", "trailing_strategy", "initial_stop_variant"]
    def _summ(part: pd.DataFrame) -> dict[str, object]:
        x = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        return {
            "opportunities": len(part), "resolved": int(x.notna().sum()),
            "censored_rate": float(pd.to_numeric(part["censored_flag"], errors="coerce").mean()),
            "median_updates": float(pd.to_numeric(part["trail_updates"], errors="coerce").median()),
            "median_holding_minutes": float(pd.to_numeric(part["holding_minutes"], errors="coerce").median()),
            "median_mfe_pct": float(pd.to_numeric(part["mfe_until_exit_or_censor"], errors="coerce").median() * 100.0),
            "median_capture_ratio": float(pd.to_numeric(part["capture_ratio_to_mfe"], errors="coerce").median()),
            "reached_3pct_rate": float(pd.to_numeric(part["reached_3pct_before_exit_flag"], errors="coerce").mean()),
            "reached_5pct_rate": float(pd.to_numeric(part["reached_5pct_before_exit_flag"], errors="coerce").mean()),
            "resolved_cost2x_pf": _profit_factor(x.dropna()),
            "resolved_cost2x_expectancy": float(x.mean()) if x.notna().any() else np.nan,
        }
    overall, yearly = [], []
    for key, part in f.groupby(keys, dropna=False, sort=True):
        overall.append(dict(zip(keys, key)) | _summ(part))
    for key, part in f.groupby(keys + ["year"], dropna=False, sort=True):
        yearly.append(dict(zip(keys + ["year"], key)) | _summ(part))
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def build_displacement_anchor_atlas(trailing_events: pd.DataFrame) -> pd.DataFrame:
    """Descriptive shock-bar atlas; no shock threshold is promoted to a rule."""
    if trailing_events.empty:
        return pd.DataFrame()
    e = trailing_events.loc[trailing_events["event_type"].astype(str).str.startswith("bull_shock")].copy()
    if e.empty:
        return pd.DataFrame()
    body = pd.to_numeric(e["bar_body_return"], errors="coerce")
    e["body_abs_bucket"] = pd.cut(
        body,
        bins=[-np.inf, 0.003, 0.005, 0.0075, 0.01, np.inf],
        labels=["<0.3%", "0.3-0.5%", "0.5-0.75%", "0.75-1.0%", ">=1.0%"],
        right=False,
    )
    rows = []
    for key, part in e.groupby(["trail_tf_min", "event_type", "body_abs_bucket"], observed=True, dropna=False, sort=True):
        rows.append({
            "trail_tf_min": int(key[0]), "event_type": str(key[1]), "body_abs_bucket": str(key[2]),
            "events": len(part),
            "median_body_return_pct": float(pd.to_numeric(part["bar_body_return"], errors="coerce").median() * 100.0),
            "median_range_return_pct": float(pd.to_numeric(part["bar_range_return"], errors="coerce").median() * 100.0),
            "bullish_fvg_rate": float(pd.to_numeric(part["bullish_fvg_flag"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def r05_causal_audit(
    opportunities: pd.DataFrame,
    hierarchy_by_tf: dict[int, pd.DataFrame],
    trailing_events: pd.DataFrame,
    trailing_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({
        "check": "no_1m_trailing_timeframe",
        "rows": int(len(trailing_events)),
        "violations": int(pd.to_numeric(trailing_events.get("trail_tf_min"), errors="coerce").eq(1).sum()) if not trailing_events.empty else 0,
    })
    swing_viol = 0
    swing_rows = 0
    for tf, h in hierarchy_by_tf.items():
        if h.empty:
            continue
        for c in ("ict_it_available_time", "ict_lt_available_time"):
            q = h.loc[h[c].notna()].copy()
            swing_rows += len(q)
            swing_viol += int((pd.to_datetime(q[c], errors="coerce") <= pd.to_datetime(q["pivot_time"], errors="coerce")).sum())
    rows.append({"check": "it_lt_available_after_pivot", "rows": swing_rows, "violations": swing_viol})
    if not trailing_events.empty:
        rows.append({
            "check": "trail_event_activation_not_before_anchor",
            "rows": len(trailing_events),
            "violations": int((pd.to_datetime(trailing_events["activation_time"], errors="coerce") <= pd.to_datetime(trailing_events["anchor_time"], errors="coerce")).sum()),
        })
    if trailing_results is not None and not trailing_results.empty:
        rows.append({
            "check": "trailing_stop_monotone_not_below_initial",
            "rows": len(trailing_results),
            "violations": int((pd.to_numeric(trailing_results["final_stop_price"], errors="coerce") + EPS < pd.to_numeric(trailing_results["initial_stop_price"], errors="coerce")).sum()),
        })
    future_cols = [c for c in opportunities.columns if c.startswith(("mfe_", "mae_", "tp_", "post4h_"))]
    rows.append({"check": "future_labels_absent_from_entry_features", "rows": len(opportunities), "violations": len(future_cols)})
    return pd.DataFrame(rows)
