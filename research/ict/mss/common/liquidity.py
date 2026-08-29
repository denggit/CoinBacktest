#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal liquidity-quality features for ICT MSS research.

Every feature in this module is available at the *open of the sweep bar*.
The sweep bar's OHLC and all post-sweep outcomes are deliberately excluded from
liquidity quality classification.  Future-looking columns such as
``future_max_eventual_order_label`` are never consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

from .structure import EPS, normalize_bars


@dataclass(frozen=True)
class _RangeExtremaIndex:
    """Compact iterative segment tree for O(log n) interval min/max queries."""

    size: int
    min_tree: np.ndarray
    max_tree: np.ndarray

    @classmethod
    def build(cls, low: np.ndarray, high: np.ndarray) -> "_RangeExtremaIndex":
        n = int(len(low))
        size = 1
        while size < n:
            size <<= 1
        min_tree = np.full(size * 2, np.inf, dtype=float)
        max_tree = np.full(size * 2, -np.inf, dtype=float)
        min_tree[size : size + n] = np.asarray(low, dtype=float)
        max_tree[size : size + n] = np.asarray(high, dtype=float)
        for pos in range(size - 1, 0, -1):
            min_tree[pos] = min(min_tree[pos << 1], min_tree[(pos << 1) | 1])
            max_tree[pos] = max(max_tree[pos << 1], max_tree[(pos << 1) | 1])
        return cls(size=size, min_tree=min_tree, max_tree=max_tree)

    def query(self, left: int, right: int) -> tuple[float, float]:
        """Inclusive [left, right] min/max."""
        if right < left:
            return np.nan, np.nan
        l = int(left) + self.size
        r = int(right) + self.size
        mn = np.inf
        mx = -np.inf
        while l <= r:
            if l & 1:
                mn = min(mn, self.min_tree[l])
                mx = max(mx, self.max_tree[l])
                l += 1
            if not (r & 1):
                mn = min(mn, self.min_tree[r])
                mx = max(mx, self.max_tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return (float(mn) if np.isfinite(mn) else np.nan, float(mx) if np.isfinite(mx) else np.nan)


def _bucket_age(minutes: pd.Series) -> pd.Series:
    x = pd.to_numeric(minutes, errors="coerce")
    return pd.cut(
        x,
        bins=[-np.inf, 60, 360, 1440, 4320, 10080, 43200, np.inf],
        labels=["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "7-30d", "30d+"],
        right=False,
    ).astype("object")


def _bucket_excursion(bp: pd.Series) -> pd.Series:
    x = pd.to_numeric(bp, errors="coerce")
    return pd.cut(
        x,
        bins=[-np.inf, 25, 50, 100, 200, 500, np.inf],
        labels=["<25bp", "25-50bp", "50-100bp", "100-200bp", "200-500bp", "500bp+"],
        right=False,
    ).astype("object")


def enrich_level_sweeps_with_causal_quality(
    primary: pd.DataFrame,
    levels: pd.DataFrame,
    level_sweeps: pd.DataFrame,
    *,
    cluster_tolerances_bp: Iterable[float] = (5.0, 10.0, 25.0),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Attach pre-sweep quality features to each eventually swept HTF level.

    Safe timing:
    - activation distance uses the last fully closed execution bar before the
      level becomes actionable;
    - excursion uses bars from activation through ``sweep_pos - 1`` only;
    - clustering includes only levels already confirmed at sweep-bar open and
      not swept on an earlier bar;
    - no sweep-bar high/low, MSS, FVG, trade or future pivot order is used.
    """

    bars = normalize_bars(primary)
    if level_sweeps.empty:
        return level_sweeps.copy()
    forbidden = [c for c in levels.columns if c.startswith("future_")]
    # The columns may exist in the source table for descriptive audits, but this
    # module must never reference them when constructing causal quality.
    _ = forbidden

    out = level_sweeps.copy().reset_index(drop=True)
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    extrema = _RangeExtremaIndex.build(low, high)

    away = np.full(len(out), np.nan, dtype=float)
    activation_distance = np.full(len(out), np.nan, dtype=float)
    pre_sweep_distance = np.full(len(out), np.nan, dtype=float)
    reporter = ProgressReporter(
        label="[ict-mss] causal liquidity excursion",
        total=len(out),
        every=max(1, len(out) // 200),
        enabled=bool(show_progress),
    )
    for i, row in enumerate(out.itertuples(index=False)):
        active = int(row.active_pos)
        sweep = int(row.sweep_pos)
        price = float(row.level_price)
        side = str(row.liquidity_side)
        if 0 < active <= n:
            prev_close = float(close[active - 1])
            activation_distance[i] = (
                (prev_close / price - 1.0) * 10_000.0
                if side == "sell_side"
                else (1.0 - prev_close / price) * 10_000.0
            )
        if sweep > 0:
            prev_close = float(close[sweep - 1])
            pre_sweep_distance[i] = (
                (prev_close / price - 1.0) * 10_000.0
                if side == "sell_side"
                else (1.0 - prev_close / price) * 10_000.0
            )
        if active <= sweep - 1:
            mn, mx = extrema.query(active, sweep - 1)
            if side == "sell_side" and np.isfinite(mx):
                away[i] = max(0.0, (mx / price - 1.0) * 10_000.0)
            elif side == "buy_side" and np.isfinite(mn):
                away[i] = max(0.0, (1.0 - mn / price) * 10_000.0)
        reporter.update(i + 1)
    reporter.close()
    out["activation_distance_bp"] = activation_distance
    out["pre_sweep_distance_bp"] = pre_sweep_distance
    out["max_excursion_away_bp_before_sweep"] = away
    out["quality_feature_last_pos"] = out["sweep_pos"].astype(np.int64) - 1

    # Build a price-sorted active-level atlas.  Levels that are never swept are
    # assigned +infinity first-sweep position, so they can still contribute to
    # a genuine equal-high/equal-low pool before the current event.
    state = levels.loc[
        :, ["level_id", "liquidity_side", "level_price", "initial_available_time", "source_timeframe_min"]
    ].copy()
    first_sweep = out.groupby("level_id", observed=False)["sweep_pos"].min()
    state["first_sweep_pos"] = state["level_id"].map(first_sweep).fillna(n + 1).astype(np.int64)
    state["initial_available_time"] = pd.to_datetime(state["initial_available_time"], errors="coerce")
    sweep_times = pd.to_datetime(out["sweep_bar_time"], errors="coerce")

    tol_values = tuple(sorted(set(float(v) for v in cluster_tolerances_bp)))
    for tol in tol_values:
        out[f"cluster_count_{tol:g}bp"] = 0
        out[f"cluster_timeframe_count_{tol:g}bp"] = 0

    by_side: dict[str, dict[str, np.ndarray]] = {}
    for side, part in state.groupby("liquidity_side", sort=False, observed=False):
        p = part.sort_values("level_price", kind="mergesort")
        by_side[str(side)] = {
            "price": p["level_price"].to_numpy(dtype=float),
            "available_ns": pd.to_datetime(p["initial_available_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64),
            "first_sweep": p["first_sweep_pos"].to_numpy(dtype=np.int64),
            "timeframe": p["source_timeframe_min"].to_numpy(dtype=np.int16),
        }

    reporter = ProgressReporter(
        label="[ict-mss] active liquidity clustering",
        total=len(out),
        every=max(1, len(out) // 200),
        enabled=bool(show_progress),
    )
    cluster_counts = {tol: np.zeros(len(out), dtype=np.int16) for tol in tol_values}
    cluster_tfs = {tol: np.zeros(len(out), dtype=np.int8) for tol in tol_values}
    sweep_ns = sweep_times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for i, row in enumerate(out.itertuples(index=False)):
        side = str(row.liquidity_side)
        atlas = by_side[side]
        prices = atlas["price"]
        price = float(row.level_price)
        pos = int(row.sweep_pos)
        event_ns = int(sweep_ns[i])
        for tol in tol_values:
            frac = tol / 10_000.0
            lo = int(np.searchsorted(prices, price * (1.0 - frac), side="left"))
            hi = int(np.searchsorted(prices, price * (1.0 + frac), side="right"))
            if hi <= lo:
                continue
            active_mask = (
                (atlas["available_ns"][lo:hi] <= event_ns)
                & (atlas["first_sweep"][lo:hi] >= pos)
            )
            count = int(active_mask.sum())
            cluster_counts[tol][i] = min(count, np.iinfo(np.int16).max)
            if count:
                tfs = atlas["timeframe"][lo:hi][active_mask]
                cluster_tfs[tol][i] = int(len(np.unique(tfs)))
        reporter.update(i + 1)
    reporter.close()
    for tol in tol_values:
        out[f"cluster_count_{tol:g}bp"] = cluster_counts[tol]
        out[f"cluster_timeframe_count_{tol:g}bp"] = cluster_tfs[tol]

    out["level_age_bucket"] = _bucket_age(out["level_age_minutes_at_sweep"])
    out["excursion_away_bucket"] = _bucket_excursion(out["max_excursion_away_bp_before_sweep"])
    out["is_mature_12h"] = out["level_age_minutes_at_sweep"].ge(12 * 60)
    out["is_mature_24h"] = out["level_age_minutes_at_sweep"].ge(24 * 60)
    out["is_mature_72h"] = out["level_age_minutes_at_sweep"].ge(72 * 60)
    out["is_remote_50bp"] = out["max_excursion_away_bp_before_sweep"].ge(50.0)
    out["is_remote_100bp"] = out["max_excursion_away_bp_before_sweep"].ge(100.0)
    out["is_remote_200bp"] = out["max_excursion_away_bp_before_sweep"].ge(200.0)
    out["is_structural_major"] = out["confirmed_order_at_sweep"].ge(3) | out["source_timeframe_min"].ge(60)
    out["is_stacked_10bp"] = out["cluster_count_10bp"].ge(2)
    out["is_stacked_multi_tf_10bp"] = out["cluster_timeframe_count_10bp"].ge(2)
    return out


def aggregate_quality_to_sweep_episodes(
    episodes: pd.DataFrame,
    enriched_level_sweeps: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate level-level quality into the same-bar sweep episode."""

    if episodes.empty or enriched_level_sweeps.empty:
        return episodes.copy()
    grouped_rows: list[dict[str, object]] = []
    for (side, pos), part in enriched_level_sweeps.groupby(["liquidity_side", "sweep_pos"], sort=False, observed=False):
        grouped_rows.append(
            {
                "liquidity_side": side,
                "sweep_pos": int(pos),
                "oldest_level_age_minutes_q": float(part["level_age_minutes_at_sweep"].max()),
                "median_level_age_minutes_q": float(part["level_age_minutes_at_sweep"].median()),
                "max_confirmed_order_age_minutes_q": float(part["confirmed_order_age_minutes_at_sweep"].max()),
                "max_excursion_away_bp_before_sweep": float(part["max_excursion_away_bp_before_sweep"].max()),
                "median_excursion_away_bp_before_sweep": float(part["max_excursion_away_bp_before_sweep"].median()),
                "max_activation_distance_bp": float(part["activation_distance_bp"].max()),
                "max_confirmed_prominence_bp_q": float(part["confirmed_prominence_bp_at_sweep"].max()),
                "max_pivot_rejection_fraction_q": float(part["pivot_rejection_fraction"].max()),
                "max_cluster_count_5bp": int(part["cluster_count_5bp"].max()),
                "max_cluster_count_10bp": int(part["cluster_count_10bp"].max()),
                "max_cluster_count_25bp": int(part["cluster_count_25bp"].max()),
                "max_cluster_timeframe_count_10bp": int(part["cluster_timeframe_count_10bp"].max()),
            }
        )
    agg = pd.DataFrame(grouped_rows)
    out = episodes.merge(agg, on=["liquidity_side", "sweep_pos"], how="left", validate="one_to_one")
    # Fixed, semantic hypotheses.  These names are candidate taxonomies, not a
    # claim that the market truly treats them as liquidity until outcomes prove it.
    out["lq_structural_major"] = out["max_confirmed_order"].ge(3) | out["max_timeframe_min"].ge(60)
    out["lq_mature_12h"] = out["oldest_level_age_minutes_q"].ge(12 * 60)
    out["lq_mature_24h"] = out["oldest_level_age_minutes_q"].ge(24 * 60)
    out["lq_mature_72h"] = out["oldest_level_age_minutes_q"].ge(72 * 60)
    out["lq_remote_50bp"] = out["max_excursion_away_bp_before_sweep"].ge(50.0)
    out["lq_remote_100bp"] = out["max_excursion_away_bp_before_sweep"].ge(100.0)
    out["lq_remote_200bp"] = out["max_excursion_away_bp_before_sweep"].ge(200.0)
    out["lq_stacked_10bp"] = out["max_cluster_count_10bp"].ge(2)
    out["lq_stacked_multi_tf_10bp"] = out["max_cluster_timeframe_count_10bp"].ge(2)
    out["lq_major_remote"] = (
        out["lq_structural_major"]
        & out["lq_mature_12h"]
        & out["lq_remote_100bp"]
    )
    out["lq_major_remote_stacked"] = out["lq_major_remote"] & out["lq_stacked_10bp"]
    out["lq_4h_mature"] = out["max_timeframe_min"].ge(240) & out["lq_mature_24h"]
    out["lq_1hplus_remote"] = out["max_timeframe_min"].ge(60) & out["lq_remote_100bp"]
    out["liquidity_age_bucket"] = _bucket_age(out["oldest_level_age_minutes_q"])
    out["confirmed_order_age_bucket"] = _bucket_age(out["max_confirmed_order_age_minutes_q"])
    out["liquidity_excursion_bucket"] = _bucket_excursion(out["max_excursion_away_bp_before_sweep"])
    return out


def liquidity_taxonomy_counts(episodes: pd.DataFrame, *, execution_timeframe: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    dimensions = [
        ("max_timeframe_min", episodes["max_timeframe_min"].astype(str)),
        ("max_confirmed_order", episodes["max_confirmed_order"].astype(str)),
        ("age_bucket", episodes["liquidity_age_bucket"]),
        ("excursion_bucket", episodes["liquidity_excursion_bucket"]),
        ("cluster_10bp", pd.cut(episodes["max_cluster_count_10bp"], [-np.inf, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]).astype("object")),
    ]
    for name, values in dimensions:
        tmp = pd.DataFrame({"bucket": values, "side": episodes["side"]})
        for bucket, part in tmp.groupby("bucket", dropna=False, observed=False):
            rows.append(
                {
                    "execution_timeframe": execution_timeframe,
                    "dimension": name,
                    "bucket": str(bucket),
                    "episodes": int(len(part)),
                    "long_episodes": int((part["side"] == 1).sum()),
                    "short_episodes": int((part["side"] == -1).sum()),
                }
            )
    return pd.DataFrame(rows)
