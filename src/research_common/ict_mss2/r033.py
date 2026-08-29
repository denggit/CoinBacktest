#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.3 causal ICT hierarchy / pool-quality research helpers.

The original R01/R02 taxonomy used fixed pivot orders (1/2/3/5).  ICT 2022
advanced market structure is different: it is a recursive swing-on-swing
hierarchy.  This module implements the mechanical portion causally:

* ST swing: every already-confirmed base swing candidate;
* IT high: ST high with lower ST highs immediately to left and right;
  IT low is the mirror;
* LT high: IT high with lower IT highs immediately to left and right;
  LT low is the mirror.

An IT/LT label is only usable after the right-hand confirming swing itself has
become available.  Eventual hierarchy is never backfilled into earlier sweeps.

The Episode-12 alternative where an imbalance rebalance can itself classify an
intermediate-term swing is deliberately *not* silently approximated here.  It
is reported as a separate future research item because a precise causal
mechanical definition of the relevant imbalance / resulting swing must first be
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class R033Config:
    pool_tolerance_bps: float = 10.0
    market_roundtrip_cost: float = 0.0011
    target_names: tuple[str, ...] = (
        "any", "pool2", "pool2tf", "htf60", "htf240", "htf1440",
        "r2p0", "r3p0", "r5p0",
    )
    entry_triggers: tuple[str, ...] = (
        "stage_reclaim", "episode_reclaim", "mss_structural_market", "mss_structural_fvg_limit",
    )
    execution_minutes: tuple[int, ...] = (1, 2, 5)

    def validate(self) -> "R033Config":
        if self.pool_tolerance_bps <= 0:
            raise ValueError("pool_tolerance_bps must be positive")
        if self.market_roundtrip_cost < 0:
            raise ValueError("market_roundtrip_cost cannot be negative")
        if not self.target_names or not self.entry_triggers or not self.execution_minutes:
            raise ValueError("R033 target/entry/execution settings cannot be empty")
        return self


def _hierarchy_extreme(center: float, left: float, right: float, side: str) -> bool:
    if not all(np.isfinite(v) for v in (center, left, right)):
        return False
    if side == "high":
        return bool(center > left and center > right)
    if side == "low":
        return bool(center < left and center < right)
    raise ValueError("side must be high/low")


def attach_causal_ict_swing_hierarchy(lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Attach causal ST/IT/LT hierarchy to lifecycle levels.

    The hierarchy is computed independently within each source timeframe and
    side.  ``ict_*_available_time`` is the earliest timestamp at which that
    hierarchy level can be known.  ``ict_swing_class_at_sweep`` uses the start
    of the 1m sweep bar as the knowledge cutoff, which is deliberately stricter
    than using the end of the sweep bar.
    """
    required = {
        "level_id", "pivot_side", "source_timeframe", "pivot_time", "level_price",
        "initial_available_time", "sweep_bar_time_1m",
    }
    missing = sorted(required - set(lifecycle.columns))
    if missing:
        raise KeyError(f"lifecycle missing hierarchy columns {missing}")
    out = lifecycle.copy()
    for name in ("pivot_time", "initial_available_time", "sweep_bar_time_1m"):
        out[name] = pd.to_datetime(out[name], errors="coerce")
    out["ict_st_available_time"] = out["initial_available_time"]
    out["ict_it_available_time"] = pd.NaT
    out["ict_lt_available_time"] = pd.NaT

    # Use positional indexes from each independent timeframe/side sequence.
    for (_, side), idx in out.groupby(["source_timeframe", "pivot_side"], sort=False).groups.items():
        loc = list(idx)
        part = out.loc[loc].sort_values(["pivot_time", "level_id"], kind="stable")
        ids = part.index.to_numpy()
        price = pd.to_numeric(part["level_price"], errors="coerce").to_numpy(dtype=float)
        st_avail = pd.to_datetime(part["ict_st_available_time"], errors="coerce")

        it_members: list[int] = []
        it_avail_by_index: dict[int, pd.Timestamp] = {}
        for j in range(1, len(part) - 1):
            if not _hierarchy_extreme(price[j], price[j - 1], price[j + 1], str(side)):
                continue
            center_idx = int(ids[j])
            # Right ST must itself be confirmed.  That is the first instant the
            # three-ST relation can be known.
            available = max(pd.Timestamp(st_avail.iloc[j]), pd.Timestamp(st_avail.iloc[j + 1]))
            out.at[center_idx, "ict_it_available_time"] = available
            it_members.append(j)
            it_avail_by_index[j] = available

        # LT is recursively defined on already identified IT swings.  It only
        # becomes known when the right neighboring IT is known.
        for k in range(1, len(it_members) - 1):
            left_j, center_j, right_j = it_members[k - 1], it_members[k], it_members[k + 1]
            if not _hierarchy_extreme(price[center_j], price[left_j], price[right_j], str(side)):
                continue
            center_idx = int(ids[center_j])
            available = max(it_avail_by_index[center_j], it_avail_by_index[right_j])
            out.at[center_idx, "ict_lt_available_time"] = available

    sweep_cutoff = pd.to_datetime(out["sweep_bar_time_1m"], errors="coerce")
    st_avail = pd.to_datetime(out["ict_st_available_time"], errors="coerce")
    it_avail = pd.to_datetime(out["ict_it_available_time"], errors="coerce")
    lt_avail = pd.to_datetime(out["ict_lt_available_time"], errors="coerce")
    st_known = sweep_cutoff.notna() & st_avail.notna() & st_avail.le(sweep_cutoff)
    it_known = sweep_cutoff.notna() & it_avail.notna() & it_avail.le(sweep_cutoff)
    lt_known = sweep_cutoff.notna() & lt_avail.notna() & lt_avail.le(sweep_cutoff)
    rank = np.where(lt_known, 3, np.where(it_known, 2, np.where(st_known, 1, 0))).astype(np.int8)
    out["ict_st_known_at_sweep_flag"] = st_known.astype(np.int8)
    out["ict_it_known_at_sweep_flag"] = it_known.astype(np.int8)
    out["ict_lt_known_at_sweep_flag"] = lt_known.astype(np.int8)
    out["ict_swing_rank_at_sweep"] = rank
    out["ict_swing_class_at_sweep"] = pd.Categorical(
        np.select([rank == 3, rank == 2, rank == 1], ["LT", "IT", "ST"], default="unavailable"),
        categories=["unavailable", "ST", "IT", "LT"], ordered=True,
    )
    return out


def _cluster_level_rows(part: pd.DataFrame, tolerance_bps: float) -> list[pd.DataFrame]:
    if part.empty:
        return []
    frame = part.copy()
    frame["level_price"] = pd.to_numeric(frame["level_price"], errors="coerce")
    frame = frame.dropna(subset=["level_price"]).sort_values("level_price", kind="stable")
    if frame.empty:
        return []
    tol = float(tolerance_bps)
    clusters: list[list[int]] = [[int(frame.index[0])]]
    prev = float(frame.iloc[0]["level_price"])
    for idx, row in frame.iloc[1:].iterrows():
        price = float(row["level_price"])
        gap_bp = abs(price / prev - 1.0) * 10_000.0 if abs(prev) > EPS else np.inf
        if gap_bp <= tol:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])
        prev = price
    return [frame.loc[idxs].copy() for idxs in clusters]


def _pool_summary(cluster: pd.DataFrame) -> dict[str, object]:
    rank = pd.to_numeric(cluster.get("ict_swing_rank_at_sweep"), errors="coerce").fillna(0).astype(int)
    tf = pd.to_numeric(cluster.get("source_timeframe_min"), errors="coerce")
    ext50 = pd.to_numeric(cluster.get("external_50_flag"), errors="coerce").fillna(0).astype(int)
    clean = pd.to_numeric(cluster.get("clean_sweep_no_prior_touch_flag"), errors="coerce").fillna(0).astype(int)
    max_rank = int(rank.max()) if len(rank) else 0
    tf_count = int(tf.dropna().nunique())
    max_tf = int(tf.max()) if tf.notna().any() else -1
    return {
        "levels": int(len(cluster)),
        "timeframes": tf_count,
        "max_tf": max_tf,
        "max_rank": max_rank,
        "it_plus": int(max_rank >= 2),
        "lt": int(max_rank >= 3),
        "htf240": int(max_tf >= 240),
        "multi_tf": int(tf_count >= 2),
        "external50": int(ext50.gt(0).any()),
        "clean": int(clean.gt(0).any()),
        # Structural-key is deliberately categorical rather than a fitted score.
        "structural_key": int(max_rank >= 2 or max_tf >= 240 or tf_count >= 2),
    }


def attach_causal_pool_hierarchy_to_episode_stages(
    hierarchy_lifecycle: pd.DataFrame,
    episode_stages: pd.DataFrame,
    *,
    tolerance_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild cumulative pools with hierarchy composition, incrementally.

    Performance matters here: the full R02 report has tens of thousands of
    causal stages.  We therefore pre-compress lifecycle rows into lightweight
    tuples keyed by (sweep_pos, direction), then maintain one small Python list
    per episode.  No repeated DataFrame concatenation or whole-table copies are
    performed inside the stage loop.
    """
    if episode_stages.empty:
        return episode_stages.copy(), pd.DataFrame()
    required_lifecycle = {"level_id", "sweep_pos_1m", "trade_direction", "level_price", "ict_swing_rank_at_sweep"}
    missing = sorted(required_lifecycle - set(hierarchy_lifecycle.columns))
    if missing:
        raise KeyError(f"hierarchy lifecycle missing {missing}")

    levels = hierarchy_lifecycle.loc[pd.to_numeric(hierarchy_lifecycle["sweep_pos_1m"], errors="coerce").ge(0)].copy()
    levels["sweep_pos_1m"] = pd.to_numeric(levels["sweep_pos_1m"], errors="coerce").astype(int)
    levels["trade_direction"] = pd.to_numeric(levels["trade_direction"], errors="coerce").astype(int)
    price = pd.to_numeric(levels["level_price"], errors="coerce")
    rank = pd.to_numeric(levels["ict_swing_rank_at_sweep"], errors="coerce").fillna(0).astype(int)
    tf = pd.to_numeric(levels.get("source_timeframe_min"), errors="coerce").fillna(-1).astype(int)
    ext = pd.to_numeric(levels.get("external_50_flag"), errors="coerce").fillna(0).astype(int)
    clean = pd.to_numeric(levels.get("clean_sweep_no_prior_touch_flag"), errors="coerce").fillna(0).astype(int)
    by_stage_key: dict[tuple[int, int], list[tuple[float, int, int, int, int, str]]] = {}
    for sp, dr, px, rk, tfi, ex, cl, lid in zip(
        levels["sweep_pos_1m"], levels["trade_direction"], price, rank, tf, ext, clean, levels["level_id"]
    ):
        if not np.isfinite(px):
            continue
        by_stage_key.setdefault((int(sp), int(dr)), []).append((float(px), int(rk), int(tfi), int(ex), int(cl), str(lid)))

    def cluster_records(records: list[tuple[float, int, int, int, int, str]]) -> list[list[tuple[float, int, int, int, int, str]]]:
        if not records:
            return []
        ordered = sorted(records, key=lambda x: x[0])
        result: list[list[tuple[float, int, int, int, int, str]]] = [[ordered[0]]]
        prev = ordered[0][0]
        for rec in ordered[1:]:
            px = rec[0]
            gap_bp = abs(px / prev - 1.0) * 10_000.0 if abs(prev) > EPS else np.inf
            if gap_bp <= float(tolerance_bps):
                result[-1].append(rec)
            else:
                result.append([rec])
            prev = px
        return result

    def summarize_cluster(cluster: list[tuple[float, int, int, int, int, str]]) -> dict[str, object]:
        ranks = [r[1] for r in cluster]
        tfs = {r[2] for r in cluster if r[2] >= 0}
        max_rank = max(ranks, default=0)
        max_tf = max(tfs, default=-1)
        tf_count = len(tfs)
        return {
            "levels": len(cluster), "timeframes": tf_count, "max_tf": max_tf, "max_rank": max_rank,
            "it_plus": int(max_rank >= 2), "lt": int(max_rank >= 3), "htf240": int(max_tf >= 240),
            "multi_tf": int(tf_count >= 2), "external50": int(any(r[3] > 0 for r in cluster)),
            "clean": int(any(r[4] > 0 for r in cluster)),
            "structural_key": int(max_rank >= 2 or max_tf >= 240 or tf_count >= 2),
        }

    output_rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    source = episode_stages.sort_values(["episode_id", "episode_stage_no", "sweep_pos_1m", "stage_id"], kind="stable")
    current_episode = None
    cumulative: list[tuple[float, int, int, int, int, str]] = []
    for stage in source.itertuples(index=False):
        stage_dict = stage._asdict()
        episode_id = stage_dict["episode_id"]
        if episode_id != current_episode:
            current_episode = episode_id
            cumulative = []
        key = (int(stage_dict["sweep_pos_1m"]), int(stage_dict["trade_direction"]))
        cumulative.extend(by_stage_key.get(key, ()))
        clusters = cluster_records(cumulative)
        stats = [summarize_cluster(c) for c in clusters]
        row = dict(stage_dict)
        row.update({
            "ict_price_pools_cum": len(stats),
            "ict_st_only_pools_cum": sum(x["max_rank"] <= 1 for x in stats),
            "ict_it_plus_pools_cum": sum(x["it_plus"] for x in stats),
            "ict_lt_pools_cum": sum(x["lt"] for x in stats),
            "ict_htf240_pools_cum": sum(x["htf240"] for x in stats),
            "ict_multi_tf_pools_cum": sum(x["multi_tf"] for x in stats),
            "ict_external50_pools_cum": sum(x["external50"] for x in stats),
            "ict_clean_pools_cum": sum(x["clean"] for x in stats),
            "ict_structural_key_pools_cum": sum(x["structural_key"] for x in stats),
            "ict_strongest_pool_rank_cum": max((x["max_rank"] for x in stats), default=0),
            "ict_max_pool_timeframes_cum": max((x["timeframes"] for x in stats), default=0),
        })
        elapsed = max(1, int(row.get("episode_elapsed_minutes", 0)) + 1)
        row["ict_structural_key_pools_per_min_cum"] = row["ict_structural_key_pools_cum"] / elapsed
        row["ict_it_plus_pools_per_min_cum"] = row["ict_it_plus_pools_cum"] / elapsed
        output_rows.append(row)
        for pool_no, (cluster, stat) in enumerate(zip(clusters, stats), start=1):
            prices = [r[0] for r in cluster]
            pool_rows.append({
                "episode_id": episode_id,
                "stage_id": row["stage_id"],
                "episode_stage_no": int(row["episode_stage_no"]),
                "pool_no_cum": pool_no,
                "pool_price_min": min(prices), "pool_price_max": max(prices),
                "pool_level_ids": "|".join(r[5] for r in cluster),
                **stat,
            })
    return pd.DataFrame(output_rows), pd.DataFrame(pool_rows)


def build_hierarchy_stage_cohorts(stages: pd.DataFrame) -> pd.DataFrame:
    """Create compact first-crossing hierarchy cohorts.

    The main cohort table stays deliberately small.  Detailed quality x raw-N
    interactions are evaluated separately at fixed N-pool crossings so they do
    not explode the trade table or tempt post-hoc threshold selection.
    """
    if stages.empty:
        return pd.DataFrame()
    definitions = {
        "first_any_pool": lambda x: pd.Series(True, index=x.index),
        "first_structural_key_pool": lambda x: pd.to_numeric(x["ict_structural_key_pools_cum"], errors="coerce").ge(1),
        "first_it_plus_pool": lambda x: pd.to_numeric(x["ict_it_plus_pools_cum"], errors="coerce").ge(1),
        "first_lt_pool": lambda x: pd.to_numeric(x["ict_lt_pools_cum"], errors="coerce").ge(1),
        "first_htf240_pool": lambda x: pd.to_numeric(x["ict_htf240_pools_cum"], errors="coerce").ge(1),
        "first_multi_tf_pool": lambda x: pd.to_numeric(x["ict_multi_tf_pools_cum"], errors="coerce").ge(1),
        "first_external50_pool": lambda x: pd.to_numeric(x["ict_external50_pools_cum"], errors="coerce").ge(1),
        "first_clean_pool": lambda x: pd.to_numeric(x["ict_clean_pools_cum"], errors="coerce").ge(1),
        "first_key_plus_ge2_total": lambda x: pd.to_numeric(x["ict_structural_key_pools_cum"], errors="coerce").ge(1) & pd.to_numeric(x["ict_price_pools_cum"], errors="coerce").ge(2),
        "first_key_plus_ge3_total": lambda x: pd.to_numeric(x["ict_structural_key_pools_cum"], errors="coerce").ge(1) & pd.to_numeric(x["ict_price_pools_cum"], errors="coerce").ge(3),
        "first_key_plus_ge4_total": lambda x: pd.to_numeric(x["ict_structural_key_pools_cum"], errors="coerce").ge(1) & pd.to_numeric(x["ict_price_pools_cum"], errors="coerce").ge(4),
    }
    rows: list[pd.DataFrame] = []
    base = stages.sort_values(["episode_id", "episode_stage_no", "sweep_pos_1m", "stage_id"], kind="stable")
    for name, fn in definitions.items():
        mask = fn(base).fillna(False)
        part = base.loc[mask].drop_duplicates("episode_id", keep="first").copy()
        if part.empty:
            continue
        part["hierarchy_cohort"] = name
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def attach_cohorts_to_trades(trades: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or cohorts.empty:
        return pd.DataFrame()
    cohort_cols = [
        "stage_id", "hierarchy_cohort", "ict_price_pools_cum", "ict_structural_key_pools_cum",
        "ict_it_plus_pools_cum", "ict_lt_pools_cum", "ict_htf240_pools_cum",
        "ict_multi_tf_pools_cum", "ict_external50_pools_cum", "ict_clean_pools_cum",
        "ict_strongest_pool_rank_cum", "ict_structural_key_pools_per_min_cum", "episode_elapsed_minutes",
    ]
    available = [c for c in cohort_cols if c in cohorts.columns]
    mapping = cohorts[available].drop_duplicates(["stage_id", "hierarchy_cohort"], keep="first")
    return trades.merge(mapping, on="stage_id", how="inner", validate="many_to_many")


def hierarchy_causal_audit(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame([{"check": "hierarchy_rows", "rows": 0, "violations": 0}])
    sweep = pd.to_datetime(lifecycle["sweep_bar_time_1m"], errors="coerce")
    st = pd.to_datetime(lifecycle["ict_st_available_time"], errors="coerce")
    it = pd.to_datetime(lifecycle["ict_it_available_time"], errors="coerce")
    lt = pd.to_datetime(lifecycle["ict_lt_available_time"], errors="coerce")
    it_known = pd.to_numeric(lifecycle["ict_it_known_at_sweep_flag"], errors="coerce").fillna(0).eq(1)
    lt_known = pd.to_numeric(lifecycle["ict_lt_known_at_sweep_flag"], errors="coerce").fillna(0).eq(1)
    rows = [
        {"check": "st_known_before_sweep", "rows": int(sweep.notna().sum()), "violations": int((sweep.notna() & st.gt(sweep)).sum())},
        {"check": "it_label_not_used_before_available", "rows": int(it_known.sum()), "violations": int((it_known & it.gt(sweep)).sum())},
        {"check": "lt_label_not_used_before_available", "rows": int(lt_known.sum()), "violations": int((lt_known & lt.gt(sweep)).sum())},
        {"check": "lt_implies_it", "rows": int(lt_known.sum()), "violations": int((lt_known & ~it_known).sum())},
    ]
    return pd.DataFrame(rows)


def profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    gains = float(x.loc[x > 0].sum())
    losses = float(-x.loc[x < 0].sum())
    if losses <= EPS:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def metric_row(frame: pd.DataFrame, net_col: str) -> dict[str, object]:
    x = pd.to_numeric(frame.get(net_col), errors="coerce").dropna() if net_col in frame.columns else pd.Series(dtype=float)
    return {
        "opportunities": int(len(frame)),
        "resolved": int(len(x)),
        "coverage": float(len(x) / len(frame)) if len(frame) else np.nan,
        "mean_net": float(x.mean()) if len(x) else np.nan,
        "median_net": float(x.median()) if len(x) else np.nan,
        "win_rate": float((x > 0).mean()) if len(x) else np.nan,
        "pf": profit_factor(x),
        "sum_net": float(x.sum()) if len(x) else np.nan,
    }


def grouped_metrics(frame: pd.DataFrame, group_cols: Sequence[str], net_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in frame.columns]
    grouped = [((), frame)] if not cols else frame.groupby(cols, dropna=False, observed=False, sort=True)
    rows: list[dict[str, object]] = []
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(cols, keys)}
        row.update(metric_row(part, net_col))
        rows.append(row)
    return pd.DataFrame(rows)


DISPLACEMENT_RESEARCH_FEATURES: tuple[str, ...] = (
    "displacement_atr",
    "displacement_speed_atr_per_min",
    "displacement_leg_range_atr",
    "max_directional_body_atr",
    "directional_body_share",
    "path_efficiency",
    "break_distance_atr",
    "mss_body_atr",
    "mss_body_ratio",
    "fvg_count_in_leg",
    "fvg_density_per_bar",
    "largest_fvg_width_atr",
    "attack_displacement_atr",
    "attack_path_efficiency",
    "attack_speed_atr_per_min",
    "reversal_attack_distance_ratio",
    "reversal_attack_speed_ratio",
)


def mss_reference_causal_audit(trades: pd.DataFrame) -> pd.DataFrame:
    """Audit both pre-sweep and newly allowed post-sweep ST MSS references."""
    if trades.empty:
        return pd.DataFrame([{"check": "mss_rows", "rows": 0, "violations": 0}])
    f = trades.copy()
    ref_mode = f.get("reference_mode", pd.Series("", index=f.index)).astype(str)
    pivot = pd.to_numeric(f.get("mss_reference_pivot_pos"), errors="coerce")
    sweep = pd.to_numeric(f.get("sweep_exec_pos"), errors="coerce")
    signal = pd.to_numeric(f.get("signal_exec_pos"), errors="coerce")
    ref_available = pd.to_datetime(f.get("mss_reference_available_time"), errors="coerce")
    sweep_bar_start = pd.to_datetime(f.get("sweep_exec_bar_time"), errors="coerce")
    signal_bar_start = pd.to_datetime(f.get("signal_bar_time"), errors="coerce")
    signal_available = pd.to_datetime(f.get("signal_available_time"), errors="coerce")
    entry_time = pd.to_datetime(f.get("entry_time"), errors="coerce")
    post = ref_mode.eq("post_sweep_st")
    pre = ref_mode.isin(["recent", "structural"])
    rows = [
        {
            "check": "pre_sweep_reference_is_pre_sweep_and_known",
            "rows": int(pre.sum()),
            "violations": int((pre & ((pivot >= sweep) | (ref_available > sweep_bar_start))).sum()),
        },
        {
            "check": "post_sweep_st_reference_forms_after_sweep",
            "rows": int(post.sum()),
            "violations": int((post & (pivot <= sweep)).sum()),
        },
        {
            "check": "post_sweep_st_reference_known_before_break_bar",
            "rows": int(post.sum()),
            "violations": int((post & (ref_available > signal_bar_start)).sum()),
        },
        {
            "check": "mss_reference_precedes_signal_position",
            "rows": int((pre | post).sum()),
            "violations": int(((pre | post) & (pivot >= signal)).sum()),
        },
        {
            "check": "mss_signal_available_after_reference",
            "rows": int((pre | post).sum()),
            "violations": int(((pre | post) & (signal_available <= ref_available)).sum()),
        },
        {
            "check": "market_or_limit_entry_not_before_signal_available",
            "rows": int(entry_time.notna().sum()),
            "violations": int((entry_time.notna() & (entry_time < signal_available)).sum()),
        },
    ]
    return pd.DataFrame(rows)


def build_displacement_payoff_atlas(
    trades: pd.DataFrame,
    *,
    net_col: str = "target_htf240_net_return_cost2x",
    min_train_rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Study displacement continuously without defining a hard strength rule.

    Quartile cut points are learned only from 2023-2024 rows inside each
    execution-timeframe/reference-mode group, then frozen for 2025-2026.  This
    intentionally permits non-monotonic results: Q2/Q3 are allowed to beat Q4.
    A separate attack-relative table directly tests whether reversals weaker
    than the move into the extreme can still have positive expectancy.
    """
    if trades.empty or net_col not in trades.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    f = trades.copy()
    f["entry_time"] = pd.to_datetime(f.get("entry_time"), errors="coerce")
    f = f.loc[f.get("trigger_type", pd.Series("", index=f.index)).astype(str).str.endswith("_market")].copy()
    if f.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_cutoff = pd.Timestamp("2025-01-01")
    recent_cutoff = pd.Timestamp("2025-10-01")
    threshold_rows: list[dict[str, object]] = []
    summary_rows: list[pd.DataFrame] = []

    group_cols = [c for c in ("trade_direction", "execution_minutes", "reference_mode") if c in f.columns]
    grouped = f.groupby(group_cols, dropna=False, observed=False, sort=True)
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        key_map = {name: value for name, value in zip(group_cols, keys)}
        train = part.loc[part["entry_time"] < train_cutoff]
        for feature in DISPLACEMENT_RESEARCH_FEATURES:
            if feature not in part.columns:
                continue
            x_train = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x_train) < int(min_train_rows):
                continue
            q = np.nanquantile(x_train.to_numpy(dtype=float), [0.25, 0.50, 0.75])
            if not np.isfinite(q).all():
                continue
            edges = np.unique(q.astype(float))
            if len(edges) < 2:
                continue
            threshold_rows.append({**key_map, "feature": feature, "train_rows": int(len(x_train)), "q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2])})
            values = pd.to_numeric(part[feature], errors="coerce").to_numpy(dtype=float)
            bucket_idx = np.searchsorted(q, values, side="right")
            bucket = np.where(np.isfinite(values), np.asarray(["Q1", "Q2", "Q3", "Q4"], dtype=object)[np.clip(bucket_idx, 0, 3)], "missing")
            tagged = part.copy()
            tagged["displacement_feature"] = feature
            tagged["displacement_bucket"] = bucket
            scopes = {
                "all": pd.Series(True, index=tagged.index),
                "train_2023_2024": tagged["entry_time"] < train_cutoff,
                "forward_2025_2026": tagged["entry_time"] >= train_cutoff,
                "recent_2025q4_2026h1": tagged["entry_time"] >= recent_cutoff,
            }
            for scope, mask in scopes.items():
                sub = tagged.loc[mask & tagged["displacement_bucket"].ne("missing")]
                if sub.empty:
                    continue
                sm = grouped_metrics(sub, ["displacement_feature", "displacement_bucket"], net_col)
                for kname, kval in key_map.items():
                    sm[kname] = kval
                sm["split"] = scope
                summary_rows.append(sm)

    # Directly test the user's hypothesis that a profitable reversal need not be
    # stronger than the attack into the extreme.  These are descriptive buckets,
    # never admission thresholds.
    ratio_rows: list[pd.DataFrame] = []
    ratio_defs = {
        "reversal_attack_distance_ratio": [-np.inf, 0.5, 0.8, 1.0, 1.5, 2.0, np.inf],
        "reversal_attack_speed_ratio": [-np.inf, 0.5, 0.8, 1.0, 1.5, 2.0, np.inf],
    }
    labels = ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.5", "1.5-2.0", ">=2.0"]
    for feature, bins in ratio_defs.items():
        if feature not in f.columns:
            continue
        tagged = f.copy()
        tagged["relative_strength_feature"] = feature
        tagged["relative_strength_bucket"] = pd.cut(
            pd.to_numeric(tagged[feature], errors="coerce"), bins=bins, labels=labels, right=False,
        )
        tagged["split"] = np.where(tagged["entry_time"] < train_cutoff, "train_2023_2024", "forward_2025_2026")
        sm = grouped_metrics(
            tagged.dropna(subset=["relative_strength_bucket"]),
            ["trade_direction", "execution_minutes", "reference_mode", "relative_strength_feature", "relative_strength_bucket", "split"],
            net_col,
        )
        if not sm.empty:
            ratio_rows.append(sm)
    return (
        pd.concat(summary_rows, ignore_index=True, sort=False) if summary_rows else pd.DataFrame(),
        pd.DataFrame(threshold_rows),
        pd.concat(ratio_rows, ignore_index=True, sort=False) if ratio_rows else pd.DataFrame(),
    )
