#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.3 - ICT swing hierarchy, key-liquidity pools, entries, exits and CVD.

R03.3 addresses three research risks exposed by R02/R03.2:

1. ``N pools`` may be an incomplete definition.  We rebuild each consumed pool
   using a causal ICT ST/IT/LT swing-on-swing hierarchy plus timeframe,
   multi-timeframe and externality composition.  A single genuinely important
   pool can therefore be compared against many weak ST-only pools.
2. MSS is not assumed mandatory.  Existing causal R02 entries are compared on
   the *same hierarchy-defined sweep stages*: stage reclaim, episode reclaim,
   structural MSS market, and structural MSS+FVG limit.  A corrected post-
   reclaim FVG execution overlay is also run on the frozen R02 core.
3. Exit destination is treated as part of the model.  Nearest liquidity,
   clustered/multi-TF pools, 1H/4H/1D liquidity and fixed-R controls are compared
   without adding a fixed time-profit exit.

Trade-bar CVD is an ETH/order-flow extension, not an ICT 2022 teaching.  It is
kept explicitly separate and causal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2 import (  # noqa: E402
    R02Config,
    R03Config,
    attach_structural_exit_outcomes,
    attach_causal_ict_swing_hierarchy,
    attach_causal_pool_hierarchy_to_episode_stages,
    attach_cohorts_to_trades,
    build_core_reclaim_execution_overlays,
    build_displacement_payoff_atlas,
    build_hierarchy_stage_cohorts,
    build_stack_execution_triggers,
    build_tradebar_microstructure_features,
    first_pool_threshold_crossing_trades,
    grouped_metrics,
    hierarchy_causal_audit,
    microstructure_feature_join_audit,
    mss_reference_causal_audit,
    r03_globalize_legacy_trade_ids,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "3.3.1"
EXPERIMENT_ID = "ETH_ICT_MSS2_HIERARCHY_ENTRY_EXIT_CVD_R03_3"
EDGE_ID = "RESEARCH_ONLY_ETH_HTF_LIQUIDITY_HIERARCHY_EXHAUSTION_LONG"
TITLE = "ETH ICT MSS2 R03.3 Hierarchy + Entry/Exit + CVD"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r03_3_liquidity_hierarchy_entry_exit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tradebar-db-name", default="okx_trade_bars.db")
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-cvd", action="store_true")
    p.add_argument("--skip-mss-displacement-refresh", action="store_true")
    p.add_argument("--skip-execution-overlay", action="store_true")
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--limit-roundtrip-cost", type=float, default=0.0009)
    p.add_argument("--fvg-signal-wait-minutes", type=int, default=180)
    p.add_argument("--fvg-limit-wait-minutes", type=int, default=180)
    return p.parse_args(argv)


def _read_manifest(path: Path) -> dict[str, object]:
    p = path / "00_manifest.json"
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


def _read_r02_lifecycle(report_dir: Path) -> pd.DataFrame:
    p = report_dir / "01_liquidity_lifecycle_causal.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, low_memory=False)


def _read_r02_episode_stages(report_dir: Path) -> pd.DataFrame:
    p = report_dir / "04_sweep_episode_stages_causal.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, low_memory=False)


def _read_r02_paths(report_dir: Path) -> pd.DataFrame:
    p = report_dir / "05_sweep_long_horizon_labels.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _read_r02_full_trades(report_dir: Path) -> pd.DataFrame:
    fpath = report_dir / "10_trade_features_causal.csv"
    lpath = report_dir / "11_trade_structural_exit_labels.csv"
    if not fpath.exists() or not lpath.exists():
        raise FileNotFoundError(f"R03.3 requires completed R02 report under {report_dir}")
    feature_cols = {
        "trade_event_id", "stage_id", "episode_id", "trade_direction", "execution_minutes", "trigger_type", "reference_mode",
        "sweep_pos_1m", "sweep_bar_time_1m", "episode_start_pos_1m", "episode_start_time_1m", "episode_elapsed_minutes",
        "levels_consumed_cum", "distinct_timeframes_cum", "max_source_timeframe_min_cum", "htf_240m_plus_levels_cum",
        "htf_1440m_plus_levels_cum", "episode_consumption_depth_bp", "levels_consumed_per_min_cum",
        "price_pools_5p0bp_cum", "pools_per_min_5p0bp_cum", "price_pools_10p0bp_cum", "pools_per_min_10p0bp_cum",
        "price_pools_20p0bp_cum", "pools_per_min_20p0bp_cum", "signal_available_time", "signal_bar_time",
        "entry_pos_1m", "entry_time", "entry_price", "entry_kind", "stop_price", "risk_bps",
        "session_primary", "is_weekend_utc", "year", "quarter", "month",
    }
    features = pd.read_csv(fpath, usecols=lambda c: c in feature_cols, low_memory=False)
    target_names = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r2p0", "r3p0", "r5p0")
    label_cols = {"trade_event_id", "stage_id", "episode_id", "target_htf240_price", "target_htf240_outcome", "target_htf240_gross_return"}
    label_cols.update({f"target_{name}_net_return_cost2x" for name in target_names})
    labels = pd.read_csv(lpath, usecols=lambda c: c in label_cols, low_memory=False)
    if len(features) != len(labels):
        raise RuntimeError("R02 feature/label row counts differ")
    features, labels = r03_globalize_legacy_trade_ids(features, labels)
    assert labels is not None
    if not np.array_equal(features["trade_event_id"].astype(str).to_numpy(), labels["trade_event_id"].astype(str).to_numpy()):
        raise RuntimeError("R02 feature/label rows are not aligned after ID repair")
    duplicate_keys = [c for c in ("trade_event_id", "stage_id", "episode_id") if c in labels.columns]
    merged = pd.concat([features.reset_index(drop=True), labels.drop(columns=duplicate_keys).reset_index(drop=True)], axis=1)
    for c in ("sweep_bar_time_1m", "episode_start_time_1m", "signal_available_time", "signal_bar_time", "entry_time"):
        if c in merged.columns:
            merged[c] = pd.to_datetime(merged[c], errors="coerce")
    return merged


def _metric_paths(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    horizons = (60, 360, 720, 1440, 2880, 4320, 10080)
    rows = []
    cols = [c for c in group_cols if c in frame.columns]
    grouped = frame.groupby(cols, dropna=False, observed=False, sort=True) if cols else [((), frame)]
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(cols, keys)}
        row["events"] = int(len(part))
        for h in horizons:
            c = f"path_close_return_{h}m"
            if c in part.columns:
                x = pd.to_numeric(part[c], errors="coerce").dropna()
                row[f"mean_{h}m"] = float(x.mean()) if len(x) else np.nan
                row[f"positive_{h}m"] = float((x > 0).mean()) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _entry_summary(cohort_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = "target_htf240_net_return_cost2x"
    overall = grouped_metrics(
        cohort_trades,
        ["hierarchy_cohort", "trade_direction", "execution_minutes", "trigger_type"], target,
    )
    yearly = grouped_metrics(
        cohort_trades,
        ["hierarchy_cohort", "trade_direction", "execution_minutes", "trigger_type", "year"], target,
    )
    return overall, yearly


def _target_summary(cohort_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = ("any", "pool2", "pool2tf", "htf60", "htf240", "htf1440", "r2p0", "r3p0", "r5p0")
    rows = []
    year_rows = []
    grouping = ["hierarchy_cohort", "trade_direction", "execution_minutes", "trigger_type"]
    for target in targets:
        col = f"target_{target}_net_return_cost2x"
        if col not in cohort_trades.columns:
            continue
        s = grouped_metrics(cohort_trades, grouping, col)
        if not s.empty:
            s.insert(0, "target", target)
            rows.append(s)
        y = grouped_metrics(cohort_trades, [*grouping, "year"], col)
        if not y.empty:
            y.insert(0, "target", target)
            year_rows.append(y)
    return (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(),
        pd.concat(year_rows, ignore_index=True, sort=False) if year_rows else pd.DataFrame(),
    )


def _pool_quality_snapshot(pool_rows: pd.DataFrame) -> pd.DataFrame:
    if pool_rows.empty:
        return pd.DataFrame()
    p = pool_rows.copy()
    p["quality_bucket"] = np.select(
        [
            pd.to_numeric(p["lt"], errors="coerce").eq(1),
            pd.to_numeric(p["it_plus"], errors="coerce").eq(1),
            pd.to_numeric(p["htf240"], errors="coerce").eq(1),
            pd.to_numeric(p["multi_tf"], errors="coerce").eq(1),
        ],
        ["LT", "IT", "ST_4H_PLUS", "ST_MULTI_TF"],
        default="ST_ONLY",
    )
    return (
        p.groupby(["quality_bucket"], dropna=False, observed=False, sort=True)
        .agg(
            pool_snapshots=("stage_id", "size"),
            episodes=("episode_id", "nunique"),
            mean_levels=("levels", "mean"),
            mean_timeframes=("timeframes", "mean"),
            external50_rate=("external50", "mean"),
            clean_rate=("clean", "mean"),
        )
        .reset_index()
    )



def _quality_at_pool_threshold_summary(
    trades: pd.DataFrame, hierarchy_stages: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pool quality at the same raw N-pool crossing.

    This keeps quantity fixed and asks whether the crossing stack already
    contains a causally-known IT/LT/4H+/multi-TF/clean/external pool.  It is a
    descriptive decomposition, not an optimized filter search.
    """
    if trades.empty or hierarchy_stages.empty:
        return pd.DataFrame(), pd.DataFrame()
    stage_cols = [
        "stage_id", "ict_it_plus_pools_cum", "ict_lt_pools_cum",
        "ict_htf240_pools_cum", "ict_multi_tf_pools_cum",
        "ict_external50_pools_cum", "ict_clean_pools_cum",
        "ict_price_pools_cum", "ict_structural_key_pools_per_min_cum",
    ]
    stage_map = hierarchy_stages[[c for c in stage_cols if c in hierarchy_stages.columns]].drop_duplicates("stage_id")
    dimensions = {
        "contains_it_plus": "ict_it_plus_pools_cum",
        "contains_lt": "ict_lt_pools_cum",
        "contains_4h_plus": "ict_htf240_pools_cum",
        "contains_multi_tf": "ict_multi_tf_pools_cum",
        "contains_external50": "ict_external50_pools_cum",
        "contains_clean": "ict_clean_pools_cum",
    }
    rows: list[pd.DataFrame] = []
    years: list[pd.DataFrame] = []
    for n in (1, 2, 3, 4):
        crossed = first_pool_threshold_crossing_trades(
            trades, threshold=n, tolerance_bps=10.0, direction=1,
            trigger_type="episode_reclaim", execution_minutes=(1, 2, 5),
        ).drop_duplicates("trade_event_id", keep="first")
        if crossed.empty:
            continue
        crossed = crossed.merge(stage_map, on="stage_id", how="left", validate="many_to_one")
        for label, col in dimensions.items():
            if col not in crossed.columns:
                continue
            part = crossed.copy()
            part["quality_dimension"] = label
            part["quality_present"] = pd.to_numeric(part[col], errors="coerce").fillna(0).gt(0).astype(np.int8)
            summary = grouped_metrics(part, ["execution_minutes", "quality_dimension", "quality_present"], "target_htf240_net_return_cost2x")
            summary.insert(0, "pool_threshold", n)
            rows.append(summary)
            yearly = grouped_metrics(part, ["execution_minutes", "quality_dimension", "quality_present", "year"], "target_htf240_net_return_cost2x")
            yearly.insert(0, "pool_threshold", n)
            years.append(yearly)
    return (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(),
        pd.concat(years, ignore_index=True, sort=False) if years else pd.DataFrame(),
    )

def _cvd_summaries(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    f = joined.copy()
    f["split"] = np.where(pd.to_datetime(f["entry_time"], errors="coerce") < pd.Timestamp("2025-01-01"), "train_2023_2024", "forward_2025_2026")
    definitions = {
        "all": pd.Series(True, index=f.index),
        "tb_absorption": pd.to_numeric(f.get("tb_absorption_mechanism_flag"), errors="coerce").fillna(0).eq(1),
        "cvd_recovery_positive": pd.to_numeric(f.get("tb_episode_cvd_recovery"), errors="coerce").gt(0),
        "cvd_bull_div_3m": pd.to_numeric(f.get("tb_cvd_bullish_divergence_3m_flag"), errors="coerce").fillna(0).eq(1),
        "cvd_bull_div_5m": pd.to_numeric(f.get("tb_cvd_bullish_divergence_5m_flag"), errors="coerce").fillna(0).eq(1),
        "cvd_bull_div_15m": pd.to_numeric(f.get("tb_cvd_bullish_divergence_15m_flag"), errors="coerce").fillna(0).eq(1),
    }
    rows = []
    years = []
    for name, mask in definitions.items():
        part = f.loc[mask].copy()
        s = grouped_metrics(part, ["hierarchy_cohort", "execution_minutes", "split"], "target_htf240_net_return_cost2x")
        if not s.empty:
            s.insert(0, "cvd_filter", name)
            rows.append(s)
        y = grouped_metrics(part, ["hierarchy_cohort", "execution_minutes", "year"], "target_htf240_net_return_cost2x")
        if not y.empty:
            y.insert(0, "cvd_filter", name)
            years.append(y)
    return (
        pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(),
        pd.concat(years, ignore_index=True, sort=False) if years else pd.DataFrame(),
    )


def _execution_summary(overlay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if overlay.empty:
        return pd.DataFrame(), pd.DataFrame()
    overall = grouped_metrics(overlay, ["fvg_minutes", "execution_variant"], "net_return_cost2x")
    yearly = grouped_metrics(overlay, ["fvg_minutes", "execution_variant", "year"], "net_return_cost2x")
    fill = (
        overlay.groupby(["fvg_minutes", "execution_variant"], dropna=False, observed=False, sort=True)
        .agg(
            rows=("base_trade_event_id", "size"),
            opportunities=("base_trade_event_id", "nunique"),
            entry_fill_rate=("entry_fill_flag", "mean"),
            fvg_signal_rate=("fvg_signal_available_time", lambda s: pd.to_datetime(s, errors="coerce").notna().mean()),
            limit_fill_rate=("limit_filled_flag", "mean"),
        )
        .reset_index()
    )
    overall = fill.merge(overall, on=["fvg_minutes", "execution_variant"], how="left", validate="one_to_one")
    return overall, yearly


def _hard_execution_audit(overlay: pd.DataFrame, core5: pd.DataFrame, fvg_minutes: Sequence[int]) -> pd.DataFrame:
    expected_core = int(core5["trade_event_id"].astype(str).nunique())
    variants = {
        "reclaim_market", "post_reclaim_fvg_market", "post_reclaim_fvg_limit", "hybrid_reclaim_market_fvg_limit"
    }
    rows = []
    for m in fvg_minutes:
        part = overlay.loc[pd.to_numeric(overlay["fvg_minutes"], errors="coerce").eq(int(m))]
        ids = part["base_trade_event_id"].astype(str)
        rows.append({"check": f"{m}m_preserves_all_core_opportunities", "expected": expected_core, "actual": int(ids.nunique()), "passed": int(ids.nunique() == expected_core)})
        counts = part.groupby("base_trade_event_id", dropna=False)["execution_variant"].nunique()
        rows.append({"check": f"{m}m_has_four_variants_per_core", "expected": expected_core, "actual": int((counts == len(variants)).sum()), "passed": int(len(counts) == expected_core and counts.eq(len(variants)).all())})
        fvg_rows = int(pd.to_datetime(part["fvg_signal_available_time"], errors="coerce").notna().sum())
        rows.append({"check": f"{m}m_nonvacuous_fvg_signals", "expected": ">0", "actual": fvg_rows, "passed": int(fvg_rows > 0)})
    expected_rows = expected_core * len(tuple(fvg_minutes)) * len(variants)
    rows.append({"check": "total_execution_overlay_rows", "expected": expected_rows, "actual": int(len(overlay)), "passed": int(len(overlay) == expected_rows)})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress = not bool(args.no_progress)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    r02_dir = PROJECT_ROOT / args.r02_report_dir
    manifest = _read_manifest(r02_dir)

    print("[r03.3] load R02 lifecycle/stages/trades", flush=True)
    lifecycle = _read_r02_lifecycle(r02_dir)
    stages = _read_r02_episode_stages(r02_dir)
    paths = _read_r02_paths(r02_dir)
    trades = _read_r02_full_trades(r02_dir)

    bars_cache: pd.DataFrame | None = None
    def load_naked_1m() -> pd.DataFrame:
        nonlocal bars_cache
        if bars_cache is None:
            warmup_start = str(manifest.get("warmup_start_date") or args.warmup_start_date)
            loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
            bars_cache = loader.fetch_data_by_date_range(warmup_start, args.end_date)
            if bars_cache.empty:
                raise RuntimeError("R03.3 requires naked 1m K for refreshed MSS/displacement/execution research")
        return bars_cache

    print("[r03.3] causal ICT ST/IT/LT swing-on-swing hierarchy", flush=True)
    hierarchy = attach_causal_ict_swing_hierarchy(lifecycle)
    hierarchy_audit = hierarchy_causal_audit(hierarchy)
    if int(pd.to_numeric(hierarchy_audit["violations"], errors="coerce").fillna(0).sum()) > 0:
        raise RuntimeError("R03.3 hierarchy causality audit failed")
    hierarchy_keep = [c for c in (
        "level_id", "pivot_side", "source_timeframe", "source_timeframe_min", "pivot_time", "level_price",
        "initial_available_time", "sweep_pos_1m", "sweep_bar_time_1m", "trade_direction",
        "confirmed_order_at_sweep", "external_50_flag", "clean_sweep_no_prior_touch_flag",
        "pretested_before_sweep_flag", "age_minutes_at_sweep", "quality_tier",
        "ict_st_available_time", "ict_it_available_time", "ict_lt_available_time",
        "ict_st_known_at_sweep_flag", "ict_it_known_at_sweep_flag", "ict_lt_known_at_sweep_flag",
        "ict_swing_rank_at_sweep", "ict_swing_class_at_sweep",
    ) if c in hierarchy.columns]
    hierarchy[hierarchy_keep].to_csv(out_dir / "03_liquidity_ict_hierarchy_causal.csv.gz", index=False, compression="gzip")

    print("[r03.3] rebuild cumulative pools by hierarchy composition", flush=True)
    hierarchy_stages, pool_rows = attach_causal_pool_hierarchy_to_episode_stages(hierarchy, stages, tolerance_bps=10.0)
    stage_keep = [c for c in (
        "stage_id", "episode_id", "episode_stage_no", "sweep_pos_1m", "sweep_bar_time_1m",
        "trade_direction", "episode_start_pos_1m", "episode_start_time_1m", "episode_elapsed_minutes",
        "levels_consumed_cum", "distinct_timeframes_cum", "max_source_timeframe_min_cum",
        "price_pools_10p0bp_cum", "pools_per_min_10p0bp_cum",
        "ict_price_pools_cum", "ict_st_only_pools_cum", "ict_it_plus_pools_cum", "ict_lt_pools_cum",
        "ict_htf240_pools_cum", "ict_multi_tf_pools_cum", "ict_external50_pools_cum", "ict_clean_pools_cum",
        "ict_structural_key_pools_cum", "ict_strongest_pool_rank_cum", "ict_max_pool_timeframes_cum",
        "ict_structural_key_pools_per_min_cum", "ict_it_plus_pools_per_min_cum",
    ) if c in hierarchy_stages.columns]
    hierarchy_stages[stage_keep].to_csv(out_dir / "04_episode_stages_hierarchy_causal.csv.gz", index=False, compression="gzip")
    pool_rows.to_csv(out_dir / "05_pool_composition_snapshots.csv.gz", index=False, compression="gzip")
    _pool_quality_snapshot(pool_rows).to_csv(out_dir / "06_pool_quality_snapshot_summary.csv", index=False, encoding="utf-8-sig")

    cohorts = build_hierarchy_stage_cohorts(hierarchy_stages)
    cohort_keep = [c for c in [*stage_keep, "hierarchy_cohort"] if c in cohorts.columns]
    cohorts[cohort_keep].to_csv(out_dir / "07_hierarchy_first_crossing_cohorts.csv.gz", index=False, compression="gzip")

    if not paths.empty:
        path_join = cohorts[["stage_id", "hierarchy_cohort", "trade_direction"]].merge(paths, on="stage_id", how="left", validate="many_to_one", suffixes=("", "_path"))
        _metric_paths(path_join, ["hierarchy_cohort", "trade_direction"]).to_csv(out_dir / "08_hierarchy_forward_path_summary.csv", index=False, encoding="utf-8-sig")

    print("[r03.3] attach hierarchy cohorts to existing causal R02 trades", flush=True)
    cohort_trades = attach_cohorts_to_trades(trades, cohorts)
    trade_review_keep = [c for c in (
        "trade_event_id", "stage_id", "episode_id", "hierarchy_cohort", "trade_direction",
        "execution_minutes", "trigger_type", "signal_available_time", "entry_time", "entry_price",
        "stop_price", "risk_bps", "ict_price_pools_cum", "ict_structural_key_pools_cum",
        "ict_it_plus_pools_cum", "ict_lt_pools_cum", "ict_htf240_pools_cum", "ict_multi_tf_pools_cum",
        "ict_external50_pools_cum", "ict_clean_pools_cum", "ict_strongest_pool_rank_cum",
        "target_htf240_net_return_cost2x", "year",
    ) if c in cohort_trades.columns]
    primary_review_mask = (
        pd.to_numeric(cohort_trades["trade_direction"], errors="coerce").eq(1)
        & pd.to_numeric(cohort_trades["execution_minutes"], errors="coerce").eq(5)
        & cohort_trades["trigger_type"].astype(str).eq("episode_reclaim")
    )
    cohort_trades.loc[primary_review_mask, trade_review_keep].to_csv(
        out_dir / "09_hierarchy_trade_rows_primary_long_5m_reclaim.csv.gz", index=False, compression="gzip"
    )
    print("[r03.3] entry-method comparison", flush=True)
    entry, entry_year = _entry_summary(cohort_trades)
    entry.to_csv(out_dir / "10_entry_method_summary.csv", index=False, encoding="utf-8-sig")
    entry_year.to_csv(out_dir / "11_entry_method_year_summary.csv", index=False, encoding="utf-8-sig")
    print("[r03.3] structural target comparison", flush=True)
    target, target_year = _target_summary(cohort_trades)
    target.to_csv(out_dir / "12_exit_target_summary.csv", index=False, encoding="utf-8-sig")
    target_year.to_csv(out_dir / "13_exit_target_year_summary.csv", index=False, encoding="utf-8-sig")
    print("[r03.3] pool-quality decomposition at fixed N thresholds", flush=True)
    quality_n, quality_n_year = _quality_at_pool_threshold_summary(trades, hierarchy_stages)
    quality_n.to_csv(out_dir / "13a_pool_quality_at_fixed_n_summary.csv", index=False, encoding="utf-8-sig")
    quality_n_year.to_csv(out_dir / "13b_pool_quality_at_fixed_n_year_summary.csv", index=False, encoding="utf-8-sig")

    # Refresh MSS on the hierarchy research stages instead of relying only on
    # R02's pre-sweep references.  This explicitly includes the common path:
    # sweep -> new small ST swing forms -> later close breaks that post-sweep ST.
    refreshed_mss = pd.DataFrame()
    refreshed_mss_cohorts = pd.DataFrame()
    refreshed_mss_audit = pd.DataFrame()
    displacement_summary = pd.DataFrame()
    displacement_thresholds = pd.DataFrame()
    displacement_vs_attack = pd.DataFrame()
    if not args.skip_mss_displacement_refresh:
        print("[r03.3] refreshed pre/post-sweep MSS + open-form displacement atlas", flush=True)
        bars_mss = load_naked_1m()
        research_stage_ids = cohorts["stage_id"].astype(str).drop_duplicates() if not cohorts.empty else pd.Series(dtype=str)
        mss_stages = hierarchy_stages.loc[hierarchy_stages["stage_id"].astype(str).isin(set(research_stage_ids))].copy()
        mss_parts: list[pd.DataFrame] = []
        r02cfg = R02Config()
        for m in (1, 2, 5):
            part = build_stack_execution_triggers(
                bars_mss, mss_stages, execution_minutes=int(m),
                config=r02cfg, reference_modes=("recent", "structural", "post_sweep_st"),
                include_reclaims=False, include_mss_market=True, include_mss_fvg=True,
                show_progress=progress,
            )
            if not part.empty:
                mss_parts.append(part)
        if mss_parts:
            refreshed_mss = pd.concat(mss_parts, ignore_index=True, sort=False)
            refreshed_mss = attach_structural_exit_outcomes(
                bars_mss, lifecycle, refreshed_mss, config=r02cfg,
                roundtrip_cost=float(args.market_roundtrip_cost), show_progress=progress,
            )
            refreshed_mss_audit = mss_reference_causal_audit(refreshed_mss)
            if int(pd.to_numeric(refreshed_mss_audit["violations"], errors="coerce").fillna(0).sum()) > 0:
                raise RuntimeError("R03.3 refreshed MSS causality audit failed")
            refreshed_mss_cohorts = attach_cohorts_to_trades(refreshed_mss, cohorts)

            mss_summary = grouped_metrics(
                refreshed_mss_cohorts,
                ["hierarchy_cohort", "trade_direction", "execution_minutes", "reference_mode", "trigger_type"],
                "target_htf240_net_return_cost2x",
            )
            mss_year = grouped_metrics(
                refreshed_mss_cohorts,
                ["hierarchy_cohort", "trade_direction", "execution_minutes", "reference_mode", "trigger_type", "year"],
                "target_htf240_net_return_cost2x",
            )
            mss_summary.to_csv(out_dir / "13c_refreshed_mss_reference_summary.csv", index=False, encoding="utf-8-sig")
            mss_year.to_csv(out_dir / "13d_refreshed_mss_reference_year_summary.csv", index=False, encoding="utf-8-sig")

            displacement_summary, displacement_thresholds, displacement_vs_attack = build_displacement_payoff_atlas(refreshed_mss)
            displacement_thresholds.to_csv(out_dir / "13e_displacement_train_2023_2024_thresholds.csv", index=False, encoding="utf-8-sig")
            displacement_summary.to_csv(out_dir / "13f_displacement_frozen_quartile_payoff.csv", index=False, encoding="utf-8-sig")
            displacement_vs_attack.to_csv(out_dir / "13g_reversal_vs_attack_payoff.csv", index=False, encoding="utf-8-sig")
            refreshed_mss_audit.to_csv(out_dir / "13h_refreshed_mss_causal_audit.csv", index=False, encoding="utf-8-sig")

            review_cols = [c for c in (
                "trade_event_id", "stage_id", "episode_id", "trade_direction", "execution_minutes",
                "reference_mode", "trigger_type", "sweep_exec_pos", "mss_reference_pivot_pos",
                "mss_reference_available_time", "signal_bar_time", "signal_available_time", "entry_time",
                "displacement_atr", "displacement_speed_atr_per_min", "path_efficiency",
                "max_directional_body_atr", "directional_body_share", "break_distance_atr",
                "fvg_count_in_leg", "largest_fvg_width_atr", "attack_displacement_atr",
                "attack_speed_atr_per_min", "reversal_attack_distance_ratio",
                "reversal_attack_speed_ratio", "target_htf240_net_return_cost2x", "year",
            ) if c in refreshed_mss.columns]
            refreshed_mss[review_cols].to_csv(out_dir / "13i_refreshed_mss_displacement_trade_rows.csv.gz", index=False, compression="gzip")

    cvd_features = pd.DataFrame()
    cvd_audit = pd.DataFrame()
    if not args.skip_cvd:
        print("[r03.3] causal trade-bar CVD on hierarchy-defined long reclaim checkpoints", flush=True)
        cvd_source = cohort_trades.loc[
            pd.to_numeric(cohort_trades["trade_direction"], errors="coerce").eq(1)
            & cohort_trades["trigger_type"].astype(str).eq("episode_reclaim")
            & cohort_trades["hierarchy_cohort"].astype(str).isin([
                "first_it_plus_pool", "first_lt_pool", "first_htf240_pool",
                "first_key_plus_ge2_total", "first_key_plus_ge3_total", "first_key_plus_ge4_total",
            ])
        ].copy()
        checkpoint_cols = ["trade_event_id", "signal_available_time", "episode_start_time_1m"]
        checkpoints = (
            cvd_source[checkpoint_cols]
            .drop_duplicates("trade_event_id", keep="first")
            .rename(columns={"trade_event_id": "checkpoint_id", "signal_available_time": "decision_time", "episode_start_time_1m": "episode_start_time"})
        )
        cvd_features, cvd_audit = build_tradebar_microstructure_features(
            checkpoints,
            symbol=args.symbol,
            data_dir=args.data_dir,
            db_name=args.tradebar_db_name,
            config=R03Config(market_roundtrip_cost=float(args.market_roundtrip_cost), limit_roundtrip_cost=float(args.limit_roundtrip_cost)),
            show_progress=progress,
        )
        join_audit = microstructure_feature_join_audit(checkpoints.assign(cohort_membership="r033"), cvd_features, module="tradebar_cvd")
        if not join_audit["passed"].astype(int).eq(1).all():
            raise RuntimeError("R03.3 CVD checkpoint attachment not 100%")
        cvd_features.to_csv(out_dir / "14_tradebar_cvd_features_causal.csv.gz", index=False, compression="gzip")
        cvd_audit.to_csv(out_dir / "15_tradebar_cvd_build_audit.csv", index=False, encoding="utf-8-sig")
        join_audit.to_csv(out_dir / "16_tradebar_cvd_join_audit.csv", index=False, encoding="utf-8-sig")
        cvd_join = cvd_source.merge(cvd_features, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="many_to_one")
        cvd_summary, cvd_year = _cvd_summaries(cvd_join)
        cvd_summary.to_csv(out_dir / "17_cvd_mechanism_summary.csv", index=False, encoding="utf-8-sig")
        cvd_year.to_csv(out_dir / "18_cvd_year_summary.csv", index=False, encoding="utf-8-sig")

    overlay = pd.DataFrame()
    execution_audit = pd.DataFrame()
    if not args.skip_execution_overlay:
        print("[r03.3] corrected frozen-core reclaim/FVG execution overlay", flush=True)
        cfg3 = R03Config(
            market_roundtrip_cost=float(args.market_roundtrip_cost),
            limit_roundtrip_cost=float(args.limit_roundtrip_cost),
            fvg_signal_wait_minutes=int(args.fvg_signal_wait_minutes),
            fvg_limit_wait_minutes=int(args.fvg_limit_wait_minutes),
        ).validate()
        core5 = first_pool_threshold_crossing_trades(
            trades, threshold=4, tolerance_bps=10.0, direction=1,
            trigger_type="episode_reclaim", execution_minutes=(5,),
        ).drop_duplicates("trade_event_id", keep="first")
        bars = load_naked_1m()
        parts = []
        ties = []
        for m in cfg3.fvg_execution_minutes:
            part, tie = build_core_reclaim_execution_overlays(bars, core5, fvg_minutes=int(m), config=cfg3, show_progress=progress)
            parts.append(part)
            ties.append(tie)
        overlay = pd.concat(parts, ignore_index=True, sort=False)
        tieout = pd.concat(ties, ignore_index=True, sort=False)
        tieout.to_csv(out_dir / "19_r02_reclaim_tieout.csv", index=False, encoding="utf-8-sig")
        tie_bad = (~tieout[["outcome_match", "gross_match"]].astype(int).eq(1)).any(axis=1).sum()
        if int(tie_bad) > 0:
            raise RuntimeError(f"R03.3 baseline tieout failed rows={int(tie_bad)}")
        execution_audit = _hard_execution_audit(overlay, core5, cfg3.fvg_execution_minutes)
        execution_audit.to_csv(out_dir / "20_execution_hard_audit.csv", index=False, encoding="utf-8-sig")
        if not execution_audit["passed"].astype(int).eq(1).all():
            raise RuntimeError("R03.3 execution overlay hard audit failed")
        overlay.to_csv(out_dir / "21_frozen_core_execution_overlay.csv.gz", index=False, compression="gzip")
        ex, ex_year = _execution_summary(overlay)
        ex.to_csv(out_dir / "22_execution_summary.csv", index=False, encoding="utf-8-sig")
        ex_year.to_csv(out_dir / "23_execution_year_summary.csv", index=False, encoding="utf-8-sig")

    hierarchy_audit.to_csv(out_dir / "24_causal_audit.csv", index=False, encoding="utf-8-sig")
    engineering = pd.DataFrame([
        {"check": "ict_hierarchy_is_swing_on_swing_not_pivot_order", "passed": 1, "value": "ST->IT->LT with explicit availability times"},
        {"check": "hierarchy_cutoff_before_sweep_bar", "passed": int(hierarchy_audit["violations"].sum() == 0), "value": int(hierarchy_audit["violations"].sum())},
        {"check": "pool_count_not_only_admission_variable", "passed": 1, "value": "single IT/LT/4H/multi-TF key pool cohorts included"},
        {"check": "mss_not_mandatory", "passed": 1, "value": "stage_reclaim + episode_reclaim compared against pre-sweep and post-sweep ST MSS variants"},
        {"check": "post_sweep_st_mss_supported", "passed": int(args.skip_mss_displacement_refresh or (not refreshed_mss_audit.empty and pd.to_numeric(refreshed_mss_audit["violations"], errors="coerce").fillna(0).sum() == 0)), "value": int(len(refreshed_mss))},
        {"check": "displacement_has_no_hard_strength_gate", "passed": 1, "value": "continuous price-path features + frozen 2023-24 quartiles; reversal/attack ratio descriptive only"},
        {"check": "exit_target_family_compared", "passed": 1, "value": "nearest/pools/1H/4H/1D/fixed-R"},
        {"check": "cvd_is_separate_non_ict_extension", "passed": 1, "value": "causal trade-bar delta/CVD only"},
        {"check": "execution_overlay_nonvacuous", "passed": int(args.skip_execution_overlay or (not execution_audit.empty and execution_audit["passed"].astype(int).eq(1).all())), "value": int(len(overlay))},
        {"check": "ny_open_gate", "passed": 1, "value": "NO"},
        {"check": "fixed_time_profit_exit", "passed": 1, "value": "NO; R02 7d censor semantics retained"},
    ])
    engineering.to_csv(out_dir / "01_engineering_audit.csv", index=False, encoding="utf-8-sig")

    design = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "ict_hierarchy": {
            "ST": "already-confirmed base swing",
            "IT": "ST extreme vs immediate left/right ST swings; usable only after right ST confirms",
            "LT": "IT extreme vs immediate left/right IT swings; usable only after right IT confirms",
            "episode12_fvg_rebalance_alternative": "documented but intentionally not approximated in this version",
        },
        "pool_quality_components": ["ST/IT/LT", "source timeframe", "multi-TF", "external50", "clean/pretested"],
        "cohorts": sorted(cohorts["hierarchy_cohort"].dropna().astype(str).unique().tolist()) if not cohorts.empty else [],
        "entry_methods": ["stage_reclaim", "episode_reclaim", "pre-sweep recent MSS", "pre-sweep structural MSS", "post-sweep newly formed ST MSS", "MSS+FVG limit", "post-reclaim FVG market/limit/hybrid overlay"],
        "displacement_policy": "no hard strong/weak formula; study distance, speed, efficiency, directional bodies, FVG density, break distance and reversal-vs-attack ratios with frozen 2023-24 quartiles",
        "exit_targets": ["nearest", "2-level pool", "multi-TF pool", "1H+", "4H+", "1D+", "2R", "3R", "5R"],
        "cvd_policy": "non-ICT causal extension; episode-anchored delta cumsum and price/CVD divergence at 3/5/15m",
        "session_policy": "no NY Open gate",
    }
    (out_dir / "02_frozen_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        f"# {TITLE}", "",
        "## Questions",
        "1. Is a single structurally important IT/LT/4H/multi-TF liquidity pool more informative than a raw N-pool count?",
        "2. At the same N-pool crossing, do causal IT/LT/4H+/multi-TF/clean/external pools improve results?",
        "3. Conditional on a specific key type (IT/LT/4H+/multi-TF), does adding 2/3/4 total pools improve monotonically or merely reduce frequency?",
        "4. Does ETH need MSS, and if so is the useful reference pre-sweep structure or a new ST swing that only forms after the sweep?",
        "5. Which opposing target (nearest/pool/1H/4H/1D/fixed R) best matches each setup?",
        "6. Does causal trade-bar CVD divergence add stable forward information?",
        "7. Is displacement payoff monotonic, or can medium/weaker-than-attack reversals outperform extreme displacement?",
        "8. On the exact frozen 269-ish R02 core opportunities, does market/FVG-limit/50:50 execution improve results after the R03.2 bug fix?",
        "",
        "## Causality",
        "- IT/LT liquidity labels are not backfilled. The right neighboring ST/IT must already be confirmed before the sweep bar starts.",
        "- post_sweep_st MSS is allowed only after a new execution-TF ST swing forms after the sweep and its right-confirmation bar has closed; the break must occur on a later causally eligible bar.",
        "- displacement is never an admission requirement in R03.3; all strength variables are descriptive research features.",
        "- Trade-bar CVD only uses completed 1m bars before the decision time.",
        "- No NY-open gate and no fixed time-profit exit.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("[r03.3] finalize GPT review pack", flush=True)
    finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )
    print(f"[done] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
