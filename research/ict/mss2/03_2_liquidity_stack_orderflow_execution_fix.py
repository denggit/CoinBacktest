#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.2 - corrected microstructure checkpoint grain + frozen-core execution study.

R03.2 repairs two issues found after reviewing the real R03 report:

1. ``core_ge4`` and ``expand_ge3`` are episode-related cohorts but are not the
   same concrete checkpoint rows.  Trade-bar and footprint features are now
   extracted for the exact union of both checkpoint-ID sets and row-attachment
   coverage is hard-audited.
2. The old first-FVG overlay changed the signal.  R03.2 instead freezes the
   profitable R02 opportunity definition (10bp >=4 pools, Long, 5m episode
   reclaim, structural stop, opposing 4H liquidity target) and changes only
   execution after that same reclaim decision.

No NY-open gate. No fixed time-profit exit.  Seven days remains right-censoring.
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
    R03Config,
    attach_footprint_microstructure_features,
    build_core_reclaim_execution_overlays,
    build_microstructure_checkpoint_union,
    build_tradebar_microstructure_features,
    first_pool_threshold_crossing_trades,
    microstructure_feature_join_audit,
    r03_causal_audit,
    r03_globalize_legacy_trade_ids,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "3.2.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_STACK_ORDERFLOW_EXECUTION_R03_2"
EDGE_ID = "RESEARCH_ONLY_ETH_HTF_MULTI_LIQUIDITY_STACK_EXHAUSTION_LONG"
TITLE = "ETH ICT MSS2 R03.2 Corrected Order-Flow + Frozen-Core Execution"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r03_2_liquidity_stack_orderflow_execution_fix"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tradebar-db-name", default="okx_trade_bars.db")
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--footprint-db-name", default="okx_range_footprints.db")
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-footprint", action="store_true")
    p.add_argument("--skip-execution-overlay", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--limit-roundtrip-cost", type=float, default=0.0009)
    p.add_argument("--fvg-signal-wait-minutes", type=int, default=180)
    p.add_argument("--fvg-limit-wait-minutes", type=int, default=180)
    p.add_argument("--footprint-chunk-days", type=int, default=120)
    return p.parse_args(argv)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(x.loc[x > 0].sum())
    losses = float(-x.loc[x < 0].sum())
    if losses <= 1e-12:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _metric(frame: pd.DataFrame, net_col: str) -> dict[str, object]:
    row: dict[str, object] = {"trades_total": int(len(frame))}
    if frame.empty or net_col not in frame.columns:
        return row
    x0 = pd.to_numeric(frame[net_col], errors="coerce")
    x = x0.dropna()
    row.update({
        "resolved": int(len(x)),
        "censored_or_missing": int(len(frame) - len(x)),
        "mean_net": float(x.mean()) if len(x) else np.nan,
        "median_net": float(x.median()) if len(x) else np.nan,
        "win_rate": float((x > 0).mean()) if len(x) else np.nan,
        "pf": _pf(x),
        "sum_net": float(x.sum()) if len(x) else np.nan,
    })
    return row


def _group_metric(frame: pd.DataFrame, group_cols: Sequence[str], net_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in frame.columns]
    grouped = [((), frame)] if not cols else frame.groupby(cols, dropna=False, observed=False, sort=True)
    rows: list[dict[str, object]] = []
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(cols, keys)}
        row.update(_metric(part, net_col))
        rows.append(row)
    return pd.DataFrame(rows)


def _read_manifest(r02_dir: Path) -> dict[str, object]:
    path = r02_dir / "00_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_r02_report(report_dir: Path) -> pd.DataFrame:
    feature_path = report_dir / "10_trade_features_causal.csv"
    label_path = report_dir / "11_trade_structural_exit_labels.csv"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"R03.2 requires completed R02 report under {report_dir}")
    feature_cols = [
        "trade_event_id", "stage_id", "episode_id", "trade_direction", "execution_minutes", "trigger_type",
        "sweep_pos_1m", "sweep_bar_time_1m", "episode_start_pos_1m", "episode_start_time_1m",
        "episode_elapsed_minutes", "levels_consumed_cum", "distinct_timeframes_cum",
        "max_source_timeframe_min_cum", "htf_240m_plus_levels_cum", "htf_1440m_plus_levels_cum",
        "episode_consumption_depth_bp", "levels_consumed_per_min_cum", "price_pools_5p0bp_cum",
        "pools_per_min_5p0bp_cum", "price_pools_10p0bp_cum", "pools_per_min_10p0bp_cum",
        "price_pools_20p0bp_cum", "pools_per_min_20p0bp_cum", "signal_available_time", "signal_bar_time",
        "entry_pos_1m", "entry_time", "entry_price", "entry_kind", "stop_price", "risk_bps",
        "session_primary", "is_weekend_utc", "year", "quarter", "month",
    ]
    features = pd.read_csv(feature_path, usecols=lambda c: c in set(feature_cols), low_memory=False)
    label_cols = [
        "trade_event_id", "stage_id", "episode_id",
        "target_htf240_price", "target_htf240_outcome", "target_htf240_exit_pos",
        "target_htf240_holding_minutes", "target_htf240_gross_return",
        "target_htf240_net_return_base", "target_htf240_net_return_cost2x", "target_htf240_net_return_cost3x",
    ]
    labels = pd.read_csv(label_path, usecols=lambda c: c in set(label_cols), low_memory=False)
    if len(features) != len(labels):
        raise RuntimeError("R02 feature/label row counts differ")
    features, labels = r03_globalize_legacy_trade_ids(features, labels)
    assert labels is not None
    if not np.array_equal(features["trade_event_id"].astype(str).to_numpy(), labels["trade_event_id"].astype(str).to_numpy()):
        raise RuntimeError("R02 feature/label rows are not aligned after ID repair")
    merged = pd.concat(
        [features.reset_index(drop=True), labels.drop(columns=["trade_event_id", "stage_id", "episode_id"]).reset_index(drop=True)],
        axis=1,
    )
    for name in ("sweep_bar_time_1m", "episode_start_time_1m", "signal_available_time", "signal_bar_time", "entry_time"):
        if name in merged.columns:
            merged[name] = pd.to_datetime(merged[name], errors="coerce")
    return merged


def _candidate_sets(r02: pd.DataFrame, cfg: R03Config) -> pd.DataFrame:
    frames = []
    for threshold, name in ((cfg.pool_threshold_expand, "expand_ge3"), (cfg.pool_threshold_core, "core_ge4")):
        part = first_pool_threshold_crossing_trades(
            r02, threshold=threshold, tolerance_bps=cfg.pool_tolerance_bps,
            direction=1, trigger_type="episode_reclaim", execution_minutes=(1, 2, 5),
        )
        part["cohort"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _split_name(ts: pd.Series) -> pd.Series:
    x = pd.to_datetime(ts, errors="coerce")
    out = pd.Series("forward_2025_2026", index=x.index, dtype="string")
    out.loc[x < pd.Timestamp("2025-01-01")] = "train_2023_2024"
    out.loc[x >= pd.Timestamp("2025-10-01")] = "late_2025Q4_2026H1"
    return out


def _candidate_performance(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = "target_htf240_net_return_cost2x"
    overall = _group_metric(candidates, ["cohort", "execution_minutes"], target)
    yearly = _group_metric(candidates, ["cohort", "execution_minutes", "year"], target)
    htf = candidates.copy()
    htf["stack_has_4h_plus"] = pd.to_numeric(htf["max_source_timeframe_min_cum"], errors="coerce").ge(240)
    htf_summary = _group_metric(htf, ["cohort", "execution_minutes", "stack_has_4h_plus"], target)
    return overall, yearly, htf_summary


def _tradebar_summaries(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = joined.copy()
    frame["split"] = _split_name(frame["entry_time"])
    net = "target_htf240_net_return_cost2x"
    rows = []
    flags = {
        "all": pd.Series(True, index=frame.index),
        "tb_absorption": frame.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1),
        "tb_flow_recovery": frame.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1),
    }
    flags["tb_absorption_and_recovery"] = flags["tb_absorption"] & flags["tb_flow_recovery"]
    for name, mask in flags.items():
        summ = _group_metric(frame.loc[mask], ["cohort", "execution_minutes", "split"], net)
        if not summ.empty:
            summ.insert(0, "mechanism", name)
            rows.append(summ)
    mechanism = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()

    freq_rows = []
    for minutes in (1, 2, 5):
        core = frame.loc[frame["cohort"].eq("core_ge4") & pd.to_numeric(frame["execution_minutes"], errors="coerce").eq(minutes)]
        broad = frame.loc[frame["cohort"].eq("expand_ge3") & pd.to_numeric(frame["execution_minutes"], errors="coerce").eq(minutes)]
        selections = {
            "core_ge4_all": core,
            "core_ge4_tb_absorption": core.loc[core.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)],
            "core_ge4_tb_flow_recovery": core.loc[core.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1)],
            "expand_ge3_all": broad,
            "expand_ge3_tb_absorption": broad.loc[broad.get("tb_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)],
            "expand_ge3_tb_flow_recovery": broad.loc[broad.get("tb_flow_recovery_flag", 0).fillna(0).astype(int).eq(1)],
        }
        for name, part in selections.items():
            row = {"execution_minutes": minutes, "selection": name}
            row.update(_metric(part, net))
            freq_rows.append(row)
    return mechanism, pd.DataFrame(freq_rows)


def _footprint_summaries(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = joined.copy()
    frame["split"] = _split_name(frame["entry_time"])
    valid = frame.get("fp_causal_valid", pd.Series(False, index=frame.index, dtype="boolean")).astype("boolean").fillna(False).astype(bool)
    coverage = (
        frame.assign(fp_valid=valid)
        .groupby(["cohort", "execution_minutes", "split"], dropna=False, observed=False, sort=True)
        .agg(events=("trade_event_id", "size"), attached_rows=("checkpoint_id", lambda s: s.notna().sum()), covered=("fp_valid", "sum"), coverage=("fp_valid", "mean"))
        .reset_index()
    )
    matched = frame.loc[valid].copy()
    net = "target_htf240_net_return_cost2x"
    rows = []
    for name, mask in (
        ("covered_all", pd.Series(True, index=matched.index)),
        ("fp_absorption", matched.get("fp_absorption_mechanism_flag", 0).fillna(0).astype(int).eq(1)),
        ("fp_delta_recovery", matched.get("fp_delta_recovery_flag", 0).fillna(0).astype(int).eq(1)),
    ):
        summ = _group_metric(matched.loc[mask], ["cohort", "execution_minutes", "split"], net)
        if not summ.empty:
            summ.insert(0, "mechanism", name)
            rows.append(summ)
    return coverage, pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _position_alignment_audit(bars: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    idx = bars.index
    rows = []
    for pos_col, time_col, name in (
        ("sweep_pos_1m", "sweep_bar_time_1m", "core_sweep_pos_matches_time"),
        ("episode_start_pos_1m", "episode_start_time_1m", "core_episode_start_pos_matches_time"),
        ("entry_pos_1m", "entry_time", "core_entry_pos_matches_time"),
    ):
        pos = pd.to_numeric(core[pos_col], errors="coerce")
        ts = pd.to_datetime(core[time_col], errors="coerce")
        valid = pos.notna() & pos.ge(0) & ts.notna()
        bad = 0
        for p0, t0 in zip(pos.loc[valid].astype(int), ts.loc[valid]):
            if p0 < 0 or p0 >= len(idx) or pd.Timestamp(idx[p0]) != pd.Timestamp(t0):
                bad += 1
        rows.append({"check": name, "rows": int(valid.sum()), "violations": int(bad), "missing": int((~valid).sum())})
    return pd.DataFrame(rows)


def _execution_summary(overlay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if overlay.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for (minutes, variant), part in overlay.groupby(["fvg_minutes", "execution_variant"], dropna=False, observed=False, sort=True):
        row = {"fvg_minutes": int(minutes), "execution_variant": variant, "opportunities": int(part["base_trade_event_id"].nunique())}
        if variant == "post_reclaim_fvg_limit":
            row["fill_rate"] = float(pd.to_numeric(part["entry_fill_flag"], errors="coerce").fillna(0).mean())
        elif variant == "hybrid_reclaim_market_fvg_limit":
            row["fill_rate"] = float(pd.to_numeric(part["limit_filled_flag"], errors="coerce").fillna(0).mean())
        else:
            row["fill_rate"] = float(pd.to_numeric(part["entry_fill_flag"], errors="coerce").fillna(0).mean())
        row.update(_metric(part, "net_return_cost2x"))
        rows.append(row)
    overall = pd.DataFrame(rows)
    yearly = _group_metric(overlay, ["fvg_minutes", "execution_variant", "year"], "net_return_cost2x")
    return overall, yearly



def _execution_causal_audit(overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame([{"check": "execution_overlay_rows", "rows": 0, "violations": 0}])
    rows = []
    base_signal = pd.to_datetime(overlay["base_signal_available_time"], errors="coerce")
    fvg_signal = pd.to_datetime(overlay["fvg_signal_available_time"], errors="coerce")
    entry = pd.to_datetime(overlay["entry_time"], errors="coerce")
    filled = pd.to_numeric(overlay["entry_fill_flag"], errors="coerce").fillna(0).eq(1)
    fvg_variant = overlay["execution_variant"].astype(str).isin(["post_reclaim_fvg_market", "post_reclaim_fvg_limit"])
    has_fvg = fvg_signal.notna()
    rows.append({
        "check": "fvg_signal_not_before_reclaim_signal",
        "rows": int(has_fvg.sum()),
        "violations": int((has_fvg & fvg_signal.lt(base_signal)).sum()),
    })
    rows.append({
        "check": "filled_fvg_entry_not_before_fvg_signal",
        "rows": int((filled & fvg_variant).sum()),
        "violations": int((filled & fvg_variant & entry.lt(fvg_signal)).sum()),
    })
    target_nunique = overlay.groupby(["fvg_minutes", "base_trade_event_id"], dropna=False)["target_htf240_price"].nunique(dropna=False)
    stop_nunique = overlay.groupby(["fvg_minutes", "base_trade_event_id"], dropna=False)["stop_price"].nunique(dropna=False)
    rows.append({
        "check": "same_frozen_target_within_opportunity",
        "rows": int(len(target_nunique)),
        "violations": int((target_nunique.ne(1)).sum()),
    })
    rows.append({
        "check": "same_frozen_stop_within_opportunity",
        "rows": int(len(stop_nunique)),
        "violations": int((stop_nunique.ne(1)).sum()),
    })
    counts = overlay.groupby(["fvg_minutes", "base_trade_event_id", "execution_variant"], dropna=False).size()
    rows.append({
        "check": "one_row_per_opportunity_variant",
        "rows": int(len(counts)),
        "violations": int((counts.ne(1)).sum()),
    })
    return pd.DataFrame(rows)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R03Config(
        market_roundtrip_cost=float(args.market_roundtrip_cost),
        limit_roundtrip_cost=float(args.limit_roundtrip_cost),
        fvg_signal_wait_minutes=int(args.fvg_signal_wait_minutes),
        fvg_limit_wait_minutes=int(args.fvg_limit_wait_minutes),
        footprint_chunk_days=int(args.footprint_chunk_days),
    ).validate()
    progress = not bool(args.no_progress)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    r02_dir = PROJECT_ROOT / args.r02_report_dir
    manifest_r02 = _read_manifest(r02_dir)

    print("[r03.2] load R02 causal trade/outcome report", flush=True)
    r02 = _read_r02_report(r02_dir)
    candidates = _candidate_sets(r02, cfg)
    overall, yearly, htf = _candidate_performance(candidates)
    overall.to_csv(out_dir / "03_candidate_cohort_performance.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "04_candidate_year_performance.csv", index=False, encoding="utf-8-sig")
    htf.to_csv(out_dir / "05_candidate_htf_involvement.csv", index=False, encoding="utf-8-sig")

    checkpoints, union_audit = build_microstructure_checkpoint_union(candidates)
    union_audit.to_csv(out_dir / "06_checkpoint_union_audit.csv", index=False, encoding="utf-8-sig")
    checkpoint_frame = checkpoints[["checkpoint_id", "decision_time", "episode_start_time"]].copy()

    print("[r03.2] causal 1m trade-bar microstructure on >=3 U >=4 checkpoints", flush=True)
    tb_features, tb_audit = build_tradebar_microstructure_features(
        checkpoint_frame, symbol=args.symbol, data_dir=args.data_dir, db_name=args.tradebar_db_name,
        config=cfg, show_progress=progress,
    )
    tb_join_audit = microstructure_feature_join_audit(checkpoints, tb_features, module="tradebar")
    tb_features.to_csv(out_dir / "07_tradebar_features_causal.csv.gz", index=False, compression="gzip")
    tb_audit.to_csv(out_dir / "08_tradebar_build_audit.csv", index=False, encoding="utf-8-sig")
    tb_join_audit.to_csv(out_dir / "09_tradebar_join_coverage_audit.csv", index=False, encoding="utf-8-sig")
    if not bool(tb_join_audit["passed"].astype(int).eq(1).all()):
        raise RuntimeError("R03.2 trade-bar checkpoint attachment is not one-to-one/100%; see 09_tradebar_join_coverage_audit.csv")
    tb_join = candidates.merge(tb_features, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="many_to_one")
    if tb_join["checkpoint_id"].isna().any():
        raise RuntimeError("R03.2 trade-bar join unexpectedly lost candidate rows")
    tb_mech, tb_freq = _tradebar_summaries(tb_join)
    tb_mech.to_csv(out_dir / "10_tradebar_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    tb_freq.to_csv(out_dir / "11_tradebar_frequency_recovery.csv", index=False, encoding="utf-8-sig")

    fp_features = pd.DataFrame()
    fp_audit = pd.DataFrame()
    fp_join_audit = pd.DataFrame()
    if not args.skip_footprint:
        print("[r03.2] causal r0020/step1 footprint on same checkpoint union", flush=True)
        fp_features, fp_audit = attach_footprint_microstructure_features(
            checkpoint_frame[["checkpoint_id", "decision_time"]], symbol=args.symbol, data_dir=args.data_dir,
            range_db_name=args.range_db_name, footprint_db_name=args.footprint_db_name,
            config=cfg, show_progress=progress,
        )
        fp_join_audit = microstructure_feature_join_audit(checkpoints, fp_features, module="footprint")
        fp_features.to_csv(out_dir / "12_footprint_features_causal.csv.gz", index=False, compression="gzip")
        fp_audit.to_csv(out_dir / "13_footprint_build_audit.csv", index=False, encoding="utf-8-sig")
        fp_join_audit.to_csv(out_dir / "14_footprint_join_coverage_audit.csv", index=False, encoding="utf-8-sig")
        if not bool(fp_join_audit["passed"].astype(int).eq(1).all()):
            raise RuntimeError("R03.2 footprint checkpoint row attachment is not one-to-one/100%; see 14_footprint_join_coverage_audit.csv")
        fp_join = candidates.merge(fp_features, left_on="trade_event_id", right_on="checkpoint_id", how="left", validate="many_to_one")
        if fp_join["checkpoint_id"].isna().any():
            raise RuntimeError("R03.2 footprint join unexpectedly lost candidate rows")
        fp_cov, fp_mech = _footprint_summaries(fp_join)
        fp_cov.to_csv(out_dir / "15_footprint_coverage.csv", index=False, encoding="utf-8-sig")
        fp_mech.to_csv(out_dir / "16_footprint_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"status": "disabled_by_cli"}]).to_csv(out_dir / "15_footprint_coverage.csv", index=False, encoding="utf-8-sig")

    overlay = pd.DataFrame()
    tieout = pd.DataFrame()
    pos_audit = pd.DataFrame()
    core5 = candidates.loc[
        candidates["cohort"].eq("core_ge4")
        & pd.to_numeric(candidates["execution_minutes"], errors="coerce").eq(cfg.baseline_execution_minutes)
    ].copy()
    core5 = core5.drop_duplicates("trade_event_id", keep="first")
    if not args.skip_execution_overlay:
        print(f"[r03.2] frozen 5m reclaim core execution overlay opportunities={len(core5)}", flush=True)
        warmup_start = str(manifest_r02.get("warmup_start_date") or args.warmup_start_date)
        manifest_symbol = str(manifest_r02.get("symbol") or args.symbol)
        if manifest_symbol != str(args.symbol):
            raise RuntimeError(f"R02/R03.2 symbol mismatch: {manifest_symbol!r} vs {args.symbol!r}")
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        bars = loader.fetch_data_by_date_range(warmup_start, args.end_date)
        if bars.empty:
            raise RuntimeError("no naked 1m K data for R03.2 execution overlay")
        pos_audit = _position_alignment_audit(bars, core5)
        pos_audit.to_csv(out_dir / "17_execution_position_alignment_audit.csv", index=False, encoding="utf-8-sig")
        if int(pd.to_numeric(pos_audit["violations"], errors="coerce").fillna(0).sum()) > 0:
            raise RuntimeError("R03.2 core position/time alignment failed")

        overlays = []
        ties = []
        for fvg_minutes in cfg.fvg_execution_minutes:
            part, tie = build_core_reclaim_execution_overlays(
                bars, core5, fvg_minutes=int(fvg_minutes), config=cfg, show_progress=progress,
            )
            overlays.append(part)
            ties.append(tie)
        overlay = pd.concat(overlays, ignore_index=True, sort=False) if overlays else pd.DataFrame()
        tieout = pd.concat(ties, ignore_index=True, sort=False) if ties else pd.DataFrame()
        tieout.to_csv(out_dir / "18_r02_reclaim_baseline_tieout.csv", index=False, encoding="utf-8-sig")
        tie_bad = int((pd.to_numeric(tieout.get("outcome_match"), errors="coerce").fillna(0).ne(1) | pd.to_numeric(tieout.get("gross_match"), errors="coerce").fillna(0).ne(1)).sum()) if len(tieout) else 0
        if tie_bad:
            raise RuntimeError(f"R03.2 baseline recomputation does not tie to R02 for {tie_bad} rows; refusing execution comparison")
        overlay.to_csv(out_dir / "19_frozen_core_execution_overlay_trades.csv.gz", index=False, compression="gzip")
        ex_overall, ex_year = _execution_summary(overlay)
        ex_overall.to_csv(out_dir / "20_frozen_core_execution_summary.csv", index=False, encoding="utf-8-sig")
        ex_year.to_csv(out_dir / "21_frozen_core_execution_year_summary.csv", index=False, encoding="utf-8-sig")

    causal = r03_causal_audit(tb_features, fp_features, pd.DataFrame())
    execution_causal = _execution_causal_audit(overlay)
    causal = pd.concat([causal, execution_causal], ignore_index=True, sort=False)
    causal.to_csv(out_dir / "22_causal_audit.csv", index=False, encoding="utf-8-sig")
    execution_bad = int(pd.to_numeric(execution_causal.get("violations"), errors="coerce").fillna(0).sum()) if len(execution_causal) else 0
    if execution_bad:
        raise RuntimeError(f"R03.2 execution causal audit failed: {execution_bad} violations")
    engineering_rows = [
        {"check": "r02_trade_event_id_unique_after_repair", "passed": int(not r02["trade_event_id"].duplicated().any()), "value": int(r02["trade_event_id"].nunique())},
        {"check": "candidate_trade_event_id_unique_within_cohort", "passed": int(not candidates.duplicated(["cohort", "trade_event_id"]).any()), "value": int(len(candidates))},
        {"check": "microstructure_union_not_assumed_subset", "passed": 1, "value": "expand_ge3 U core_ge4 concrete trade IDs"},
        {"check": "tradebar_row_attachment_100pct", "passed": int(tb_join_audit["passed"].astype(int).eq(1).all()), "value": int(len(tb_features))},
        {"check": "footprint_row_attachment_100pct", "passed": int(args.skip_footprint or fp_join_audit["passed"].astype(int).eq(1).all()), "value": int(len(fp_features))},
        {"check": "footprint_missing_cache_not_zero_signal", "passed": 1, "value": "fp_causal_valid controls covered subset"},
        {"check": "execution_uses_same_frozen_r02_signal_stop_target", "passed": 1, "value": "5m episode_reclaim + stored stop_price + stored target_htf240_price"},
        {"check": "r02_reclaim_baseline_tieout", "passed": int(args.skip_execution_overlay or len(tieout) > 0 and tieout[["outcome_match", "gross_match"]].astype(int).eq(1).all().all()), "value": int(len(tieout))},
        {"check": "ny_open_used_as_gate", "passed": 1, "value": "NO"},
        {"check": "time_profit_exit_used", "passed": 1, "value": "NO; 7d is censor only"},
    ]
    engineering = pd.DataFrame(engineering_rows)
    engineering.to_csv(out_dir / "01_engineering_audit.csv", index=False, encoding="utf-8-sig")

    design = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "config": asdict(cfg),
        "corrected_microstructure_grain": "exact union of concrete first>=3 and first>=4 R02 trade checkpoints",
        "frozen_core": "Long + 10bp first >=4 pool crossing + 5m episode_reclaim + R02 structural stop + R02 opposing 4H target",
        "execution_variants": [
            "reclaim_market",
            "post_reclaim_fvg_market",
            "post_reclaim_fvg_limit",
            "hybrid_reclaim_market_fvg_limit",
        ],
        "execution_fvg_timeframes": list(cfg.fvg_execution_minutes),
        "target_policy": "never re-select after FVG; use exact target_htf240_price frozen by R02",
        "censor_policy": "same absolute 7d opportunity horizon from original reclaim entry",
        "session_policy": "no NY-open gate",
    }
    (out_dir / "02_frozen_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        f"# {TITLE}", "",
        "R03.2 is a correction, not a new threshold search.", "",
        "## Hard corrections",
        "- Microstructure is extracted for the exact union of >=3 and >=4 concrete trade checkpoints.",
        "- Feature row attachment must be one-to-one and 100%; footprint data validity can still be partial via fp_causal_valid.",
        "- FVG execution is evaluated only after the frozen profitable 5m episode-reclaim core signal.",
        "- Structural stop and opposing 4H target are copied from R02 and never re-selected later.",
        "- Original reclaim-market outcomes are recomputed from naked 1m K and must tie exactly to R02 before overlays are trusted.",
        "- No NY Open gate and no fixed time-profit exit.", "",
        "## What to judge",
        "1. After the corrected join, does core_ge4 Trade Bar / Footprint add stable information across train/forward splits?",
        "2. Does any fixed >=3 microstructure mechanism recover frequency without falling below PF 1 after 2x costs?",
        "3. On the exact same 5m reclaim opportunities, does FVG market, FVG limit, or 50:50 improve 2x-cost expectancy/PF without losing yearly robustness?",
        "4. Treat FVG as execution only; do not promote first-FVG-alone as a new signal.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "r02_rows": int(len(r02)),
        "candidate_rows": int(len(candidates)),
        "checkpoint_union_rows": int(len(checkpoints)),
        "tradebar_rows": int(len(tb_features)),
        "footprint_rows": int(len(fp_features)),
        "core5_opportunities": int(len(core5)),
        "execution_overlay_rows": int(len(overlay)),
        "baseline_tieout_rows": int(len(tieout)),
        "causal_violations": int(pd.to_numeric(causal.get("violations"), errors="coerce").fillna(0).sum()) if len(causal) else 0,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] R03.2 report -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
