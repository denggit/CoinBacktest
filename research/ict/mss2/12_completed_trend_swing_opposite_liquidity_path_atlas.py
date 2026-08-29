#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R12 completed-trend Swing Sweep -> Opposite Liquidity Path Atlas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.r12 import (  # noqa: E402
    R12Config,
    build_completed_trend_physical_liquidity,
    build_opposite_liquidity_paths,
    build_root_sweep_events,
    prepare_completed_trend_contexts,
    r12_causal_audit,
    summarize_landmark_uplift,
    summarize_path_outcomes,
    summarize_root_taxonomy,
    summarize_success_failure_features,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "12.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_COMPLETED_TREND_SWING_OPPOSITE_LIQUIDITY_PATH_R12"
EDGE_ID = "RESEARCH_ONLY_COMPLETED_TREND_LIQUIDITY_PATH"
TITLE = "ETH ICT MSS2 R12 Completed-Trend Swing Sweep to Opposite Liquidity Path Atlas"
DEFAULT_R08_DIR = "data/reports/research/ict/mss2/r08_1_full_trend_ict_structure_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r12_completed_trend_swing_opposite_liquidity_path_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r08-dir", default=DEFAULT_R08_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--region-cluster-bps", type=float, default=10.0)
    p.add_argument("--path-horizon-days", type=int, default=30)
    p.add_argument("--landmark-max-minutes", type=int, default=360)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _load_r08(path: Path, end: pd.Timestamp):
    mp = path / "00_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"R08.1 manifest missing: {mp}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    covered = pd.Timestamp(manifest.get("research_end_date"))
    if covered < end:
        raise RuntimeError(f"R08.1 only covers through {covered}; rerun through {end}")
    native = pd.read_csv(path / "05_trend_qualified_key_liquidity.csv.gz")
    nested_path = path / "05b_nested_lower_tf_liquidity.csv.gz"
    nested = pd.read_csv(nested_path) if nested_path.exists() else pd.DataFrame()
    return native, nested, manifest


def _monthly(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    q = paths.copy()
    q["month"] = pd.to_datetime(q["root_sweep_time"], errors="coerce").dt.to_period("M").astype(str)
    rows = []
    for (month, side), p in q.groupby(["month", "root_side"], sort=True):
        rows.append({
            "month": month, "root_side": side, "events": len(p),
            "direct_opposite_delivery": int(p["path_outcome"].eq("direct_opposite_delivery").sum()),
            "cascade_then_opposite_delivery": int(p["path_outcome"].eq("cascade_then_opposite_delivery").sum()),
            "same_side_continuation": int(p["path_outcome"].eq("same_side_continuation_no_opposite_hit").sum()),
        })
    return pd.DataFrame(rows)


def _manual(out: Path, paths: pd.DataFrame) -> None:
    d = out / "manual_review"; d.mkdir(parents=True, exist_ok=True)
    if paths.empty:
        return
    x = paths.sort_values("root_sweep_time", kind="stable")
    x.loc[x["path_outcome"].eq("direct_opposite_delivery")].tail(20).to_csv(d/"01_recent_20_direct_opposite_delivery.csv", index=False, encoding="utf-8-sig")
    x.loc[x["path_outcome"].eq("same_side_continuation_no_opposite_hit")].tail(20).to_csv(d/"02_recent_20_same_side_continuation_failures.csv", index=False, encoding="utf-8-sig")
    x.loc[x["path_outcome"].eq("cascade_then_opposite_delivery")].tail(20).to_csv(d/"03_recent_20_cascade_then_opposite_delivery.csv", index=False, encoding="utf-8-sig")
    keep = [c for c in [
        "root_event_id","root_sweep_time","root_side","root_zone_low","root_zone_high","root_swing_ids",
        "root_max_swing_tf_min","root_lt_count","root_oldest_age_days","root_max_known_trend_tf_min",
        "root_max_known_trend_move_pct","root_native_context_any","root_nested_context_any","root_sweep_depth_bps",
        "root_rejection_wick_share","root_same_bar_full_reclaim_flag","opposite_1_zone_low","opposite_1_zone_high",
        "opposite_1_touch_price","opposite_1_roles","deeper_same_side_zone_low","deeper_same_side_zone_high",
        "deeper_same_side_touch_price","path_outcome","opposite_1_touch_time","deeper_same_side_touch_time",
        "reclaim_available_time","post_sweep_st_mss_1m_available_time","post_sweep_st_mss_2m_available_time",
        "post_sweep_st_mss_5m_available_time","first_directional_fvg_available_time"
    ] if c in x.columns]
    x.loc[:,keep].tail(60).to_csv(d/"04_recent_60_compact_chart_check.csv",index=False,encoding="utf-8-sig")
    (d/"README.md").write_text(
        "# R12 manual review\n\n"
        "Compare direct opposite-delivery successes with same-side continuation failures. Every root is a physical completed-trend IT/LT first sweep. "
        "The opposite target and deeper same-side competing liquidity are frozen when the sweep bar closes; the first-passage race begins on the next 1m bar. "
        "Reclaim/MSS/FVG are diagnostic landmarks only, not strategy filters.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start_date); end = pd.Timestamp(args.end_date)
    cfg = R12Config(
        region_cluster_bps=float(args.region_cluster_bps),
        path_horizon_minutes=int(args.path_horizon_days)*24*60,
        landmark_max_minutes=int(args.landmark_max_minutes),
    ).validate()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("[r12] load R08.1 completed-trend swing contexts", flush=True)
    native, nested, r08_manifest = _load_r08(Path(args.r08_dir), end)
    contexts = prepare_completed_trend_contexts(native, nested)
    print("[r12] load bare 1m K", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    print("[r12] physical completed-trend IT/LT lifecycle", flush=True)
    physical = build_completed_trend_physical_liquidity(bars, contexts)
    print("[r12] first physical swing sweeps", flush=True)
    roots = build_root_sweep_events(bars,physical,contexts,research_start=start,research_end=end,config=cfg)
    print("[r12] opposite-liquidity vs deeper-same-side first-passage paths", flush=True)
    paths = build_opposite_liquidity_paths(bars,physical,contexts,roots,config=cfg,progress=not args.no_progress)

    outcome = summarize_path_outcomes(paths)
    taxonomy = summarize_root_taxonomy(paths)
    features = summarize_success_failure_features(paths)
    landmarks = summarize_landmark_uplift(paths)
    monthly = _monthly(paths)
    audit = r12_causal_audit(contexts,physical,roots,paths)

    manifest = {
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"title":TITLE,
        "symbol":args.symbol,"warmup_start_date":args.warmup_start_date,"research_start_date":args.start_date,"research_end_date":args.end_date,
        "r08_report":args.r08_dir,"r08_manifest":r08_manifest,
        "market_semantics":"ETH continuous 24/7; date reporting only",
        "liquidity_universe":"physical ITH/ITL/LTH/LTL active after a causally completed R08.1 native or nested-lower-TF trend context; ST and invalid higher-TF projections excluded",
        "physical_dedup":"one swing_id once across contexts",
        "primary_path_test":"nearest frozen opposite completed-trend liquidity vs nearest deeper same-side completed-trend liquidity, starting next 1m bar",
        "path_horizon_days":args.path_horizon_days,"region_cluster_bps":cfg.region_cluster_bps,
        "strategy_status":"path atlas only; no entry/SL/TP promoted",
    }
    (out/"00_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    contexts.to_csv(out/"01_completed_trend_liquidity_contexts.csv.gz",index=False,compression="gzip")
    physical.to_csv(out/"02_physical_completed_trend_liquidity_lifecycle.csv.gz",index=False,compression="gzip")
    roots.to_csv(out/"03_root_completed_trend_swing_sweeps.csv.gz",index=False,compression="gzip")
    paths.to_csv(out/"04_opposite_liquidity_path_rows.csv.gz",index=False,compression="gzip")
    outcome.to_csv(out/"05_path_outcome_summary.csv",index=False)
    taxonomy.to_csv(out/"06_root_taxonomy_outcome_summary.csv",index=False)
    features.to_csv(out/"07_success_vs_failure_feature_summary.csv",index=False)
    landmarks.to_csv(out/"08_confirmation_landmark_uplift.csv",index=False)
    monthly.to_csv(out/"09_monthly_path_counts.csv",index=False)
    audit.to_csv(out/"10_causal_audit.csv",index=False)
    pd.DataFrame([
        {"check":"valid_completed_trend_context_rows","value":len(contexts)},
        {"check":"unique_physical_completed_trend_levels","value":len(physical)},
        {"check":"research_root_sweep_events","value":len(roots)},
        {"check":"path_rows","value":len(paths)},
        {"check":"causal_audit_violations","value":int(pd.to_numeric(audit.get("violations"),errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]).to_csv(out/"11_engineering_audit.csv",index=False)
    (out/"R12_RESEARCH_NOTES.md").write_text(
        "# R12 research notes\n\nR12 replaces the broad R11/R11.1 map framing with the requested path-first study: completed-trend historical Swing liquidity only. "
        "The goal is to compare paths that reverse to frozen opposite liquidity against paths that continue through deeper same-side liquidity, then study causal differentiating features.\n",
        encoding="utf-8",
    )
    _manual(out,paths)
    finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(f"[r12] done -> {out}",flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
