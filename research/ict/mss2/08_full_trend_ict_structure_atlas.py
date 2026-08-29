#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R08: full-trend ICT structure atlas before any new strategy research.

Goal
----
Validate that CoinBacktest identifies ICT market structure the way a human
would before those levels are reused as liquidity.  This version does NOT
optimize a trading strategy.

Key rules
---------
* Classical recursive hierarchy: ST -> IT -> LT on each chart timeframe.
* ST swings are construction-only and never become trend-qualified liquidity.
* A large historical trend is evaluated from one LT anchor to the opposite LT
  anchor, not from an arbitrary rolling-window high/low.
* Internal ITH/ITL sequences must progress monotonically with the trend.
* The trend is not activated for future liquidity research until a post-terminal
  IT-level close-through BOS confirms that the prior leg ended.
* 3% / 5% / 7% are research-scale labels only; they do not redefine ICT swings.
"""
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
from src.research_common.ict_mss2.r08 import (  # noqa: E402
    R08Config,
    build_bos_events,
    build_completed_trend_legs,
    build_multi_timeframe_hierarchy,
    build_trend_qualified_liquidity,
    build_projection_impact_atlas,
    r08_causal_audit,
    summarize_hierarchy,
    summarize_key_liquidity,
    summarize_projection_impact,
    summarize_trend_scales,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "8.1.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_FULL_TREND_STRUCTURE_ATLAS_R08_1"
EDGE_ID = "RESEARCH_ONLY_ICT_STRUCTURE_FOUNDATION"
TITLE = "ETH ICT MSS2 R08.1 Full-Trend ICT Structure Atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r08_1_full_trend_ict_structure_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--structure-timeframes", default="15m,30m,1H,4H")
    p.add_argument("--trend-scales-pct", default="3,5,7")
    p.add_argument("--min-it-swings-per-side", type=int, default=2)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_timeframes(text: str) -> tuple[tuple[str, int], ...]:
    mapping = {"1m":1, "2m":2, "5m":5, "15m":15, "30m":30, "1H":60, "4H":240}
    out=[]
    for raw in str(text).split(','):
        tf=raw.strip()
        if tf not in mapping:
            raise ValueError(f"unsupported structure timeframe: {tf}")
        out.append((tf,mapping[tf]))
    return tuple(out)


def _parse_scales(text: str) -> tuple[float, ...]:
    vals=tuple(sorted(set(float(x.strip())/100.0 for x in str(text).split(',') if x.strip())))
    if not vals or any(x<=0 for x in vals):
        raise ValueError("trend scales must be positive percentages")
    return vals


def _manual_review(out: Path, legs: pd.DataFrame, liquidity: pd.DataFrame, bos: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    d=out/"manual_review"; d.mkdir(parents=True,exist_ok=True)
    l=legs.copy()
    if not l.empty:
        l["leg_available_time"]=pd.to_datetime(l["leg_available_time"],errors="coerce")
        l=l.loc[l["leg_available_time"].between(start,end,inclusive="both")]
        l=l.loc[l["trend_qualified_ge3_flag"].eq(1)].sort_values("leg_available_time",kind="stable").tail(30)
        keep=[c for c in [
            "trend_leg_id","source_timeframe","trend_direction","origin_class","origin_time","origin_price",
            "terminal_class","terminal_time","terminal_price","trend_move_pct","trend_duration_minutes",
            "it_high_count","it_low_count","ith_sequence","itl_sequence","ith_monotonic_flag","itl_monotonic_flag",
            "reversal_bos_structure","reversal_bos_reference_time","reversal_bos_reference_price",
            "reversal_bos_available_time","reversal_bos_close_price","leg_available_time",
            "scale_ge_03pct_flag","scale_ge_05pct_flag","scale_ge_07pct_flag"
        ] if c in l.columns]
        l.loc[:,keep].to_csv(d/"01_recent_30_completed_clean_trend_legs.csv",index=False,encoding="utf-8-sig")
        for tf in ("15m","1H","4H"):
            q=l.loc[l["source_timeframe"].eq(tf)]
            if not q.empty:
                q.loc[:,keep].tail(20).to_csv(d/f"01b_recent_20_{tf}_completed_trend_legs.csv",index=False,encoding="utf-8-sig")
    q=liquidity.copy()
    if not q.empty:
        q["liquidity_activation_time"]=pd.to_datetime(q["liquidity_activation_time"],errors="coerce")
        q=q.loc[q["liquidity_activation_time"].between(start,end,inclusive="both") & q["active_at_activation_flag"].eq(1)]
        q=q.sort_values("liquidity_activation_time",kind="stable").tail(60)
        keep=[c for c in [
            "trend_leg_id","source_timeframe","trend_direction","trend_move_pct","trend_origin_time","trend_terminal_time",
            "liquidity_side","swing_role","pivot_time","level_price","liquidity_activation_time",
            "first_sweep_after_activation_time","scale_ge_03pct_flag","scale_ge_05pct_flag","scale_ge_07pct_flag"
        ] if c in q.columns]
        q.loc[:,keep].to_csv(d/"02_recent_60_active_key_liquidity_levels.csv",index=False,encoding="utf-8-sig")
    b=bos.copy()
    if not b.empty:
        b["bos_available_time"]=pd.to_datetime(b["bos_available_time"],errors="coerce")
        b=b.loc[b["bos_available_time"].between(start,end,inclusive="both")].sort_values("bos_available_time",kind="stable").tail(40)
        b.to_csv(d/"03_recent_40_it_lt_bos_events.csv",index=False,encoding="utf-8-sig")
    (d/"README.md").write_text(
        "# R08 manual ICT structure review\n\n"
        "Start with `01_recent_30_completed_clean_trend_legs.csv`.  For each row, open the stated source timeframe and verify the full move from origin LT anchor to terminal LT anchor.  The `ith_sequence` and `itl_sequence` fields list every classical intermediate swing used inside the leg.\n\n"
        "Then inspect `02_recent_60_active_key_liquidity_levels.csv`: these are canonical native-timeframe IT/LT levels only. Nested lower-timeframe IT/LT is written separately to `05b_nested_lower_tf_liquidity.csv.gz` and is not mixed into this manual file. ST-only swings are intentionally absent. A level already consumed by the BOS/confirmation is excluded from this active file.\n\n"
        "R08 is a structure-validation study, not a trading-strategy report. Do not judge it by PF or returns.\n",
        encoding="utf-8",
    )


def _source_basis(out: Path) -> None:
    (out/"R08_ICT_SOURCE_BASIS.md").write_text(
        "# R08 ICT source basis\n\n"
        "Primary concept source used for the mechanical hierarchy: ICT 2022 Mentorship Episode 12 (Market Structure for Precision Technicians).\n\n"
        "Source transcript: https://pickscribe.com/v/8GkQfdAXZP0\n\n"
        "Cross-check transcript: https://info.quagmyre.com/xwiki/bin/view/Forex/The-Inner-Circle-Trader/ICT-Youtube-Series-2022/ICT-YT-2022-02-25-ICT-Mentorship-2022-Episode-12-srt/\n\n"
        "## Quantization boundary\n"
        "- Classical ITH/ITL: an ST swing that is more extreme than the immediately adjacent same-side ST swings.\n"
        "- Classical LTH/LTL: recursive mirror on the IT sequence.\n"
        "- Episode 12 also discusses imbalance-rebalance swings as intermediate-term structure. R08 intentionally does not merge that discretionary extension into the classical labels before manual chart validation.\n"
        "- The >=3% / >=5% / >=7% completed-trend scales are CoinBacktest research sensitivities, not canonical ICT thresholds.\n"
        "- R08's post-terminal IT-BOS requirement is an explicit mechanical implementation of the research requirement that the whole trend must be seen and its reversal confirmed before the historical leg is used to qualify future liquidity.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv)
    cfg=R08Config(
        structure_timeframes=_parse_timeframes(args.structure_timeframes),
        trend_scales=_parse_scales(args.trend_scales_pct),
        min_it_swings_per_side=int(args.min_it_swings_per_side),
    ).validate()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    progress=not args.no_progress
    print("[r08] load bare 1m K",flush=True)
    loader=OKXDataLoader(symbol=args.symbol,timeframe="1m",db_dir=args.data_dir)
    bars=loader.fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    start=pd.Timestamp(args.start_date); end=pd.Timestamp(args.end_date)
    print("[r08] classical ST -> IT -> LT hierarchy",flush=True)
    hierarchy,bar_map=build_multi_timeframe_hierarchy(bars,config=cfg,progress=progress)
    print("[r08] complete LT-to-LT trend legs + IT-BOS",flush=True)
    legs=build_completed_trend_legs(hierarchy,bar_map,config=cfg,progress=progress)
    print("[r08] IT/LT BOS atlas",flush=True)
    bos=build_bos_events(hierarchy,bar_map,progress=progress)
    print("[r08] classified trend-qualified historical liquidity",flush=True)
    liquidity_all=build_trend_qualified_liquidity(bars,hierarchy,legs,config=cfg,progress=progress)
    native=liquidity_all.loc[liquidity_all["projection_scope"].eq("native")].copy() if not liquidity_all.empty else pd.DataFrame()
    nested=liquidity_all.loc[liquidity_all["projection_scope"].eq("nested_lower_tf")].copy() if not liquidity_all.empty else pd.DataFrame()
    rejected=liquidity_all.loc[liquidity_all["projection_scope"].eq("invalid_higher_tf_projection")].copy() if not liquidity_all.empty else pd.DataFrame()
    print("[r08] projection impact audit",flush=True)
    impact=build_projection_impact_atlas(bars,liquidity_all,research_start=start,research_end=end)
    impact_summary=summarize_projection_impact(impact)

    hierarchy.to_csv(out/"01_classical_recursive_swing_hierarchy.csv.gz",index=False,compression="gzip")
    summarize_hierarchy(hierarchy).to_csv(out/"02_hierarchy_summary.csv",index=False)
    legs.to_csv(out/"03_completed_lt_to_lt_trend_legs.csv.gz",index=False,compression="gzip")
    summarize_trend_scales(legs,config=cfg).to_csv(out/"04_completed_trend_scale_summary.csv",index=False)
    native.to_csv(out/"05_trend_qualified_key_liquidity.csv.gz",index=False,compression="gzip")
    nested.to_csv(out/"05b_nested_lower_tf_liquidity.csv.gz",index=False,compression="gzip")
    rejected.to_csv(out/"05c_rejected_higher_tf_projection.csv.gz",index=False,compression="gzip")
    summarize_key_liquidity(liquidity_all).to_csv(out/"06_key_liquidity_summary.csv",index=False)
    impact.to_csv(out/"06b_projection_impact_rows.csv.gz",index=False,compression="gzip")
    impact_summary.to_csv(out/"06c_projection_impact_summary.csv",index=False)
    bos.to_csv(out/"07_it_lt_bos_events.csv.gz",index=False,compression="gzip")
    audit=r08_causal_audit(hierarchy,legs,liquidity_all)
    audit.to_csv(out/"08_causal_audit.csv",index=False)
    eng=pd.DataFrame([
        {"check":"bare_1m_rows","value":len(bars)},
        {"check":"classical_st_rows","value":len(hierarchy)},
        {"check":"classical_it_rows","value":int(hierarchy.get('is_it',pd.Series(dtype=int)).sum()) if len(hierarchy) else 0},
        {"check":"classical_lt_rows","value":int(hierarchy.get('is_lt',pd.Series(dtype=int)).sum()) if len(hierarchy) else 0},
        {"check":"completed_leg_candidates","value":len(legs)},
        {"check":"clean_completed_ge3_legs","value":int(legs.get('trend_qualified_ge3_flag',pd.Series(dtype=int)).sum()) if len(legs) else 0},
        {"check":"classified_liquidity_rows","value":len(liquidity_all)},
        {"check":"canonical_native_rows","value":len(native)},
        {"check":"nested_lower_tf_rows","value":len(nested)},
        {"check":"rejected_higher_tf_projection_rows","value":len(rejected)},
        {"check":"active_canonical_native_rows","value":int(native.get('active_at_activation_flag',pd.Series(dtype=int)).sum()) if len(native) else 0},
        {"check":"causal_audit_violations","value":int(pd.to_numeric(audit.get('violations'),errors='coerce').fillna(0).sum()) if len(audit) else 0},
    ])
    eng.to_csv(out/"09_engineering_audit.csv",index=False)
    manifest={
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"title":TITLE,
        "symbol":args.symbol,"warmup_start_date":args.warmup_start_date,"research_start_date":args.start_date,
        "research_end_date":args.end_date,"structure_timeframes":[tf for tf,_ in cfg.structure_timeframes],
        "trend_scales_pct":[x*100 for x in cfg.trend_scales],"min_it_swings_per_side":cfg.min_it_swings_per_side,
        "purpose":"Validate classical ICT large-structure hierarchy and separate native vs nested liquidity before any strategy reuse; ST swings are construction-only.",
        "imbalance_rebalance_it_status":"deliberately separate / not merged into classical labels in R08.1",
        "projection_fix":"canonical 05 file is native timeframe only; nested lower-TF kept separately; higher-TF into lower-trend rejected",
    }
    (out/"00_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    _source_basis(out)
    _manual_review(out,legs,native,bos,start,end)
    finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(f"[r08] done -> {out}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
