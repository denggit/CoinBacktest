#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R09: ICT liquidity quality ladder x execution atlas.

R09 consumes R08.1's corrected full-trend liquidity foundation and asks a
trading question without collapsing the universe to only A+ setups:

1. How many independent SSL/BSL root sweep opportunities exist across 15m/30m/
   1H/4H completed-trend contexts?
2. How does context quality change win-rate / expectancy / PF without hard
   filtering away the broad opportunity set?
3. On the exact same root opportunity, does sweep-immediate, reclaim, MSS, or
   confirmation+FVG-limit execution best convert directional edge into a
   tradable distribution?
4. Can later 15m multi-IT liquidity cascades diagnose structural breakdown
   without leaking that future information into the initial setup tier?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.core import MSS2Config  # noqa: E402
from src.research_common.ict_mss2.r02 import R02Config, build_stack_execution_triggers  # noqa: E402
from src.research_common.ict_mss2.r09 import (  # noqa: E402
    R09Config,
    attach_r09_outcomes,
    build_immediate_entries,
    build_physical_liquidity_sweeps,
    build_reclaim_fvg_limit_entries,
    build_root_sweep_episodes,
    r09_causal_audit,
    summarize_cascade_diagnostic,
    summarize_execution_grid,
    summarize_quality_ladder,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "9.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_LIQUIDITY_QUALITY_EXECUTION_R09"
EDGE_ID = "ICT_FULL_TREND_LIQUIDITY_QUALITY_EXECUTION"
TITLE = "ETH ICT MSS2 R09 Liquidity Quality x Execution Atlas"
DEFAULT_R08_DIR = "data/reports/research/ict/mss2/r08_1_full_trend_ict_structure_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r09_liquidity_quality_execution_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r08-dir", default=DEFAULT_R08_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--execution-minutes", default="1,2,5")
    p.add_argument("--episode-gap-minutes", type=int, default=15)
    p.add_argument("--confirmation-minutes", type=int, default=180)
    p.add_argument("--fvg-wait-minutes", type=int, default=180)
    p.add_argument("--stop-buffer-bps", type=float, default=2.0)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_ints(text: str) -> tuple[int, ...]:
    vals = tuple(sorted(set(int(x.strip()) for x in str(text).split(",") if x.strip())))
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("execution minutes must be positive")
    return vals


def _load_r08(r08_dir: Path, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = r08_dir / "00_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"R08.1 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    covered = pd.Timestamp(manifest.get("research_end_date"))
    if covered < end:
        raise RuntimeError(f"R08.1 only covers through {covered}; rerun R08.1 through {end} first")
    native = pd.read_csv(r08_dir / "05_trend_qualified_key_liquidity.csv.gz")
    nested = pd.read_csv(r08_dir / "05b_nested_lower_tf_liquidity.csv.gz")
    return native, nested, manifest


def _rewrite_trade_ids(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["trade_event_id"] = [f"{prefix}_{i+1:09d}" for i in range(len(out))]
    if "limit_variant" not in out.columns:
        out["limit_variant"] = np.where(out["entry_kind"].astype(str).eq("fvg_limit"), "proximal", "none")
    if "entry_source" not in out.columns:
        out["entry_source"] = out["trigger_type"].astype(str)
    return out


def _summary_mark_1d(labeled: pd.DataFrame, *, market_cost: float, limit_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labeled.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = labeled.copy()
    x["year"] = pd.to_datetime(x["entry_time"], errors="coerce").dt.year
    per_row_cost = np.where(x["entry_kind"].astype(str).eq("fvg_limit"), float(limit_cost), float(market_cost))
    x["net_1d"] = pd.to_numeric(x["mark_return_1440m"], errors="coerce") - per_row_cost
    groups = ["liquidity_side","context_tier","trigger_type","execution_minutes","entry_kind","limit_variant"]
    def pf(v: pd.Series) -> float:
        v=pd.to_numeric(v,errors="coerce").dropna(); p=float(v[v>0].sum()); n=float(-v[v<0].sum())
        return (p/n) if n>1e-12 else (np.inf if p>1e-12 else np.nan)
    def one(p: pd.DataFrame) -> pd.Series:
        v=pd.to_numeric(p["net_1d"],errors="coerce").dropna()
        return pd.Series({"trades":len(p),"resolved":len(v),"win_rate_1d":float((v>0).mean()) if len(v) else np.nan,"mean_net_1d":float(v.mean()) if len(v) else np.nan,"median_net_1d":float(v.median()) if len(v) else np.nan,"pf_1d":pf(v),"positive_expectancy_flag":int(len(v)>0 and v.mean()>0)})
    overall=x.groupby(groups,dropna=False,sort=True).apply(one,include_groups=False).reset_index()
    year=x.groupby(groups+["year"],dropna=False,sort=True).apply(one,include_groups=False).reset_index()
    return overall,year


def _manual_review(out: Path, roots: pd.DataFrame, labeled: pd.DataFrame) -> None:
    d=out/"manual_review"; d.mkdir(parents=True,exist_ok=True)
    root_cols=[c for c in ["episode_id","liquidity_side","sweep_bar_time_1m","context_tier","root_physical_level_count","root_native_level_count","root_nested_level_count","root_max_context_tf_min","root_max_swing_tf_min","root_trend_move_pct_max","root_max_liquidity_age_days","root_swing_ids","root_trend_leg_ids","future_cascade_level_count_15m"] if c in roots.columns]
    roots.sort_values("sweep_bar_time_1m",kind="stable").tail(20).loc[:,root_cols].to_csv(d/"01_recent_20_root_sweep_events.csv",index=False,encoding="utf-8-sig")
    if labeled.empty:
        return
    keep=[c for c in ["trade_event_id","episode_id","liquidity_side","context_tier","trigger_type","execution_minutes","entry_kind","limit_variant","sweep_bar_time_1m","signal_available_time","entry_time","entry_price","structural_extreme_pre_entry","stop_price","risk_bps","r2p0_target_price","r2p0_outcome","r2p0_exit_time","r2p0_net_return_cost2x","mark_return_1440m","mfe_1440m","mae_1440m","root_swing_ids","root_trend_leg_ids"] if c in labeled.columns]
    specs=[("02_recent_10_immediate.csv",labeled["trigger_type"].eq("sweep_immediate")),
           ("03_recent_10_reclaim.csv",labeled["trigger_type"].eq("episode_reclaim")),
           ("04_recent_10_mss.csv",labeled["trigger_type"].astype(str).str.contains("mss_.*_market",regex=True)),
           ("05_recent_10_fvg_limit.csv",labeled["entry_kind"].eq("fvg_limit"))]
    for fn,mask in specs:
        q=labeled.loc[mask].sort_values("entry_time",kind="stable").tail(10)
        q.loc[:,keep].to_csv(d/fn,index=False,encoding="utf-8-sig")
    (d/"README.md").write_text(
        "# R09 manual chart review\n\n"
        "Use `01_recent_20_root_sweep_events.csv` to verify that the physical IT/LT liquidity and context tier look correct on chart.\n"
        "Use the execution files to inspect sweep-immediate, reclaim, MSS and FVG-limit entries. `structural_extreme_pre_entry` is the chart extreme; `stop_price` includes the fixed 2bps execution buffer.\n"
        "The R2 column is a research target, not a frozen final TP rule.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); progress=not args.no_progress
    start=pd.Timestamp(args.start_date); end=pd.Timestamp(args.end_date)
    cfg=R09Config(episode_gap_minutes=args.episode_gap_minutes,confirmation_minutes=args.confirmation_minutes,fvg_wait_minutes=args.fvg_wait_minutes,stop_buffer_bps=args.stop_buffer_bps).validate()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    r08_dir=Path(args.r08_dir)
    print("[r09] load R08.1 corrected native/nested liquidity",flush=True)
    native,nested,r08_manifest=_load_r08(r08_dir,end)
    print("[r09] load bare 1m K",flush=True)
    bars=OKXDataLoader(symbol=args.symbol,timeframe="1m",db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    if bars.empty: raise RuntimeError("No 1m OHLCV rows returned")
    print("[r09] physical liquidity dedupe + causal root sweep episodes",flush=True)
    physical=build_physical_liquidity_sweeps(native,nested,research_start=start,research_end=end)
    roots,minute_detail=build_root_sweep_episodes(physical,bars,config=cfg)
    quality=summarize_quality_ladder(roots,research_start=start,research_end=end)

    print("[r09] sweep-immediate entries",flush=True)
    imm=build_immediate_entries(bars,roots,config=cfg)
    executions=[_rewrite_trade_ids(imm,"R09_IMM")]
    r02cfg=R02Config(max_confirmation_minutes=cfg.confirmation_minutes,max_fvg_wait_minutes=cfg.fvg_wait_minutes,stop_buffer_bps=cfg.stop_buffer_bps)
    basecfg=MSS2Config()
    reclaim_tables=[]
    print("[r09] reclaim / MSS / MSS+FVG execution atlas",flush=True)
    for tf in _parse_ints(args.execution_minutes):
        tr=build_stack_execution_triggers(
            bars,roots,execution_minutes=tf,base_config=basecfg,config=r02cfg,
            reference_modes=("structural","post_sweep_st"),include_reclaims=True,include_mss_market=True,include_mss_fvg=True,
            show_progress=progress,
        )
        if tr.empty: continue
        tr=tr.loc[~tr["trigger_type"].eq("stage_reclaim")].copy()
        tr["limit_variant"]=np.where(tr["entry_kind"].astype(str).eq("fvg_limit"),"proximal","none")
        reclaim_tables.append(tr.loc[tr["trigger_type"].eq("episode_reclaim")].copy())
        executions.append(_rewrite_trade_ids(tr,f"R09_{tf}M"))
    reclaim_all=pd.concat(reclaim_tables,ignore_index=True) if reclaim_tables else pd.DataFrame()
    print("[r09] reclaim -> FVG resting-limit overlay",flush=True)
    rfvg=build_reclaim_fvg_limit_entries(bars,reclaim_all,config=cfg,show_progress=progress)
    executions.append(_rewrite_trade_ids(rfvg,"R09_RFVG"))
    trades=pd.concat([x for x in executions if x is not None and not x.empty],ignore_index=True,sort=False) if any(not x.empty for x in executions) else pd.DataFrame()
    if not trades.empty:
        # Stable global IDs after concatenating execution families.
        trades=trades.sort_values(["entry_time","episode_id","trigger_type","execution_minutes","entry_kind","limit_variant"],kind="stable").reset_index(drop=True)
        trades["trade_event_id"]=[f"R09_TRADE_{i+1:09d}" for i in range(len(trades))]
    print("[r09] structural-stop outcomes + MFE/MAE",flush=True)
    labeled=attach_r09_outcomes(bars,trades,config=cfg,show_progress=progress)

    summary_parts=[]; year_parts=[]
    for rr in cfg.fixed_r_targets:
        s,y=summarize_execution_grid(labeled,target_r=rr,cost_multiple=2)
        if not s.empty: s.insert(0,"target_r",rr); summary_parts.append(s)
        if not y.empty: y.insert(0,"target_r",rr); year_parts.append(y)
    exec_summary=pd.concat(summary_parts,ignore_index=True) if summary_parts else pd.DataFrame()
    exec_year=pd.concat(year_parts,ignore_index=True) if year_parts else pd.DataFrame()
    coverage_rows=[]
    if not trades.empty:
        denom=max(1,int(roots["episode_id"].nunique()))
        for keys,p in trades.groupby(["trigger_type","execution_minutes","entry_kind","limit_variant"],dropna=False,sort=True):
            trigger,tf,kind,lv=keys
            filled=int(p["episode_id"].nunique())
            coverage_rows.append({"trigger_type":trigger,"execution_minutes":tf,"entry_kind":kind,"limit_variant":lv,"filled_independent_episodes":filled,"root_episode_universe":denom,"fill_coverage":filled/denom})
    fill_coverage=pd.DataFrame(coverage_rows)
    mark_summary,mark_year=_summary_mark_1d(labeled,market_cost=cfg.market_roundtrip_cost*2,limit_cost=cfg.limit_roundtrip_cost*2)
    cascade=summarize_cascade_diagnostic(labeled.loc[labeled["trigger_type"].eq("sweep_immediate")].copy(),cost_multiple=2) if not labeled.empty else pd.DataFrame()
    audit=r09_causal_audit(roots,trades)

    pd.DataFrame([{
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"symbol":args.symbol,
        "research_start_date":str(start),"research_end_date":str(end),"r08_report":str(r08_dir),
        "episode_gap_minutes":cfg.episode_gap_minutes,"stop_buffer_bps":cfg.stop_buffer_bps,
        "initial_tier_definition":"C=15m completed-trend context; B=30m; A=1H; A+=4H. No outcome information used.",
        "future_cascade_semantics":"diagnostic only; never used for initial tier or entry admission",
    }]).to_csv(out/"00_manifest.csv",index=False)
    physical.to_csv(out/"01_physical_liquidity_first_sweeps.csv.gz",index=False,compression="gzip")
    roots.to_csv(out/"02_independent_root_sweep_events.csv.gz",index=False,compression="gzip")
    quality.to_csv(out/"03_quality_ladder_frequency.csv",index=False)
    minute_detail.to_csv(out/"04_episode_sweep_minute_detail.csv.gz",index=False,compression="gzip")
    trades.to_csv(out/"05_execution_trade_candidates.csv.gz",index=False,compression="gzip")
    fill_coverage.to_csv(out/"05b_execution_fill_coverage.csv",index=False)
    labeled.to_csv(out/"06_execution_outcome_rows.csv.gz",index=False,compression="gzip")
    exec_summary.to_csv(out/"07_execution_fixed_r_summary_cost2x.csv",index=False)
    exec_year.to_csv(out/"08_execution_fixed_r_year_summary_cost2x.csv",index=False)
    mark_summary.to_csv(out/"09_execution_1d_mark_summary_cost2x.csv",index=False)
    mark_year.to_csv(out/"10_execution_1d_mark_year_summary_cost2x.csv",index=False)
    cascade.to_csv(out/"11_future_cascade_diagnostic_1d.csv",index=False)
    audit.to_csv(out/"12_causal_audit.csv",index=False)
    eng=pd.DataFrame([
        {"check":"r08_native_rows","value":len(native)}, {"check":"r08_nested_rows","value":len(nested)},
        {"check":"physical_sweeps","value":len(physical)}, {"check":"independent_root_events","value":len(roots)},
        {"check":"execution_candidates","value":len(trades)}, {"check":"causal_violations","value":int(pd.to_numeric(audit["violations"],errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]); eng.to_csv(out/"13_engineering_audit.csv",index=False)
    (out/"00_manifest.json").write_text(json.dumps({
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"title":TITLE,
        "symbol":args.symbol,"warmup_start_date":args.warmup_start_date,"research_start_date":args.start_date,"research_end_date":args.end_date,
        "r08_manifest":r08_manifest,"execution_minutes":list(_parse_ints(args.execution_minutes)),
        "purpose":"Preserve broad ICT full-trend liquidity opportunities, rank by causal context scale, and compare execution without future cascade leakage.",
    },ensure_ascii=False,indent=2),encoding="utf-8")
    _manual_review(out,roots,labeled)
    finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(f"[r09] done -> {out}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
