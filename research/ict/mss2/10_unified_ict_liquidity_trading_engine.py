#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R10: Unified ICT Liquidity Trading Engine.

Consolidation phase.  R10 deliberately stops creating independent Sweep / MSS /
FVG strategies and tests one coherent long-only lifecycle on the broad R09 SSL
universe:

    qualified SSL sweep -> 2m episode reclaim -> one position
      -> later structural MSS upgrades trade state (never creates a second trade)
      -> Base realizes 2R -> Runner follows causal 5m LTL
      -> if MSS-confirmed + 3R, Runner slows to causal 15m LTL.

No add-on in R10 v1.  No fixed time exit.  Risk schedules are frozen before R10
results and are evaluated at 1x/2x/3x costs with single-ETH-position allocation.
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
from src.research_common.ict_mss2.r05 import R05Config, build_trailing_events  # noqa: E402
from src.research_common.ict_mss2.r10 import (  # noqa: E402
    R10Config,
    attach_risk_sizing,
    build_daily_partial_equity,
    build_structural_mss_upgrade_map,
    build_unified_reclaim_base,
    r10_causal_audit,
    select_single_position,
    simulate_unified_lifecycles,
    summarize_scenario,
    summarize_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "10.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_UNIFIED_LIQUIDITY_ENGINE_R10"
EDGE_ID = "ICT_FULL_TREND_SSL_UNIFIED_ENGINE"
TITLE = "ETH ICT MSS2 R10 Unified ICT Liquidity Trading Engine"
DEFAULT_R09_DIR = "data/reports/research/ict/mss2/r09_liquidity_quality_execution_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r10_unified_ict_liquidity_trading_engine"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r09-dir", default=DEFAULT_R09_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _load_r09(path: Path, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    mp = path / "00_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"R09 manifest missing: {mp}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    covered = pd.Timestamp(manifest.get("research_end_date"))
    if covered < end:
        raise RuntimeError(f"R09 only covers through {covered}; rerun R09 through {end} first")
    fp = path / "06_execution_outcome_rows.csv.gz"
    if not fp.exists():
        raise FileNotFoundError(f"R09 outcome rows missing: {fp}")
    return pd.read_csv(fp), manifest


def _months(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(1e-9, float((end - start) / pd.Timedelta(days=30.4375)))


def _scorecards(executed: pd.DataFrame, bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: R10Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    cards=[]; eq_parts=[]
    months=_months(start,end)
    groups=["lifecycle_variant","risk_schedule","cost_scale"]
    for key,p in executed.groupby(groups,dropna=False,sort=True):
        eq=build_daily_partial_equity(p,bars,market_roundtrip_cost=cfg.market_roundtrip_cost)
        if eq.empty:
            continue
        card=summarize_scenario(p,eq,months=months)
        cards.append(dict(zip(groups,key))|card)
        q=eq.copy(); q.insert(0,"cost_scale",key[2]); q.insert(0,"risk_schedule",key[1]); q.insert(0,"lifecycle_variant",key[0]); eq_parts.append(q)
    return pd.DataFrame(cards), (pd.concat(eq_parts,ignore_index=True,sort=False) if eq_parts else pd.DataFrame())


def _monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    rows=[]
    groups=["lifecycle_variant","risk_schedule","cost_scale"]
    for key,p in equity.groupby(groups,dropna=False,sort=True):
        q=p.set_index("date")["equity"].sort_index().resample("ME").last().dropna().pct_change().dropna()
        for t,v in q.items():
            rows.append(dict(zip(groups,key))|{"month":pd.Timestamp(t),"return":float(v)})
    return pd.DataFrame(rows)


def _tier_summary(executed: pd.DataFrame) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame()
    rows=[]
    groups=["lifecycle_variant","risk_schedule","cost_scale","context_tier"]
    for key,p in executed.groupby(groups,dropna=False,sort=True):
        v=pd.to_numeric(p["strategy_equity_return"],errors="coerce").dropna()
        gp=float(v[v>0].sum()); gl=float(-v[v<0].sum()); pf=gp/gl if gl>1e-12 else (np.inf if gp>1e-12 else np.nan)
        rows.append(dict(zip(groups,key))|{
            "trades":len(p),"resolved":len(v),"win_rate":float((v>0).mean()) if len(v) else np.nan,
            "trade_pf":pf,"mean_equity_return":float(v.mean()) if len(v) else np.nan,
            "mss_upgrade_rate":float(pd.to_numeric(p["mss_upgrade_flag"],errors="coerce").mean()),
            "base_target_hit_rate":float(pd.to_numeric(p["base_target_hit_flag"],errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def _manual_review(out: Path, executed: pd.DataFrame) -> None:
    d=out/"manual_review"; d.mkdir(parents=True,exist_ok=True)
    if executed.empty:
        return
    # Fixed scenario for human chart review; not selected from R10 performance.
    q=executed.loc[
        executed["lifecycle_variant"].astype(str).eq("base75_2r_runner25")
        & executed["risk_schedule"].astype(str).eq("quality_scaled")
        & pd.to_numeric(executed["cost_scale"],errors="coerce").eq(2.0)
    ].sort_values("entry_time",kind="stable").tail(20)
    keep=[c for c in [
        "episode_id","context_tier","entry_time","entry_price","initial_stop_price","initial_risk_return",
        "base_target_price","base_target_hit_flag","base_exit_time","base_exit_price","mss_upgrade_time","mss_upgrade_flag",
        "major_upgrade_time","major_upgrade_flag","trail_updates_5m","trail_updates_15m","final_stop_price","exit_time","exit_price",
        "gross_return_unit_notional","notional_multiple","risk_budget_fraction","strategy_equity_return","root_swing_ids","root_trend_leg_ids"
    ] if c in q.columns]
    q.loc[:,keep].to_csv(d/"01_recent_20_unified_positions.csv",index=False,encoding="utf-8-sig")
    (d/"README.md").write_text(
        "# R10 manual review\n\n"
        "Inspect `01_recent_20_unified_positions.csv` on chart. The scenario is fixed to base75/runner25 + quality_scaled + 2x solely for review consistency, not because it is the best R10 result.\n"
        "`initial_stop_price` is the R09 sweep/reclaim structural invalidation including the fixed 2bps execution buffer. `base_target_price` is the frozen 2R partial target. MSS is a later state upgrade, not a second entry.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); progress=not args.no_progress
    start=pd.Timestamp(args.start_date); end=pd.Timestamp(args.end_date)
    cfg=R10Config().validate()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    print("[r10] load R09 outcomes",flush=True)
    r09,r09_manifest=_load_r09(Path(args.r09_dir),end)
    print("[r10] freeze one unified 2m reclaim base per SSL episode",flush=True)
    base=build_unified_reclaim_base(r09,execution_minutes=cfg.execution_minutes)
    mss=build_structural_mss_upgrade_map(base,r09,execution_minutes=cfg.execution_minutes)
    print("[r10] load bare 1m K",flush=True)
    bars=OKXDataLoader(symbol=args.symbol,timeframe="1m",db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    print("[r10] causal 5m/15m LTL runner anchors",flush=True)
    trailing=build_trailing_events(bars,config=R05Config(trail_minutes=(5,15)),)
    print("[r10] unified Base + Runner lifecycle",flush=True)
    paths=simulate_unified_lifecycles(base,bars,trailing,mss,config=cfg,show_progress=progress)
    print("[r10] fixed risk schedules x 1x/2x/3x costs",flush=True)
    sized=attach_risk_sizing(paths,config=cfg)
    executed,overlap=select_single_position(sized)
    print("[r10] daily MTM equity + smoothness scorecards",flush=True)
    cards,equity=_scorecards(executed,bars,start,end,cfg)
    years=summarize_years(executed)
    months=_monthly_returns(equity)
    tiers=_tier_summary(executed)
    audit=r10_causal_audit(base,paths,sized)

    manifest={
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"title":TITLE,
        "symbol":args.symbol,"warmup_start_date":args.warmup_start_date,"research_start_date":args.start_date,"research_end_date":args.end_date,
        "r09_report":args.r09_dir,"r09_manifest":r09_manifest,
        "unified_entry":"SSL only; 2m episode reclaim; next-open market entry",
        "mss_semantics":"later 2m structural MSS is state upgrade only; never opens a second position",
        "fvg_semantics":"diagnostic/execution research remains in R09; R10 v1 does not create separate FVG trades",
        "initial_stop":"R09 causal sweep/reclaim structural extreme + fixed 2bps buffer",
        "base_target":"2R research-frozen partial target for Base+Runner variants",
        "runner":"after 2R, BE from next 1m then 5m LTL; after structural MSS + 3R, later anchors slow to 15m LTL",
        "addon":"disabled in R10 v1",
        "risk_schedules":[list(x) for x in cfg.risk_schedules],"cost_scales":list(cfg.cost_scales),
    }
    (out/"00_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([{
        "script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"symbol":args.symbol,"research_start_date":args.start_date,"research_end_date":args.end_date,
        "base_entries":len(base),"lifecycle_rows":len(paths),"sized_rows":len(sized),"executed_rows":len(executed),
    }]).to_csv(out/"00_manifest.csv",index=False)
    base.to_csv(out/"01_unified_base_entries.csv.gz",index=False,compression="gzip")
    mss.to_csv(out/"02_structural_mss_state_upgrades.csv.gz",index=False,compression="gzip")
    paths.to_csv(out/"03_unified_lifecycle_paths.csv.gz",index=False,compression="gzip")
    sized.to_csv(out/"04_risk_sized_scenarios.csv.gz",index=False,compression="gzip")
    executed.to_csv(out/"05_single_position_executed_trades.csv.gz",index=False,compression="gzip")
    overlap.to_csv(out/"06_overlap_audit.csv",index=False)
    cards.to_csv(out/"07_portfolio_equity_scorecard.csv",index=False)
    years.to_csv(out/"08_portfolio_year_summary.csv",index=False)
    months.to_csv(out/"09_portfolio_monthly_returns.csv.gz",index=False,compression="gzip")
    tiers.to_csv(out/"10_executed_tier_summary.csv",index=False)
    equity.to_csv(out/"11_daily_mtm_equity.csv.gz",index=False,compression="gzip")
    audit.to_csv(out/"12_causal_risk_audit.csv",index=False)
    focus=cards.loc[pd.to_numeric(cards.get("cost_scale"),errors="coerce").eq(2.0)].copy() if not cards.empty else pd.DataFrame()
    if not focus.empty:
        focus=focus.sort_values(["positive_month_rate","max_drawdown_daily_mtm","total_return"],ascending=[False,False,False],kind="stable")
    focus.to_csv(out/"13_cost2x_equity_focus.csv",index=False)
    _manual_review(out,executed)
    finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(f"[r10] done -> {out}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
