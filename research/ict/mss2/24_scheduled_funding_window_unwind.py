#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R24 — scheduled funding-window unwind."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa:E402
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa:E402
from src.research_common.ict_mss2.r24 import (  # noqa:E402
    R24Config,build_funding_window_events,build_r24_gate,r24_causal_audit,
    simulate_funding_unwind,summarize_r24,summarize_r24_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa:E402

TITLE="ETH ICT MSS2 R24 Scheduled Funding-Window Unwind"
EXPERIMENT_ID="ETH_ICT_MSS2_SCHEDULED_FUNDING_WINDOW_UNWIND_R24"
EDGE_ID="RESEARCH_ONLY_SCHEDULED_FUNDING_WINDOW_UNWIND"
DEFAULT_OUT_DIR="data/reports/research/ict/mss2/r24_scheduled_funding_window_unwind"


def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser(description=TITLE); p.add_argument("--symbol",default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date",default="2022-01-01"); p.add_argument("--end-date",default="2025-06-30 23:59:59")
    p.add_argument("--out-dir",default=DEFAULT_OUT_DIR); p.add_argument("--skip-review-pack",action="store_true"); return p.parse_args(argv)


def _manual(out:Path,trades:pd.DataFrame)->None:
    d=out/"manual_review"; d.mkdir(parents=True,exist_ok=True); closed=trades.loc[trades["path_status"].eq("included")].copy()
    closed.sort_values("entry_time").tail(80).to_csv(d/"01_recent_80.csv",index=False)
    closed.sort_values("net_return_cost2x",ascending=False).head(40).to_csv(d/"02_best_40.csv",index=False)
    closed.sort_values("net_return_cost2x").head(40).to_csv(d/"03_worst_40.csv",index=False)
    (d/"README.md").write_text("# R24 manual review\n\nVerify completed pre-settlement hour, fixed clock entry, reversal direction, ATR barriers, stop-first path, and next-clock timeout.\n",encoding="utf-8")


def main(argv:Sequence[str]|None=None)->int:
    args=parse_args(argv); cfg=R24Config().validate()
    if pd.Timestamp(args.end_date)>=cfg.embargo_start: raise ValueError("R24 end date must remain before July embargo")
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    print("[r24] load bare ETH 1m through src.data_feed",flush=True)
    bars=OKXDataLoader(args.symbol,"1m").fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    coverage=data_coverage_audit(bars,requested_start=pd.Timestamp(args.warmup_start_date),requested_end=pd.Timestamp(args.end_date))
    events=build_funding_window_events(bars,config=cfg); pieces=[]
    for target_r in cfg.target_rs:
        for split,start,end in (("discovery",cfg.discovery_start,cfg.validation_start),("validation",cfg.validation_start,cfg.embargo_start)):
            for direction in (1,-1):
                part=simulate_funding_unwind(bars,events,target_r=target_r,direction=direction,split=split,split_start=start,split_end=end,config=cfg)
                if not part.empty: pieces.append(part)
    trades=pd.concat(pieces,ignore_index=True,sort=False) if pieces else pd.DataFrame()
    if trades.empty: raise RuntimeError("R24 produced no paths")
    score=summarize_r24(trades); years=summarize_r24_years(trades); gate=build_r24_gate(score,years); audit=r24_causal_audit(trades,config=cfg)
    manifest={"experiment_id":EXPERIMENT_ID,"edge_id":EDGE_ID,"title":TITLE,"market":args.symbol,"window":[args.warmup_start_date,args.end_date],"signal":{"schedule_hours":cfg.schedule_hours,"sigma_hours":cfg.sigma_hours,"impulse_z":cfg.impulse_z},"execution":{"stop_atr":cfg.stop_atr,"target_rs":cfg.target_rs,"hold_hours":cfg.hold_hours},"costs":{"roundtrip":cfg.market_roundtrip_cost,"scales":cfg.cost_scales},"holdout_rows_loaded":0}
    (out/"00_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    coverage.to_csv(out/"01_data_coverage.csv",index=False)
    pd.DataFrame([{"check":"signal_events","value":len(events)},{"check":"closed_paths","value":int(trades["path_status"].eq("included").sum())},{"check":"boundary_censored","value":int(trades["path_status"].eq("boundary_censored").sum())},{"check":"holdout_rows_loaded","value":0}]).to_csv(out/"02_funnel.csv",index=False)
    events.to_csv(out/"03_events.csv.gz",index=False,compression="gzip",float_format="%.17g"); trades.to_csv(out/"04_trade_paths.csv.gz",index=False,compression="gzip",float_format="%.17g")
    score.to_csv(out/"05_scorecard.csv",index=False); years.to_csv(out/"06_years.csv",index=False); gate.to_csv(out/"07_candidate_gate.csv",index=False); audit.to_csv(out/"08_causal_audit.csv",index=False); _manual(out,trades)
    (out/"R24_GENERATED_NOTE.md").write_text("# R24 generated note\n\nScheduled funding-clock unwind; funding values are not inferred. July and holdout are absent.\n",encoding="utf-8")
    if not args.skip_review_pack: finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(score.to_string(index=False),flush=True); print(gate.to_string(index=False),flush=True); print(f"[r24] done -> {out}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())

