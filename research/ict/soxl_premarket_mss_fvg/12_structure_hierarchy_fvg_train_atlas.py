#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R12: structure hierarchy + FVG-train semantic alignment atlas.

Purpose
-------
R12 is a semantic-alignment research pass driven by manual 2026-08-05 review.
It does *not* search for one strict profitable filter.  It compares:

* early 04:00-08:30 and late 08:30-09:30 session liquidity as separate families;
* every causal low-timeframe swing vs more visible/outer structural barriers;
* multiple MSS attempts inside the same liquidity episode;
* the whole pre-break / break-candle FVG train;
* FVG entry capped by broken swing +/- 0.10;
* FVG entry capped by the FVG whose middle candle is the break candle +/- 0.10;
* direct next-open market entry after a close structure break;
* old external target vs the nearest still-active internal structure target.

No R12 structure score is a hard gate.  The atlas is designed to quantify how
many opportunities each semantic interpretation would keep/drop and whether the
manual interpretation improves forward performance.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE
from src.research_common.ict.premarket_mss_fvg import (
    EPS, NY_TZ, ReplayScenario, ResearchConfig, add_analysis_dimensions,
    build_data_quality_table, eligible_ny_dates, enforce_single_lifecycle,
    ny_date_bounds_to_source_naive, replay_attempts, source_naive_to_new_york,
    summarize_variant, slice_ny_day, aggregate_closed_bars, make_synthetic_ict_day,
)
from src.research_common.ict.premarket_mss_fvg_v2 import (
    SweepEpisodeConfig, build_all_premarket_levels_v2,
)
from src.research_common.ict.entry_expansion import (
    EntryExpansionConfig, build_intraday_15m_swing_catalog, build_intraday_15m_sweep_events,
)
from src.research_common.ict.spot_perp_overlap import (
    build_equity_proxy_data_quality_table, densify_equity_minutes_causally,
)
from src.research_common.ict.structure_entry_semantics import (
    StructureSemanticConfig, build_dual_session_liquidity_levels,
    build_visible_swing_catalog, build_causal_sweep_events_for_levels,
    build_structure_break_fvg_atlas, expand_entry_target_variants,
)
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r12_structure_hierarchy_fvg_train"


def _csv_ints(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals:
        raise ValueError("empty integer list")
    return vals


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R12 structure hierarchy + FVG-train atlas")
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="okx")
    p.add_argument("--symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--execution-timeframes", default="1,2,5")
    p.add_argument("--structure-lookback-minutes", type=int, default=150)
    p.add_argument("--entry-buffer", type=float, default=0.10)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(symbol=args.alpaca_symbol, timeframe="1Min", feed=args.alpaca_feed, adjustment=args.alpaca_adjustment, data_dir=args.data_dir)
        raw = loader.fetch_data_by_date_range(start_ny.tz_convert("UTC"), end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1), local_only=bool(args.local_only))
        if raw.empty:
            raise RuntimeError("Alpaca loader returned no data")
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars = raw.copy(); bars.index = idx.tz_convert(NY_TZ); bars.index.name = "bar_start_ny"
    else:
        offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
        start_src, end_src = ny_date_bounds_to_source_naive(args.start_date, args.end_date, source_offset_hours=offset)
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        raw = loader.load_local_data() if args.local_only else loader.fetch_data_by_date_range(start_src, end_src)
        if args.local_only and not raw.empty:
            raw = raw.loc[(raw.index >= start_src) & (raw.index <= end_src)].copy()
        if raw.empty:
            raise RuntimeError("OKX loader returned no data")
        bars = source_naive_to_new_york(raw, source_offset_hours=offset)
    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _attach_market_next_open(bars: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    if variants.empty:
        return variants
    out = variants.copy()
    market = out["entry_order_type"].astype(str).eq("market_next_open")
    out["market_next_open_price"] = np.nan
    out["market_next_open_time"] = pd.Series(pd.NaT, index=out.index, dtype=f"datetime64[ns, {NY_TZ}]")
    for (day_text, tf), idxs in out.loc[market].groupby(["ny_date", "execution_tf_minutes"], sort=False).groups.items():
        day = pd.Timestamp(day_text).date()
        one = slice_ny_day(bars, day, pd.Timestamp("04:00").time(), pd.Timestamp("16:30").time())
        frame = aggregate_closed_bars(one, int(tf))
        if frame.empty:
            continue
        fidx = pd.DatetimeIndex(frame.index)
        fav = pd.DatetimeIndex(pd.to_datetime(frame["available_time"]))
        for i in idxs:
            t = pd.Timestamp(out.at[i, "break_available_time"])
            positions = np.flatnonzero((fidx >= t).to_numpy(dtype=bool) if hasattr((fidx >= t), 'to_numpy') else np.asarray(fidx >= t))
            if positions.size == 0:
                continue
            p = int(positions[0])
            out.at[i, "market_next_open_price"] = float(frame.iloc[p]["open"])
            out.at[i, "market_next_open_time"] = pd.Timestamp(fidx[p])
            out.at[i, "entry_price"] = float(frame.iloc[p]["open"])
            out.at[i, "entry_available_time"] = pd.Timestamp(fidx[p])
    return out


def _replay_market_rows(bars: pd.DataFrame, rows: pd.DataFrame, *, cost: float, risk_fraction: float, max_notional_multiple: float) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out=[]
    cache={}
    for r in rows.to_dict("records"):
        day_text=str(r["ny_date"]); is_long=str(r["trade_side"])=="LONG"
        if day_text not in cache:
            cache[day_text]=slice_ny_day(bars,pd.Timestamp(day_text).date(),pd.Timestamp("08:30").time(),pd.Timestamp("16:30").time())
        day=cache[day_text]
        entry=float(r.get("entry_price",np.nan)); stop=float(r.get("stop_price",np.nan)); target=float(r.get("target_price",np.nan)); fill=pd.Timestamp(r.get("market_next_open_time",r.get("entry_available_time")))
        risk=(entry-stop) if is_long else (stop-entry)
        base={**r,"filled":False,"fill_time":fill,"entry_price":entry,"exit_time":pd.NaT,"exit_price":np.nan,"exit_reason":"","gross_return":np.nan,"net_return":np.nan,"account_return":np.nan,"round_trip_cost":float(cost)}
        if not np.isfinite(entry) or not np.isfinite(stop) or not np.isfinite(target) or risk<=EPS:
            base["exit_reason"]="invalid_market_entry"; out.append(base); continue
        path=day.loc[pd.DatetimeIndex(day.index)>=fill]
        if path.empty:
            base["exit_reason"]="no_path"; out.append(base); continue
        exit_price=np.nan; exit_time=pd.NaT; reason=""
        for ts,bar in path.iterrows():
            st=(float(bar["low"])<=stop) if is_long else (float(bar["high"])>=stop)
            tp=(float(bar["high"])>=target) if is_long else (float(bar["low"])<=target)
            if st and tp:
                exit_price=stop; exit_time=pd.Timestamp(ts)+pd.Timedelta(minutes=1); reason="stop_first_same_bar_both_conservative"; break
            if st:
                exit_price=stop; exit_time=pd.Timestamp(ts)+pd.Timedelta(minutes=1); reason="structural_sweep_extreme_stop"; break
            if tp:
                exit_price=target; exit_time=pd.Timestamp(ts)+pd.Timedelta(minutes=1); reason="target"; break
        if not reason:
            exit_price=float(path.iloc[-1]["close"]); exit_time=pd.Timestamp(path.index[-1])+pd.Timedelta(minutes=1); reason="session_1630_close"
        gross=(exit_price/entry-1.0)*(1.0 if is_long else -1.0); net=gross-float(cost); risk_pct=risk/entry; notional=min(float(max_notional_multiple),float(risk_fraction)/risk_pct)
        base.update(filled=True,exit_time=exit_time,exit_price=exit_price,exit_reason=reason,gross_return=gross,net_return=net,gross_r=gross/risk_pct,net_r=net/risk_pct,notional_multiple=notional,account_return=net*notional)
        out.append(base)
    return pd.DataFrame(out)


def _replay_variants(bars: pd.DataFrame, variants: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if variants.empty:
        return pd.DataFrame()
    limit = variants.loc[variants["entry_order_type"].eq("limit")].copy()
    if not limit.empty:
        limit["fvg_near_edge_entry"] = pd.to_numeric(limit["entry_price"],errors="coerce")
        limit["signal_time"] = pd.to_datetime(limit["entry_available_time"])
        limit["attempt_id"] = limit["attempt_id_r12"].astype(str)
        limit_replay = replay_attempts(bars,limit,scenario=ReplayScenario(name="base",cost_multiple=1.0,order_delay_minutes=0),round_trip_cost=float(args.round_trip_cost),risk_fraction=float(args.risk_fraction),max_notional_multiple=float(args.max_notional_multiple))
    else:
        limit_replay=pd.DataFrame()
    market = variants.loc[variants["entry_order_type"].eq("market_next_open")].copy()
    market_replay = _replay_market_rows(bars,market,cost=float(args.round_trip_cost),risk_fraction=float(args.risk_fraction),max_notional_multiple=float(args.max_notional_multiple))
    return pd.concat([x for x in (limit_replay,market_replay) if not x.empty],ignore_index=True,sort=False) if (not limit_replay.empty or not market_replay.empty) else pd.DataFrame()


def _summary(life: pd.DataFrame) -> pd.DataFrame:
    if life.empty:
        return pd.DataFrame()
    rows=[]
    keys=["execution_tf","entry_model","target_model_r12"]
    for key,g in life.groupby(keys,dropna=False,sort=True):
        f=g.loc[g["filled"].fillna(False).astype(bool)]
        x=pd.to_numeric(f.get("net_return"),errors="coerce").dropna()
        gains=float(x[x>0].sum()); losses=float(-x[x<0].sum()); pf=gains/losses if losses>0 else (np.inf if gains>0 else np.nan)
        rows.append(dict(zip(keys,key),attempts=len(g),filled_trades=len(f),fill_rate=len(f)/len(g) if len(g) else np.nan,win_rate=float((x>0).mean()) if len(x) else np.nan,profit_factor=pf,mean_net_return=float(x.mean()) if len(x) else np.nan))
    return pd.DataFrame(rows)


def _cap_retention(fvgs: pd.DataFrame) -> pd.DataFrame:
    if fvgs.empty:
        return pd.DataFrame()
    rows=[]
    for tf,g in fvgs.groupby("execution_tf",sort=True):
        rows.append({"execution_tf":tf,"fvg_candidates":len(g),"swing_plusminus_0p10_keep_rate":float(g["swing_buffer_cap_pass"].fillna(False).mean()),"break_middle_plusminus_0p10_keep_rate":float(g["break_middle_fvg_buffer_cap_pass"].fillna(False).mean()),"break_middle_fvg_available_rate":float(pd.to_numeric(g["break_middle_fvg_cap_base_entry"],errors="coerce").notna().mean())})
    allg=fvgs
    rows.append({"execution_tf":"ALL","fvg_candidates":len(allg),"swing_plusminus_0p10_keep_rate":float(allg["swing_buffer_cap_pass"].fillna(False).mean()),"break_middle_plusminus_0p10_keep_rate":float(allg["break_middle_fvg_buffer_cap_pass"].fillna(False).mean()),"break_middle_fvg_available_rate":float(pd.to_numeric(allg["break_middle_fvg_cap_base_entry"],errors="coerce").notna().mean())})
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); print(f"[write] {path.name} rows={len(df):,}",flush=True)


def run_research(bars: pd.DataFrame,args:argparse.Namespace)->dict[str,Path]:
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    days=eligible_ny_dates(bars,start_date=args.start_date,end_date=args.end_date,exclude_equity_holidays=not args.include_us_equity_holidays)
    quality=build_equity_proxy_data_quality_table(bars,days) if args.data_source=="alpaca" else build_data_quality_table(bars,days,required_coverage=float(args.required_day_coverage))
    valid=set(quality.loc[quality["coverage_pass"],"ny_date"].astype(str)); days=[pd.Timestamp(x).date() for x in sorted(valid)]
    if not days: raise RuntimeError("no valid sessions")
    cfg=StructureSemanticConfig(execution_timeframes=_csv_ints(args.execution_timeframes),structure_lookback_minutes=int(args.structure_lookback_minutes),absolute_entry_buffer=float(args.entry_buffer))
    stage=ProgressReporter(label="[research] R12 stages",total=10,every=1,enabled=not args.no_progress)
    pm=build_all_premarket_levels_v2(bars,days,pivot_left=2,pivot_right=2,episode_config=SweepEpisodeConfig())
    major=pm.loc[pm["level_type"].eq("major_15m_swing")].copy() if not pm.empty else pd.DataFrame()
    if not major.empty: major["liquidity_family"]="major_15m_swing"
    dual,running=build_dual_session_liquidity_levels(bars,days,config=cfg)
    levels=pd.concat([x for x in (dual,major) if not x.empty],ignore_index=True,sort=False) if (not dual.empty or not major.empty) else pd.DataFrame()
    stage.update(1)
    sweeps=build_causal_sweep_events_for_levels(bars,levels)
    # R11 intraday 15m physical liquidity remains a separate source family.
    intraday_cfg=EntryExpansionConfig(intraday_pivot_left=1,intraday_pivot_right=1)
    intraday_catalog=build_intraday_15m_swing_catalog(bars,days,pm,config=intraday_cfg)
    intraday_sweeps=build_intraday_15m_sweep_events(bars,intraday_catalog,config=intraday_cfg)
    if not intraday_sweeps.empty:
        intraday_sweeps=intraday_sweeps.copy(); intraday_sweeps["setup_eligible_at_sweep"]=True
    all_sweeps=pd.concat([x for x in (sweeps,intraday_sweeps) if not x.empty],ignore_index=True,sort=False) if (not sweeps.empty or not intraday_sweeps.empty) else pd.DataFrame()
    stage.update(2)
    swing_catalog=build_visible_swing_catalog(bars,days,config=cfg); stage.update(3)
    print(f"[stage 4/10] structure-break/FVG-train scan sweeps={len(all_sweeps):,} swings={len(swing_catalog):,}", flush=True)
    breaks,fvgs=build_structure_break_fvg_atlas(bars,all_sweeps,swing_catalog,config=cfg); stage.update(4)
    variants=expand_entry_target_variants(breaks,fvgs,config=cfg); variants=_attach_market_next_open(bars,variants); stage.update(5)
    life=_replay_variants(bars,variants,args); stage.update(6)
    summary=_summary(life); caps=_cap_retention(fvgs); stage.update(7)
    # Descriptive structure layer comparison: no filter is frozen from this table.
    layer_rows=[]
    if not life.empty:
        base=life.loc[life["filled"].fillna(False).astype(bool)].copy()
        for flag in ["reference_is_latest_newly_broken","reference_is_highest_visibility_newly_broken","reference_is_outermost_barrier_newly_broken"]:
            if flag not in base: continue
            for tf,g in base.loc[base[flag].fillna(False)].groupby("execution_tf",sort=True):
                x=pd.to_numeric(g["net_return"],errors="coerce").dropna(); gains=float(x[x>0].sum()); losses=float(-x[x<0].sum())
                layer_rows.append({"reference_model":flag,"execution_tf":tf,"trades":len(x),"win_rate":float((x>0).mean()) if len(x) else np.nan,"profit_factor":gains/losses if losses>0 else np.nan,"mean_net_return":float(x.mean()) if len(x) else np.nan})
    layer=pd.DataFrame(layer_rows); stage.update(8)
    golden=pd.DataFrame()
    for frame,name in ((breaks,"break"),(fvgs,"fvg"),(variants,"entry"),(life,"trade")):
        if not frame.empty and "ny_date" in frame:
            g=frame.loc[frame["ny_date"].astype(str)==str(args.golden_date)].copy(); g.insert(0,"golden_record_type",name)
            golden=pd.concat([golden,g],ignore_index=True,sort=False)
    stage.update(9)
    design=f"""# R12 Structure Hierarchy + FVG Train Atlas\n\n- Source: {args.data_source}\n- Window: {args.start_date} -> {args.end_date}\n- Early session liquidity: 04:00-08:30 ET, frozen 08:30.\n- Late premarket liquidity: 08:30-09:30 ET, frozen 09:30; running extremes are also exported causally.\n- Every 1/1 pivot is retained as a candidate, but a continuous visibility score prevents `latest pivot == meaningful swing` from being assumed.\n- One sweep may emit multiple MSS candidates.\n- FVG train only associates gaps whose middle candle is before/on the break candle.\n- Entry buffer research: broken-swing +/- {args.entry_buffer:.2f} and break-middle-FVG +/- {args.entry_buffer:.2f}; neither is a hard universe gate.\n- Close-break next-open market entry is a separate execution model.\n- If the old external target has already been consumed by signal time, that target variant is dropped while nearest internal structure remains eligible.\n"""
    (out/"00_research_design.md").write_text(design,encoding="utf-8")
    _write(quality,out/"01_data_quality.csv"); _write(dual,out/"02_dual_session_liquidity_levels.csv"); _write(running,out/"03_late_premarket_running_extremes.csv"); _write(major,out/"04_major_15m_levels.csv"); _write(all_sweeps,out/"05_sweep_events.csv"); _write(swing_catalog,out/"06_causal_swing_hierarchy.csv"); _write(breaks,out/"07_structure_break_candidates.csv"); _write(fvgs,out/"08_fvg_train_candidates.csv"); _write(caps,out/"09_entry_cap_retention.csv"); _write(variants,out/"10_entry_target_variants.csv"); _write(life,out/"11_trade_lifecycle.csv"); _write(summary,out/"12_entry_model_performance.csv"); _write(layer,out/"13_reference_layer_performance.csv"); _write(golden,out/f"14_golden_replay_{args.golden_date}.csv")
    manifest={"experiment_id":"SOXL_ICT_MSS_R12_STRUCTURE_HIERARCHY_FVG_TRAIN","data_source":args.data_source,"start_date":args.start_date,"end_date":args.end_date,"valid_sessions":len(days),"sweeps":len(all_sweeps),"structure_breaks":len(breaks),"fvg_candidates":len(fvgs),"entry_variants":len(variants),"round_trip_cost":float(args.round_trip_cost),"entry_buffer":float(args.entry_buffer),"protocol":"semantic atlas; no visibility/entry-cap filter frozen as final rule"}
    (out/"15_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    stage.update(10); stage.close()
    if not args.skip_review_pack:
        finalize_research_report(out,experiment_id=manifest["experiment_id"],edge_id="SOXL_ICT_SWEEP_MSS_STRUCTURE_SEMANTICS",title="SOXL ICT R12 Structure Hierarchy + FVG Train Atlas",print_log=True)
    return {"report_dir":out,"review_pack":out/"gpt_review_pack.zip"}


def run_self_test(args:argparse.Namespace)->int:
    bars=make_synthetic_ict_day("2026-06-02")
    with tempfile.TemporaryDirectory(prefix="soxl_r12_") as tmp:
        args.start_date=args.end_date="2026-06-02"; args.out_dir=tmp; args.include_us_equity_holidays=True; args.skip_review_pack=True; args.no_progress=True
        result=run_research(bars,args)
        if not (result["report_dir"]/"09_entry_cap_retention.csv").exists(): raise AssertionError("missing cap retention")
        if not (result["report_dir"]/"06_causal_swing_hierarchy.csv").exists(): raise AssertionError("missing swing hierarchy")
    print("[self-test] PASS",flush=True); return 0


def main(argv:Sequence[str]|None=None)->int:
    args=parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    return 0 if run_research(_load_1m(args),args) else 1


if __name__=="__main__":
    raise SystemExit(main())
