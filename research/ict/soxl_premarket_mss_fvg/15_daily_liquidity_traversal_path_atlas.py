#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R15: daily liquidity traversal path atlas.

R14 intentionally narrowed the candidate universe until it became executable,
but the resulting trade frequency was far below discretionary observation.
R15 therefore steps back *before* profitability filtering:

1. freeze a causal premarket dealing range every valid day;
2. study every day's first raid of either side and whether price later reaches
   the opposite boundary;
3. classify the full path, including shallow raids, repeated raids, reclaim and
   partial traversal;
4. attach all causal 1m/2m MSS/FVG candidates after that first raid, without
   target-state or Swing-$0.10 gating;
5. measure which execution styles actually capture the observed traversals.

Future bars are used only for outcome labels/replay.  Range levels, swing
confirmation, MSS, displacement and FVG availability remain strictly causal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE
from src.research_common.ict.daily_liquidity_path import (
    DailyLiquidityPathConfig,
    build_daily_path_outcomes,
    build_daily_range_definitions,
    path_events_to_sweep_events,
    period_label,
    replay_entry_candidates,
    summarize_entry_capture,
    summarize_path_by_period,
    summarize_path_outcomes,
)
from src.research_common.ict.premarket_mss_fvg import (
    NY_TZ,
    build_data_quality_table,
    eligible_ny_dates,
    ny_date_bounds_to_source_naive,
    source_naive_to_new_york,
)
from src.research_common.ict.semantic_consolidation import (
    add_market_next_open_choices,
    attach_market_next_open_from_bars,
    consolidate_fvg_entry_choices,
)
from src.research_common.ict.spot_perp_overlap import (
    build_equity_proxy_data_quality_table,
    densify_equity_minutes_causally,
)
from src.research_common.ict.structure_entry_semantics import (
    StructureSemanticConfig,
    build_r13_primary_break_fvg_compact,
    build_visible_swing_catalog,
)
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r15_daily_liquidity_traversal_path_atlas"


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R15 daily liquidity traversal path atlas")
    p.add_argument("--data-source", choices=("alpaca", "okx"), default="alpaca")
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
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(symbol=args.alpaca_symbol, timeframe="1Min", feed=args.alpaca_feed,
                                   adjustment=args.alpaca_adjustment, data_dir=args.data_dir)
        raw = loader.fetch_data_by_date_range(start_ny.tz_convert("UTC"), end_ny.tz_convert("UTC")-pd.Timedelta(minutes=1), local_only=bool(args.local_only))
        if raw.empty:
            raise RuntimeError("Alpaca loader returned no data")
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars = raw.copy(); bars.index = idx.tz_convert(NY_TZ); bars.index.name = "bar_start_ny"
        bars = densify_equity_minutes_causally(bars)
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
    mins = bars.index.hour*60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _build_structure_atlas(
    bars: pd.DataFrame,
    days: list,
    path_sweeps: pd.DataFrame,
    *,
    no_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = StructureSemanticConfig(execution_timeframes=(1,2), structure_lookback_minutes=180, absolute_entry_buffer=0.10)
    swings = build_visible_swing_catalog(bars, days, config=cfg)
    if path_sweeps.empty or swings.empty:
        return swings, pd.DataFrame(), pd.DataFrame()
    sweep_groups = {str(k):g for k,g in path_sweeps.groupby("ny_date", sort=True)}
    swing_groups = {str(k):g for k,g in swings.groupby("ny_date", sort=False)}
    pparts=[]; fparts=[]
    prog=ProgressReporter(label="[stage5] daily MSS/FVG path scan", total=max(1,len(sweep_groups)), every=10, enabled=not no_progress)
    for day_text, ds in sweep_groups.items():
        sw=swing_groups.get(day_text, pd.DataFrame())
        if not sw.empty:
            p,f,_=build_r13_primary_break_fvg_compact(bars, ds, sw, config=cfg)
            if not p.empty: pparts.append(p)
            if not f.empty: fparts.append(f)
        prog.update(1)
    prog.close()
    primary=pd.concat(pparts,ignore_index=True,sort=False) if pparts else pd.DataFrame()
    fvgs=pd.concat(fparts,ignore_index=True,sort=False) if fparts else pd.DataFrame()
    return swings, primary, fvgs


def _attach_entries(bars: pd.DataFrame, primary: pd.DataFrame, fvgs: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    fvg_entries = consolidate_fvg_entry_choices(primary, fvgs) if not fvgs.empty else pd.DataFrame()
    market = add_market_next_open_choices(primary)
    if not market.empty:
        market = attach_market_next_open_from_bars(bars, market)
    parts=[x for x in (fvg_entries,market) if not x.empty]
    if not parts:
        return pd.DataFrame()
    entries=pd.concat(parts, ignore_index=True, sort=False)
    keep_path=["path_event_id","range_model","traversal_complete","path_archetype","opposite_hit_time","opposite_hit_minutes_from_raid",
               "first_raid_penetration_frac_range","max_same_side_penetration_frac_range","reclaim_minutes","same_side_raid_count","max_progress_fraction",
               "upper_price","lower_price","range_width_abs","target_price"]
    path_meta=paths[[c for c in keep_path if c in paths.columns]].drop_duplicates("path_event_id").copy()
    entries=entries.merge(path_meta, left_on="event_id", right_on="path_event_id", how="left", suffixes=("","_path"), validate="many_to_one")
    if "target_price_path" in entries.columns:
        entries["target_price"] = pd.to_numeric(entries["target_price_path"], errors="coerce")
    elif "target_price" in entries.columns:
        entries["target_price"] = pd.to_numeric(entries["target_price"], errors="coerce")
    entries["entry_distance_from_swing_frac_range"] = pd.to_numeric(entries.get("entry_distance_abs_r13", np.nan), errors="coerce") / pd.to_numeric(entries["range_width_abs"], errors="coerce")
    entries["break_overshoot_frac_range"] = pd.to_numeric(entries.get("break_overshoot_abs", np.nan), errors="coerce").abs() / pd.to_numeric(entries["range_width_abs"], errors="coerce")
    entries["path_net_distance_frac_range"] = pd.to_numeric(entries.get("path_net_distance_abs", np.nan), errors="coerce") / pd.to_numeric(entries["range_width_abs"], errors="coerce")
    return entries


def _success_failure_features(primary: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    cols=["path_event_id","traversal_complete","path_archetype","first_raid_penetration_frac_range","max_same_side_penetration_frac_range","reclaim_minutes","same_side_raid_count","max_progress_fraction","range_model","range_width_abs"]

    # The compact R13 builder already carries most path metadata through the
    # sweep event into ``primary`` (including ``range_model`` and
    # ``traversal_complete``).  Merging those columns again with pandas'
    # default suffixes turns them into ``range_model_x/_y`` and
    # ``traversal_complete_x/_y``; the later groupby then raises KeyError.
    # Attach only metadata that is genuinely missing so the canonical column
    # names remain stable.
    q=primary.copy()
    missing=[c for c in cols if c != "path_event_id" and c in paths.columns and c not in q.columns]
    if missing:
        path_meta=paths[["path_event_id", *missing]].drop_duplicates("path_event_id")
        q=q.merge(path_meta, left_on="event_id", right_on="path_event_id", how="left", validate="many_to_one")

    required=("range_model","execution_tf","traversal_complete")
    absent=[c for c in required if c not in q.columns]
    if absent:
        raise RuntimeError(f"success/failure feature table missing required columns after metadata attach: {absent}")

    features=["terminal_to_break_minutes","directional_bar_fraction","path_efficiency","break_overshoot_abs","causal_visibility_percentile","two_sided_excursion_vs_prior_range","local_prominence_vs_prior_range"]
    rows=[]
    for (model,tf,success),g in q.groupby(["range_model","execution_tf","traversal_complete"],sort=True,dropna=False):
        row={"range_model":model,"execution_tf":tf,"traversal_complete":bool(success),"n_narratives":len(g),"n_events":int(g["event_id"].nunique())}
        for f in features:
            if f in g: row[f"median_{f}"]=float(pd.to_numeric(g[f],errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_opportunity_counts(entries: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    base=paths.loc[paths["first_raid_side"].isin(["high","low"])][["ny_date","path_event_id","range_model","traversal_complete"]].copy()
    if entries.empty:
        base["mss_entry_candidates"]=0; return base
    counts=entries.groupby(["ny_date","event_id"],sort=False).size().rename("mss_entry_candidates").reset_index()
    base=base.merge(counts,left_on=["ny_date","path_event_id"],right_on=["ny_date","event_id"],how="left")
    base["mss_entry_candidates"]=base["mss_entry_candidates"].fillna(0).astype(int)
    return base.drop(columns=["event_id"],errors="ignore")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str,Path]:
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    stage=ProgressReporter(label="[research] R15 stages", total=10, every=1, enabled=not args.no_progress)
    days=eligible_ny_dates(bars,start_date=args.start_date,end_date=args.end_date,exclude_equity_holidays=not args.include_us_equity_holidays)
    quality=build_equity_proxy_data_quality_table(bars,days) if args.data_source=="alpaca" else build_data_quality_table(bars,days,required_coverage=float(args.required_day_coverage))
    valid=set(quality.loc[quality["coverage_pass"].fillna(False).astype(bool),"ny_date"].astype(str)); days=[pd.Timestamp(x).date() for x in sorted(valid)]
    if not days: raise RuntimeError("no valid sessions")
    stage.update(1)

    cfg=DailyLiquidityPathConfig(round_trip_cost=float(args.round_trip_cost))
    ranges=build_daily_range_definitions(bars,days,config=cfg); stage.update(1)
    paths=build_daily_path_outcomes(bars,ranges,config=cfg); stage.update(1)
    path_summary=summarize_path_outcomes(paths); path_period=summarize_path_by_period(paths); stage.update(1)

    path_sweeps=path_events_to_sweep_events(paths)
    swings,primary,fvgs=_build_structure_atlas(bars,days,path_sweeps,no_progress=args.no_progress); stage.update(1)
    entries=_attach_entries(bars,primary,fvgs,paths); stage.update(1)
    replayed=replay_entry_candidates(bars,entries,cost=float(args.round_trip_cost)) if not entries.empty else pd.DataFrame(); stage.update(1)
    capture=summarize_entry_capture(replayed); stage.update(1)
    success_failure=_success_failure_features(primary,paths)
    daily_counts=_daily_opportunity_counts(entries,paths); stage.update(1)

    # Explicit frequency tables answer whether the market offers ~daily paths
    # before any profitability filter is imposed.
    freq_rows=[]
    for model,g in daily_counts.groupby("range_model",sort=True):
        freq_rows.append({
            "range_model":model,"valid_sessions":int(paths.loc[paths["range_model"].eq(model),"ny_date"].nunique()),
            "raid_sessions":int(g["ny_date"].nunique()),
            "sessions_with_any_causal_entry_candidate":int(g.loc[g["mss_entry_candidates"].gt(0),"ny_date"].nunique()),
            "entry_candidate_day_rate":float(g.loc[g["mss_entry_candidates"].gt(0),"ny_date"].nunique()/max(1,paths.loc[paths["range_model"].eq(model),"ny_date"].nunique())),
            "mean_candidate_count_on_raid_day":float(g["mss_entry_candidates"].mean()) if len(g) else np.nan,
            "traversal_sessions":int(g.loc[g["traversal_complete"].fillna(False).astype(bool),"ny_date"].nunique()),
        })
    frequency=pd.DataFrame(freq_rows)

    _write(quality,out/"01_data_quality.csv")
    _write(ranges,out/"02_daily_range_definitions.csv")
    _write(paths,out/"03_daily_path_outcomes.csv")
    _write(path_summary,out/"04_path_archetype_summary.csv")
    _write(path_period,out/"05_path_period_stability.csv")
    _write(primary,out/"06_causal_mss_narratives.csv")
    _write(entries,out/"07_entry_candidate_atlas.csv")
    _write(replayed,out/"08_entry_lifecycle_replay.csv")
    _write(capture,out/"09_entry_capture_summary.csv")
    _write(success_failure,out/"10_success_vs_failure_structure_features.csv")
    _write(daily_counts,out/"11_daily_opportunity_counts.csv")
    _write(frequency,out/"12_frequency_summary.csv")
    golden=paths.loc[paths["ny_date"].astype(str).eq(str(args.golden_date))].copy()
    if not replayed.empty:
        g2=replayed.loc[replayed["ny_date"].astype(str).eq(str(args.golden_date))].copy()
        if not g2.empty: golden=g2
    _write(golden,out/f"13_golden_replay_{args.golden_date}.csv")

    design=(
        "# R15 Daily Liquidity Traversal Path Atlas\n\n"
        f"- Source: {args.data_source}\n- Window: {args.start_date} -> {args.end_date}\n"
        "- This study intentionally removes R14 target-state profitability filters.\n"
        "- Every valid day receives multiple predeclared causal range models: early 04:00-08:30 extremes, full 04:00-09:30 extremes, and the most prominent causally-confirmed 15m swing pair at 08:30/09:30.\n"
        "- For each range, the first intraday raid is labelled, then the entire path toward the opposite boundary is measured.\n"
        "- Outcome labels may use future bars; MSS/FVG candidate generation never does.\n"
        "- Same physical path event is studied once per range model; no EQL/partial-consume/fresh target state is required.\n"
        "- 1m and 2m execution are studied. 5m is excluded from this first-pass entry scan and can be reintroduced later as context if needed.\n"
        "- Swing +/- $0.10 is not a gate and is not used for selection.\n"
        f"- Baseline round-trip cost in replay: {float(args.round_trip_cost):.4%}.\n"
        "- R15 is a path-discovery atlas, not a final executable strategy freeze.\n"
    )
    (out/"00_research_design.md").write_text(design,encoding="utf-8")
    manifest={"research_id":"R15","data_source":args.data_source,"start_date":args.start_date,"end_date":args.end_date,
              "valid_sessions":len(days),"range_rows":len(ranges),"path_rows":len(paths),"path_raid_events":len(path_sweeps),
              "mss_narratives":len(primary),"entry_candidates":len(entries),"replayed_candidates":len(replayed),"round_trip_cost":float(args.round_trip_cost)}
    (out/"14_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    stage.update(1); stage.close()
    if not args.skip_review_pack:
        try:
            finalize_research_report(out)
        except Exception as exc:
            print(f"[review-pack] warning: {exc}",flush=True)
    print("[done] R15 daily liquidity traversal path atlas",flush=True)
    print(path_summary.to_string(index=False) if not path_summary.empty else "[summary] no paths",flush=True)
    print(frequency.to_string(index=False) if not frequency.empty else "[frequency] empty",flush=True)
    return {"out_dir":out}


def _synthetic_day() -> pd.DataFrame:
    # Causal toy path: early range 100-110, low raid, visible reversal, then high target.
    idx=pd.date_range("2026-08-05 04:00", "2026-08-05 16:29", freq="1min", tz=NY_TZ)
    n=len(idx); base=np.full(n,105.0)
    mins=idx.hour*60+idx.minute
    # premarket oscillation creates 15m pivots and fixed 100/110 extremes.
    pre=(mins<510); x=np.linspace(0,6*np.pi,pre.sum()); base[pre]=105+4.5*np.sin(x)
    # after 08:30 drift to raid 100 then reverse to >110.
    post=np.flatnonzero(mins>=510); t=np.arange(len(post)); vals=np.where(t<45,104-4.2*t/45,99.7+11.5*np.minimum(np.maximum(t-45,0),90)/90)
    vals=np.where(t>=135,111+0.3*np.sin((t-135)/8),vals); base[post]=vals
    op=base; cl=base+0.02*np.sin(np.arange(n)); hi=np.maximum(op,cl)+0.18; lo=np.minimum(op,cl)-0.18
    return pd.DataFrame({"open":op,"high":hi,"low":lo,"close":cl,"volume":1000.0},index=idx)


def self_test() -> None:
    bars=_synthetic_day(); day=pd.Timestamp("2026-08-05").date(); cfg=DailyLiquidityPathConfig()
    ranges=build_daily_range_definitions(bars,[day],config=cfg)
    assert not ranges.empty and {"early_extreme_0400_0830","full_premarket_extreme_0400_0930"}.issubset(set(ranges["range_model"]))
    paths=build_daily_path_outcomes(bars,ranges,config=cfg)
    early=paths.loc[paths["range_model"].eq("early_extreme_0400_0830")].iloc[0]
    assert early["first_raid_side"] in {"low","high"}
    assert pd.Timestamp(early["range_available_time"]) <= pd.Timestamp(early["first_raid_time"])
    sweeps=path_events_to_sweep_events(paths)
    assert not sweeps.empty and sweeps["sweep_time"].notna().all()
    print("R15 self-test PASS")


def main(argv: Sequence[str] | None=None) -> int:
    args=parse_args(argv)
    if args.self_test:
        self_test(); return 0
    run_research(_load_1m(args),args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
