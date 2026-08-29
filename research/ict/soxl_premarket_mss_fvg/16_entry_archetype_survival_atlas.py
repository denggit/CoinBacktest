#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R16: entry-archetype survival atlas.

R15 showed that daily liquidity traversals are common; the bottleneck is not
opportunity scarcity but *where to enter*.  R16 therefore freezes R15's path
universe and compares causal entry archetypes with a special focus on avoiding
entries that are stopped almost immediately.

Entry families:
- raid reclaim -> next-open market;
- raid reclaim -> source-level retest limit;
- first-any / first-visible MSS -> next-open market;
- first-any / first-visible MSS -> break/closest FVG limit (near and CE);
- first-visible MSS -> quantitative Order Block proxy (open / midpoint);
- first-visible MSS -> OB x FVG overlap midpoint;
- 2m visible structure -> 1m FVG execution.

No Swing +/- $0.10 gate is used.  Future bars are used only by lifecycle replay
and 25/50/75/100% path labels.
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
)
from src.research_common.ict.entry_archetype_survival import (
    EntryArchetypeConfig,
    attach_approach_compression_features,
    attach_causal_entry_state,
    attach_path_metadata,
    build_hybrid_2m_structure_1m_fvg_candidates,
    build_mss_fvg_entry_candidates,
    build_mss_market_entry_candidates,
    build_ob_fvg_overlap_candidates,
    build_order_block_entry_candidates,
    build_reclaim_entry_candidates,
    period_label,
    replay_entry_survival,
    select_first_mss_narratives,
    summarize_compression_bins,
    summarize_entry_archetypes,
    summarize_fixed_feature_bins,
    summarize_stop_survival_features,
)
from src.research_common.ict.premarket_mss_fvg import (
    NY_TZ,
    build_data_quality_table,
    eligible_ny_dates,
    ny_date_bounds_to_source_naive,
    source_naive_to_new_york,
)
from src.research_common.ict.spot_perp_overlap import build_equity_proxy_data_quality_table, densify_equity_minutes_causally
from src.research_common.ict.structure_entry_semantics import StructureSemanticConfig, build_r13_primary_break_fvg_compact, build_visible_swing_catalog
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_R15_CACHE = "data/reports/research/ict/soxl/mss/r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r16_entry_archetype_survival_atlas"


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R16 entry archetype survival atlas")
    p.add_argument("--data-source", choices=("alpaca", "okx"), default="alpaca")
    p.add_argument("--symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--r15-cache-dir", default=DEFAULT_R15_CACHE)
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
        raw = loader.fetch_data_by_date_range(start_ny.tz_convert("UTC"), end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1), local_only=bool(args.local_only))
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
    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _validate_r15_manifest(cache: Path, args: argparse.Namespace) -> dict[str, object]:
    mf = cache / "14_manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"R15 cache manifest missing: {mf}")
    data = json.loads(mf.read_text(encoding="utf-8"))
    if str(data.get("data_source")) != str(args.data_source):
        raise RuntimeError(f"R15 cache data_source={data.get('data_source')} does not match {args.data_source}")
    if str(data.get("start_date")) != str(args.start_date) or str(data.get("end_date")) != str(args.end_date):
        raise RuntimeError(f"R15 cache window {data.get('start_date')}->{data.get('end_date')} does not match requested {args.start_date}->{args.end_date}")
    return data


def _read_csv_selected(path: Path, wanted: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in wanted if c in header]
    return pd.read_csv(path, usecols=use, low_memory=False)


def _load_r15_cache(cache: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = _validate_r15_manifest(cache, args)
    paths = pd.read_csv(cache / "03_daily_path_outcomes.csv", low_memory=False)
    primary_cols = [
        "ny_date","range_model","path_event_id","event_id","trade_side","first_raid_time","source_level_price","target_price","lower_price","upper_price","range_width_abs",
        "execution_tf","execution_tf_minutes","terminal_version","terminal_extreme_time","terminal_extreme_price","mss_reference_time","mss_reference_price","mss_reference_available_time",
        "causal_visibility_percentile","visibility_score","two_sided_excursion_vs_prior_range","local_prominence_vs_prior_range","break_bar_start","break_available_time","break_wick_cross","break_close_cross",
        "break_open","break_high","break_low","break_close","terminal_to_break_minutes","directional_bar_fraction","path_net_distance_abs","path_efficiency","break_overshoot_abs",
        "structure_visibility_tier_r13","reference_model_r13","narrative_attempt_sequence_r13",
    ]
    primary = _read_csv_selected(cache / "06_causal_mss_narratives.csv", primary_cols)
    entry_cols = [
        "ny_date","range_model","path_event_id","event_id","trade_side","first_raid_side","first_raid_time","first_reclaim_time",
        "source_level_price","target_price","lower_price","upper_price","range_width_abs","traversal_complete","max_progress_fraction",
        "execution_tf","execution_tf_minutes","terminal_version","terminal_extreme_time","terminal_extreme_price",
        "mss_reference_time","mss_reference_price","break_available_time","break_bar_start","break_close_cross","causal_visibility_percentile","structure_visibility_tier_r13",
        "terminal_to_break_minutes","directional_bar_fraction","path_efficiency","break_overshoot_abs","entry_model_r13","entry_order_type","entry_price","entry_available_time","stop_price",
        "fvg_near_edge_entry","fvg_far_edge","fvg_size_abs","fvg_middle_relation_to_break","target_price","nearest_internal_target_price",
    ]
    r15_entries = _read_csv_selected(cache / "07_entry_candidate_atlas.csv", entry_cols)
    for df in (paths, primary, r15_entries):
        for c in [x for x in df.columns if x.endswith("_time") or x in {"break_bar_start","entry_available_time","terminal_extreme_time","mss_reference_time","first_raid_time"}]:
            # Multi-year New York CSVs contain both -04:00 and -05:00 offsets.
            # Parse through UTC, then convert back to a single timezone-aware
            # dtype so pandas 3/4 does not fall back to mixed object datetimes.
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    return paths, primary, r15_entries, manifest


def _rebuild_minimal_r15(bars: pd.DataFrame, args: argparse.Namespace, days: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback when no R15 cache is supplied; keeps the same causal semantics."""
    pcfg = DailyLiquidityPathConfig(round_trip_cost=float(args.round_trip_cost))
    ranges = build_daily_range_definitions(bars, days, config=pcfg)
    paths = build_daily_path_outcomes(bars, ranges, config=pcfg)
    sweeps = path_events_to_sweep_events(paths)
    scfg = StructureSemanticConfig(execution_timeframes=(1,2), structure_lookback_minutes=180, absolute_entry_buffer=0.10)
    swings = build_visible_swing_catalog(bars, days, config=scfg)
    pparts: list[pd.DataFrame] = []; eparts: list[pd.DataFrame] = []
    # The fallback only needs primary narratives. FVG entries are intentionally
    # omitted; users with R15 results should use --r15-cache-dir for the full
    # R16 FVG comparison without regenerating the wide R15 entry atlas.
    for day_text, ds in sweeps.groupby("ny_date", sort=True):
        sw = swings.loc[swings["ny_date"].astype(str).eq(str(day_text))]
        if sw.empty: continue
        p, _, _ = build_r13_primary_break_fvg_compact(bars, ds, sw, config=scfg)
        if not p.empty: pparts.append(p)
    primary = pd.concat(pparts, ignore_index=True, sort=False) if pparts else pd.DataFrame()
    return paths, primary, pd.DataFrame()


def _build_candidate_catalog(bars: pd.DataFrame, paths: pd.DataFrame, primary: pd.DataFrame, r15_entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cfg = EntryArchetypeConfig(round_trip_cost=float(args.round_trip_cost))
    first_any = select_first_mss_narratives(primary, min_visibility=None)
    first_visible = select_first_mss_narratives(primary, min_visibility=cfg.visible_swing_percentile)
    parts: list[pd.DataFrame] = []
    def add(x: pd.DataFrame) -> None:
        if x is not None and not x.empty: parts.append(x)

    add(build_reclaim_entry_candidates(bars, paths, config=cfg))
    for selected, label in ((first_any, "first_any"), (first_visible, "first_visible")):
        add(build_mss_market_entry_candidates(bars, selected, label=label, config=cfg))
        if not r15_entries.empty:
            add(build_mss_fvg_entry_candidates(r15_entries, selected, label=label))
    # OB-style research is deliberately tied to the first *visible* MSS so a
    # mathematically tiny micro pivot does not define the block.
    add(build_order_block_entry_candidates(bars, first_visible, label="first_visible", config=cfg))
    add(build_ob_fvg_overlap_candidates(bars, first_visible, label="first_visible", config=cfg))
    add(build_hybrid_2m_structure_1m_fvg_candidates(bars, first_visible, label="first_visible", config=cfg))
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts, ignore_index=True, sort=False)
    q = attach_path_metadata(q, paths)
    width = pd.to_numeric(q.get("range_width_abs"), errors="coerce")
    if "break_overshoot_abs" in q:
        q["break_overshoot_frac_range"] = pd.to_numeric(q["break_overshoot_abs"], errors="coerce").abs() / width
    if "fvg_size_abs" in q:
        q["fvg_size_frac_range"] = pd.to_numeric(q["fvg_size_abs"], errors="coerce") / width
    q = attach_causal_entry_state(bars, q, config=cfg)
    # One research candidate per physical path x archetype.  Different
    # archetypes remain parallel counterfactuals, but one archetype cannot
    # multiply trades through repeated MSS attempts from the same sweep.
    q["_signal"] = pd.to_datetime(q["entry_available_time"])
    q = q.sort_values(["path_event_id","entry_archetype","_signal"], kind="mergesort").drop_duplicates(["path_event_id","entry_archetype"], keep="first").drop(columns=["_signal"]).reset_index(drop=True)
    return q


def _frequency_summary(candidates: pd.DataFrame, replayed: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows=[]
    valid_days = int(paths["ny_date"].astype(str).nunique()) if not paths.empty else 0
    for (model, arch), g in candidates.groupby(["range_model","entry_archetype"], sort=True):
        r = replayed.loc[(replayed["range_model"].eq(model)) & (replayed["entry_archetype"].eq(arch))] if not replayed.empty else pd.DataFrame()
        rows.append({
            "range_model":model,"entry_archetype":arch,"valid_days":valid_days,
            "candidate_days":int(g["ny_date"].astype(str).nunique()),"candidate_days_per_all_days":float(g["ny_date"].astype(str).nunique()/max(1,valid_days)),
            "filled_days":int(r.loc[r["filled"].fillna(False).astype(bool),"ny_date"].astype(str).nunique()) if not r.empty else 0,
            "filled_days_per_all_days":float(r.loc[r["filled"].fillna(False).astype(bool),"ny_date"].astype(str).nunique()/max(1,valid_days)) if not r.empty else 0.0,
        })
    return pd.DataFrame(rows)


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str,Path]:
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    stage=ProgressReporter(label="[research] R16 stages", total=10, every=1, enabled=not args.no_progress)
    days=eligible_ny_dates(bars,start_date=args.start_date,end_date=args.end_date,exclude_equity_holidays=not args.include_us_equity_holidays)
    quality=build_equity_proxy_data_quality_table(bars,days) if args.data_source=="alpaca" else build_data_quality_table(bars,days,required_coverage=float(args.required_day_coverage))
    valid=set(quality.loc[quality["coverage_pass"].fillna(False).astype(bool),"ny_date"].astype(str)); days=[pd.Timestamp(x).date() for x in sorted(valid)]
    if not days: raise RuntimeError("no valid sessions")
    stage.update(1)

    cache=Path(args.r15_cache_dir) if str(args.r15_cache_dir).strip() else None
    if cache is not None and cache.exists():
        print(f"[cache] loading R15 causal path cache: {cache}", flush=True)
        paths,primary,r15_entries,r15_manifest=_load_r15_cache(cache,args)
        ranges=pd.read_csv(cache/"02_daily_range_definitions.csv",low_memory=False)
    else:
        print("[cache] R15 cache not found; rebuilding minimal causal path/MSS universe (FVG variants unavailable in fallback)",flush=True)
        paths,primary,r15_entries=_rebuild_minimal_r15(bars,args,days); ranges=build_daily_range_definitions(bars,days,config=DailyLiquidityPathConfig(round_trip_cost=float(args.round_trip_cost))); r15_manifest={}
    stage.update(1)

    print(f"[stage 3/10] approach/compression features paths={len(paths):,}",flush=True)
    paths=attach_approach_compression_features(bars,paths,config=EntryArchetypeConfig(round_trip_cost=float(args.round_trip_cost))); stage.update(1)
    print(f"[stage 4/10] entry archetypes primary={len(primary):,} r15_fvg_entries={len(r15_entries):,}",flush=True)
    candidates=_build_candidate_catalog(bars,paths,primary,r15_entries,args); stage.update(1)
    print(f"[stage 5/10] replay survival candidates={len(candidates):,}",flush=True)
    replayed=replay_entry_survival(bars,candidates,config=EntryArchetypeConfig(round_trip_cost=float(args.round_trip_cost))); stage.update(1)

    score=summarize_entry_archetypes(replayed); stage.update(1)
    rp=replayed.copy()
    if not rp.empty: rp["period"]=[period_label(x) for x in rp["ny_date"]]
    period=summarize_entry_archetypes(rp,group_extra=("period",)) if not rp.empty else pd.DataFrame(); stage.update(1)
    survival=summarize_stop_survival_features(replayed)
    bins=summarize_fixed_feature_bins(replayed)
    compression=summarize_compression_bins(replayed)
    freq=_frequency_summary(candidates,replayed,paths); stage.update(1)

    _write(quality,out/"01_data_quality.csv")
    _write(ranges,out/"02_daily_range_definitions.csv")
    _write(paths,out/"03_daily_paths_with_approach_features.csv")
    _write(candidates,out/"04_entry_archetype_candidates.csv")
    _write(replayed,out/"05_entry_survival_lifecycle.csv")
    _write(score,out/"06_entry_archetype_scorecard.csv")
    _write(period,out/"07_entry_archetype_period_stability.csv")
    _write(survival,out/"08_immediate_stop_vs_survivor_features.csv")
    _write(bins,out/"09_fixed_feature_stop_risk_atlas.csv")
    _write(compression,out/"10_approach_compression_atlas.csv")
    _write(freq,out/"11_daily_entry_frequency.csv")
    golden=replayed.loc[replayed["ny_date"].astype(str).eq(str(args.golden_date))].copy() if not replayed.empty else pd.DataFrame()
    _write(golden,out/f"12_golden_replay_{args.golden_date}.csv")

    design=(
        "# R16 Entry Archetype Survival Atlas\n\n"
        f"- Source: {args.data_source}\n- Window: {args.start_date} -> {args.end_date}\n"
        "- R15 path outcomes are frozen; R16 does not select only historically profitable target states.\n"
        "- Primary question: which causal entry archetype avoids immediate SL and reaches 50/75/100% of the dealing range before the terminal-extreme stop?\n"
        "- Entry families: raid reclaim market/retest, MSS market, MSS FVG near/CE, quantitative OB proxy open/mid, OB-FVG overlap, 2m structure + 1m FVG.\n"
        "- Order Block is explicitly a quantitative proxy: the last opposing closed candle in the selected displacement leg.\n"
        "- Compression/three-wave approach is recorded as a causal feature, never a hard gate.\n"
        "- Same physical path contributes at most one candidate per archetype; repeated MSS attempts do not inflate one archetype's frequency.\n"
        "- Swing +/- $0.10 is not used. No absolute-dollar chase cap exists.\n"
        "- Immediate-stop labels (1/3/5/10m) and 25/50/75/100 first-hit outcomes use future bars only after the entry has been generated.\n"
        f"- Baseline round-trip cost: {float(args.round_trip_cost):.4%}.\n"
        "- R16 is entry discovery, not a final executable strategy freeze; any exclusion rule discovered here must later be frozen and forward-validated.\n"
    )
    (out/"00_research_design.md").write_text(design,encoding="utf-8")
    manifest={"research_id":"R16","data_source":args.data_source,"start_date":args.start_date,"end_date":args.end_date,"valid_sessions":len(days),
              "r15_cache":str(cache) if cache else "","r15_manifest":r15_manifest,"path_rows":len(paths),"primary_mss_rows":len(primary),"entry_candidates":len(candidates),
              "filled_candidates":int(replayed["filled"].fillna(False).sum()) if not replayed.empty else 0,"round_trip_cost":float(args.round_trip_cost)}
    (out/"13_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    stage.update(1); stage.close()
    if not args.skip_review_pack:
        try: finalize_research_report(out)
        except Exception as exc: print(f"[review-pack] warning: {exc}",flush=True)
    print("[done] R16 entry archetype survival atlas",flush=True)
    if not score.empty:
        show=[c for c in ["range_model","entry_archetype","execution_tf","filled","immediate_stop_5m_rate","milestone_50_before_stop_rate","milestone_75_before_stop_rate","milestone_100_before_stop_rate","profit_factor_exit_50","profit_factor_exit_75","profit_factor_exit_100"] if c in score]
        print(score[show].sort_values(["range_model","entry_archetype"]).head(80).to_string(index=False),flush=True)
    return {"out_dir":out}


def _synthetic_day() -> pd.DataFrame:
    idx=pd.date_range("2026-08-05 04:00","2026-08-05 16:29",freq="1min",tz=NY_TZ)
    n=len(idx); base=np.full(n,105.0); mins=idx.hour*60+idx.minute
    pre=mins<510; x=np.linspace(0,6*np.pi,pre.sum()); base[pre]=105+4.5*np.sin(x)
    post=np.flatnonzero(mins>=510); t=np.arange(len(post))
    # Low raid -> reclaim -> pullback -> structured displacement to the other side.
    vals=np.where(t<30,103.5-3.8*t/30,99.65+1.6*np.minimum(np.maximum(t-30,0),15)/15)
    vals=np.where(t>=45,101.25-0.7*np.minimum(t-45,12)/12,vals)
    vals=np.where(t>=57,100.55+11.0*np.minimum(t-57,110)/110,vals)
    vals=np.where(t>=167,111.5+0.25*np.sin((t-167)/6),vals); base[post]=vals
    op=base; cl=base+0.03*np.sin(np.arange(n)/2); hi=np.maximum(op,cl)+0.16; lo=np.minimum(op,cl)-0.16
    return pd.DataFrame({"open":op,"high":hi,"low":lo,"close":cl,"volume":1000.0},index=idx)


def self_test() -> None:
    bars=_synthetic_day(); day=pd.Timestamp("2026-08-05").date(); pcfg=DailyLiquidityPathConfig()
    ranges=build_daily_range_definitions(bars,[day],config=pcfg); paths=build_daily_path_outcomes(bars,ranges,config=pcfg)
    paths=attach_approach_compression_features(bars,paths)
    reclaim=build_reclaim_entry_candidates(bars,paths)
    assert not reclaim.empty and {"raid_reclaim_next_open_market","raid_reclaim_level_retest_limit"}.issubset(set(reclaim["entry_archetype"]))
    reclaim=attach_path_metadata(reclaim,paths); reclaim=attach_causal_entry_state(bars,reclaim)
    replay=replay_entry_survival(bars,reclaim)
    assert not replay.empty and {"stop_within_5m","milestone_50_before_stop","net_return_exit_50"}.issubset(replay.columns)
    score=summarize_entry_archetypes(replay)
    assert not score.empty and "immediate_stop_5m_rate" in score
    print("R16 self-test PASS")


def main(argv: Sequence[str] | None=None) -> int:
    args=parse_args(argv)
    if args.self_test:
        self_test(); return 0
    run_research(_load_1m(args),args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
