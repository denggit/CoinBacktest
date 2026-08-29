#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R18 — ICT opposite-liquidity probability hypothesis study.

Research question:
    Once one side of the causally frozen 08:30 prominent-15m liquidity pair is
    raided, how does the probability of reaching the opposite external liquidity
    change as causal ICT evidence arrives, and which entry geometry maximizes
    P(opposite TP before terminal-extreme SL)?

This is deliberately not a 25/50/75 path study and not a PnL parameter search.
It reuses R15/R16 causal caches, fits simple discovery-only logistic models, and
checks the frozen models on 2025 and 2026 without refitting.
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

from src.research_common.ict.opposite_liquidity_probability import (
    DEFAULT_ENTRY_ARCHETYPES,
    DEFAULT_RANGE_MODEL,
    ENTRY_HYPOTHESIS_FEATURES,
    EVENT_HYPOTHESIS_FEATURES,
    ProbabilityStudyConfig,
    assert_no_forbidden_features,
    build_entry_probability_dataset,
    build_event_snapshot_dataset,
    calibration_table,
    fit_cumulative_hypothesis_models,
    paired_entry_scorecard,
    stage_baseline_summary,
)

from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R18 ICT opposite-liquidity probability hypotheses")
    p.add_argument("--r15-cache-dir", default="data/reports/research/ict/soxl/mss/r15_daily_liquidity_traversal_path_atlas_alpaca_2023_2026_08")
    p.add_argument("--r16-cache-dir", default="data/reports/research/ict/soxl/mss/r16_entry_archetype_survival_atlas_alpaca_2023_2026_08")
    p.add_argument("--start-date", default="2023-07-01")
    p.add_argument("--end-date", default="2026-08-14")
    p.add_argument("--range-model", default=DEFAULT_RANGE_MODEL)
    p.add_argument("--visible-swing-percentile", type=float, default=0.50)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--out-dir", default="data/reports/research/ict/soxl/mss/r18_opposite_liquidity_probability_hypotheses")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    return p.parse_args(argv)


def _manifest(path: Path, filename: str, research_id: str, args: argparse.Namespace) -> dict[str, object]:
    p = path / filename
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if str(data.get("research_id")) != research_id:
        raise RuntimeError(f"expected {research_id} cache, got {data.get('research_id')} at {p}")
    for key in ("start_date", "end_date"):
        if str(data.get(key)) != str(getattr(args, key)):
            raise RuntimeError(f"{research_id} cache {key}={data.get(key)} != requested {getattr(args, key)}")
    return data


def _read_selected(path: Path, wanted: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in wanted if c in header]
    return pd.read_csv(path, usecols=use, low_memory=False)


def _load_r15(cache: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mf = _manifest(cache, "14_manifest.json", "R15", args)
    path_cols = [
        "ny_date","range_model","range_available_time","path_start_time","upper_price","lower_price",
        "upper_source_time","lower_source_time","upper_confirmation_time","lower_confirmation_time",
        "upper_prominence_score","lower_prominence_score","range_width_abs","first_raid_side","first_raid_time",
        "source_level_price","target_price","traversal_complete","opposite_hit_time","opposite_hit_minutes_from_raid",
        "first_raid_penetration_frac_range","first_reclaim_time","reclaim_minutes","path_event_id",
        "approach_efficiency","approach_distance_contraction_ratio","approach_recent_range_vs_prior",
        "approach_monotonic_swing_count","approach_three_swing_contraction","approach_three_swing_distance_ratio",
    ]
    narrative_cols = [
        *path_cols,
        "event_id","sweep_time","liquidity_side","level_price","liquidity_family","level_type",
        "execution_tf","execution_tf_minutes","terminal_version","terminal_extreme_time","terminal_extreme_price",
        "mss_reference_time","mss_reference_price","mss_reference_available_time","mss_reference_relation",
        "visibility_score","causal_visibility_percentile","two_sided_excursion_vs_prior_range",
        "local_prominence_vs_prior_range","reference_is_latest_newly_broken",
        "reference_is_highest_visibility_newly_broken","reference_is_outermost_barrier_newly_broken",
        "break_bar_start","break_available_time","break_wick_cross","break_close_cross","break_open","break_high",
        "break_low","break_close","terminal_to_break_minutes","directional_bar_fraction","path_net_distance_abs",
        "path_efficiency","break_overshoot_abs","narrative_attempt_sequence_r13",
    ]
    paths = _read_selected(cache / "03_daily_path_outcomes.csv", path_cols)
    narratives = _read_selected(cache / "06_causal_mss_narratives.csv", narrative_cols)
    paths = paths.loc[paths["range_model"].astype(str).eq(args.range_model)].copy()
    narratives = narratives.loc[narratives["range_model"].astype(str).eq(args.range_model)].copy()
    return paths, narratives, mf


def _load_r16(cache: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    mf = _manifest(cache, "13_manifest.json", "R16", args)
    wanted = [
        "ny_date","range_model","entry_archetype","entry_family","execution_tf","entry_order_type","trade_side",
        "path_event_id","event_id","first_raid_time","source_level_price","target_price","upper_price","lower_price",
        "upper_confirmation_time","lower_confirmation_time","upper_prominence_score","lower_prominence_score",
        "range_width_abs","first_raid_side","first_raid_penetration_frac_range","first_reclaim_time",
        "approach_efficiency","approach_distance_contraction_ratio","approach_recent_range_vs_prior",
        "approach_monotonic_swing_count","approach_three_swing_contraction",
        "entry_available_time","entry_price","stop_price","filled","fill_time","fill_wait_minutes",
        "milestone_100_before_stop","rr_to_100","net_return_exit_100",
        "causal_visibility_percentile","terminal_version","terminal_extreme_time","terminal_extreme_price",
        "mss_reference_time","mss_reference_price","mss_reference_available_time","mss_reference_relation",
        "two_sided_excursion_vs_prior_range","local_prominence_vs_prior_range",
        "reference_is_latest_newly_broken","reference_is_highest_visibility_newly_broken",
        "reference_is_outermost_barrier_newly_broken","break_bar_start","break_available_time","break_wick_cross",
        "break_close_cross","terminal_to_break_minutes","directional_bar_fraction","path_net_distance_abs",
        "path_efficiency","break_overshoot_abs","fvg_size_frac_range","initial_risk_frac_range",
        "entry_progress_fraction","signal_minutes_from_raid","raid_count_so_far_at_entry",
        "penetration_so_far_frac_range","source_reclaimed_at_entry",
    ]
    life = _read_selected(cache / "05_entry_survival_lifecycle.csv", wanted)
    life = life.loc[life["range_model"].astype(str).eq(args.range_model)].copy()
    return life, mf


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    # Compare each cumulative model to the immediately previous model inside a period.
    order = {m: i for i, m in enumerate(out["model_set"].drop_duplicates().tolist())}
    out["_order"] = out["model_set"].map(order)
    out = out.sort_values(["period", "_order"], kind="mergesort")
    out["delta_brier_vs_previous"] = out.groupby("period")["brier"].diff()
    out["delta_log_loss_vs_previous"] = out.groupby("period")["log_loss"].diff()
    out["delta_auc_vs_previous"] = out.groupby("period")["auc"].diff()
    return out.drop(columns="_order")


def _entry_fill_summary(lifecycle: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    q = lifecycle.loc[
        lifecycle["range_model"].astype(str).eq(args.range_model)
        & lifecycle["entry_archetype"].astype(str).isin(DEFAULT_ENTRY_ARCHETYPES)
    ].copy()
    if q.empty:
        return pd.DataFrame()
    if q["filled"].dtype != bool:
        q["filled"] = q["filled"].astype(str).str.lower().map({"true":True,"false":False}).fillna(False)
    rows=[]
    for arch,g in q.groupby("entry_archetype",sort=True):
        f=g.loc[g["filled"]]
        y = f["milestone_100_before_stop"].astype(str).str.lower().map({"true":1,"false":0}).dropna()
        rows.append({
            "entry_archetype":arch,
            "candidate_events":int(g["path_event_id"].nunique()),
            "filled_events":int(f["path_event_id"].nunique()),
            "fill_rate":float(f["path_event_id"].nunique()/max(g["path_event_id"].nunique(),1)),
            "tp_before_terminal_sl_rate":float(y.mean()) if len(y) else np.nan,
        })
    return pd.DataFrame(rows)


def _golden(event_pred: pd.DataFrame, entry_pred: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frames=[]
    for label,df in (("event_probability",event_pred),("entry_tp_before_sl_probability",entry_pred)):
        if df.empty or "ny_date" not in df:
            continue
        q=df.loc[df["ny_date"].astype(str).eq(str(args.golden_date))].copy()
        if not q.empty:
            q.insert(0,"probability_type",label)
            frames.append(q)
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()


def _design_text(args: argparse.Namespace) -> str:
    return f"""# SOXL ICT R18 — Opposite-Liquidity Probability Hypotheses

## Frozen research question
Universe: `{args.range_model}` only.

One side of the external-liquidity pair is raided.  The event target is only:
`opposite external liquidity reached by session end`.

For actual filled entries the second target is only:
`opposite external liquidity TP before the terminal-extreme SL`.

No 25/50/75 dealing-range milestone is used as a target, feature or filter.

## Hypotheses
- H1 liquidity context: prominence/age/approach geometry of the frozen pair;
- H2 terminal maturity: penetration, terminal evolution, reclaim and timing state;
- H3 meaningful MSS: causal swing visibility/prominence and what structure was actually broken;
- H4 displacement: efficiency, directional dominance and normalized overshoot;
- H5 mitigation/entry geometry: market vs break-FVG/CE vs OB/overlap/hybrid execution;
- H6 cross-timeframe confirmation: only confirmation already visible by the snapshot.

## Validation discipline
The logistic probability model is fitted once on 2023H2-2024.  It is not refit
or threshold-tuned on 2025 or 2026.  Reports emphasize Brier score, log loss and
calibration in addition to AUC.  Future full-day fields and post-entry MFE/MAE
are explicitly forbidden as predictors.
"""


def _self_test() -> int:
    from src.research_common.ict.opposite_liquidity_probability import (
        assert_no_forbidden_features, build_event_snapshot_dataset,
        fit_cumulative_hypothesis_models, paired_entry_scorecard,
    )
    assert_no_forbidden_features(EVENT_HYPOTHESIS_FEATURES)
    assert_no_forbidden_features(ENTRY_HYPOTHESIS_FEATURES)
    rng=np.random.default_rng(7)
    days=pd.date_range("2023-07-03",periods=120,freq="B")
    paths=[]; narr=[]
    for i,d in enumerate(days):
        ev=f"e{i}"; y=int((i%5) in (0,1))
        paths.append({"ny_date":str(d.date()),"range_model":DEFAULT_RANGE_MODEL,"path_event_id":ev,
                      "upper_price":110.,"lower_price":100.,"upper_confirmation_time":d,"lower_confirmation_time":d,
                      "upper_prominence_score":2.0,"lower_prominence_score":1.5,"range_width_abs":10.,
                      "first_raid_side":"low","first_raid_time":d+pd.Timedelta(hours=9),"source_level_price":100.,
                      "target_price":110.,"traversal_complete":bool(y),"first_raid_penetration_frac_range":0.02,
                      "first_reclaim_time":d+pd.Timedelta(hours=9,minutes=3),"approach_efficiency":.5,
                      "approach_distance_contraction_ratio":.8,"approach_recent_range_vs_prior":1.,
                      "approach_monotonic_swing_count":2,"approach_three_swing_contraction":False})
        for tf,mins in (("1m",5),("2m",7)):
            narr.append({**paths[-1],"event_id":ev,"execution_tf":tf,"causal_visibility_percentile":.6,
                         "break_available_time":d+pd.Timedelta(hours=9,minutes=mins),"terminal_version":2,
                         "terminal_extreme_time":d+pd.Timedelta(hours=9,minutes=2),"terminal_extreme_price":99.5,
                         "mss_reference_time":d+pd.Timedelta(hours=8,minutes=55),"mss_reference_available_time":d+pd.Timedelta(hours=8,minutes=56),
                         "mss_reference_relation":"pre_sweep","two_sided_excursion_vs_prior_range":1.2+y*.2+rng.normal(0,.05),
                         "local_prominence_vs_prior_range":1.0,"reference_is_latest_newly_broken":True,
                         "reference_is_highest_visibility_newly_broken":True,"reference_is_outermost_barrier_newly_broken":bool(y),
                         "break_wick_cross":True,"break_close_cross":bool(y),"terminal_to_break_minutes":3.,
                         "directional_bar_fraction":.6+y*.1,"path_net_distance_abs":2.0,"path_efficiency":.5+y*.2,
                         "break_overshoot_abs":.5+y*.3,"narrative_attempt_sequence_r13":1})
    e=build_event_snapshot_dataset(pd.DataFrame(paths),pd.DataFrame(narr))
    assert len(e)==240 and e["target_opposite_by_eod"].isin([0,1]).all()
    m,p,c=fit_cumulative_hypothesis_models(e,target="target_opposite_by_eod",groups=EVENT_HYPOTHESIS_FEATURES,
                                           group_order=list(EVENT_HYPOTHESIS_FEATURES))
    assert not m.empty and not p.empty and not c.empty
    s,pa=paired_entry_scorecard(pd.DataFrame({"path_event_id":["a","a","b","b"],"entry_archetype":["x","y","x","y"],
                                              "target_tp_before_terminal_sl":[1,0,0,1],"rr_to_100":[2,2,2,2]}))
    assert len(s)==2 and len(pa)==1 and int(pa.iloc[0]["common_filled_events"])==2
    print("R18 self-test PASS",flush=True)
    return 0


def run_research(args: argparse.Namespace) -> bool:
    if args.self_test:
        return _self_test()==0
    assert_no_forbidden_features(EVENT_HYPOTHESIS_FEATURES)
    assert_no_forbidden_features(ENTRY_HYPOTHESIS_FEATURES)
    r15=Path(args.r15_cache_dir); r16=Path(args.r16_cache_dir)
    stage=ProgressReporter(label="[research] R18 stages", total=8, every=1, enabled=not args.no_progress)
    print("[stage 1/8] load causal R15/R16 caches",flush=True)
    paths,narratives,m15=_load_r15(r15,args)
    lifecycle,m16=_load_r16(r16,args); stage.update(1)
    cfg=ProbabilityStudyConfig(range_model=args.range_model,visible_swing_percentile=float(args.visible_swing_percentile))

    print(f"[stage 2/8] event snapshots paths={len(paths):,} narratives={len(narratives):,}",flush=True)
    event=build_event_snapshot_dataset(paths,narratives,config=cfg)
    baseline=stage_baseline_summary(paths,event,range_model=args.range_model); stage.update(1)

    print(f"[stage 3/8] opposite-liquidity probability models snapshots={len(event):,}",flush=True)
    event_metrics=[]; event_preds=[]; event_coefs=[]; event_cals=[]
    group_order=list(EVENT_HYPOTHESIS_FEATURES)
    for snapshot_stage,g in event.groupby("stage",sort=True):
        print(f"  [model] {snapshot_stage} rows={len(g):,}",flush=True)
        met,pred,coef=fit_cumulative_hypothesis_models(g,target="target_opposite_by_eod",groups=EVENT_HYPOTHESIS_FEATURES,group_order=group_order)
        if not met.empty:
            met.insert(0,"stage",snapshot_stage); event_metrics.append(_metric_deltas(met))
        if not pred.empty:
            event_preds.append(pred); cal=calibration_table(pred,"target_opposite_by_eod"); cal.insert(0,"stage",snapshot_stage); event_cals.append(cal)
        if not coef.empty:
            coef.insert(0,"stage",snapshot_stage); event_coefs.append(coef)
    event_metrics_df=pd.concat(event_metrics,ignore_index=True) if event_metrics else pd.DataFrame()
    event_pred_df=pd.concat(event_preds,ignore_index=True) if event_preds else pd.DataFrame()
    event_coef_df=pd.concat(event_coefs,ignore_index=True) if event_coefs else pd.DataFrame()
    event_cal_df=pd.concat(event_cals,ignore_index=True) if event_cals else pd.DataFrame(); stage.update(1)

    print("[stage 4/8] build filled-entry probability dataset",flush=True)
    entry=build_entry_probability_dataset(lifecycle,event,config=cfg,entry_archetypes=DEFAULT_ENTRY_ARCHETYPES)
    fill_summary=_entry_fill_summary(lifecycle,args)
    score,pairs=paired_entry_scorecard(entry); stage.update(1)

    print(f"[stage 5/8] TP-before-SL probability model fills={len(entry):,}",flush=True)
    entry_metrics=entry_pred=entry_coef=entry_cal=pd.DataFrame()
    if not entry.empty:
        entry_order=["H1_liquidity_context","H2_terminal_maturity","H3_mss_structure","H4_displacement","H5_mitigation_entry","H6_cross_tf"]
        entry_metrics,entry_pred,entry_coef=fit_cumulative_hypothesis_models(
            entry,target="target_tp_before_terminal_sl",groups=ENTRY_HYPOTHESIS_FEATURES,group_order=entry_order)
        entry_metrics=_metric_deltas(entry_metrics)
        entry_cal=calibration_table(entry_pred,"target_tp_before_terminal_sl")
    stage.update(1)

    print("[stage 6/8] golden replay + reports",flush=True)
    golden=_golden(event_pred_df,entry_pred,args)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"00_research_design.md").write_text(_design_text(args),encoding="utf-8")
    _write(baseline,out/"01_sweep_mss_opposite_baseline.csv")
    _write(event,out/"02_event_snapshot_dataset.csv")
    _write(event_metrics_df,out/"03_event_hypothesis_incremental_metrics.csv")
    _write(event_cal_df,out/"04_event_probability_calibration.csv")
    _write(event_coef_df,out/"05_event_logistic_coefficients.csv")
    _write(fill_summary,out/"06_entry_fill_and_tp_first_summary.csv")
    _write(score,out/"07_entry_filled_scorecard.csv")
    _write(pairs,out/"08_entry_paired_same_sweep_comparison.csv")
    _write(entry_metrics,out/"09_entry_probability_incremental_metrics.csv")
    _write(entry_cal,out/"10_entry_probability_calibration.csv")
    _write(entry_coef,out/"11_entry_logistic_coefficients.csv")
    _write(golden,out/f"12_golden_replay_{args.golden_date}.csv")
    manifest={
        "research_id":"R18","data_source":"cached_R15_R16_causal_atlases","start_date":args.start_date,"end_date":args.end_date,
        "range_model":args.range_model,"visible_swing_percentile":float(args.visible_swing_percentile),
        "r15_cache":str(r15),"r16_cache":str(r16),"r15_manifest":m15,"r16_manifest":m16,
        "path_rows":len(paths),"narrative_rows":len(narratives),"event_snapshot_rows":len(event),"filled_entry_rows":len(entry),
        "entry_archetypes":list(DEFAULT_ENTRY_ARCHETYPES),"feature_hypotheses":{**EVENT_HYPOTHESIS_FEATURES,"H5_mitigation_entry":ENTRY_HYPOTHESIS_FEATURES["H5_mitigation_entry"]},
        "validation":"fit discovery_2023H2_2024 once; evaluate unchanged on validation_2025 and forward_2026",
    }
    (out/"13_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); stage.update(1)
    print("[stage 7/8] audit no 25/50/75 predictors/targets",flush=True)
    forbidden_tokens=("milestone_25","milestone_50","milestone_75")
    model_feature_text=json.dumps(manifest["feature_hypotheses"],ensure_ascii=False)
    assert not any(t in model_feature_text for t in forbidden_tokens)
    stage.update(1)
    print("[stage 8/8] finalize review pack",flush=True)
    if not args.skip_review_pack:
        try:
            finalize_research_report(out)
        except Exception as exc:
            print(f"[review-pack] warning: {exc}",flush=True)
    stage.update(1); stage.close()
    print(f"[done] {out}",flush=True)
    return True


def main(argv: Sequence[str] | None=None) -> int:
    args=parse_args(argv)
    return 0 if run_research(args) else 1


if __name__=="__main__":
    raise SystemExit(main())
