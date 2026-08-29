#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R07: broaden ICT research beyond the existing SSL-exhaustion Long sleeve.

Families studied
----------------
1. Proper BSL reversal Short / SSL reversal Long confirmation audit using the
   already-causal R02/R03.3 entries.  A sweep-only path is never treated as an
   actual trade.
2. Liquidity expansion continuation, both directions: close-through key
   liquidity -> directional FVG -> resting limit retracement entry.
3. Reversal FVG corridor scalp: after an episode reclaim has already confirmed
   direction, wait with a limit order inside the first directional FVG and use
   either structural liquidity or a causally-existing opposite FVG as target.

R07 is a breadth atlas, not a portfolio promotion.  It also reports monthly
opportunity overlap with R06 to see whether new families can fill the long
underwater periods without pretending unvalidated families are capital-ready.
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
from src.research_common.ict_mss2.r07 import (  # noqa: E402
    R07Config,
    build_family_complementarity,
    build_fvg_lifecycle,
    build_liquidity_expansion_continuations,
    build_reversal_confirmation_atlas,
    build_reversal_fvg_corridor_scalps,
    r07_causal_audit,
    summarize_family_outcomes,
    summarize_fvg_target_scalps,
    summarize_reversal_atlas,
    summarize_reversal_target_grid,
    summarize_family_target_grid,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "7.1.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_ICT_FAMILY_EXPANSION_ATLAS_R07"
EDGE_ID = "RESEARCH_ONLY_ETH_ICT_COMPLEMENTARY_FAMILIES"
TITLE = "ETH ICT MSS2 R07 ICT Family Expansion Atlas"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_R033_DIR = "data/reports/research/ict/mss2/r03_3_liquidity_hierarchy_entry_exit"
DEFAULT_R06_DIR = "data/reports/research/ict/mss2/r06_adaptive_risk_position_lifecycle"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r07_ict_family_expansion_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--r033-report-dir", default=DEFAULT_R033_DIR)
    p.add_argument("--r06-report-dir", default=DEFAULT_R06_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit-market-roundtrip-cost", type=float, default=0.0008)
    p.add_argument("--stop-buffer-bps", type=float, default=2.0)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _optional(path: Path) -> pd.DataFrame:
    return _read(path) if path.exists() else pd.DataFrame()



def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_upstream_coverage(r02: Path, r033: Path, r06: Path, stages: pd.DataFrame, hierarchy: pd.DataFrame, requested_end: str) -> None:
    """Refuse silent mixed-window research.

    R07 depends on persisted upstream reports.  Changing only R07's end date must
    never make an old June report look like an August study.
    """
    req = pd.Timestamp(requested_end)
    r02m = _read_json(r02 / "00_manifest.json")
    r06m = _read_json(r06 / "00_manifest.json")
    checks = [
        ("R02", r02m.get("research_end_date") or r02m.get("end_date")),
        ("R06", r06m.get("end_date") or r06m.get("research_end_date")),
    ]
    stale = []
    for name, value in checks:
        if not value:
            stale.append(f"{name}: missing end-date metadata")
            continue
        if pd.Timestamp(value) < req:
            stale.append(f"{name}: {value}")
    if stale:
        raise RuntimeError(
            "R07 requested end_date=" + str(requested_end) + " but upstream coverage is stale (" + "; ".join(stale) + "). "
            "Rerun R02 -> R03.3 -> R05 -> R06 with the same end date before R07."
        )
    if "stage_id" in stages.columns and "stage_id" in hierarchy.columns:
        base_ids = pd.Index(stages["stage_id"].astype(str).dropna().unique())
        hier_ids = pd.Index(hierarchy["stage_id"].astype(str).dropna().unique())
        missing = base_ids.difference(hier_ids)
        if len(missing):
            raise RuntimeError(
                f"R03.3 hierarchy is stale/incomplete versus R02: missing {len(missing):,} R02 stage_ids. "
                "Rerun R03.3 with the same end date before R07."
            )

def _engineering_audit(
    bars: pd.DataFrame,
    lifecycle: pd.DataFrame,
    stages: pd.DataFrame,
    hierarchy: pd.DataFrame,
    reversal: pd.DataFrame,
    continuation: pd.DataFrame,
    corridor: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame([
        {"check":"bare_1m_rows","value":len(bars)},
        {"check":"liquidity_lifecycle_rows","value":len(lifecycle)},
        {"check":"r02_episode_stage_rows","value":len(stages)},
        {"check":"r033_hierarchy_stage_rows","value":len(hierarchy)},
        {"check":"proper_reversal_trade_rows","value":len(reversal)},
        {"check":"continuation_limit_trade_rows","value":len(continuation)},
        {"check":"corridor_limit_trade_rows","value":len(corridor)},
        {"check":"continuation_market_entries","value":int((continuation.get("entry_kind",pd.Series(dtype=str)).astype(str)!="fvg_limit").sum()) if len(continuation) else 0},
        {"check":"corridor_market_entries","value":int((corridor.get("entry_kind",pd.Series(dtype=str)).astype(str)!="fvg_limit").sum()) if len(corridor) else 0},
    ])


def _reversal_direction_summary(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame()
    a=atlas.loc[atlas.get("source_table","").astype(str).eq("r02_actual_entry")].copy()
    if a.empty:
        return pd.DataFrame()
    x=pd.to_numeric(a.get("target_htf240_net_return_cost2x"),errors="coerce")
    a=a.assign(_net=x)
    rows=[]
    for key,p in a.groupby(["family","trigger_type","execution_minutes"],dropna=False,sort=True):
        v=p["_net"].dropna()
        gp=float(v.loc[v>0].sum()); gl=float(-v.loc[v<0].sum())
        rows.append({"family":key[0],"trigger_type":key[1],"execution_minutes":int(key[2]),"trades":len(p),"resolved":len(v),"pf_2x_htf240":gp/gl if gl>1e-12 else np.nan,"mean_net_2x_htf240":float(v.mean()) if len(v) else np.nan})
    return pd.DataFrame(rows)



def _manual_recent_rows(frame: pd.DataFrame, *, target: str = "htf240", n: int = 10, logic: str = "") -> pd.DataFrame:
    """Compact latest executable examples for chart-by-chart manual review.

    This is a review artifact only.  It never feeds a research feature or rule.
    """
    if frame is None or frame.empty or "entry_time" not in frame.columns:
        return pd.DataFrame()
    x = frame.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], errors="coerce")
    x = x.dropna(subset=["entry_time"]).sort_values("entry_time", kind="stable")
    if x.empty:
        return pd.DataFrame()
    outcome_col = f"target_{target}_outcome" if f"target_{target}_outcome" in x.columns else ("fvg_target_outcome" if "fvg_target_outcome" in x.columns else "")
    if outcome_col:
        resolved = x.loc[x[outcome_col].astype(str).isin(["target", "stop"])].copy()
        if not resolved.empty:
            x = resolved
    x = x.tail(int(n)).copy()
    if logic:
        x["manual_review_logic"] = logic
    preferred = [
        "family", "quality_rule", "episode_id", "stage_id", "trade_event_id",
        "trade_direction", "execution_minutes", "trigger_type",
        "sweep_available_time_1m", "signal_bar_time", "signal_available_time",
        "entry_kind", "limit_variant", "entry_time", "entry_price",
        "fvg_lower", "fvg_upper", "fvg_proximal", "fvg_ce",
        "stop_variant", "stop_price",
        f"target_{target}_price", f"target_{target}_outcome",
        f"target_{target}_exit_time", f"target_{target}_holding_minutes",
        f"target_{target}_net_return_base", f"target_{target}_net_return_cost2x",
        "fvg_target_price", "fvg_target_outcome", "fvg_target_exit_time",
        "fvg_target_holding_minutes", "fvg_target_net_return_base",
        "fvg_target_net_return_cost2x", "manual_review_logic",
    ]
    cols = [c for c in preferred if c in x.columns]
    return x.loc[:, cols].reset_index(drop=True)


def _write_manual_review(out: Path, reversal: pd.DataFrame, cont: pd.DataFrame, cont_out: pd.DataFrame, corridor: pd.DataFrame, corr_out: pd.DataFrame, corr_fvg: pd.DataFrame) -> None:
    review = out / "manual_review"
    review.mkdir(parents=True, exist_ok=True)

    # Proper reversal uses the existing confirmed R02/R03.3 entries.  Keep Long/Short separate.
    if reversal is not None and not reversal.empty:
        for direction, name in ((1, "01_recent_10_ssl_reversal_long.csv"), (-1, "02_recent_10_bsl_reversal_short.csv")):
            part = reversal.loc[pd.to_numeric(reversal.get("trade_direction"), errors="coerce").eq(direction)].copy()
            rows = _manual_recent_rows(part, target="htf240", logic="Confirmed liquidity reversal entry from the causal R02/R03.3 trigger; structural stop and frozen opposing 4H-liquidity target are shown for chart review.")
            rows.to_csv(review / name, index=False, encoding="utf-8-sig")

    # Use one fixed canonical sensitivity for visual validation so the same episode is not repeated four times.
    if cont_out is not None and not cont_out.empty:
        c = cont_out.copy()
        if "limit_variant" in c.columns:
            c = c.loc[c["limit_variant"].astype(str).eq("proximal")]
        if "stop_variant" in c.columns:
            c = c.loc[c["stop_variant"].astype(str).eq("episode_structural")]
        rows = _manual_recent_rows(c, target="htf240", logic="Liquidity close-through/acceptance -> directional FVG -> resting proximal limit; stop is the episode structural invalidation and TP is the frozen opposing 4H-liquidity target.")
        rows.to_csv(review / "03_recent_10_continuation_proximal.csv", index=False, encoding="utf-8-sig")

    if corr_out is not None and not corr_out.empty:
        c = corr_out.copy()
        if "limit_variant" in c.columns:
            c = c.loc[c["limit_variant"].astype(str).eq("proximal")]
        if "stop_variant" in c.columns:
            c = c.loc[c["stop_variant"].astype(str).eq("fvg_invalidation")]
        rows = _manual_recent_rows(c, target="htf240", logic="Reversal already confirmed -> first same-direction FVG -> resting proximal limit; local FVG invalidation stop; structural 4H target is included only as a comparison objective.")
        rows.to_csv(review / "04_recent_10_fvg_corridor_structural_target.csv", index=False, encoding="utf-8-sig")

    if corridor is not None and not corridor.empty and corr_fvg is not None and not corr_fvg.empty:
        keys = [c for c in ("trade_event_id", "fvg_target_price") if c in corridor.columns and c in corr_fvg.columns]
        if keys:
            cf = corridor.merge(corr_fvg, on=keys, how="left", validate="one_to_one", suffixes=("", "_label"))
            if "limit_variant" in cf.columns:
                cf = cf.loc[cf["limit_variant"].astype(str).eq("proximal")]
            if "stop_variant" in cf.columns:
                cf = cf.loc[cf["stop_variant"].astype(str).eq("fvg_invalidation")]
            rows = _manual_recent_rows(cf, logic="Reversal confirmed -> resting proximal FVG limit -> causally-existing opposite FVG target.  If that target was touched before the entry limit filled, the order is stale/cancelled and is absent from this executable review file.")
            rows.to_csv(review / "05_recent_10_fvg_to_fvg_corridor.csv", index=False, encoding="utf-8-sig")

    (review / "README.md").write_text(
        "# Manual chart review\n\n"
        "These files contain the latest executable examples for visual K-line validation. They are review artifacts only and are never used as causal features or selection rules.\n\n"
        "Start with `03_recent_10_continuation_proximal.csv`, `04_recent_10_fvg_corridor_structural_target.csv`, and `05_recent_10_fvg_to_fvg_corridor.csv` for the new R07 families. Use `01/02` to manually compare confirmed SSL/BSL reversals.\n",
        encoding="utf-8",
    )

def _source_basis_markdown() -> str:
    return """# R07 ICT source basis and quantization map

R07 does **not** assume every ICT narrative transfers to ETH.  The source material is used only to define candidate mechanics; ETH data decides whether they have edge.

## Source-derived candidate mechanics

- **Episode 3 — Internal Range Liquidity & Market Structure Shifts**: after sell-side liquidity is taken, a break of a short-term high can matter; symmetrically, after buy-side liquidity is taken, a later break of a short-term low can matter.  The lesson explicitly describes a bearish MSS after buy-side is taken, then a bearish FVG retracement used for a short.  R07 therefore does not equate BSL sweep with an immediate short.
- **Episode 6 — FVG + MSS institutional order-flow model**: displacement/MSS creates the trade idea, while the FVG retracement is the entry mechanism.  R07 continuation/scalp entries are resting limits; no market chase is allowed for the small-range FVG family.
- **Episode 16 — Multiple setups**: a displaced move can create an FVG, price can return into it, and the first objective can be nearby liquidity while a remainder can pursue a later objective.  This motivates the corridor scalp as a separate family from the long-run runner.
- **Episode 18 — Order-block validity**: an order block is discussed together with an imbalance; R07 records whether the pre-FVG opposite candle overlaps the imbalance (`ob_overlap_flag`) rather than calling every last opposite candle an order block.
- **Episode 23/26/40 — multiple objectives / partials / FVG and liquidity draws**: R07 keeps structural-liquidity targets and opposite-FVG targets separate so a small scalp is not judged with the same exit as a multi-day reversal.

## Quantization rules frozen in R07

1. BSL/SSL reversal entries come from actual causal reclaim/MSS/FVG entries already produced by R02/R03.3.  Sweep-only rows are descriptive controls only.
2. Continuation requires a **close through** already-swept key liquidity, then a directional FVG.  Entry is a limit at either FVG proximal edge or consequent encroachment (CE).  Both are frozen sensitivity variants, not optimized thresholds.
3. A continuation `ob_overlap_flag` is descriptive: an opposite candle before the FVG/displacement must overlap the imbalance.  It is not an admission gate.
4. FVG corridor scalp starts only after direction has already been confirmed by episode reclaim.  It waits for the first same-direction FVG and uses a resting proximal/CE limit.  It never enters at the FVG signal close.
5. Opposite-FVG targets must already exist and remain not fully rebalanced at the signal time.  Future FVGs cannot become targets retroactively.
6. Same-bar limit-fill/target ambiguity is pessimistic: stop can trigger on the fill bar, target begins on the next 1m bar.
7. No NY Open filter is used.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    cfg=R07Config(limit_market_roundtrip_cost=float(args.limit_market_roundtrip_cost),stop_buffer_bps=float(args.stop_buffer_bps)).validate()
    r02=Path(args.r02_report_dir); r033=Path(args.r033_report_dir); r06=Path(args.r06_report_dir)

    print("[r07] load R02/R03.3 causal research state",flush=True)
    lifecycle=_read(r02/"01_liquidity_lifecycle_causal.csv", usecols=["level_id","pivot_side","level_price","source_timeframe_min","active_pos_1m","sweep_pos_1m"])
    stages=_read(r02/"04_sweep_episode_stages_causal.csv", usecols=["stage_id","sweep_available_time_1m","episode_start_pos_1m","min_consumed_level_price_cum","max_consumed_level_price_cum","episode_extreme_so_far","liquidity_side"])
    features=_read(r02/"10_trade_features_causal.csv", usecols=["trade_event_id","stage_id","episode_id","trade_direction","execution_minutes","trigger_type","entry_time","signal_available_time","stop_price"])
    reversal_targets=("any","pool2","pool2tf","htf60","htf240","htf1440","r1p0","r2p0","r3p0","r5p0")
    reversal_label_cols={"trade_event_id","stage_id","episode_id","target_htf240_holding_minutes"}
    reversal_label_cols.update({f"target_{name}_net_return_cost2x" for name in reversal_targets})
    labels=_read(r02/"11_trade_structural_exit_labels.csv", usecols=lambda c: c in reversal_label_cols)
    hierarchy=_read(r033/"04_episode_stages_hierarchy_causal.csv.gz")
    refreshed=_optional(r033/"13i_refreshed_mss_displacement_trade_rows.csv.gz")
    r06base=_read(r06/"03_base_opportunities_causal.csv.gz", usecols=["entry_time"]) if (r06/"03_base_opportunities_causal.csv.gz").exists() else pd.DataFrame()
    _assert_upstream_coverage(r02, r033, r06, stages, hierarchy, args.end_date)

    print("[r07] proper BSL/SSL reversal confirmation audit",flush=True)
    reversal=build_reversal_confirmation_atlas(features,labels,hierarchy,refreshed)
    rev_sum,rev_year=summarize_reversal_atlas(reversal)
    rev_targets,rev_targets_year=summarize_reversal_target_grid(reversal)
    rev_dir=_reversal_direction_summary(reversal)

    print("[r07] load bare 1m K",flush=True)
    loader=OKXDataLoader(symbol=args.symbol,timeframe="1m",db_dir=args.data_dir)
    bars=loader.fetch_data_by_date_range(args.warmup_start_date,args.end_date)
    if bars.empty: raise RuntimeError("R07 bare 1m K is empty")

    print("[r07] causal FVG lifecycle 1m/2m/5m",flush=True)
    fvg=build_fvg_lifecycle(bars,execution_minutes=cfg.execution_minutes,show_progress=not args.no_progress)

    print("[r07] liquidity expansion -> FVG limit continuation",flush=True)
    cont,cont_out=build_liquidity_expansion_continuations(bars,hierarchy,stages,lifecycle,config=cfg,show_progress=not args.no_progress)
    cont_sum,cont_year=summarize_family_outcomes(cont,cont_out)
    cont_targets,cont_targets_year=summarize_family_target_grid(cont,cont_out)

    print("[r07] confirmed reversal -> FVG limit corridor scalp",flush=True)
    corridor,corr_out,corr_fvg=build_reversal_fvg_corridor_scalps(bars,features,hierarchy,fvg,lifecycle,config=cfg,show_progress=not args.no_progress)
    corr_struct,corr_struct_year=summarize_family_outcomes(corridor,corr_out)
    corr_targets,corr_targets_year=summarize_family_target_grid(corridor,corr_out)
    corr_fvg_sum,corr_fvg_year=summarize_fvg_target_scalps(corridor,corr_fvg)

    monthly,overlap=build_family_complementarity(r06base,[cont,corridor],start_date=args.start_date,end_date=args.end_date)
    audit=r07_causal_audit(cont,corridor,fvg)
    eng=_engineering_audit(bars,lifecycle,stages,hierarchy,reversal,cont,corridor)

    # Compact outputs first; large row tables remain gzip review sources.
    eng.to_csv(out/"01_engineering_audit.csv",index=False)
    pd.DataFrame([{"script_version":SCRIPT_VERSION,"experiment_id":EXPERIMENT_ID,"symbol":args.symbol,"warmup_start":args.warmup_start_date,"start":args.start_date,"end":args.end_date,"limit_market_roundtrip_cost":cfg.limit_market_roundtrip_cost,"note":"R07 is breadth discovery; no family is auto-promoted."}]).to_csv(out/"02_frozen_design.csv",index=False)
    rev_sum.to_csv(out/"03_reversal_confirmation_summary.csv",index=False); rev_year.to_csv(out/"04_reversal_confirmation_year_summary.csv",index=False); rev_dir.to_csv(out/"05_bsl_ssl_reversal_direction_summary.csv",index=False)
    rev_targets.to_csv(out/"06_reversal_target_grid.csv",index=False); rev_targets_year.to_csv(out/"07_reversal_target_grid_year.csv",index=False)
    cont_sum.to_csv(out/"08_continuation_htf240_summary.csv",index=False); cont_year.to_csv(out/"09_continuation_htf240_year.csv",index=False)
    cont_targets.to_csv(out/"10_continuation_target_grid.csv",index=False); cont_targets_year.to_csv(out/"11_continuation_target_grid_year.csv",index=False)
    corr_struct.to_csv(out/"12_fvg_corridor_htf240_summary.csv",index=False); corr_struct_year.to_csv(out/"13_fvg_corridor_htf240_year.csv",index=False)
    corr_targets.to_csv(out/"14_fvg_corridor_structural_target_grid.csv",index=False); corr_targets_year.to_csv(out/"15_fvg_corridor_structural_target_grid_year.csv",index=False)
    corr_fvg_sum.to_csv(out/"16_fvg_corridor_opposite_fvg_target_summary.csv",index=False); corr_fvg_year.to_csv(out/"17_fvg_corridor_opposite_fvg_target_year.csv",index=False)
    monthly.to_csv(out/"18_family_monthly_opportunity_counts.csv",index=False); overlap.to_csv(out/"19_family_same_hour_overlap.csv",index=False)
    audit.to_csv(out/"20_causal_audit.csv",index=False)
    reversal.to_csv(out/"21_reversal_confirmation_trade_rows.csv.gz",index=False,compression="gzip")
    cont.to_csv(out/"22_continuation_trade_features.csv.gz",index=False,compression="gzip"); cont_out.to_csv(out/"23_continuation_structural_exit_rows.csv.gz",index=False,compression="gzip")
    corridor.to_csv(out/"24_fvg_corridor_trade_features.csv.gz",index=False,compression="gzip"); corr_out.to_csv(out/"25_fvg_corridor_structural_exit_rows.csv.gz",index=False,compression="gzip"); corr_fvg.to_csv(out/"26_fvg_corridor_fvg_target_labels.csv.gz",index=False,compression="gzip")
    fvg.to_csv(out/"27_fvg_lifecycle_causal.csv.gz",index=False,compression="gzip")
    _write_manual_review(out, reversal, cont, cont_out, corridor, corr_out, corr_fvg)
    (out/"R07_ICT_SOURCE_BASIS.md").write_text(_source_basis_markdown(),encoding="utf-8")

    readme="""# R07 ICT Family Expansion Atlas\n\nR07 deliberately broadens the research instead of further filtering the existing Long reversal sleeve.  Read `03-05` first to answer whether properly-confirmed BSL reversal Shorts really lack edge.  Read `06-07` for close-through-liquidity continuation.  Read `08-11` for the small FVG-limit corridor family.  `12-13` asks whether new families add opportunity in months/hours where the R06 sleeve is inactive.\n\n**Do not compare small FVG corridor trades with the same target architecture as multi-day reversals.** The report therefore keeps structural-liquidity and opposite-FVG targets separate. No NY Open gate is used.\n"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    (out/"GPT_REVIEW_PROMPT.md").write_text("Review R07 as a breadth/complementarity study. Reject any claim that BSL sweep itself is a short signal. Prioritize: (1) BSL reversal with actual confirmation, (2) continuation long/short cross-year 2x-cost stability, (3) FVG corridor maker-entry economics after 2x/3x costs, (4) monthly complementarity with R06, (5) causal audit. Do not auto-select the maximum PF subgroup.\n",encoding="utf-8")
    finalize_research_report(out,experiment_id=EXPERIMENT_ID,edge_id=EDGE_ID,title=TITLE)
    print(f"[r07] done -> {out}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
