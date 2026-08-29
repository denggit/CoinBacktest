#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R05: ETH liquidity reversal entry timing + structural stop/runner atlas.

R05 keeps the R04 major-reversal discovery intact but investigates the risk
architecture around it:

* same causal liquidity stage, compare 1m / 2m / 5m episode-reclaim entries;
* compare several *structural* initial stops instead of one episode-extreme SL;
* measure MAE before +0.5/+0.75/+1/+2/+3/+5% targets;
* test runner trailing only on 2m/5m/15m ITL/LTL or unusually large bullish
  displacement anchors.  1m trailing is intentionally excluded;
* no fixed TP is promoted.  3%/5% are right-tail diagnostics only, and 14d is
  censoring only.
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
from src.research_common.ict_mss2 import r03_globalize_legacy_trade_ids  # noqa: E402
from src.research_common.ict_mss2.r05 import (  # noqa: E402
    R05Config,
    attach_initial_structural_stops,
    build_displacement_anchor_atlas,
    build_execution_swing_hierarchy,
    build_exclusive_opportunity_buckets,
    build_initial_stop_target_atlas,
    build_quality_entry_universe,
    build_trailing_events,
    r05_causal_audit,
    simulate_structural_trailing,
    summarize_initial_stop_atlas,
    summarize_exclusive_opportunity_buckets,
    summarize_initial_stop_by_bucket,
    summarize_mae_by_bucket,
    summarize_trailing_by_bucket,
    summarize_mae_before_target,
    summarize_trailing_results,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "5.1.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_ENTRY_TIMING_STRUCTURAL_STOP_RUNNER_R05"
EDGE_ID = "RESEARCH_ONLY_ETH_LIQUIDITY_REVERSAL_STOP_RUNNER"
TITLE = "ETH ICT MSS2 R05 Entry Timing + Structural Stop / Runner Atlas"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_R033_DIR = "data/reports/research/ict/mss2/r03_3_liquidity_hierarchy_entry_exit"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r05_entry_timing_structural_stop_runner_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--r033-report-dir", default=DEFAULT_R033_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--stop-buffer-bps", type=float, default=2.0)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _load_r02_features(path: Path) -> pd.DataFrame:
    f = _read_csv(path / "10_trade_features_causal.csv")
    # Globalize legacy IDs without requiring the future-label table.
    f, _ = r03_globalize_legacy_trade_ids(f, None)
    return f


def _quality_entry_summary(opps: pd.DataFrame) -> pd.DataFrame:
    if opps.empty:
        return pd.DataFrame()
    rows = []
    for key, part in opps.groupby(["quality_rule", "execution_minutes"], dropna=False, sort=True):
        entry = pd.to_numeric(part["entry_price"], errors="coerce")
        ep_stop = pd.to_numeric(part.get("stop_episode_extreme"), errors="coerce")
        risk = (entry - ep_stop) / entry
        lag = (pd.to_datetime(part["entry_time"], errors="coerce") - pd.to_datetime(part["sweep_bar_time_1m"], errors="coerce")) / pd.Timedelta(minutes=1)
        rows.append({
            "quality_rule": key[0], "execution_minutes": int(key[1]), "opportunities": len(part),
            "episodes": int(part["episode_id"].nunique()),
            "median_minutes_sweep_to_entry": float(pd.to_numeric(lag, errors="coerce").median()),
            "p75_minutes_sweep_to_entry": float(pd.to_numeric(lag, errors="coerce").quantile(0.75)),
            "median_episode_stop_risk_pct": float(risk.median() * 100.0),
        })
    return pd.DataFrame(rows)


def _initial_stop_availability(opps: pd.DataFrame) -> pd.DataFrame:
    if opps.empty:
        return pd.DataFrame()
    stop_cols = [c for c in opps.columns if c.startswith("stop_") and not c.endswith("_price")]
    rows = []
    for key, part in opps.groupby(["quality_rule", "execution_minutes"], dropna=False, sort=True):
        for c in stop_cols:
            x = pd.to_numeric(part[c], errors="coerce")
            entry = pd.to_numeric(part["entry_price"], errors="coerce")
            risk = (entry - x) / entry
            rows.append({
                "quality_rule": key[0], "execution_minutes": int(key[1]),
                "stop_variant": c.replace("stop_", ""), "opportunities": len(part),
                "available": int(x.notna().sum()), "coverage": float(x.notna().mean()),
                "median_risk_pct": float(risk.median() * 100.0) if x.notna().any() else np.nan,
                "p75_risk_pct": float(risk.quantile(0.75) * 100.0) if x.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = R05Config(
        market_roundtrip_cost=float(args.market_roundtrip_cost),
        stop_buffer_bps=float(args.stop_buffer_bps),
    ).validate()

    print("[r05] load R02 causal entries + R03.3 hierarchy stages", flush=True)
    r02 = _load_r02_features(Path(args.r02_report_dir))
    hierarchy_stages = _read_csv(Path(args.r033_report_dir) / "04_episode_stages_hierarchy_causal.csv.gz")
    opps = build_quality_entry_universe(r02, hierarchy_stages)
    if opps.empty:
        raise RuntimeError("R05 built zero quality entry opportunities")

    print("[r05] load bare 1m K", flush=True)
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if bars.empty:
        raise RuntimeError("R05 bare 1m K is empty")

    print("[r05] causal 2m/5m/15m ST/IT/LT structure", flush=True)
    hier_by_tf: dict[int, pd.DataFrame] = {}
    for tf in cfg.trail_minutes:
        print(f"[r05] hierarchy {tf}m", flush=True)
        hier_by_tf[int(tf)] = build_execution_swing_hierarchy(bars, int(tf))

    print("[r05] structural initial-stop atlas", flush=True)
    opps = attach_initial_structural_stops(
        opps, bars, hierarchy_by_tf=hier_by_tf, config=cfg, show_progress=not args.no_progress
    )
    entry_summary = _quality_entry_summary(opps)
    stop_availability = _initial_stop_availability(opps)
    initial_outcomes, mae_before_tp = build_initial_stop_target_atlas(opps, bars, config=cfg, show_progress=not args.no_progress)
    stop_summary, stop_year = summarize_initial_stop_atlas(initial_outcomes)
    mae_summary = summarize_mae_before_target(mae_before_tp)

    print("[r05] mutually-exclusive realized opportunity buckets", flush=True)
    exclusive_buckets = build_exclusive_opportunity_buckets(opps, bars, config=cfg)
    bucket_summary, bucket_year = summarize_exclusive_opportunity_buckets(exclusive_buckets)
    bucket_stop_summary, bucket_stop_year = summarize_initial_stop_by_bucket(initial_outcomes, exclusive_buckets)
    bucket_mae_summary = summarize_mae_by_bucket(mae_before_tp, exclusive_buckets)

    print("[r05] causal structural/displacement trailing anchors", flush=True)
    trailing_events = build_trailing_events(bars, config=cfg)
    displacement_atlas = build_displacement_anchor_atlas(trailing_events)

    # Runner research is intentionally limited to the two quality families that
    # R04 showed as the most interesting frequency/quality trade-off.  This is
    # not a new admission optimization.
    runner = opps.loc[opps["quality_rule"].isin(["n3_4h_or_lt", "n4_4h_or_lt"])].copy()
    print(f"[r05] structural runner simulations opportunities={len(runner):,}", flush=True)
    trailing = simulate_structural_trailing(runner, bars, trailing_events, config=cfg, show_progress=not args.no_progress)
    trail_summary, trail_year = summarize_trailing_results(trailing)
    bucket_trail_summary, bucket_trail_year = summarize_trailing_by_bucket(trailing, exclusive_buckets)

    causal = r05_causal_audit(opps, hier_by_tf, trailing_events, trailing)
    violations = int(pd.to_numeric(causal["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"R05 causal audit failed violations={violations}")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "market_roundtrip_cost": cfg.market_roundtrip_cost,
        "stop_buffer_bps": cfg.stop_buffer_bps,
        "entry_timeframes": list(cfg.entry_minutes),
        "trailing_timeframes": list(cfg.trail_minutes),
        "max_horizon_minutes": cfg.max_horizon_minutes,
        "semantics": [
            "1m/2m/5m entry comparison uses the exact same first qualifying causal liquidity stage per episode.",
            "No 1m trailing stop is implemented.",
            "ITL/LTL trailing anchors become usable only after recursive right-side confirmation; activation starts on the next eligible 1m bar.",
            "Bullish displacement anchors use causal rolling percentile context; q90/q95/q99 are research bins, not promoted strategy thresholds.",
            "Trailing stops only ratchet upward and never loosen.",
            "3%/5% targets are right-tail diagnostics only. 14d is censoring, not a forced exit.",
            "Exclusive future opportunity buckets are reporting labels only: <0.3%, 0.3-1%, 1-3%, 3-5%, >=5% MFE before the frozen episode-extreme thesis stop. They never enter causal features.",
            "Nested target tables remain separate because they answer upgrade probabilities; exclusive bucket reports answer what genuinely short/medium/swing/major paths look like without long-tail contamination.",
        ],
        "entry_rows": int(len(opps)),
        "episodes": int(opps["episode_id"].nunique()),
        "trailing_anchor_events": int(len(trailing_events)),
        "causal_violations": violations,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"metric": "entry_rows", "value": len(opps)},
        {"metric": "unique_episodes", "value": opps["episode_id"].nunique()},
        {"metric": "trailing_anchor_events", "value": len(trailing_events)},
        {"metric": "causal_violations", "value": violations},
    ]).to_csv(out_dir / "01_engineering_audit.csv", index=False)
    (out_dir / "02_frozen_design.json").write_text(json.dumps({
        "quality_rules": sorted(opps["quality_rule"].astype(str).unique().tolist()),
        "entry_timeframes": list(cfg.entry_minutes),
        "initial_stop_variants": [
            "episode_extreme", "qualifying_stage_extreme", "reclaim_leg_extreme", "signal_bar_extreme",
            "itl_2m_at_entry", "itl_5m_at_entry", "itl_15m_at_entry",
        ],
        "runner_trailing_variants": [
            "itl_2m", "itl_5m", "itl_15m", "ltl_5m", "ltl_15m",
            "shock95_2m", "shock95_5m", "shock95_fvg_5m", "shock99_5m", "shock99_fvg_5m",
            "shock95_15m", "shock95_fvg_15m", "itl15_or_shock95_5m", "itl5_15_or_shock95_5m",
        ],
        "important": "No fixed TP is promoted; fixed targets only benchmark stop architecture and right-tail preservation.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep raw review outputs compressed and summaries compact.
    opp_cols = [c for c in opps.columns if not c.startswith("_")]
    opps.loc[:, opp_cols].copy().to_csv(out_dir / "03_entry_opportunities_causal.csv.gz", index=False, compression="gzip")
    entry_summary.to_csv(out_dir / "04_entry_timing_summary.csv", index=False)
    stop_availability.to_csv(out_dir / "05_initial_stop_availability.csv", index=False)
    stop_summary.to_csv(out_dir / "06_initial_stop_target_summary.csv", index=False)
    stop_year.to_csv(out_dir / "07_initial_stop_target_year_summary.csv", index=False)
    mae_summary.to_csv(out_dir / "08_mae_before_target_summary.csv", index=False)
    mae_before_tp.to_csv(out_dir / "09_mae_before_target_rows.csv.gz", index=False, compression="gzip")
    trailing_events.to_csv(out_dir / "10_trailing_anchor_events_causal.csv.gz", index=False, compression="gzip")
    displacement_atlas.to_csv(out_dir / "11_displacement_anchor_atlas.csv", index=False)
    trail_summary.to_csv(out_dir / "12_structural_trailing_summary.csv", index=False)
    trail_year.to_csv(out_dir / "13_structural_trailing_year_summary.csv", index=False)
    trailing.to_csv(out_dir / "14_structural_trailing_trade_rows.csv.gz", index=False, compression="gzip")
    causal.to_csv(out_dir / "15_causal_audit.csv", index=False)
    exclusive_buckets.to_csv(out_dir / "16_exclusive_opportunity_bucket_rows.csv.gz", index=False, compression="gzip")
    bucket_summary.to_csv(out_dir / "17_exclusive_opportunity_bucket_summary.csv", index=False)
    bucket_year.to_csv(out_dir / "18_exclusive_opportunity_bucket_year_summary.csv", index=False)
    bucket_stop_summary.to_csv(out_dir / "19_exclusive_bucket_initial_stop_target_summary.csv", index=False)
    bucket_stop_year.to_csv(out_dir / "20_exclusive_bucket_initial_stop_target_year_summary.csv", index=False)
    bucket_mae_summary.to_csv(out_dir / "21_exclusive_bucket_mae_before_target_summary.csv", index=False)
    bucket_trail_summary.to_csv(out_dir / "22_exclusive_bucket_structural_trailing_summary.csv", index=False)
    bucket_trail_year.to_csv(out_dir / "23_exclusive_bucket_structural_trailing_year_summary.csv", index=False)

    bucket_defs = {
        "under_0p3": "Resolved path never reaches +0.3% before frozen episode-extreme thesis stop / 14d censor.",
        "short_0p3_1p0": "Resolved path MFE is >=0.3% and <1.0% before thesis invalidation.",
        "medium_1p0_3p0": "Resolved path MFE is >=1.0% and <3.0% before thesis invalidation.",
        "swing_3p0_5p0": "Resolved path MFE is >=3.0% and <5.0% before thesis invalidation.",
        "major_ge_5p0": "Resolved path MFE reaches >=5.0% before thesis invalidation.",
    }
    bucket_root = out_dir / "opportunity_buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    for bucket_name, description in bucket_defs.items():
        bdir = bucket_root / bucket_name
        bdir.mkdir(parents=True, exist_ok=True)
        exclusive_buckets.loc[exclusive_buckets["opportunity_bucket"].eq(bucket_name)].to_csv(
            bdir / "01_opportunity_rows.csv.gz", index=False, compression="gzip"
        )
        bucket_summary.loc[bucket_summary["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "02_overview.csv", index=False)
        bucket_year.loc[bucket_year["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "03_year_overview.csv", index=False)
        bucket_stop_summary.loc[bucket_stop_summary["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "04_initial_stop_target_summary.csv", index=False)
        bucket_stop_year.loc[bucket_stop_year["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "05_initial_stop_target_year_summary.csv", index=False)
        bucket_mae_summary.loc[bucket_mae_summary["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "06_mae_before_target_summary.csv", index=False)
        bucket_trail_summary.loc[bucket_trail_summary["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "07_structural_trailing_summary.csv", index=False)
        bucket_trail_year.loc[bucket_trail_year["opportunity_bucket"].eq(bucket_name)].to_csv(bdir / "08_structural_trailing_year_summary.csv", index=False)
        (bdir / "README.md").write_text(
            f"# {bucket_name}\n\n{description}\n\n"
            "This directory is an exclusive *future-outcome* diagnostic cohort. It is intentionally separated from other return ranges so >=5% long-tail winners do not contaminate short-rebound statistics. The bucket label must never be used as a causal entry feature.\n",
            encoding="utf-8",
        )

    readme = "# R05 Entry Timing + Structural Stop / Runner Atlas\n\n"
    readme += "R05 does not replace the R04 5% discovery with a fixed 5% TP. It studies why the current episode-extreme stop is wide, whether 1m/2m entries improve timing/risk relative to 5m, and whether 2m/5m/15m ITL/LTL or strong bullish displacement anchors can trail a runner without destroying the 3%-5% right tail.\n\n"
    readme += "Important: 1m may be used for entry and path execution, but **never for trailing-stop structure** in R05.\n"
    readme += "The MAE-before-target table is the primary input for designing any future short-rebound stop; do not infer a tight stop from full-horizon MAE.\n\n"
    readme += "## Exclusive opportunity reports\n\nR05.1 additionally writes mutually-exclusive future-outcome cohorts under `opportunity_buckets/`: `<0.3%`, `0.3-1%`, `1-3%`, `3-5%`, and `>=5%`, defined by MFE achieved before the frozen episode-extreme thesis stop (or complete 14d censor). These reports are descriptive future labels only. They prevent a >=5% winner from also contaminating the short-rebound cohort. The original nested target atlas is retained separately for upgrade-probability research.\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r05] done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
