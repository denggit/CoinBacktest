#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 — Reversal Quality & Entry Discovery.

This version does not train ML or promote a strategy.  It compares completed-
trend liquidity sweeps that deliver directly to frozen opposite liquidity with
paths that reach deeper same-side liquidity first, then measures the earliest
causal separation and a small, predeclared entry family.
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
from src.research_common.ict_mss2.r13 import (  # noqa: E402
    R13Config,
    attach_reversal_quality_features,
    build_entry_candidate_outcomes,
    build_feature_bin_atlas,
    data_coverage_audit,
    prepare_reversal_comparison_universe,
    r13_causal_audit,
    summarize_direct_failure_divergence,
    summarize_entry_models,
    summarize_entry_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "13.0.1"
EXPERIMENT_ID = "ETH_ICT_MSS2_REVERSAL_QUALITY_ENTRY_DISCOVERY_R13"
EDGE_ID = "RESEARCH_ONLY_COMPLETED_TREND_REVERSAL_QUALITY"
TITLE = "ETH ICT MSS2 R13 Reversal Quality & Entry Discovery"
DEFAULT_R12_DIR = "data/reports/research/ict/mss2/r12_completed_trend_swing_opposite_liquidity_path_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r13_reversal_quality_entry_discovery"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r12-dir", default=DEFAULT_R12_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--discovery-end", default="2024-12-31 23:59:59")
    p.add_argument("--validation-end", default="2025-06-30 23:59:59")
    p.add_argument("--holdout-start", default="2025-08-01 00:00:00")
    p.add_argument("--unseal-holdout", action="store_true", help="Explicit final evaluation only; never use during R13 rule discovery")
    return p.parse_args(argv)


def _load_r12(path: Path, requested_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    manifest_path = path / "00_manifest.json"
    rows_path = path / "04_opposite_liquidity_path_rows.csv.gz"
    if not manifest_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"R12 report incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pd.Timestamp(manifest.get("research_end_date")) < requested_end:
        raise RuntimeError(f"R12 manifest ends before requested R13 end: {manifest.get('research_end_date')}")
    return pd.read_csv(rows_path), manifest


def _manual_review(out: Path, features: pd.DataFrame, entries: pd.DataFrame) -> None:
    d = out / "manual_review"
    d.mkdir(parents=True, exist_ok=True)
    if features.empty:
        return
    ordered = features.sort_values("root_sweep_time", kind="stable")
    direct = ordered.loc[ordered["direct_reversal_label"].eq(1)]
    failure = ordered.loc[ordered["direct_reversal_label"].eq(0)]
    direct.tail(25).to_csv(d / "01_recent_25_direct_reversals.csv", index=False, encoding="utf-8-sig")
    failure.tail(25).to_csv(d / "02_recent_25_same_side_first_failures.csv", index=False, encoding="utf-8-sig")
    if not entries.empty:
        filled = entries.loc[entries["entry_status"].eq("filled")].sort_values("entry_time", kind="stable")
        filled.tail(40).to_csv(d / "03_recent_40_filled_entry_comparisons.csv", index=False, encoding="utf-8-sig")
        losers = filled.loc[filled["outcome"].eq("sl_first")].sort_values("entry_time", kind="stable")
        losers.tail(25).to_csv(d / "04_recent_25_false_reversal_entries.csv", index=False, encoding="utf-8-sig")
        winners = filled.loc[filled["outcome"].eq("tp_first")].copy()
        winners = winners.sort_values(["structural_rr", "entry_time"], ascending=[False, True], kind="stable")
        winners.tail(25).to_csv(d / "05_high_structural_rr_winners.csv", index=False, encoding="utf-8-sig")
    for year, part in ordered.groupby("year", sort=True):
        sample = pd.concat([
            part.loc[part["direct_reversal_label"].eq(1)].tail(5),
            part.loc[part["direct_reversal_label"].eq(0)].tail(5),
        ]).sort_values("root_sweep_time", kind="stable")
        sample.to_csv(d / f"06_year_{int(year)}_representative_paths.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R13 manual review\n\n"
        "Direct reversal means frozen opposite completed-trend liquidity was touched before deeper same-side liquidity. "
        "Cascade-then-opposite is intentionally a direct-reversal failure because the structural invalidation was reached first. "
        "Entry rows use next-eligible-bar execution; same-bar TP/SL is stop-first and an FVG target cannot be credited on its fill bar.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    cfg = R13Config(
        discovery_end=pd.Timestamp(args.discovery_end),
        validation_end=pd.Timestamp(args.validation_end),
        holdout_start=pd.Timestamp(args.holdout_start),
    ).validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r13] load R12 completed-trend first-passage paths", flush=True)
    r12, r12_manifest = _load_r12(Path(args.r12_dir), end)
    print("[r13] load bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    coverage = data_coverage_audit(bars, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=end)
    covered = coverage.loc[coverage["check"].eq("requested_end_covered"), "value"]
    if covered.empty or int(covered.iloc[0]) != 1:
        actual = coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]
        raise RuntimeError(f"Bare 1m data ends at {actual}, before requested R13 end {end}; do not publish a false coverage manifest")

    print("[r13] seal holdout and build direct-vs-same-side-first comparison", flush=True)
    universe, seal = prepare_reversal_comparison_universe(r12, config=cfg, include_holdout=bool(args.unseal_holdout))
    print("[r13] sweep morphology + expected response + reclaim/MSS/FVG quality", flush=True)
    features = attach_reversal_quality_features(bars, universe, config=cfg)
    print("[r13] causal entry candidates + frozen structural TP/SL", flush=True)
    entries = build_entry_candidate_outcomes(bars, features, config=cfg)
    entry_summary = summarize_entry_models(entries)
    entry_years = summarize_entry_years(entries)
    definitions, bins, monotonicity = build_feature_bin_atlas(features, entries=entries)
    divergence = summarize_direct_failure_divergence(features)
    audit = r13_causal_audit(features, entries, holdout_start=cfg.holdout_start)

    actual_start = coverage.loc[coverage["check"].eq("actual_start"), "value"].iloc[0]
    actual_end = coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "requested_warmup_start_date": args.warmup_start_date,
        "requested_research_start_date": args.start_date,
        "requested_research_end_date": args.end_date,
        "actual_bare_1m_start": str(actual_start),
        "actual_bare_1m_end": str(actual_end),
        "r12_report": args.r12_dir,
        "r12_manifest": r12_manifest,
        "splits": {
            "discovery_end": str(cfg.discovery_end),
            "validation_end": str(cfg.validation_end),
            "embargo": "2025-07-01 through 2025-07-31 (30d path-label purge)",
            "holdout_start": str(cfg.holdout_start),
            "holdout_unsealed": bool(args.unseal_holdout),
        },
        "comparison": "direct opposite delivery vs deeper same-side liquidity first; cascade-then-opposite is failure for direct reversal thesis",
        "entry_models": ["root_next_open", "same_bar_reclaim", "response_15m_market", "reclaim_market", "fvg_market", "mss_1m_market", "mss_2m_market", "mss_5m_market", "reclaim_fvg_proximal_limit", "mss_2m_fvg_proximal_limit"],
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "limit_roundtrip": cfg.limit_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "research only; no strategy/risk schedule promoted",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_actual_data_coverage_audit.csv", index=False)
    seal.to_csv(out / "02_holdout_seal.csv", index=False)
    universe.to_csv(out / "03_direct_vs_same_side_first_universe.csv.gz", index=False, compression="gzip")
    features.to_csv(out / "04_reversal_quality_feature_rows.csv.gz", index=False, compression="gzip")
    divergence.to_csv(out / "05_direct_failure_divergence_by_year.csv", index=False)
    definitions.to_csv(out / "06_discovery_frozen_bin_definitions.csv", index=False)
    bins.to_csv(out / "07_feature_bin_atlas.csv", index=False)
    monotonicity.to_csv(out / "08_feature_monotonicity.csv", index=False)
    entries.to_csv(out / "09_entry_candidate_outcome_rows.csv.gz", index=False, compression="gzip")
    entry_summary.to_csv(out / "10_entry_model_summary_cost2x.csv", index=False)
    entry_years.to_csv(out / "11_entry_model_year_summary_cost2x.csv", index=False)
    audit.to_csv(out / "12_causal_audit.csv", index=False)
    pd.DataFrame([
        {"check": "comparison_events", "value": len(universe)},
        {"check": "feature_rows", "value": len(features)},
        {"check": "entry_candidate_rows", "value": len(entries)},
        {"check": "causal_audit_violations", "value": int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]).to_csv(out / "13_engineering_audit.csv", index=False)
    (out / "R13_RESEARCH_NOTES.md").write_text(
        "# R13 generated research note\n\n"
        "R13 is a diagnostic and entry-discovery atlas, not a promoted strategy. Holdout is sealed by default. "
        "Review discovery-to-validation monotonicity and yearly entry summaries before freezing any rule.\n",
        encoding="utf-8",
    )
    _manual_review(out, features, entries)
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r13] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
