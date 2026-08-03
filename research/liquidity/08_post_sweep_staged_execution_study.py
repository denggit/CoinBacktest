#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R08 fixed-total-risk staged execution and structure-success study.

R08 deliberately stops trying to predict the exact final low. It asks whether
keeping a small probe while delaying the rest of a fixed 1R risk budget can
reduce path risk without missing too much of fast post-sweep opportunity.

Important boundaries:
- every scheme sees the same event universe;
- structure triggers use only closed R04 checkpoint information;
- a trigger becomes active on the next bar (signal elapsed + 1);
- R07 Footprint opportunity flags are reporting strata only, never hard filters;
- no parameter grid or threshold mining is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.post_sweep_staged_execution import (  # noqa: E402
    PostSweepStagedExecutionConfig,
    add_trigger_flags,
    build_fill_table,
    causal_audit,
    data_quality,
    earliest_trigger_rows,
    load_r04,
    load_r07_opportunity,
    missed_opportunity,
    opportunity_stratification,
    relative_to_baseline,
    research_brief,
    scheme_specs,
    scheme_summary,
    simulate_schemes,
    structure_outcome_atlas,
    trigger_coverage,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "08_post_sweep_staged_execution_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_POST_SWEEP_STAGED_EXECUTION_R08"
EDGE_ID = "RESEARCH_ONLY_POST_SWEEP_FIXED_RISK_STAGED_EXECUTION"
TITLE = "ETH Post-Sweep Fixed-Risk Staged Execution Study R08"
DEFAULT_R04_DIR = "data/reports/research/liquidity/post_sweep_continuation_exhaustion_r04"
DEFAULT_R07_DIR = "data/reports/research/liquidity/post_sweep_footprint_books_absorption_r07"
DEFAULT_OUT_DIR = "data/reports/research/liquidity/post_sweep_staged_execution_r08"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--r04-dir", default=DEFAULT_R04_DIR)
    p.add_argument("--r07-dir", default=DEFAULT_R07_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--max-events", type=int, default=0, help="Deterministic smoke cap; 0 uses all events.")
    p.add_argument("--max-deployment-minutes", type=int, default=60)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-rate-per-side", type=float, default=0.00010)
    p.add_argument("--stressed-slippage-rate-per-side", type=float, default=0.00020)
    p.add_argument("--sample-rows", type=int, default=50_000)
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _cfg(args: argparse.Namespace) -> PostSweepStagedExecutionConfig:
    return PostSweepStagedExecutionConfig(
        max_deployment_minutes=int(args.max_deployment_minutes),
        fee_rate_per_side=float(args.fee_rate_per_side),
        slippage_rate_per_side=float(args.slippage_rate_per_side),
        stressed_slippage_rate_per_side=float(args.stressed_slippage_rate_per_side),
        report_sample_rows=int(args.sample_rows),
    ).validate()


def run_self_test() -> None:
    path = pd.DataFrame(
        {
            "checkpoint_id": ["E_C1", "E_C2", "E_C3", "E_C4", "E_C5", "E_C6"],
            "zone_event_id": ["E"] * 6,
            "event_kind": ["SWEEP"] * 6,
            "period": ["EARLY_2023_2024"] * 6,
            "checkpoint_available_time": pd.date_range("2025-01-01 00:01:00", periods=6, freq="1min"),
            "entry_reference_time": pd.date_range("2025-01-01 00:01:00", periods=6, freq="1min"),
            "elapsed_bars": [1, 2, 3, 4, 5, 6],
            "checkpoint_high": [100, 99, 100, 101, 102, 103],
            "checkpoint_low": [99, 98, 98.5, 99.5, 100.5, 101],
            "checkpoint_close": [99, 98.5, 99.5, 100.5, 101.5, 102],
            "entry_reference_price": [99, 98.5, 99.5, 100.5, 101.5, 102],
            "no_new_low_3bars": [False, False, False, True, True, True],
            "no_new_low_5bars": [False, False, False, False, True, True],
            "no_new_low_10bars": [False] * 6,
            "micro_high_break_3bars": [False, False, False, True, True, True],
            "micro_high_break_5bars": [False, False, False, False, True, True],
            "micro_high_break_10bars": [False] * 6,
            "zone_floor_reclaimed": [False, False, False, True, True, True],
            "zone_ceiling_reclaimed": [False] * 6,
            "future_mfe_15m": [0.01] * 6,
            "future_mae_15m": [-0.01] * 6,
            "future_mfe_60m": [0.02] * 6,
            "future_mae_60m": [-0.01] * 6,
            "future_mfe_180m": [0.03] * 6,
            "future_mae_180m": [-0.02] * 6,
            "future_no_lower_low_60m": [True] * 6,
            "future_reversal_dominant_60m": [True] * 6,
            "future_large_mfe_0p5_180m": [True] * 6,
            "future_large_mfe_1_180m": [True] * 6,
            "future_large_mfe_2_180m": [True] * 6,
            "future_close_return_180m": [0.01] * 6,
        }
    )
    cfg = PostSweepStagedExecutionConfig(horizons=(5,), max_deployment_minutes=60).validate()
    flagged = add_trigger_flags(path)
    triggers = earliest_trigger_rows(flagged, cfg.max_deployment_minutes)
    fills = build_fill_table(triggers, scheme_specs())
    replay = simulate_schemes(flagged, fills, scheme_specs(), cfg, progress=False)
    if replay.empty or int((fills["trigger_name"] == "EARLY_REJECTION").sum()) == 0:
        raise RuntimeError("R08 self-test failed staged trigger/fill path")
    if int((pd.to_datetime(fills["entry_time"]) < pd.to_datetime(fills["signal_time"])).sum()) != 0:
        raise RuntimeError("R08 self-test detected entry before signal")
    print(f"[self-test] passed schemes={replay['scheme'].nunique()} fills={len(fills)}", flush=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def run(args: argparse.Namespace) -> Path:
    if args.self_test:
        run_self_test()
        return PROJECT_ROOT / args.out_dir
    cfg = _cfg(args)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("end-date must not precede start-date")
    r04_dir = PROJECT_ROOT / args.r04_dir
    r07_dir = PROJECT_ROOT / args.r07_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"[run] {TITLE} v{SCRIPT_VERSION}", flush=True)
    print(f"[window] {start} -> {end}", flush=True)
    print(f"[source] R04={r04_dir}", flush=True)
    print(f"[source] R07={r07_dir}", flush=True)
    print("[design] same event universe; fixed 1R risk; causal next-bar staged fills; no hard Footprint filter", flush=True)

    print("[stage] load R04 checkpoint path and labels", flush=True)
    path, events = load_r04(r04_dir, start, end, max_events=int(args.max_events))
    print(f"[universe] events={len(events):,} checkpoints={len(path):,}", flush=True)

    print("[stage] build fixed causal structure triggers", flush=True)
    path = add_trigger_flags(path)
    triggers = earliest_trigger_rows(path, cfg.max_deployment_minutes)
    print(f"[triggers] rows={len(triggers):,}", flush=True)

    print("[stage] build staged fills and replay sparse checkpoint paths", flush=True)
    schemes = scheme_specs()
    fills = build_fill_table(triggers, schemes)
    replay = simulate_schemes(path, fills, schemes, cfg, progress=not args.no_progress)
    print(f"[replay] rows={len(replay):,} fills={len(fills):,}", flush=True)

    print("[stage] attach R07 opportunity strata for reporting only", flush=True)
    opportunity, thresholds = load_r07_opportunity(r07_dir, set(events["zone_event_id"].astype(str)))
    print(f"[footprint-strata] events={len(opportunity):,} thresholds={thresholds}", flush=True)

    reports = {
        "01_data_quality.csv": data_quality(path, events, triggers, fills, replay),
        "02_trigger_coverage.csv": trigger_coverage(triggers, len(events)),
        "03_scheme_summary.csv": scheme_summary(replay, cfg),
        "04_scheme_relative_to_full_entry.csv": relative_to_baseline(replay, cfg),
        "05_missed_opportunity_summary.csv": missed_opportunity(replay, events),
        "06_structure_success_atlas.csv": structure_outcome_atlas(triggers),
        "07_footprint_opportunity_stratification.csv": opportunity_stratification(replay, opportunity, cfg),
        "08_causal_audit.csv": causal_audit(fills),
    }
    for name, frame in reports.items():
        _write(frame, out_dir / name)

    sample_n = min(cfg.report_sample_rows, len(replay))
    replay.sort_values(["period", "zone_event_id", "scheme"], kind="mergesort").head(sample_n).to_csv(out_dir / "09_execution_event_sample.csv", index=False)
    fills.sort_values(["period", "zone_event_id", "scheme", "stage_sequence"], kind="mergesort").head(cfg.report_sample_rows).to_csv(out_dir / "10_fill_sample.csv", index=False)
    replay.to_csv(out_dir / "11_execution_event_table.csv.gz", index=False, compression="gzip")
    fills.to_csv(out_dir / "12_fill_table.csv.gz", index=False, compression="gzip")
    events.to_csv(out_dir / "13_event_label_table.csv.gz", index=False, compression="gzip")
    if not opportunity.empty:
        opportunity.to_csv(out_dir / "14_event_footprint_opportunity_table.csv.gz", index=False, compression="gzip")
    (out_dir / "15_research_brief.md").write_text(research_brief(reports["03_scheme_summary.csv"], reports["04_scheme_relative_to_full_entry.csv"], reports["05_missed_opportunity_summary.csv"]), encoding="utf-8")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "start_date": str(start),
        "end_date": str(end),
        "config": asdict(cfg),
        "schemes": [asdict(s) for s in schemes],
        "event_count": len(events),
        "checkpoint_count": len(path),
        "fill_count": len(fills),
        "footprint_thresholds": thresholds,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "R08 replays R04 sparse checkpoint bars; exact full 1m path replay is required before promotion.",
            "Footprint opportunity score is a reporting stratum only and never removes an event.",
            "This is execution research, not a finalized TP/SL strategy backtest.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = []
    for name in ("01_data_quality.csv", "08_causal_audit.csv"):
        frame = reports[name]
        failed.extend(frame.loc[frame["status"] == "FAIL", "check"].astype(str).tolist())
    if failed:
        raise RuntimeError(f"R08 quality/causal gate failed: {failed}")
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
