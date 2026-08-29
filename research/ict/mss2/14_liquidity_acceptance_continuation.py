#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R14 — Completed-Trend Liquidity Acceptance/Continuation Atlas."""
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
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa: E402
from src.research_common.ict_mss2.r14 import (  # noqa: E402
    R14Config,
    attach_acceptance_features,
    build_continuation_entries,
    prepare_continuation_universe,
    r14_causal_audit,
    summarize_continuation_models,
    summarize_continuation_months,
    summarize_continuation_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "14.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_LIQUIDITY_ACCEPTANCE_CONTINUATION_R14"
EDGE_ID = "RESEARCH_ONLY_COMPLETED_TREND_ACCEPTANCE_CONTINUATION"
TITLE = "ETH ICT MSS2 R14 Liquidity Acceptance/Continuation"
DEFAULT_R12_DIR = "data/reports/research/ict/mss2/r12_completed_trend_swing_opposite_liquidity_path_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r14_liquidity_acceptance_continuation"


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
    p.add_argument("--unseal-holdout", action="store_true", help="Explicit final evaluation only; never use during R14 discovery")
    return p.parse_args(argv)


def _load_r12(path: Path, requested_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    manifest_path = path / "00_manifest.json"
    rows_path = path / "04_opposite_liquidity_path_rows.csv.gz"
    if not manifest_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"R12 report incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pd.Timestamp(manifest.get("research_end_date")) < requested_end:
        raise RuntimeError(f"R12 manifest ends before requested R14 end: {manifest.get('research_end_date')}")
    return pd.read_csv(rows_path), manifest


def _manual_review(out: Path, features: pd.DataFrame, entries: pd.DataFrame) -> None:
    d = out / "manual_review"
    d.mkdir(parents=True, exist_ok=True)
    if features.empty:
        return
    cols = [c for c in [
        "root_event_id", "root_sweep_time", "root_side", "root_zone_low", "root_zone_high",
        "root_bar_close", "deeper_same_side_touch_price", "path_outcome", "same_side_first_label",
        "root_close_outside_flag", "accept_5m_outside_close_share", "accept_15m_outside_close_share",
    ] if c in features]
    features.sort_values("root_sweep_time", kind="stable").tail(60)[cols].to_csv(
        d / "01_recent_60_acceptance_paths.csv", index=False, encoding="utf-8-sig"
    )
    if not entries.empty:
        filled = entries.loc[entries["entry_status"].eq("filled")].sort_values("entry_time", kind="stable")
        filled.tail(60).to_csv(d / "02_recent_60_filled_entries.csv", index=False, encoding="utf-8-sig")
        filled.loc[filled["outcome"].eq("tp_first")].tail(30).to_csv(d / "03_recent_30_continuation_winners.csv", index=False, encoding="utf-8-sig")
        filled.loc[filled["outcome"].eq("sl_first")].tail(30).to_csv(d / "04_recent_30_reclaim_failures.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R14 manual review\n\n"
        "BSL sweeps imply long continuation and SSL sweeps imply short continuation. "
        "The target is frozen deeper same-side completed-trend liquidity. Full reclaim beyond the swept region plus 2bps invalidates. "
        "Signals use closed bars and enter at the next 1m open; same-bar TP/SL is stop-first.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    end = pd.Timestamp(args.end_date)
    cfg = R14Config(
        research_start=pd.Timestamp(args.start_date),
        discovery_end=pd.Timestamp(args.discovery_end),
        validation_end=pd.Timestamp(args.validation_end),
        holdout_start=pd.Timestamp(args.holdout_start),
    ).validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r14] load R12 completed-trend paths", flush=True)
    r12, r12_manifest = _load_r12(Path(args.r12_dir), end)
    print("[r14] load bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    coverage = data_coverage_audit(bars, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=end)
    covered = coverage.loc[coverage["check"].eq("requested_end_covered"), "value"]
    if covered.empty or int(covered.iloc[0]) != 1:
        actual = coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]
        raise RuntimeError(f"Bare 1m data ends at {actual}, before requested R14 end {end}")

    print("[r14] seal holdout and build deeper-same-side universe", flush=True)
    universe, seal = prepare_continuation_universe(r12, config=cfg, include_holdout=bool(args.unseal_holdout))
    print("[r14] root/5m/15m acceptance features", flush=True)
    features = attach_acceptance_features(bars, universe, config=cfg)
    print("[r14] causal continuation entries and frozen TP/SL", flush=True)
    entries = build_continuation_entries(bars, features, config=cfg)
    summary = summarize_continuation_models(entries, config=cfg)
    years = summarize_continuation_years(entries)
    months = summarize_continuation_months(entries)
    audit = r14_causal_audit(features, entries, holdout_start=cfg.holdout_start)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "requested_warmup_start_date": args.warmup_start_date,
        "requested_research_start_date": args.start_date,
        "requested_research_end_date": args.end_date,
        "actual_bare_1m_start": str(coverage.loc[coverage["check"].eq("actual_start"), "value"].iloc[0]),
        "actual_bare_1m_end": str(coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]),
        "r12_report": args.r12_dir,
        "r12_manifest": r12_manifest,
        "splits": {
            "discovery_end": str(cfg.discovery_end), "validation_end": str(cfg.validation_end),
            "embargo": "2025-07-01 through 2025-07-31 (30d path-label purge)",
            "holdout_start": str(cfg.holdout_start), "holdout_unsealed": bool(args.unseal_holdout),
        },
        "mechanism": "completed-trend liquidity sweep -> causal outside acceptance -> deeper same-side liquidity continuation",
        "continuation_direction": "BSL sweep long; SSL sweep short",
        "target": "root-time frozen deeper same-side completed-trend liquidity touch",
        "stop": "far edge of swept root region plus 2bps reclaim buffer",
        "entry_models": ["root_close_outside"] + [
            f"accept_{w}m_p{int(round(s * 100)):03d}" for w in cfg.acceptance_windows_minutes for s in cfg.persistence_shares
        ],
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "strategy_status": "research only; market-entry edge gate before any execution overlay or portfolio promotion",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_actual_data_coverage_audit.csv", index=False)
    seal.to_csv(out / "02_holdout_seal.csv", index=False)
    universe.to_csv(out / "03_continuation_universe.csv.gz", index=False, compression="gzip")
    features.to_csv(out / "04_acceptance_feature_rows.csv.gz", index=False, compression="gzip")
    entries.to_csv(out / "05_continuation_entry_rows.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "06_entry_model_scorecard.csv", index=False)
    years.to_csv(out / "07_entry_model_years.csv", index=False)
    months.to_csv(out / "08_entry_model_months.csv", index=False)
    audit.to_csv(out / "09_causal_audit.csv", index=False)
    pd.DataFrame([
        {"check": "universe_rows", "value": len(universe)},
        {"check": "feature_rows", "value": len(features)},
        {"check": "entry_rows", "value": len(entries)},
        {"check": "causal_audit_violations", "value": int(pd.to_numeric(audit.get("violations"), errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]).to_csv(out / "10_engineering_audit.csv", index=False)
    (out / "R14_RESEARCH_NOTES.md").write_text(
        "# R14 generated research note\n\n"
        "R14 is a continuation edge gate, not a promoted strategy. The holdout is sealed by default. "
        "Do not add FVG/order-flow execution or risk tiers unless simple market acceptance is positive at 2x cost in discovery and validation, across years, and after top-winner removal.\n",
        encoding="utf-8",
    )
    _manual_review(out, features, entries)
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r14] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
