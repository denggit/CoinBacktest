#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R15 — SSL root-acceptance fixed-R first-passage atlas."""
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
from src.research_common.ict_mss2.r15 import (  # noqa: E402
    R15Config,
    build_fixed_r_first_passage,
    prepare_fixed_r_universe,
    r15_causal_audit,
    summarize_fixed_r,
    summarize_fixed_r_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "15.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_ACCEPTANCE_FIXED_R_FIRST_PASSAGE_R15"
EDGE_ID = "RESEARCH_ONLY_SSL_ACCEPTANCE_FIXED_R_PATH"
TITLE = "ETH ICT MSS2 R15 SSL Acceptance Fixed-R First Passage"
DEFAULT_R14_DIR = "data/reports/research/ict/mss2/r14_liquidity_acceptance_continuation"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r15_acceptance_fixed_r_first_passage"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r14-dir", default=DEFAULT_R14_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def _load_r14(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest_path = path / "00_manifest.json"
    features_path = path / "04_acceptance_feature_rows.csv.gz"
    entries_path = path / "05_continuation_entry_rows.csv.gz"
    seal_path = path / "02_holdout_seal.csv"
    for required in (manifest_path, features_path, entries_path, seal_path):
        if not required.exists():
            raise FileNotFoundError(f"R14 report incomplete: {required}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("splits", {}).get("holdout_unsealed")):
        raise RuntimeError("R15 refuses an R14 report generated with holdout unsealed")
    return pd.read_csv(entries_path), pd.read_csv(features_path), manifest, pd.read_csv(seal_path)


def _manual_review(out: Path, paths: pd.DataFrame) -> None:
    d = out / "manual_review"
    d.mkdir(parents=True, exist_ok=True)
    if paths.empty:
        return
    for multiple, p in paths.groupby("r_target", sort=True):
        name = str(multiple).replace(".", "p")
        p.sort_values("entry_time", kind="stable").tail(30).to_csv(d / f"r{name}_recent_30.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R15 manual review\n\n"
        "All rows reuse the frozen R14 SSL root-close-outside short entry and full-region-reclaim stop. "
        "Only the diagnostic target changes. Same-bar target/stop is stop-first; there is no time exit or runner.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R15Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r15] load sealed R14 features and entries", flush=True)
    r14_entries, r14_features, r14_manifest, seal = _load_r14(Path(args.r14_dir))
    universe = prepare_fixed_r_universe(r14_entries, r14_features)
    print("[r15] load bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")
    coverage = data_coverage_audit(bars, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=pd.Timestamp(args.end_date))
    if int(coverage.loc[coverage["check"].eq("requested_end_covered"), "value"].iloc[0]) != 1:
        raise RuntimeError("R15 requested end is not covered by bare 1m data")

    print("[r15] exact 0.5R/1R/2R/3R first passage", flush=True)
    paths = build_fixed_r_first_passage(bars, universe, config=cfg)
    summary = summarize_fixed_r(paths, config=cfg)
    years = summarize_fixed_r_years(paths)
    audit = r15_causal_audit(paths, holdout_start=cfg.holdout_start)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "requested_end_date": args.end_date,
        "actual_bare_1m_start": str(coverage.loc[coverage["check"].eq("actual_start"), "value"].iloc[0]),
        "actual_bare_1m_end": str(coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]),
        "r14_report": args.r14_dir,
        "r14_manifest": r14_manifest,
        "frozen_entry": "SSL root close outside -> next 1m open short",
        "frozen_stop": "R14 full-region reclaim plus 2bps",
        "diagnostic_targets_r": list(cfg.r_multiples),
        "same_bar_policy": "stop first",
        "time_exit": "none; 30d R12 horizon is censoring only",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "holdout_unsealed": False,
        "strategy_status": "first-passage target diagnostic only; no fixed-R target or lifecycle promoted",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_actual_data_coverage_audit.csv", index=False)
    seal.to_csv(out / "02_holdout_seal.csv", index=False)
    universe.to_csv(out / "03_frozen_ssl_root_acceptance_universe.csv.gz", index=False, compression="gzip")
    paths.to_csv(out / "04_fixed_r_first_passage_rows.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "05_fixed_r_scorecard.csv", index=False)
    years.to_csv(out / "06_fixed_r_years.csv", index=False)
    audit.to_csv(out / "07_causal_audit.csv", index=False)
    pd.DataFrame([
        {"check": "frozen_entries", "value": len(universe)},
        {"check": "first_passage_rows", "value": len(paths)},
        {"check": "causal_audit_violations", "value": int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]).to_csv(out / "08_engineering_audit.csv", index=False)
    (out / "R15_RESEARCH_NOTES.md").write_text(
        "# R15 generated research note\n\n"
        "This is an exact first-passage target diagnostic on the frozen R14 SSL root-acceptance entry. "
        "It does not promote fixed-R exits, a runner, risk sizing, or a portfolio. Holdout remains sealed.\n",
        encoding="utf-8",
    )
    _manual_review(out, paths)
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r15] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
