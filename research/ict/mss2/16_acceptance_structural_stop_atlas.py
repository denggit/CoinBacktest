#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R16 — SSL acceptance structural/behavioral stop atlas."""
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
from src.research_common.ict_mss2.r16 import (  # noqa: E402
    R16Config,
    build_stop_model_outcomes,
    prepare_stop_atlas_universe,
    r16_causal_audit,
    summarize_stop_models,
    summarize_stop_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "16.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_ACCEPTANCE_STRUCTURAL_STOP_ATLAS_R16"
EDGE_ID = "RESEARCH_ONLY_SSL_ACCEPTANCE_STOP_SEMANTICS"
TITLE = "ETH ICT MSS2 R16 SSL Acceptance Structural Stop Atlas"
DEFAULT_R14_DIR = "data/reports/research/ict/mss2/r14_liquidity_acceptance_continuation"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r16_acceptance_structural_stop_atlas"


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
    files = {
        "manifest": path / "00_manifest.json",
        "seal": path / "02_holdout_seal.csv",
        "features": path / "04_acceptance_feature_rows.csv.gz",
        "entries": path / "05_continuation_entry_rows.csv.gz",
    }
    for p in files.values():
        if not p.exists():
            raise FileNotFoundError(f"R14 report incomplete: {p}")
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if bool(manifest.get("splits", {}).get("holdout_unsealed")):
        raise RuntimeError("R16 refuses an R14 report generated with holdout unsealed")
    return pd.read_csv(files["entries"]), pd.read_csv(files["features"]), manifest, pd.read_csv(files["seal"])


def _manual_review(out: Path, paths: pd.DataFrame) -> None:
    d = out / "manual_review"
    d.mkdir(parents=True, exist_ok=True)
    if paths.empty:
        return
    for model, p in paths.groupby("stop_model", sort=True):
        p.sort_values("entry_time", kind="stable").tail(40).to_csv(d / f"{model}_recent_40.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R16 manual review\n\n"
        "All variants use the same frozen SSL root-close acceptance short entry and deeper same-side target. "
        "Only invalidation changes. A target on the same bar as a touch stop or reclaim close is pessimistically a failure.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R16Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("[r16] load sealed R14 root-acceptance universe", flush=True)
    entries, features, r14_manifest, seal = _load_r14(Path(args.r14_dir))
    universe = prepare_stop_atlas_universe(entries, features)
    print("[r16] load bare 1m K through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    coverage = data_coverage_audit(bars, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=pd.Timestamp(args.end_date))
    if coverage.empty or int(coverage.loc[coverage["check"].eq("requested_end_covered"), "value"].iloc[0]) != 1:
        raise RuntimeError("R16 requested end is not covered by bare 1m data")
    print("[r16] region touch vs root extreme vs close-reclaim", flush=True)
    paths = build_stop_model_outcomes(bars, universe, config=cfg)
    summary = summarize_stop_models(paths, config=cfg)
    years = summarize_stop_years(paths)
    audit = r16_causal_audit(paths, holdout_start=cfg.holdout_start)
    manifest = {
        "script_version": SCRIPT_VERSION, "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID, "title": TITLE, "symbol": args.symbol,
        "requested_end_date": args.end_date,
        "actual_bare_1m_start": str(coverage.loc[coverage["check"].eq("actual_start"), "value"].iloc[0]),
        "actual_bare_1m_end": str(coverage.loc[coverage["check"].eq("actual_end"), "value"].iloc[0]),
        "r14_report": args.r14_dir, "r14_manifest": r14_manifest,
        "frozen_entry": "SSL root close outside -> next 1m open short",
        "frozen_target": "root-time deeper same-side completed-trend liquidity touch",
        "stop_models": ["region_edge_touch", "root_bar_extreme_touch", "close_reclaim_plus_extreme"],
        "behavioral_exit": "first close above root zone high -> next 1m open; root-bar-extreme hard stop remains active",
        "same_bar_policy": "target tied with touch stop or reclaim bar is failure",
        "costs": {"market_roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "holdout_unsealed": False,
        "strategy_status": "stop-semantics atlas only; no model promoted automatically",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_actual_data_coverage_audit.csv", index=False)
    seal.to_csv(out / "02_holdout_seal.csv", index=False)
    universe.to_csv(out / "03_frozen_ssl_root_acceptance_universe.csv.gz", index=False, compression="gzip")
    paths.to_csv(out / "04_stop_model_outcomes.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "05_stop_model_scorecard.csv", index=False)
    years.to_csv(out / "06_stop_model_years.csv", index=False)
    audit.to_csv(out / "07_causal_audit.csv", index=False)
    pd.DataFrame([
        {"check": "frozen_entries", "value": len(universe)},
        {"check": "stop_model_rows", "value": len(paths)},
        {"check": "causal_audit_violations", "value": int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) if len(audit) else 0},
    ]).to_csv(out / "08_engineering_audit.csv", index=False)
    (out / "R16_RESEARCH_NOTES.md").write_text(
        "# R16 generated research note\n\n"
        "This atlas changes only invalidation semantics for the frozen R14 SSL root acceptance entry. "
        "Review 2x/3x, years and top-winner removal before the mandatory strategic reset. Holdout remains sealed.\n",
        encoding="utf-8",
    )
    _manual_review(out, paths)
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r16] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
