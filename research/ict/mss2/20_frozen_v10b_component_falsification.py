#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R20 — frozen LF V10B component visible-window falsification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_common.ict_mss2.r20 import (  # noqa: E402
    R20Config,
    build_r20_gate,
    prepare_r20_trades,
    r20_causal_audit,
    summarize_r20_components,
    summarize_r20_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.sleeve_lib.lf_v10b.selector import run_lf_v10b_leg  # noqa: E402

TITLE = "ETH ICT MSS2 R20 Frozen LF V10B Component Falsification"
EXPERIMENT_ID = "ETH_ICT_MSS2_FROZEN_LF_V10B_COMPONENT_FALSIFICATION_R20"
EDGE_ID = "RESEARCH_ONLY_CONTAMINATED_PRIOR_LF_V10B_COMPONENTS"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r20_frozen_v10b_component_falsification"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    # The 20:00-labelled 4H bar closes at the July boundary, so validation
    # loading stops before that bar begins.
    parser.add_argument("--end-date", default="2025-06-30 19:59:59")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        initial_capital=1000.0,
        fee_rate=0.0,
        slippage_pct=0.0,
        slippage=0.0,
        lf_preset="turbo",
        lf_bear_preset="high",
        lf_bull_preset="high",
        lf_priority_mode="reclaim_first",
        lf_global_risk_scale=1.30,
        lf_micro_filter_mode="soft",
    )


def _manual_review(out: Path, trades: pd.DataFrame) -> None:
    directory = out / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    included = trades.loc[trades["path_status"].eq("included")].copy()
    included.sort_values("entry_time", kind="stable").tail(80).to_csv(directory / "01_recent_80.csv", index=False)
    included.sort_values("net_return_cost2x", ascending=False, kind="stable").head(40).to_csv(directory / "02_best_40_cost2x.csv", index=False)
    included.sort_values("net_return_cost2x", kind="stable").head(40).to_csv(directory / "03_worst_40_cost2x.csv", index=False)
    for component, part in included.groupby("component", sort=True):
        safe = str(component).lower().replace(" / ", "_").replace(" ", "_")
        part.sort_values("entry_time", kind="stable").tail(30).to_csv(directory / f"04_{safe}_recent_30.csv", index=False)
    (directory / "README.md").write_text(
        "# R20 manual review\n\nVerify the frozen 4H signal at `signal_time`, next-4H-open entry, "
        "zero-cost average entry/exit path, component identity, and boundary status. R20 is a contaminated-prior "
        "falsification and contains no July or holdout rows.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R20Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("R20 end date must remain before the July embargo")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r20] run frozen current LF V10B path at zero execution cost through visible validation only", flush=True)
    raw, _equity, features = run_lf_v10b_leg(_runner_args(args))
    trades = prepare_r20_trades(raw, config=cfg)
    scorecard = summarize_r20_components(trades, config=cfg)
    years = summarize_r20_years(trades)
    gate = build_r20_gate(scorecard)
    audit = r20_causal_audit(trades, features, config=cfg)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "source": "current src/sleeve_lib/lf_v10b and src/edge_lib/lf_* definitions",
        "provenance": "contaminated prior selected on overlapping 2023-2026 history; R20 is falsification only",
        "requested_window": [args.start_date, args.end_date],
        "splits": {"discovery": "2023-2024", "validation": "2025H1 (visible, not independent)", "embargo_start": str(cfg.embargo_start), "holdout_start": str(cfg.holdout_start)},
        "holdout_rows_loaded": 0,
        "primary_unit": "unlevered signed price return from zero-cost average entry to exit",
        "roundtrip_cost": cfg.market_roundtrip_cost,
        "cost_scales": list(cfg.cost_scales),
        "promotion_status": "cannot promote; at most eligible for genuinely new forward incubation",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_index = pd.to_datetime(features.index, errors="coerce")
    feature_deltas = pd.Series(feature_index).diff().dropna()
    pd.DataFrame(
        [
            {"check": "data_end_before_embargo", "value": int(pd.Timestamp(args.end_date) < cfg.embargo_start)},
            {"check": "holdout_rows_loaded", "value": 0},
            {"check": "feature_rows", "value": len(features)},
            {"check": "feature_start", "value": str(feature_index.min())},
            {"check": "feature_end", "value": str(feature_index.max())},
            {"check": "feature_duplicate_timestamps", "value": int(feature_index.duplicated().sum())},
            {"check": "feature_nonexact_4h_intervals", "value": int(feature_deltas.ne(pd.Timedelta(hours=4)).sum())},
            {"check": "raw_zero_cost_trades", "value": len(raw)},
            {"check": "included_visible_trades", "value": int(trades["path_status"].eq("included").sum())},
            {"check": "boundary_or_right_edge_censored", "value": int((~trades["path_status"].eq("included")).sum())},
            {"check": "maximum_trade_exit", "value": str(pd.to_datetime(trades["exit_time"]).max())},
        ]
    ).to_csv(out / "01_boundary_and_seal_audit.csv", index=False)
    trades.to_csv(out / "02_unlevered_trade_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    scorecard.to_csv(out / "03_component_scorecard.csv", index=False)
    years.to_csv(out / "04_component_years.csv", index=False)
    gate.to_csv(out / "05_forward_incubation_gate.csv", index=False)
    audit.to_csv(out / "06_causal_and_arithmetic_audit.csv", index=False)
    _manual_review(out, trades)
    (out / "R20_GENERATED_NOTE.md").write_text(
        "# R20 generated note\n\nThis report uses a historically contaminated prior. Validation is visible, "
        "not untouched; July and holdout data are absent. No component is selected into a composite.\n",
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(scorecard.to_string(index=False), flush=True)
    print(gate.to_string(index=False), flush=True)
    print(f"[r20] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
