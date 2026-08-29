#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R23 — frozen panic-wick structural Long falsification."""
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

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.ict_mss2.r23 import (  # noqa: E402
    R23Config,
    build_panic_features,
    build_priority_union_events,
    build_r23_gate,
    r23_causal_audit,
    simulate_frozen_panic_long,
    summarize_r23,
    summarize_r23_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

TITLE = "ETH ICT MSS2 R23 Frozen Panic-Wick Structural Long"
EXPERIMENT_ID = "ETH_ICT_MSS2_FROZEN_PANIC_WICK_LONG_R23"
EDGE_ID = "RESEARCH_ONLY_FROZEN_PANIC_WICK_STRUCTURAL_LONG"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r23_frozen_panic_wick_structural_long"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _manual_review(out: Path, trades: pd.DataFrame) -> None:
    directory = out / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    closed.sort_values("entry_time").tail(80).to_csv(directory / "01_recent_80.csv", index=False)
    closed.sort_values("net_return_cost2x", ascending=False).head(40).to_csv(directory / "02_best_40.csv", index=False)
    closed.sort_values("net_return_cost2x").head(40).to_csv(directory / "03_worst_40.csv", index=False)
    for year, part in closed.groupby(pd.to_datetime(closed["entry_time"]).dt.year, sort=True):
        part.sort_values("entry_time").tail(25).to_csv(directory / f"04_{int(year)}_recent_25.csv", index=False)
    (directory / "README.md").write_text(
        "# R23 manual review\n\nVerify the source-observed panic wick, priority-union reason, event+3m entry, distinct low sweeps, event-high reclaim, causal higher-low trail, and next-open exit.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R23Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("R23 end date must remain before the July embargo")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r23] load cached trade-derived 1m bars through src.data_feed", flush=True)
    raw = OKXTradeBarLoader(args.symbol, "1m").fetch_data_by_date_range(
        args.warmup_start_date, args.end_date, build_missing=False
    )
    if raw.empty:
        raise RuntimeError("R23 trade-bar source is empty")
    expected = len(pd.date_range(pd.Timestamp(args.warmup_start_date), pd.Timestamp(args.end_date).floor("min"), freq="1min"))
    gaps = raw.index.to_series().diff().gt(pd.Timedelta(minutes=1))
    print("[r23] reconstruct frozen features and priority union", flush=True)
    features = build_panic_features(raw, config=cfg)
    events = build_priority_union_events(features, config=cfg)
    discovery = simulate_frozen_panic_long(
        features, events, split="discovery", split_start=cfg.discovery_start, split_end=cfg.validation_start, config=cfg
    )
    validation = simulate_frozen_panic_long(
        features, events, split="validation", split_start=cfg.validation_start, split_end=cfg.embargo_start, config=cfg
    )
    trades = pd.concat([discovery, validation], ignore_index=True, sort=False)
    if trades.empty:
        raise RuntimeError("R23 produced no paths")
    score = summarize_r23(trades)
    years = summarize_r23_years(trades)
    gate = build_r23_gate(score, years)
    audit = r23_causal_audit(trades, events, config=cfg)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "market": args.symbol,
        "window": [args.warmup_start_date, args.end_date],
        "historical_selection_status": "full_window_contaminated_prior",
        "frozen_policy": "priority_union + multi_sweep_deeper_higher_low_trail + entry_delay_2",
        "thresholds": cfg.__dict__,
        "costs": {"roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "splits": {"discovery": "2023-2024 reset", "validation": "2025H1 reset", "embargo_start": str(cfg.embargo_start), "holdout_start": str(cfg.holdout_start)},
        "holdout_rows_loaded": 0,
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([
        {"check": "requested_minutes", "value": expected},
        {"check": "source_rows", "value": len(raw)},
        {"check": "missing_minutes", "value": expected - len(raw)},
        {"check": "gap_runs", "value": int(gaps.sum())},
        {"check": "maximum_gap", "value": str(raw.index.to_series().diff().max())},
        {"check": "regularized_minutes", "value": len(features)},
        {"check": "eligible_events", "value": len(events)},
        {"check": "closed_trades", "value": int(trades["path_status"].eq("included").sum())},
        {"check": "data_gap_censored", "value": int(trades["path_status"].eq("data_gap_censored").sum())},
        {"check": "boundary_censored", "value": int(trades["path_status"].eq("boundary_censored").sum())},
        {"check": "holdout_rows_loaded", "value": 0},
        {"check": "historical_v1_entry_policies", "value": 3},
        {"check": "historical_v1_exit_modes", "value": 9},
        {"check": "historical_v1_delays", "value": 3},
        {"check": "historical_v1_visible_combinations", "value": 81},
        {"check": "historical_v1_1_exit_upgrade_modes", "value": 7},
    ]).to_csv(out / "01_provenance_data_audit.csv", index=False)
    events.to_csv(out / "02_events.csv.gz", index=False, compression="gzip", float_format="%.17g")
    trades.to_csv(out / "03_trade_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    score.to_csv(out / "04_scorecard.csv", index=False)
    years.to_csv(out / "05_years.csv", index=False)
    gate.to_csv(out / "06_candidate_gate.csv", index=False)
    audit.to_csv(out / "07_causal_audit.csv", index=False)
    _manual_review(out, trades)
    (out / "R23_GENERATED_NOTE.md").write_text(
        "# R23 generated note\n\nFrozen historical panic-wick Long falsification. July and holdout outcomes are absent.\n",
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(score.to_string(index=False), flush=True)
    print(years.to_string(index=False), flush=True)
    print(gate.to_string(index=False), flush=True)
    print(f"[r23] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

