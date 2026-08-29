#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R21 — canonical daily channel trend following."""
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
from src.research_common.ict_mss2.r13 import data_coverage_audit  # noqa: E402
from src.research_common.ict_mss2.r21 import (  # noqa: E402
    R21Config,
    build_daily_channel_features,
    r21_causal_audit,
    simulate_daily_channel,
    summarize_r21,
    summarize_r21_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

TITLE = "ETH ICT MSS2 R21 Canonical Daily Channel Trend Following"
EXPERIMENT_ID = "ETH_ICT_MSS2_CANONICAL_DAILY_CHANNEL_TREND_R21"
EDGE_ID = "RESEARCH_ONLY_DAILY_DONCHIAN_LONG_SHORT"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r21_canonical_daily_channel_trend"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _gate(score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = score.loc[score["model"].eq("D20_X10")]
    sensitivity = score.loc[score["model"].eq("D55_X20")]
    for direction in ("Long", "Short"):
        reasons = []
        p = primary.loc[primary["direction"].eq(direction)].set_index("research_split")
        s = sensitivity.loc[(sensitivity["direction"].eq(direction)) & sensitivity["research_split"].eq("discovery")]
        for split, minimum in (("discovery", 8), ("validation", 2)):
            if split not in p.index:
                reasons.append(f"missing_{split}")
                continue
            row = p.loc[split]
            if int(row["trades"]) < minimum:
                reasons.append(f"{split}_sample")
            if float(row["net_pf_cost2x"]) < 1.4:
                reasons.append(f"{split}_pf2x")
            if float(row["mean_net_return_cost2x"]) <= 0:
                reasons.append(f"{split}_expectancy")
        if "discovery" in p.index and float(p.loc["discovery", "net_sum_cost2x_top5_removed"]) <= 0:
            reasons.append("discovery_top5")
        if s.empty or float(s.iloc[0]["mean_net_return_cost2x"]) <= 0:
            reasons.append("sensitivity_discovery")
        rows.append({"direction": direction, "research_candidate": int(not reasons), "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons)})
    return pd.DataFrame(rows)


def _manual_review(out: Path, trades: pd.DataFrame) -> None:
    directory = out / "manual_review"
    directory.mkdir(parents=True, exist_ok=True)
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    closed.sort_values("entry_time").tail(60).to_csv(directory / "01_recent_60.csv", index=False)
    closed.sort_values("net_return_cost2x", ascending=False).head(30).to_csv(directory / "02_best_30.csv", index=False)
    closed.sort_values("net_return_cost2x").head(30).to_csv(directory / "03_worst_30.csv", index=False)
    for (model, year), part in closed.groupby(["model", pd.to_datetime(closed["entry_time"]).dt.year], sort=True):
        part.sort_values("entry_time").tail(15).to_csv(directory / f"04_{model.lower()}_{int(year)}_recent_15.csv", index=False)
    (directory / "README.md").write_text(
        "# R21 manual review\n\nVerify prior completed daily channels, signal-bar close, next-day 00:00 entry, "
        "fixed 2ATR stop, first 1m stop touch or next-open channel exit, and split-boundary censoring.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R21Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("R21 end date must remain before the July embargo")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r21] load bare OKX 1m through src.data_feed", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m").fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    coverage = data_coverage_audit(bars, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=pd.Timestamp(args.end_date))
    print("[r21] build completed daily channels and ATR", flush=True)
    daily = build_daily_channel_features(bars, config=cfg)

    pieces = []
    for model in cfg.models:
        for split, start, end in (
            ("discovery", cfg.discovery_start, cfg.validation_start),
            ("validation", cfg.validation_start, cfg.embargo_start),
        ):
            for direction in (1, -1):
                pieces.append(simulate_daily_channel(bars, daily, model=model, direction=direction, split=split, split_start=start, split_end=end, config=cfg))
    trades = pd.concat([part for part in pieces if not part.empty], ignore_index=True, sort=False) if any(not part.empty for part in pieces) else pd.DataFrame()
    score = summarize_r21(trades, config=cfg)
    years = summarize_r21_years(trades)
    gate = _gate(score)
    audit = r21_causal_audit(trades, config=cfg)

    manifest = {
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID, "title": TITLE,
        "market": args.symbol, "window": [args.warmup_start_date, args.end_date],
        "models": [{"name": m.name, "entry_window": m.entry_window, "exit_window": m.exit_window} for m in cfg.models],
        "atr_window": cfg.atr_window, "stop_atr": cfg.stop_atr,
        "splits": {"discovery": "2023-2024 reset simulation", "validation": "2025H1 reset simulation", "embargo_start": str(cfg.embargo_start), "holdout_start": str(cfg.holdout_start)},
        "holdout_rows_loaded": 0, "costs": {"roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "same_bar_policy": "stop_first; no fixed target", "positioning": "unlevered; Long/Short separate; one position; no add-ons",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    coverage.to_csv(out / "01_data_coverage.csv", index=False)
    pd.DataFrame(
        [
            {"check": "daily_rows", "value": len(daily)},
            {"check": "daily_start", "value": str(daily.index.min())},
            {"check": "daily_end", "value": str(daily.index.max())},
            {"check": "daily_duplicate_timestamps", "value": int(daily.index.duplicated().sum())},
            {"check": "holdout_rows_loaded", "value": 0},
            {"check": "closed_trades", "value": int(trades["path_status"].eq("included").sum())},
            {"check": "boundary_censored", "value": int(trades["path_status"].eq("boundary_censored").sum())},
        ]
    ).to_csv(out / "02_data_and_boundary_audit.csv", index=False)
    trades.to_csv(out / "03_trade_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    score.to_csv(out / "04_scorecard.csv", index=False)
    years.to_csv(out / "05_years.csv", index=False)
    gate.to_csv(out / "06_candidate_gate.csv", index=False)
    audit.to_csv(out / "07_causal_audit.csv", index=False)
    _manual_review(out, trades)
    (out / "R21_GENERATED_NOTE.md").write_text(
        "# R21 generated note\n\nCanonical daily channel trend following. Long/Short and split simulations are separate; July and holdout outcomes are absent.\n",
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(score.to_string(index=False), flush=True)
    print(gate.to_string(index=False), flush=True)
    print(f"[r21] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

