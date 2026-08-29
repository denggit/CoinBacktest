#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R22 — BTC-led ETH catch-up first passage."""
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
from src.research_common.ict_mss2.r22 import (  # noqa: E402
    R22Config,
    build_catchup_events,
    build_cross_market_features,
    build_r22_gate,
    r22_causal_audit,
    simulate_catchup,
    summarize_r22,
    summarize_r22_years,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

TITLE = "ETH ICT MSS2 R22 BTC-Led ETH Catch-Up First Passage"
EXPERIMENT_ID = "ETH_ICT_MSS2_BTC_LED_ETH_CATCHUP_R22"
EDGE_ID = "RESEARCH_ONLY_BTC_LED_ETH_CATCHUP"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r22_btc_led_eth_catchup"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--eth-symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--btc-symbol", default="BTC-USDT-SWAP")
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
    for (target_r, year), part in closed.groupby(["target_r", pd.to_datetime(closed["entry_time"]).dt.year], sort=True):
        part.sort_values("entry_time").tail(20).to_csv(directory / f"04_r{int(target_r)}_{int(year)}_recent_20.csv", index=False)
    (directory / "README.md").write_text(
        "# R22 manual review\n\nVerify the completed BTC impulse, prior-only beta/sigmas, ETH lag, next-hour ETH open, frozen ATR stop/target, and exact 1m first passage.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R22Config().validate()
    if pd.Timestamp(args.end_date) >= cfg.embargo_start:
        raise ValueError("R22 end date must remain before the July embargo")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r22] load aligned ETH/BTC 1m through src.data_feed", flush=True)
    eth = OKXDataLoader(symbol=args.eth_symbol, timeframe="1m").fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    btc = OKXDataLoader(symbol=args.btc_symbol, timeframe="1m").fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    eth_coverage = data_coverage_audit(eth, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=pd.Timestamp(args.end_date))
    eth_coverage.insert(0, "market", "ETH")
    btc_coverage = data_coverage_audit(btc, requested_start=pd.Timestamp(args.warmup_start_date), requested_end=pd.Timestamp(args.end_date))
    btc_coverage.insert(0, "market", "BTC")

    print("[r22] build prior-only beta, sigma, and catch-up events", flush=True)
    features = build_cross_market_features(eth, btc, config=cfg)
    events = build_catchup_events(features)
    pieces = []
    for target_r in cfg.target_rs:
        for split, start, end in (
            ("discovery", cfg.discovery_start, cfg.validation_start),
            ("validation", cfg.validation_start, cfg.embargo_start),
        ):
            for direction in (1, -1):
                part = simulate_catchup(
                    eth,
                    events,
                    target_r=target_r,
                    direction=direction,
                    split=split,
                    split_start=start,
                    split_end=end,
                    config=cfg,
                )
                if not part.empty:
                    pieces.append(part)
    trades = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if trades.empty:
        raise RuntimeError("R22 produced no trades")
    score = summarize_r22(trades, config=cfg)
    years = summarize_r22_years(trades)
    gate = build_r22_gate(score, years)
    audit = r22_causal_audit(trades, config=cfg)

    common_minutes = eth.index.intersection(btc.index)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "markets": [args.eth_symbol, args.btc_symbol],
        "window": [args.warmup_start_date, args.end_date],
        "signal": {
            "beta_hours": cfg.beta_hours,
            "btc_sigma_hours": cfg.btc_sigma_hours,
            "residual_sigma_hours": cfg.residual_sigma_hours,
            "impulse_z": cfg.impulse_z,
            "lag_z": cfg.lag_z,
        },
        "execution": {"stop_atr": cfg.stop_atr, "target_rs": list(cfg.target_rs), "max_hold_hours": cfg.max_hold_hours},
        "costs": {"roundtrip": cfg.market_roundtrip_cost, "scales": list(cfg.cost_scales)},
        "splits": {"discovery": "2023-2024 reset", "validation": "2025H1 reset", "embargo_start": str(cfg.embargo_start), "holdout_start": str(cfg.holdout_start)},
        "holdout_rows_loaded": 0,
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.concat([eth_coverage, btc_coverage], ignore_index=True).to_csv(out / "01_data_coverage.csv", index=False)
    pd.DataFrame([
        {"check": "eth_1m_rows", "value": len(eth)},
        {"check": "btc_1m_rows", "value": len(btc)},
        {"check": "aligned_1m_rows", "value": len(common_minutes)},
        {"check": "eth_only_minutes", "value": len(eth.index.difference(btc.index))},
        {"check": "btc_only_minutes", "value": len(btc.index.difference(eth.index))},
        {"check": "complete_hour_rows", "value": len(features)},
        {"check": "signal_events", "value": len(events)},
        {"check": "closed_paths", "value": int(trades["path_status"].eq("included").sum())},
        {"check": "boundary_censored_paths", "value": int(trades["path_status"].eq("boundary_censored").sum())},
        {"check": "holdout_rows_loaded", "value": 0},
    ]).to_csv(out / "02_data_and_funnel.csv", index=False)
    events.to_csv(out / "03_signal_events.csv.gz", index=False, compression="gzip", float_format="%.17g")
    trades.to_csv(out / "04_trade_paths.csv.gz", index=False, compression="gzip", float_format="%.17g")
    score.to_csv(out / "05_scorecard.csv", index=False)
    years.to_csv(out / "06_years.csv", index=False)
    gate.to_csv(out / "07_candidate_gate.csv", index=False)
    audit.to_csv(out / "08_causal_audit.csv", index=False)
    _manual_review(out, trades)
    (out / "R22_GENERATED_NOTE.md").write_text(
        "# R22 generated note\n\nBTC-led ETH catch-up with prior-only beta/sigmas. July and holdout outcomes are absent.\n",
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(score.to_string(index=False), flush=True)
    print(gate.to_string(index=False), flush=True)
    print(f"[r22] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

