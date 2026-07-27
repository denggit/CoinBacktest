#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI orchestration for the ETH long martingale backtest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE

from .config import EngineConfig, MartingaleVariant, VARIANTS
from .data import load_bar_data, run_bar_replay, run_raw_trade_replay
from .engine import MartingaleEngine
from .reporting import _json_dump, print_comparison, write_engine_outputs

SCRIPT_NAME = "eth_martingale_limit_long_backtest"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/backtest/mf/eth_martingale_limit_long"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--variant",
        choices=["all", *VARIANTS.keys()],
        default="all",
        help="Run all three frozen variants or one named variant.",
    )
    parser.add_argument(
        "--data-source",
        choices=["trade_bar", "range_bar", "raw_trade"],
        default="trade_bar",
    )
    parser.add_argument("--trade-bar-timeframe", default="1m")
    parser.add_argument("--range-pct", type=float, default=0.0020)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument(
        "--capital-utilization",
        type=float,
        default=1.0,
        help="Fraction of cycle capital reserved for the complete ladder.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.00055,
        help="Fee charged per fill side. Default 0.055%% => about 0.11%% round trip.",
    )
    parser.add_argument(
        "--maintenance-margin-rate",
        type=float,
        default=0.005,
        help="Approximate maintenance-margin rate for liquidation audit.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Read existing local cache/raw ZIP only; do not build/download missing days.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--leave-open-end",
        action="store_true",
        help="Do not force-close an open cycle at the end of the test window.",
    )
    parser.add_argument("--skip-full-report", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--trades-url-template",
        default=DEFAULT_OKX_TRADES_URL_TEMPLATE,
    )
    args = parser.parse_args(argv)

    if pd.Timestamp(args.end_date) < pd.Timestamp(args.start_date):
        parser.error("--end-date must be >= --start-date")
    if args.chunksize <= 0:
        parser.error("--chunksize must be > 0")
    if args.range_pct <= 0:
        parser.error("--range-pct must be > 0")
    EngineConfig(
        initial_capital=args.initial_capital,
        capital_utilization=args.capital_utilization,
        fee_rate=args.fee_rate,
        maintenance_margin_rate=args.maintenance_margin_rate,
        force_close_end=not args.leave_open_end,
    ).validate()
    return args


def selected_variants(name: str) -> list[MartingaleVariant]:
    if name == "all":
        return [VARIANTS[key] for key in ("midterm", "aggressive", "longterm")]
    return [VARIANTS[name]]


def build_engines(args: argparse.Namespace) -> list[MartingaleEngine]:
    config = EngineConfig(
        initial_capital=float(args.initial_capital),
        capital_utilization=float(args.capital_utilization),
        fee_rate=float(args.fee_rate),
        maintenance_margin_rate=float(args.maintenance_margin_rate),
        force_close_end=not bool(args.leave_open_end),
    )
    return [MartingaleEngine(variant, config) for variant in selected_variants(args.variant)]


def run(args: argparse.Namespace) -> pd.DataFrame:
    engines = build_engines(args)
    print(
        f"[config] source={args.data_source} symbol={args.symbol} "
        f"range={args.start_date}->{args.end_date} variants={[e.variant.key for e in engines]}",
        flush=True,
    )
    print(
        f"[cost] fee_rate_per_side={args.fee_rate:.5%} "
        f"round_trip_base={2 * args.fee_rate:.5%} maintenance_margin={args.maintenance_margin_rate:.3%}",
        flush=True,
    )

    if args.data_source in {"trade_bar", "range_bar"}:
        print("[stage] load cached/build bar data", flush=True)
        bars = load_bar_data(args)
        print(f"[data] rows={len(bars):,} first={bars.index[0]} last={bars.index[-1]}", flush=True)
        run_bar_replay(
            bars,
            engines,
            source=args.data_source,
            progress_enabled=not bool(args.no_progress),
        )
        del bars
    else:
        print("[stage] stream raw trades", flush=True)
        run_raw_trade_replay(
            args,
            engines,
            progress_enabled=not bool(args.no_progress),
        )

    for engine in engines:
        engine.finalize(force_close=not bool(args.leave_open_end))

    summaries = [write_engine_outputs(engine, args) for engine in engines]
    comparison = pd.DataFrame(summaries)
    root_out = Path(args.out_dir) / args.data_source
    root_out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(root_out / "00_variant_comparison.csv", index=False)
    _json_dump(
        root_out / "00_run_manifest.json",
        {
            "script": SCRIPT_NAME,
            "created_at": pd.Timestamp.utcnow(),
            "arguments": vars(args),
            "variants": [asdict(engine.variant) for engine in engines],
            "comparison_file": "00_variant_comparison.csv",
        },
    )
    print_comparison(comparison, root_out)
    return comparison


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
