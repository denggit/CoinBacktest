#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download public OKX inputs for Estimated Liquidation Heatmap V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_derivatives_loader import OKXDerivativesLoader  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild OKX OI/Funding/Mark/Liquidation data for the estimated liquidation heatmap")
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--oi-period", default="5m", choices=["5m", "15m", "30m", "1H", "2H", "4H", "1D"], help="Preferred OI period. The loader auto-upgrades when OKX's latest-1,440-entry window cannot reach start-date.")
    parser.add_argument("--mark-timeframe", default="1m")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["open_interest", "funding_rate", "mark_price", "liquidation"],
        choices=["open_interest", "funding_rate", "mark_price", "liquidation"],
    )
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loader = OKXDerivativesLoader(symbol=args.symbol, data_dir=args.data_dir)
    print(f"[run] OKX liquidation-map inputs | {args.symbol} | {args.start_date} -> {args.end_date}", flush=True)
    print("[note] Public endpoints are used; no API key or account setup is required.", flush=True)
    print(
        "[note] OI uses OKX contract-level open-interest-history with instId; "
        "the endpoint exposes only the latest 1,440 entries per period.",
        flush=True,
    )
    print(
        "[note] --oi-period is a preferred resolution. The loader automatically selects "
        "the finest coarser period that can still reach --start-date.",
        flush=True,
    )

    def progress(name: str):
        last = {"value": -1}

        def callback(rows: int) -> None:
            if rows != last["value"] and (rows < 500 or rows % 500 == 0):
                print(f"[{name}] rows={rows:,}", flush=True)
                last["value"] = rows

        return callback

    results = {}
    errors: dict[str, str] = {}

    def run_dataset(name: str, stage_label: str, fetcher) -> None:
        print(f"[stage] {stage_label}", flush=True)
        try:
            results[name] = fetcher()
        except Exception as exc:  # Continue so one public endpoint does not hide the rest.
            errors[name] = str(exc)
            print(f"[error] {name}: {exc}", flush=True)

    if "open_interest" in args.datasets:
        run_dataset(
            "open_interest",
            "open interest history",
            lambda: loader.fetch_open_interest_history(
                args.start_date,
                args.end_date,
                period=args.oi_period,
                progress=progress("open_interest"),
            ),
        )
    if "funding_rate" in args.datasets:
        run_dataset(
            "funding_rate",
            "funding rate history",
            lambda: loader.fetch_funding_rate_history(
                args.start_date,
                args.end_date,
                progress=progress("funding_rate"),
            ),
        )
    if "mark_price" in args.datasets:
        run_dataset(
            "mark_price",
            "mark price history",
            lambda: loader.fetch_mark_price_history(
                args.start_date,
                args.end_date,
                timeframe=args.mark_timeframe,
                progress=progress("mark_price"),
            ),
        )
    if "liquidation" in args.datasets:
        run_dataset(
            "liquidation",
            "liquidation orders",
            lambda: loader.fetch_liquidation_orders(
                args.start_date,
                args.end_date,
                progress=progress("liquidation"),
            ),
        )


    print("[done] downloaded/persisted", flush=True)
    for name, frame in results.items():
        note = getattr(frame, "attrs", {}).get("availability_note", "") if frame is not None else ""
        if frame is None or frame.empty:
            if note:
                print(f"  {name}: 0 rows | {note}", flush=True)
            elif name == "funding_rate":
                print("  funding_rate: 0 rows (requested range may precede OKX coverage)", flush=True)
            else:
                print(f"  {name}: 0 rows (endpoint returned no data for the requested range)", flush=True)
        else:
            print(f"  {name}: {len(frame):,} rows | {frame.index.min()} -> {frame.index.max()}", flush=True)
            if name == "open_interest":
                attrs = getattr(frame, "attrs", {})
                source = attrs.get("oi_source", "unknown")
                requested_period = attrs.get("requested_period", args.oi_period)
                effective_period = attrs.get("effective_period", requested_period)
                usd_rows = int(frame.get("oi_usd", []).notna().sum()) if "oi_usd" in frame else 0
                ccy_rows = int(frame.get("oi_ccy", []).notna().sum()) if "oi_ccy" in frame else 0
                print(
                    f"    source: {source} | requested_period={requested_period} | "
                    f"effective_period={effective_period} | oi_usd rows={usd_rows:,} | oi_ccy rows={ccy_rows:,}",
                    flush=True,
                )
                if usd_rows == 0 and ccy_rows > 0:
                    print("    note: heatmap engine will causally convert oi_ccy × aligned mark price to USD OI", flush=True)
            if note:
                print(f"    note: {note}", flush=True)
    if errors:
        print("[errors] one or more datasets failed:", flush=True)
        for name, message in errors.items():
            print(f"  {name}: {message}", flush=True)
    print(f"  database: {loader.db_path}", flush=True)
    print("[note] Funding count near 90 rows for one month is normal because funding is periodic, not 1-minute data.", flush=True)
    print("[note] liquidation=0 is expected for historical backfill and does not block the estimated heatmap; open_interest is required.", flush=True)
    print("[note] The heatmap is an estimate. Missing datasets reduce confidence; they are never fabricated.", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
