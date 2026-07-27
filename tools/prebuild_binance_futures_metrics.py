#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download Binance USD-M futures 5-minute metrics from the official archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.binance_futures_metrics_loader import (  # noqa: E402
    EXPECTED_ROWS_PER_DAY,
    BinanceFuturesMetricsLoader,
)
from src.research_common.progress import ProgressReporter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official Binance USD-M daily metrics archives into a local SQLite cache"
    )
    parser.add_argument("--symbol", default="ETHUSDT", help="Binance USD-M symbol; ETH-USDT-SWAP is also accepted")
    parser.add_argument("--start-date", default="2022-01-01", help="First official UTC archive day")
    parser.add_argument("--end-date", default="2026-06-30", help="Last official UTC archive day")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent daily downloads; SQLite writes remain serialized")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-verify-checksum", action="store_true", help="Skip official SHA256 verification")
    parser.add_argument("--require-checksum", action="store_true", help="Fail a day when its CHECKSUM file is unavailable")
    parser.add_argument("--no-keep-raw", action="store_true", help="Do not retain official ZIP archives under data/binance/raw")
    parser.add_argument("--inspect-day", default=None, help="Fetch and parse one UTC day without writing SQLite")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loader = BinanceFuturesMetricsLoader(
        symbol=args.symbol,
        data_dir=args.data_dir,
        timeout=args.timeout,
    )
    verify_checksum = not args.no_verify_checksum

    print(
        f"[run] Binance USD-M futures metrics | {loader.symbol} | "
        f"UTC days {args.start_date} -> {args.end_date} | period=5m workers={max(1, args.workers)}",
        flush=True,
    )
    print("[note] Source: official data.binance.vision daily metrics archives; no API key is required.", flush=True)
    print(
        "[note] Archive days are UTC. Returned loader timestamps follow the CoinBacktest project timezone; "
        "strategy features use a configurable causal publication lag.",
        flush=True,
    )
    print(
        "[note] Raw ZIP files are retained by default for reproducibility. Resume skips completed days; "
        "--force-rebuild replaces them.",
        flush=True,
    )

    if args.inspect_day:
        result = loader.inspect_archive_day(
            args.inspect_day,
            verify_checksum=verify_checksum,
            require_checksum=args.require_checksum,
            retries=args.retries,
        )
        print(
            f"[inspect] day={result.day_utc} status={result.status} rows={result.rows:,} "
            f"checksum_verified={result.checksum_verified} url={result.source_url}",
            flush=True,
        )
        if result.frame is not None and not result.frame.empty:
            print(result.frame.head(3).to_string(index=False), flush=True)
            print(result.frame.tail(3).to_string(index=False), flush=True)
        return 0 if result.status in {"complete", "partial"} else 1

    total_days = (pd.Timestamp(args.end_date).date() - pd.Timestamp(args.start_date).date()).days + 1
    reporter = ProgressReporter(
        label="[binance-metrics] days",
        total=max(0, int(total_days)),
        every=max(1, int(total_days // 100) if total_days > 100 else 1),
    )
    last_message = {"status": ""}

    def on_progress(done: int, total: int, result) -> None:  # type: ignore[no-untyped-def]
        reporter.total = total
        reporter.update(done, force=result.status in {"missing", "error", "partial"})
        if result.status in {"missing", "error", "partial"}:
            message = (
                f"[{result.day_utc}] status={result.status} rows={result.rows:,} "
                f"checksum={result.checksum_verified} error={result.error or '-'}"
            )
            if message != last_message["status"]:
                print(message, flush=True)
                last_message["status"] = message

    summary = loader.download_history(
        args.start_date,
        args.end_date,
        workers=args.workers,
        force_rebuild=args.force_rebuild,
        verify_checksum=verify_checksum,
        require_checksum=args.require_checksum,
        keep_raw=not args.no_keep_raw,
        retries=args.retries,
        progress=on_progress,
    )
    reporter.close()

    coverage = loader.coverage()
    requested = loader.load_archive_days(args.start_date, args.end_date, index_mode="none")
    expected_rows = summary.requested_days * EXPECTED_ROWS_PER_DAY
    completeness = len(requested) / expected_rows if expected_rows else 0.0

    print("\n[summary]", flush=True)
    print(f"  requested_days: {summary.requested_days:,}", flush=True)
    print(f"  downloaded_days: {summary.downloaded_days:,}", flush=True)
    print(f"  skipped_days: {summary.skipped_days:,}", flush=True)
    print(f"  partial_days: {summary.partial_days:,}", flush=True)
    print(f"  missing_days: {summary.missing_days:,}", flush=True)
    print(f"  error_days: {summary.error_days:,}", flush=True)
    print(f"  rows_written_this_run: {summary.rows_written:,}", flush=True)
    print(f"  requested_rows_in_db: {len(requested):,} / expected≈{expected_rows:,} ({completeness:.3%})", flush=True)
    if not requested.empty:
        print(f"  local_timestamp_range: {requested['timestamp'].min()} -> {requested['timestamp'].max()}", flush=True)
        print(f"  oi_usd_non_null: {int(requested['sum_open_interest_value'].notna().sum()):,}", flush=True)
    print(
        f"  total_db_coverage: rows={coverage.rows:,} days_complete={coverage.complete_days:,} "
        f"days_partial={coverage.partial_days:,} days_missing={coverage.missing_days:,} days_error={coverage.error_days:,}",
        flush=True,
    )
    print(f"  database: {summary.db_path}", flush=True)
    print(f"  elapsed_seconds: {summary.elapsed_seconds:.1f}", flush=True)

    if summary.error_days:
        print("[result] completed with errors; rerun the same command to resume failed days.", flush=True)
        return 1
    print("[result] completed. Rerunning the same command is resume-safe.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
