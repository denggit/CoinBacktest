#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild OKX Range Bar Footprints into SQLite.

This tool is optimized for multiple range thresholds: raw trades are read and
normalized once per day/chunk, then fed into all requested RangeBarBuilder
instances with the same price bucket size.

Example:
    python tools/prebuild_okx_range_footprints.py \
        --symbol ETH-USDT-SWAP \
        --start-date 2023-01-01 \
        --end-date 2026-06-15 \
        --range-pcts 0.0015 0.0020 0.0025 \
        --price-step 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_range_bar_loader import (  # noqa: E402
    DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
    DEFAULT_OKX_TRADES_URL_TEMPLATE,
    DEFAULT_RANGE_PCTS,
    RangeBarBuilder,
    iter_trade_csv_chunks,
    normalize_trade_chunk_fast,
    range_code,
)
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader, price_step_code  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger("OKXRangeFootprintPrebuilder")


@dataclass
class BuildResult:
    symbol: str
    range_pct: float
    range_code: str
    price_step: float
    step_code: str
    utc_start: str
    utc_end: str
    status: str
    footprints_written: int
    bars_closed: int
    chunks_read: int
    elapsed_seconds: float
    table_name: str
    db_path: str
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prebuild OKX Range Bar Footprints from official trades ZIP files.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", required=True, help="UTC start day, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="UTC end day, YYYY-MM-DD")
    p.add_argument("--range-pcts", nargs="+", type=float, default=list(DEFAULT_RANGE_PCTS), help="Range thresholds, e.g. 0.0015 0.002 0.0025")
    p.add_argument("--price-step", type=float, default=1.0, help="Footprint price bucket size, e.g. 1 or 0.5")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_range_footprints.db")
    p.add_argument("--url-template", default=DEFAULT_OKX_TRADES_URL_TEMPLATE)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--flush-rows", type=int, default=200_000, help="Flush buffered footprint rows to SQLite after this many rows per range. Larger is faster but uses more memory.")
    p.add_argument("--contract-value", type=float, default=None)
    p.add_argument("--large-trade-notional-threshold", type=float, default=DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD)
    p.add_argument("--utc-timestamps", action="store_true", help="Store UTC-naive timestamps instead of matching OKXDataLoader timezone.")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--sleep-sec", type=float, default=0.0)
    return p.parse_args(argv)


def parse_day(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc


def date_range(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError(f"--end-date must be >= --start-date, got {start} -> {end}")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def make_loader(args: argparse.Namespace, range_pct: float) -> OKXRangeFootprintLoader:
    kwargs = {
        "symbol": args.symbol,
        "range_pct": range_pct,
        "price_step": args.price_step,
        "data_dir": args.data_dir,
        "db_name": args.db_name,
        "trades_url_template": args.url_template,
        "contract_value": args.contract_value,
        "large_trade_notional_threshold": args.large_trade_notional_threshold,
        "align_with_okx_loader_timezone": not bool(args.utc_timestamps),
    }
    if args.contract_value is None:
        kwargs.pop("contract_value")
    return OKXRangeFootprintLoader(**kwargs)


def _footprint_end_day(row: dict) -> date:
    return pd.Timestamp(row["end_ts"]).date()


def _empty_result(args: argparse.Namespace, loader: OKXRangeFootprintLoader, start: date, end: date, range_pct: float, status: str, elapsed: float) -> BuildResult:
    return BuildResult(
        symbol=args.symbol,
        range_pct=range_pct,
        range_code=range_code(range_pct),
        price_step=float(args.price_step),
        step_code=price_step_code(float(args.price_step)),
        utc_start=start.isoformat(),
        utc_end=end.isoformat(),
        status=status,
        footprints_written=0,
        bars_closed=0,
        chunks_read=0,
        elapsed_seconds=elapsed,
        table_name=loader.table_name,
        db_path=str(loader.db_path),
    )


def prebuild_multi_ranges(args: argparse.Namespace, start: date, end: date) -> list[BuildResult]:
    started = time.time()
    days = list(date_range(start, end))
    range_pcts = [float(x) for x in args.range_pcts]
    loaders = {rp: make_loader(args, rp) for rp in range_pcts}

    if args.dry_run:
        elapsed = time.time() - started
        return [_empty_result(args, loader, start, end, rp, "dry_run", elapsed) for rp, loader in loaders.items()]

    missing_days: dict[float, set[date]] = {}
    for rp, loader in loaders.items():
        if args.force_rebuild:
            loader._delete_cached_days(days)
            missing_days[rp] = set(days)
        else:
            missing_days[rp] = {d for d in days if not loader._has_coverage(d)}

    if not any(missing_days.values()):
        elapsed = time.time() - started
        logger.info(
            "[RANGE-FOOTPRINT-PREBUILD-SKIP-ALL] symbol=%s ranges=%s step=%s days=%s->%s already cached",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            args.price_step,
            start,
            end,
        )
        return [_empty_result(args, loader, start, end, rp, "skipped", elapsed) for rp, loader in loaders.items()]

    builders = {
        rp: RangeBarBuilder(
            range_pct=rp,
            contract_value=loaders[rp].contract_value,
            large_trade_notional_threshold=loaders[rp].large_trade_notional_threshold,
            price_step=float(args.price_step),
        )
        for rp in range_pcts
    }
    footprints_written = {rp: 0 for rp in range_pcts}
    bars_closed = {rp: 0 for rp in range_pcts}
    chunks_read = 0

    logger.info(
        "[RANGE-FOOTPRINT-PREBUILD-START] symbol=%s ranges=%s step=%s days=%s->%s db=%s chunksize=%s force=%s mode=one_pass_multi_range",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        args.price_step,
        start,
        end,
        next(iter(loaders.values())).db_path,
        args.chunksize,
        bool(args.force_rebuild),
    )

    def flush_pending(rp: float, pending: list[dict]) -> int:
        if not pending:
            return 0
        loader = loaders[rp]
        df = loader._footprints_to_frame(pending)
        loader._upsert_footprints(df)
        flushed = len(df)
        pending.clear()
        return flushed

    flush_rows = max(1, int(args.flush_rows))
    for day in days:
        raw_file = next(iter(loaders.values()))._ensure_raw_trade_file(day)
        logger.info(
            "[RANGE-FOOTPRINT-MULTI-DAY-START] symbol=%s ranges=%s step=%s utc_day=%s raw=%s",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            args.price_step,
            day,
            raw_file,
        )
        day_rows = {rp: 0 for rp in range_pcts}
        day_bars = {rp: 0 for rp in range_pcts}
        pending_footprints: dict[float, list[dict]] = {rp: [] for rp in range_pcts}
        for raw in iter_trade_csv_chunks(raw_file, chunksize=int(args.chunksize)):
            chunks_read += 1
            chunk = normalize_trade_chunk_fast(raw)
            for rp in range_pcts:
                bars, footprints = builders[rp].process_chunk(chunk)
                if bars:
                    closed_count = sum(1 for b in bars if pd.Timestamp(b["end_ts"]).date() in missing_days[rp])
                    bars_closed[rp] += closed_count
                    day_bars[rp] += sum(1 for b in bars if pd.Timestamp(b["end_ts"]).date() == day and day in missing_days[rp])
                if not footprints:
                    continue
                footprints_to_write = [fp for fp in footprints if _footprint_end_day(fp) in missing_days[rp]]
                if not footprints_to_write:
                    continue
                pending_footprints[rp].extend(footprints_to_write)
                footprints_written[rp] += len(footprints_to_write)
                day_rows[rp] += len(footprints_to_write)
                if len(pending_footprints[rp]) >= flush_rows:
                    flush_pending(rp, pending_footprints[rp])
        for rp in range_pcts:
            flush_pending(rp, pending_footprints[rp])
            if day in missing_days[rp]:
                loaders[rp]._mark_coverage(day, rows=day_rows[rp], bars=day_bars[rp])
        logger.info(
            "[RANGE-FOOTPRINT-MULTI-DAY-DONE] symbol=%s step=%s utc_day=%s rows_by_range=%s bars_by_range=%s chunks_read_total=%s",
            args.symbol,
            args.price_step,
            day,
            ",".join(f"{range_code(rp)}:{day_rows[rp]}" for rp in range_pcts),
            ",".join(f"{range_code(rp)}:{day_bars[rp]}" for rp in range_pcts),
            chunks_read,
        )
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    elapsed = time.time() - started
    results: list[BuildResult] = []
    for rp in range_pcts:
        status = "built" if footprints_written[rp] > 0 or missing_days[rp] else "skipped"
        loader = loaders[rp]
        results.append(
            BuildResult(
                symbol=args.symbol,
                range_pct=rp,
                range_code=range_code(rp),
                price_step=float(args.price_step),
                step_code=price_step_code(float(args.price_step)),
                utc_start=start.isoformat(),
                utc_end=end.isoformat(),
                status=status,
                footprints_written=int(footprints_written[rp]),
                bars_closed=int(bars_closed[rp]),
                chunks_read=chunks_read,
                elapsed_seconds=elapsed,
                table_name=loader.table_name,
                db_path=str(loader.db_path),
            )
        )
    logger.info(
        "[RANGE-FOOTPRINT-PREBUILD-DONE] symbol=%s ranges=%s step=%s total_footprints=%s raw_chunks_read=%s elapsed=%.2fs",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        args.price_step,
        sum(footprints_written.values()),
        chunks_read,
        elapsed,
    )
    return results


def print_summary(results: list[BuildResult]) -> None:
    print("=" * 120)
    print("OKX Range Footprint prebuild summary")
    print(f"jobs               : {len(results)}")
    print(f"built              : {sum(1 for r in results if r.status == 'built')}")
    print(f"skipped            : {sum(1 for r in results if r.status == 'skipped')}")
    print(f"failed             : {sum(1 for r in results if r.status == 'failed')}")
    print(f"footprints_written : {sum(r.footprints_written for r in results)}")
    print(f"bars_closed        : {sum(r.bars_closed for r in results)}")
    print(f"raw_chunks_read    : {max((r.chunks_read for r in results), default=0)}")
    print("=" * 120)
    for r in results:
        print(json.dumps(asdict(r), ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = parse_day(args.start_date, "--start-date")
    end = parse_day(args.end_date, "--end-date")
    try:
        results = prebuild_multi_ranges(args, start, end)
    except Exception as exc:
        logger.exception("[RANGE-FOOTPRINT-PREBUILD-FAILED] symbol=%s ranges=%s step=%s error=%s", args.symbol, args.range_pcts, args.price_step, exc)
        if not args.continue_on_error:
            print_summary([
                BuildResult(
                    symbol=args.symbol,
                    range_pct=0.0,
                    range_code="multi",
                    price_step=float(args.price_step),
                    step_code=price_step_code(float(args.price_step)),
                    utc_start=start.isoformat(),
                    utc_end=end.isoformat(),
                    status="failed",
                    footprints_written=0,
                    bars_closed=0,
                    chunks_read=0,
                    elapsed_seconds=0.0,
                    table_name="",
                    db_path=str(Path(args.data_dir or PROJECT_ROOT / "data") / args.db_name),
                    error=repr(exc),
                )
            ])
            return 1
        results = []
    print_summary(results)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
