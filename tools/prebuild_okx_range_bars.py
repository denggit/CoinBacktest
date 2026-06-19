#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild OKX trade-derived Range Bars into SQLite.

This tool is optimized for multiple range thresholds: raw trades are read and
normalized once per day/chunk, then fed into all requested RangeBarBuilder
instances.  That avoids reading the same multi-year tick ZIPs once per range.

Example:
    python tools/prebuild_okx_range_bars.py \
        --symbol ETH-USDT-SWAP \
        --start-date 2023-01-01 \
        --end-date 2026-06-15 \
        --range-pcts 0.0015 0.0020 0.0025
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
    DEFAULT_RANGE_PCTS,
    DEFAULT_OKX_TRADES_URL_TEMPLATE,
    OKXRangeBarLoader,
    RangeBarBuilder,
    iter_trade_csv_chunks,
    normalize_trade_chunk_fast,
    range_code,
)
from src.utils.log import get_logger  # noqa: E402

logger = get_logger("OKXRangeBarPrebuilder")


@dataclass
class BuildResult:
    symbol: str
    range_pct: float
    range_code: str
    utc_start: str
    utc_end: str
    status: str
    bars_written: int
    chunks_read: int
    elapsed_seconds: float
    table_name: str
    db_path: str
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prebuild OKX Range Bars from official trades ZIP files.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", required=True, help="UTC start day, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="UTC end day, YYYY-MM-DD")
    p.add_argument("--range-pcts", nargs="+", type=float, default=list(DEFAULT_RANGE_PCTS), help="Range thresholds, e.g. 0.0015 0.002 0.0025")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_range_bars.db")
    p.add_argument("--url-template", default=DEFAULT_OKX_TRADES_URL_TEMPLATE)
    p.add_argument("--chunksize", type=int, default=300_000)
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


def make_loader(args: argparse.Namespace, range_pct: float) -> OKXRangeBarLoader:
    kwargs = {
        "symbol": args.symbol,
        "range_pct": range_pct,
        "data_dir": args.data_dir,
        "db_name": args.db_name,
        "trades_url_template": args.url_template,
        "contract_value": args.contract_value,
        "large_trade_notional_threshold": args.large_trade_notional_threshold,
        "align_with_okx_loader_timezone": not bool(args.utc_timestamps),
    }
    if args.contract_value is None:
        kwargs.pop("contract_value")
    return OKXRangeBarLoader(**kwargs)


def _bar_end_day(bar: dict) -> date:
    return pd.Timestamp(bar["end_ts"]).date()


def _empty_result(args: argparse.Namespace, loader: OKXRangeBarLoader, start: date, end: date, range_pct: float, status: str, elapsed: float) -> BuildResult:
    return BuildResult(
        symbol=args.symbol,
        range_pct=range_pct,
        range_code=range_code(range_pct),
        utc_start=start.isoformat(),
        utc_end=end.isoformat(),
        status=status,
        bars_written=0,
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
            "[RANGE-BAR-PREBUILD-SKIP-ALL] symbol=%s ranges=%s days=%s->%s already cached",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            start,
            end,
        )
        return [_empty_result(args, loader, start, end, rp, "skipped", elapsed) for rp, loader in loaders.items()]

    builders = {
        rp: RangeBarBuilder(
            range_pct=rp,
            contract_value=loaders[rp].contract_value,
            large_trade_notional_threshold=loaders[rp].large_trade_notional_threshold,
            price_step=None,
        )
        for rp in range_pcts
    }
    written = {rp: 0 for rp in range_pcts}
    chunks_read = 0

    logger.info(
        "[RANGE-BAR-PREBUILD-START] symbol=%s ranges=%s days=%s->%s db=%s chunksize=%s force=%s mode=one_pass_multi_range",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        start,
        end,
        next(iter(loaders.values())).db_path,
        args.chunksize,
        bool(args.force_rebuild),
    )

    for day in days:
        raw_file = next(iter(loaders.values()))._ensure_raw_trade_file(day)
        logger.info(
            "[RANGE-BAR-MULTI-DAY-START] symbol=%s ranges=%s utc_day=%s raw=%s",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            day,
            raw_file,
        )
        day_rows = {rp: 0 for rp in range_pcts}
        # Buffer all closed bars for this UTC day and write once per range/day.
        # This avoids opening SQLite connections and committing for every chunk.
        pending_bars: dict[float, list[dict]] = {rp: [] for rp in range_pcts}
        for raw in iter_trade_csv_chunks(raw_file, chunksize=int(args.chunksize)):
            chunks_read += 1
            chunk = normalize_trade_chunk_fast(raw)
            for rp in range_pcts:
                bars, _ = builders[rp].process_chunk(chunk)
                if not bars:
                    continue
                bars_to_write = [b for b in bars if _bar_end_day(b) in missing_days[rp]]
                if not bars_to_write:
                    continue
                pending_bars[rp].extend(bars_to_write)
                written[rp] += len(bars_to_write)
                day_rows[rp] += len(bars_to_write)
        for rp in range_pcts:
            if pending_bars[rp]:
                loader = loaders[rp]
                df = loader._bars_to_frame(pending_bars[rp])
                loader._upsert_bars(df)
            if day in missing_days[rp]:
                loaders[rp]._mark_coverage(day, rows=day_rows[rp])
        logger.info(
            "[RANGE-BAR-MULTI-DAY-DONE] symbol=%s utc_day=%s rows_by_range=%s chunks_read_total=%s",
            args.symbol,
            day,
            ",".join(f"{range_code(rp)}:{day_rows[rp]}" for rp in range_pcts),
            chunks_read,
        )
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    elapsed = time.time() - started
    results: list[BuildResult] = []
    for rp in range_pcts:
        status = "built" if written[rp] > 0 or missing_days[rp] else "skipped"
        loader = loaders[rp]
        results.append(
            BuildResult(
                symbol=args.symbol,
                range_pct=rp,
                range_code=range_code(rp),
                utc_start=start.isoformat(),
                utc_end=end.isoformat(),
                status=status,
                bars_written=int(written[rp]),
                chunks_read=chunks_read,
                elapsed_seconds=elapsed,
                table_name=loader.table_name,
                db_path=str(loader.db_path),
            )
        )
    logger.info(
        "[RANGE-BAR-PREBUILD-DONE] symbol=%s ranges=%s total_bars=%s raw_chunks_read=%s elapsed=%.2fs",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        sum(written.values()),
        chunks_read,
        elapsed,
    )
    return results


def print_summary(results: list[BuildResult]) -> None:
    print("=" * 110)
    print("OKX Range Bar prebuild summary")
    print(f"jobs            : {len(results)}")
    print(f"built           : {sum(1 for r in results if r.status == 'built')}")
    print(f"skipped         : {sum(1 for r in results if r.status == 'skipped')}")
    print(f"failed          : {sum(1 for r in results if r.status == 'failed')}")
    print(f"bars_written    : {sum(r.bars_written for r in results)}")
    print(f"raw_chunks_read : {max((r.chunks_read for r in results), default=0)}")
    print("=" * 110)
    for r in results:
        print(json.dumps(asdict(r), ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = parse_day(args.start_date, "--start-date")
    end = parse_day(args.end_date, "--end-date")
    try:
        results = prebuild_multi_ranges(args, start, end)
    except Exception as exc:
        logger.exception("[RANGE-BAR-PREBUILD-FAILED] symbol=%s ranges=%s error=%s", args.symbol, args.range_pcts, exc)
        if not args.continue_on_error:
            print_summary([
                BuildResult(
                    symbol=args.symbol,
                    range_pct=0.0,
                    range_code="multi",
                    utc_start=start.isoformat(),
                    utc_end=end.isoformat(),
                    status="failed",
                    bars_written=0,
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
