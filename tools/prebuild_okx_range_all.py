#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild OKX Range Bars and Range Footprints in one raw-trades pass.

Purpose:
    Read each OKX trades ZIP/chunk only once, then simultaneously build:
      - data/okx_range_bars.db
      - data/okx_range_footprints.db

This is faster than running prebuild_okx_range_bars.py and
prebuild_okx_range_footprints.py separately because raw ZIP decompression and
pandas CSV parsing are shared.

Example:
    python tools/prebuild_okx_range_all.py \
        --symbol ETH-USDT-SWAP \
        --start-date 2022-01-01 \
        --end-date 2026-06-15 \
        --range-pcts 0.0015 0.002 0.0025 \
        --price-step 1 \
        --chunksize 1000000 \
        --flush-rows 1000000
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
    OKXRangeBarLoader,
    RangeBarBuilder,
    iter_trade_csv_chunks,
    normalize_trade_chunk_fast,
    range_code,
)
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader, price_step_code  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger("OKXRangeAllPrebuilder")


@dataclass
class RangeAllResult:
    symbol: str
    range_pct: float
    range_code: str
    price_step: float
    step_code: str
    utc_start: str
    utc_end: str
    status: str
    bars_written: int
    footprints_written: int
    bars_closed_for_footprint: int
    chunks_read: int
    elapsed_seconds: float
    range_bar_table: str
    footprint_table: str
    range_bar_db_path: str
    footprint_db_path: str
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prebuild OKX Range Bars and Range Footprints from official trades ZIP files in one pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", required=True, help="UTC start day, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="UTC end day, YYYY-MM-DD, inclusive")
    p.add_argument("--range-pcts", nargs="+", type=float, default=list(DEFAULT_RANGE_PCTS), help="Range thresholds, e.g. 0.0015 0.002 0.0025")
    p.add_argument("--price-step", type=float, default=1.0, help="Footprint price bucket size, e.g. 1 or 0.5")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--range-bars-db-name", default="okx_range_bars.db")
    p.add_argument("--footprints-db-name", default="okx_range_footprints.db")
    p.add_argument("--url-template", default=DEFAULT_OKX_TRADES_URL_TEMPLATE)
    p.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per raw trades CSV chunk. Use smaller values on low-memory servers.")
    p.add_argument("--flush-rows", type=int, default=1_000_000, help="Flush buffered footprint rows to SQLite after this many rows per range.")
    p.add_argument("--contract-value", type=float, default=None)
    p.add_argument("--large-trade-notional-threshold", type=float, default=DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD)
    p.add_argument("--utc-timestamps", action="store_true", help="Store UTC-naive timestamps instead of matching OKXDataLoader timezone.")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-bars", action="store_true", help="Only build footprints; do not write range-bar DB.")
    p.add_argument("--skip-footprints", action="store_true", help="Only build range bars; do not write footprint DB.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep after each UTC day.")
    p.add_argument("--warmup-days", type=int, default=1, help="When resuming partial cache, start this many UTC days before the first missing day to rebuild cross-day active range-bar state.")
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


def make_bar_loader(args: argparse.Namespace, range_pct: float) -> OKXRangeBarLoader:
    kwargs = {
        "symbol": args.symbol,
        "range_pct": range_pct,
        "data_dir": args.data_dir,
        "db_name": args.range_bars_db_name,
        "trades_url_template": args.url_template,
        "contract_value": args.contract_value,
        "large_trade_notional_threshold": args.large_trade_notional_threshold,
        "align_with_okx_loader_timezone": not bool(args.utc_timestamps),
    }
    if args.contract_value is None:
        kwargs.pop("contract_value")
    return OKXRangeBarLoader(**kwargs)


def make_footprint_loader(args: argparse.Namespace, range_pct: float) -> OKXRangeFootprintLoader:
    kwargs = {
        "symbol": args.symbol,
        "range_pct": range_pct,
        "price_step": args.price_step,
        "data_dir": args.data_dir,
        "db_name": args.footprints_db_name,
        "trades_url_template": args.url_template,
        "contract_value": args.contract_value,
        "large_trade_notional_threshold": args.large_trade_notional_threshold,
        "align_with_okx_loader_timezone": not bool(args.utc_timestamps),
    }
    if args.contract_value is None:
        kwargs.pop("contract_value")
    return OKXRangeFootprintLoader(**kwargs)


def _end_day(row: dict) -> date:
    return pd.Timestamp(row["end_ts"]).date()


def _empty_result(
    args: argparse.Namespace,
    range_pct: float,
    start: date,
    end: date,
    status: str,
    elapsed: float,
    bar_loader: OKXRangeBarLoader | None,
    fp_loader: OKXRangeFootprintLoader | None,
) -> RangeAllResult:
    return RangeAllResult(
        symbol=args.symbol,
        range_pct=range_pct,
        range_code=range_code(range_pct),
        price_step=float(args.price_step),
        step_code=price_step_code(float(args.price_step)),
        utc_start=start.isoformat(),
        utc_end=end.isoformat(),
        status=status,
        bars_written=0,
        footprints_written=0,
        bars_closed_for_footprint=0,
        chunks_read=0,
        elapsed_seconds=elapsed,
        range_bar_table="" if bar_loader is None else bar_loader.table_name,
        footprint_table="" if fp_loader is None else fp_loader.table_name,
        range_bar_db_path="" if bar_loader is None else str(bar_loader.db_path),
        footprint_db_path="" if fp_loader is None else str(fp_loader.db_path),
    )


def prebuild_range_all(args: argparse.Namespace, start: date, end: date) -> list[RangeAllResult]:
    if args.skip_bars and args.skip_footprints:
        raise ValueError("--skip-bars and --skip-footprints cannot both be set")
    if float(args.price_step) <= 0:
        raise ValueError("--price-step must be > 0")

    started = time.time()
    days = list(date_range(start, end))
    range_pcts = [float(x) for x in args.range_pcts]

    bar_loaders = {} if args.skip_bars else {rp: make_bar_loader(args, rp) for rp in range_pcts}
    fp_loaders = {} if args.skip_footprints else {rp: make_footprint_loader(args, rp) for rp in range_pcts}
    raw_loader = next(iter(bar_loaders.values()), None) or next(iter(fp_loaders.values()))

    if args.dry_run:
        elapsed = time.time() - started
        return [
            _empty_result(args, rp, start, end, "dry_run", elapsed, bar_loaders.get(rp), fp_loaders.get(rp))
            for rp in range_pcts
        ]

    missing_bar_days: dict[float, set[date]] = {rp: set() for rp in range_pcts}
    missing_fp_days: dict[float, set[date]] = {rp: set() for rp in range_pcts}

    for rp in range_pcts:
        if rp in bar_loaders:
            if args.force_rebuild:
                bar_loaders[rp]._delete_cached_days(days)
                missing_bar_days[rp] = set(days)
            else:
                missing_bar_days[rp] = {d for d in days if not bar_loaders[rp]._has_coverage(d)}
        if rp in fp_loaders:
            if args.force_rebuild:
                fp_loaders[rp]._delete_cached_days(days)
                missing_fp_days[rp] = set(days)
            else:
                missing_fp_days[rp] = {d for d in days if not fp_loaders[rp]._has_coverage(d)}

    needed_days = [
        d
        for d in days
        if any(d in missing_bar_days[rp] or d in missing_fp_days[rp] for rp in range_pcts)
    ]
    if not needed_days:
        elapsed = time.time() - started
        logger.info(
            "[RANGE-ALL-PREBUILD-SKIP-ALL] symbol=%s ranges=%s step=%s days=%s->%s already cached",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            args.price_step,
            start,
            end,
        )
        return [
            _empty_result(args, rp, start, end, "skipped", elapsed, bar_loaders.get(rp), fp_loaders.get(rp))
            for rp in range_pcts
        ]

    warmup_days = max(0, int(args.warmup_days))
    effective_start = max(start, needed_days[0] - timedelta(days=warmup_days))
    effective_end = needed_days[-1]
    process_days = [d for d in days if effective_start <= d <= effective_end]

    builders = {
        rp: RangeBarBuilder(
            range_pct=rp,
            contract_value=(bar_loaders.get(rp) or fp_loaders[rp]).contract_value,
            large_trade_notional_threshold=(bar_loaders.get(rp) or fp_loaders[rp]).large_trade_notional_threshold,
            price_step=None if args.skip_footprints else float(args.price_step),
        )
        for rp in range_pcts
    }

    bars_written = {rp: 0 for rp in range_pcts}
    footprints_written = {rp: 0 for rp in range_pcts}
    bars_closed_for_fp = {rp: 0 for rp in range_pcts}
    chunks_read = 0
    flush_rows = max(1, int(args.flush_rows))

    def flush_footprints(rp: float, pending: list[dict]) -> int:
        if not pending:
            return 0
        loader = fp_loaders[rp]
        df = loader._footprints_to_frame(pending)
        loader._upsert_footprints(df)
        flushed = len(df)
        pending.clear()
        return flushed

    logger.info(
        "[RANGE-ALL-PREBUILD-START] symbol=%s ranges=%s step=%s days=%s->%s effective_days=%s->%s warmup_days=%s bars_db=%s footprints_db=%s chunksize=%s flush_rows=%s force=%s mode=one_pass_bars_and_footprints",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        args.price_step,
        start,
        end,
        effective_start,
        effective_end,
        warmup_days,
        "SKIP" if args.skip_bars else next(iter(bar_loaders.values())).db_path,
        "SKIP" if args.skip_footprints else next(iter(fp_loaders.values())).db_path,
        args.chunksize,
        flush_rows,
        bool(args.force_rebuild),
    )

    for day in process_days:
        raw_file = raw_loader._ensure_raw_trade_file(day)
        logger.info(
            "[RANGE-ALL-DAY-START] symbol=%s ranges=%s step=%s utc_day=%s raw=%s",
            args.symbol,
            ",".join(range_code(x) for x in range_pcts),
            args.price_step,
            day,
            raw_file,
        )
        day_bar_rows = {rp: 0 for rp in range_pcts}
        day_fp_rows = {rp: 0 for rp in range_pcts}
        day_fp_bars = {rp: 0 for rp in range_pcts}
        pending_bars: dict[float, list[dict]] = {rp: [] for rp in range_pcts}
        pending_fps: dict[float, list[dict]] = {rp: [] for rp in range_pcts}

        for raw in iter_trade_csv_chunks(raw_file, chunksize=int(args.chunksize)):
            chunks_read += 1
            chunk = normalize_trade_chunk_fast(raw)
            for rp in range_pcts:
                bars, footprints = builders[rp].process_chunk(chunk)
                if bars:
                    if rp in bar_loaders:
                        bars_to_write = [b for b in bars if _end_day(b) in missing_bar_days[rp]]
                        if bars_to_write:
                            pending_bars[rp].extend(bars_to_write)
                            bars_written[rp] += len(bars_to_write)
                            day_bar_rows[rp] += len(bars_to_write)
                    if rp in fp_loaders:
                        fp_bar_count = sum(1 for b in bars if _end_day(b) in missing_fp_days[rp])
                        bars_closed_for_fp[rp] += fp_bar_count
                        day_fp_bars[rp] += sum(1 for b in bars if _end_day(b) == day and day in missing_fp_days[rp])
                if footprints and rp in fp_loaders:
                    fps_to_write = [fp for fp in footprints if _end_day(fp) in missing_fp_days[rp]]
                    if fps_to_write:
                        pending_fps[rp].extend(fps_to_write)
                        footprints_written[rp] += len(fps_to_write)
                        day_fp_rows[rp] += len(fps_to_write)
                        if len(pending_fps[rp]) >= flush_rows:
                            flush_footprints(rp, pending_fps[rp])

        for rp in range_pcts:
            if rp in bar_loaders and pending_bars[rp]:
                loader = bar_loaders[rp]
                df = loader._bars_to_frame(pending_bars[rp])
                loader._upsert_bars(df)
            if rp in fp_loaders:
                flush_footprints(rp, pending_fps[rp])
            if rp in bar_loaders and day in missing_bar_days[rp]:
                bar_loaders[rp]._mark_coverage(day, rows=day_bar_rows[rp])
            if rp in fp_loaders and day in missing_fp_days[rp]:
                fp_loaders[rp]._mark_coverage(day, rows=day_fp_rows[rp], bars=day_fp_bars[rp])

        logger.info(
            "[RANGE-ALL-DAY-DONE] symbol=%s step=%s utc_day=%s bars_by_range=%s footprints_by_range=%s fp_bars_by_range=%s chunks_read_total=%s",
            args.symbol,
            args.price_step,
            day,
            ",".join(f"{range_code(rp)}:{day_bar_rows[rp]}" for rp in range_pcts),
            ",".join(f"{range_code(rp)}:{day_fp_rows[rp]}" for rp in range_pcts),
            ",".join(f"{range_code(rp)}:{day_fp_bars[rp]}" for rp in range_pcts),
            chunks_read,
        )
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    elapsed = time.time() - started
    results: list[RangeAllResult] = []
    for rp in range_pcts:
        built_any = bool(missing_bar_days[rp] or missing_fp_days[rp])
        results.append(
            RangeAllResult(
                symbol=args.symbol,
                range_pct=rp,
                range_code=range_code(rp),
                price_step=float(args.price_step),
                step_code=price_step_code(float(args.price_step)),
                utc_start=start.isoformat(),
                utc_end=end.isoformat(),
                status="built" if built_any else "skipped",
                bars_written=int(bars_written[rp]),
                footprints_written=int(footprints_written[rp]),
                bars_closed_for_footprint=int(bars_closed_for_fp[rp]),
                chunks_read=chunks_read,
                elapsed_seconds=elapsed,
                range_bar_table="" if rp not in bar_loaders else bar_loaders[rp].table_name,
                footprint_table="" if rp not in fp_loaders else fp_loaders[rp].table_name,
                range_bar_db_path="" if rp not in bar_loaders else str(bar_loaders[rp].db_path),
                footprint_db_path="" if rp not in fp_loaders else str(fp_loaders[rp].db_path),
            )
        )

    logger.info(
        "[RANGE-ALL-PREBUILD-DONE] symbol=%s ranges=%s step=%s total_bars=%s total_footprints=%s raw_chunks_read=%s elapsed=%.2fs",
        args.symbol,
        ",".join(range_code(x) for x in range_pcts),
        args.price_step,
        sum(bars_written.values()),
        sum(footprints_written.values()),
        chunks_read,
        elapsed,
    )
    return results


def print_summary(results: list[RangeAllResult]) -> None:
    print("=" * 130)
    print("OKX Range ALL prebuild summary")
    print(f"jobs                  : {len(results)}")
    print(f"built                 : {sum(1 for r in results if r.status == 'built')}")
    print(f"skipped               : {sum(1 for r in results if r.status == 'skipped')}")
    print(f"failed                : {sum(1 for r in results if r.status == 'failed')}")
    print(f"bars_written          : {sum(r.bars_written for r in results)}")
    print(f"footprints_written    : {sum(r.footprints_written for r in results)}")
    print(f"fp_bars_closed        : {sum(r.bars_closed_for_footprint for r in results)}")
    print(f"raw_chunks_read       : {max((r.chunks_read for r in results), default=0)}")
    print("=" * 130)
    for r in results:
        print(json.dumps(asdict(r), ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = parse_day(args.start_date, "--start-date")
    end = parse_day(args.end_date, "--end-date")
    try:
        results = prebuild_range_all(args, start, end)
    except Exception as exc:
        logger.exception("[RANGE-ALL-PREBUILD-FAILED] symbol=%s ranges=%s step=%s error=%s", args.symbol, args.range_pcts, args.price_step, exc)
        if not args.continue_on_error:
            print_summary([
                RangeAllResult(
                    symbol=args.symbol,
                    range_pct=0.0,
                    range_code="multi",
                    price_step=float(args.price_step),
                    step_code=price_step_code(float(args.price_step)),
                    utc_start=start.isoformat(),
                    utc_end=end.isoformat(),
                    status="failed",
                    bars_written=0,
                    footprints_written=0,
                    bars_closed_for_footprint=0,
                    chunks_read=0,
                    elapsed_seconds=0.0,
                    range_bar_table="",
                    footprint_table="",
                    range_bar_db_path=str(Path(args.data_dir or PROJECT_ROOT / "data") / args.range_bars_db_name),
                    footprint_db_path=str(Path(args.data_dir or PROJECT_ROOT / "data") / args.footprints_db_name),
                    error=repr(exc),
                )
            ])
            return 1
        results = []
    print_summary(results)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
