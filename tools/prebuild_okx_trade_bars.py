#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild OKX trade-aggregated bars into local SQLite cache.

Purpose:
    Run this during idle time to aggregate local/downloaded OKX official trades
    ZIP files into ``data/okx_trade_bars.db``. Real backtests can then read the
    aggregated 1m/5m/order-flow bars directly from DB.

Key design:
    - DB coverage first: already completed days are skipped by default.
    - Raw local first: read ``data/okx/raw/trades/<symbol>/*.zip`` first.
    - Missing raw day: let OKXTradeBarLoader/OKXTickLoader download it.
    - Streaming persistence: aggregate raw trades chunk by chunk and write
      completed bars to DB immediately. The last bar of each chunk is carried
      over to the next chunk, so chunk-boundary bars are not cut incorrectly.
    - Interrupt-safe: coverage is marked only after a full UTC day is finished.
      If interrupted mid-day, rerun with the same command; already-written bars
      will be upserted/overwritten safely.

Examples:
    python tools/prebuild_okx_trade_bars.py \
      --symbol ETH-USDT-SWAP \
      --start-date 2022-01-01 \
      --end-date 2022-01-31 \
      --timeframes 1s 5s 1m 5m

    python tools/prebuild_okx_trade_bars.py \
      --symbol ETH-USDT-SWAP \
      --start-date 2022-01-01 \
      --end-date 2022-12-31 \
      --timeframes 1s \
      --chunksize 500000 \
      --large-trade-notional-threshold 100000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_feed.okx_trade_bar_loader import (  # noqa: E402
    DEFAULT_OKX_TRADES_URL_TEMPLATE,
    OKXTradeBarLoader,
)

try:
    from src.utils.log import get_logger

    logger = get_logger("OKXTradeBarPrebuilder")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("OKXTradeBarPrebuilder")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


SUPPORTED_TIMEFRAMES_HELP = "1s 5s 10s 15s 30s 1m 5m 15m 30m 1H 4H 1D"


@dataclass
class DayBuildResult:
    symbol: str
    timeframe: str
    utc_day: str
    status: str
    rows_written: int
    chunks_read: int
    elapsed_seconds: float
    error: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-aggregate OKX official trades ZIP files into local SQLite trade-bar cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP", help="OKX instrument id.")
    p.add_argument("--start-date", required=True, help="UTC raw trade start day, YYYY-MM-DD.")
    p.add_argument("--end-date", required=True, help="UTC raw trade end day, YYYY-MM-DD, inclusive.")
    p.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m"],
        help=(
            "One or more target bar timeframes. Supports positive integer second/minute/hour/day "
            f"bars, e.g. {SUPPORTED_TIMEFRAMES_HELP}."
        ),
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Project data directory. Default: <project>/data.")
    p.add_argument("--db-name", default="okx_trade_bars.db", help="SQLite DB filename under data-dir.")
    p.add_argument("--chunksize", type=int, default=300_000, help="Rows per raw trades CSV chunk.")
    p.add_argument("--contract-value", type=float, default=None, help="Notional multiplier. Default is inferred by loader.")
    p.add_argument("--large-trade-notional-threshold", type=float, default=100_000.0, help="Large trade threshold in quote notional.")
    p.add_argument("--url-template", default=DEFAULT_OKX_TRADES_URL_TEMPLATE, help="Official OKX trades ZIP URL template.")
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild even if coverage table says the UTC day is already cached.")
    p.add_argument("--dry-run", action="store_true", help="Print planned work without reading/writing files.")
    p.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between UTC days.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue remaining days/timeframes if one day fails.")
    p.add_argument(
        "--utc-timestamps",
        action="store_true",
        help="Store UTC-naive timestamps instead of matching OKXDataLoader TIMEZONE-shifted timestamps.",
    )
    p.add_argument(
        "--log-every-chunks",
        type=int,
        default=10,
        help="Progress log interval by raw CSV chunks. 0 disables chunk progress logs.",
    )
    return p.parse_args(argv)


def parse_day(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got: {value!r}") from exc


def date_range(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError(f"--end-date must be >= --start-date, got {start} -> {end}")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def make_loader(args: argparse.Namespace, timeframe: str) -> OKXTradeBarLoader:
    kwargs = {
        "symbol": args.symbol,
        "timeframe": timeframe,
        "data_dir": args.data_dir,
        "db_name": args.db_name,
        "trades_url_template": args.url_template,
        "contract_value": args.contract_value,
        "large_trade_notional_threshold": args.large_trade_notional_threshold,
        "align_with_okx_loader_timezone": not bool(args.utc_timestamps),
    }
    if args.contract_value is None:
        kwargs.pop("contract_value")
    return OKXTradeBarLoader(**kwargs)


def merge_partial_bars(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge unfinalized UTC partial bars while preserving chunk order."""
    non_empty = [p for p in partials if p is not None and not p.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty).sort_index(kind="stable")
    grouped = combined.groupby(level=0, sort=True)
    return pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "trades_count": grouped["trades_count"].sum(),
            "buy_volume": grouped["buy_volume"].sum(),
            "sell_volume": grouped["sell_volume"].sum(),
            "notional": grouped["notional"].sum(),
            "buy_notional": grouped["buy_notional"].sum(),
            "sell_notional": grouped["sell_notional"].sum(),
            "buy_trades_count": grouped["buy_trades_count"].sum(),
            "sell_trades_count": grouped["sell_trades_count"].sum(),
            "price_size_sum": grouped["price_size_sum"].sum(),
            "large_buy_notional": grouped["large_buy_notional"].sum(),
            "large_sell_notional": grouped["large_sell_notional"].sum(),
            "large_buy_trades_count": grouped["large_buy_trades_count"].sum(),
            "large_sell_trades_count": grouped["large_sell_trades_count"].sum(),
            "large_trades_count": grouped["large_trades_count"].sum(),
            "max_trade_notional": grouped["max_trade_notional"].max(),
            "max_trade_size": grouped["max_trade_size"].max(),
        }
    )


def finalize_and_write(loader: OKXTradeBarLoader, partial: pd.DataFrame) -> int:
    """Convert UTC partial bars to final schema and upsert them into SQLite."""
    if partial is None or partial.empty:
        return 0
    bars = loader._combine_partial_bars([partial])  # intentional: public tool for this project, uses loader internals
    if bars.empty:
        return 0
    loader._upsert_bars(bars)
    return len(bars)


def prebuild_one_day(
    loader: OKXTradeBarLoader,
    day: date,
    *,
    chunksize: int,
    force_rebuild: bool,
    dry_run: bool,
    log_every_chunks: int,
) -> DayBuildResult:
    start_time = time.time()
    if not force_rebuild and loader._has_coverage(day):
        return DayBuildResult(
            symbol=loader.symbol,
            timeframe=loader.timeframe,
            utc_day=day.isoformat(),
            status="skipped_cached",
            rows_written=0,
            chunks_read=0,
            elapsed_seconds=time.time() - start_time,
        )

    if dry_run:
        raw_file = loader.tick_loader.find_local_trade_file(day, template=loader.trades_url_template)
        status = "would_build_local" if raw_file else "would_download_then_build"
        return DayBuildResult(
            symbol=loader.symbol,
            timeframe=loader.timeframe,
            utc_day=day.isoformat(),
            status=status,
            rows_written=0,
            chunks_read=0,
            elapsed_seconds=time.time() - start_time,
        )

    raw_file = loader._ensure_raw_trade_file(day)
    logger.info(
        "[PREBUILD-DAY-START] symbol=%s timeframe=%s utc_day=%s raw=%s table=%s",
        loader.symbol,
        loader.timeframe,
        day,
        raw_file,
        loader.table_name,
    )

    carry: pd.DataFrame | None = None
    rows_written = 0
    chunks_read = 0

    for raw in loader._iter_trade_csv_chunks(raw_file, chunksize=chunksize):
        chunks_read += 1
        chunk = loader._normalize_trade_chunk_fast(raw)
        partial = loader._aggregate_trade_chunk_partial(chunk) if not chunk.empty else pd.DataFrame()

        merged = merge_partial_bars([carry, partial] if carry is not None else [partial])
        if merged.empty:
            carry = None
            continue

        # The latest bar may still receive trades from the next CSV chunk. Keep
        # it in memory and flush all older bars immediately.
        if len(merged) > 1:
            flushable = merged.iloc[:-1]
            carry = merged.iloc[[-1]]
            rows_written += finalize_and_write(loader, flushable)
        else:
            carry = merged

        if log_every_chunks > 0 and chunks_read % log_every_chunks == 0:
            logger.info(
                "[PREBUILD-DAY-PROGRESS] symbol=%s timeframe=%s utc_day=%s chunks=%s rows_written=%s carry_bar=%s",
                loader.symbol,
                loader.timeframe,
                day,
                chunks_read,
                rows_written,
                None if carry is None or carry.empty else str(carry.index[-1]),
            )

    # Flush the final carry bar for this UTC day.
    if carry is not None and not carry.empty:
        rows_written += finalize_and_write(loader, carry)

    loader._mark_coverage(day, rows=rows_written)
    elapsed = time.time() - start_time
    logger.info(
        "[PREBUILD-DAY-DONE] symbol=%s timeframe=%s utc_day=%s chunks=%s rows_written=%s elapsed=%.2fs",
        loader.symbol,
        loader.timeframe,
        day,
        chunks_read,
        rows_written,
        elapsed,
    )
    return DayBuildResult(
        symbol=loader.symbol,
        timeframe=loader.timeframe,
        utc_day=day.isoformat(),
        status="built",
        rows_written=rows_written,
        chunks_read=chunks_read,
        elapsed_seconds=elapsed,
    )


def prebuild_one_day_multi(
    loaders: dict[str, OKXTradeBarLoader],
    day: date,
    *,
    chunksize: int,
    force_rebuild: bool,
    dry_run: bool,
    log_every_chunks: int,
) -> list[DayBuildResult]:
    """Build every requested timeframe from one raw-trade scan.

    Trade normalization is independent of timeframe.  The old timeframe-first
    loop reopened and parsed the same daily ZIP once per timeframe; this
    day-first implementation normalizes each chunk once and fans the resulting
    trades into all active timeframe aggregators.  SQLite coverage remains the
    per-timeframe completion marker, so interruption and reruns are still safe.
    """

    started = time.time()
    results: list[DayBuildResult] = []
    active: dict[str, OKXTradeBarLoader] = {}
    for timeframe, loader in loaders.items():
        if not force_rebuild and loader._has_coverage(day):
            results.append(
                DayBuildResult(
                    symbol=loader.symbol,
                    timeframe=timeframe,
                    utc_day=day.isoformat(),
                    status="skipped_cached",
                    rows_written=0,
                    chunks_read=0,
                    elapsed_seconds=time.time() - started,
                )
            )
        else:
            active[timeframe] = loader

    if not active:
        return results

    base_loader = next(iter(active.values()))
    if dry_run:
        raw_file = base_loader.tick_loader.find_local_trade_file(day, template=base_loader.trades_url_template)
        status = "would_build_local" if raw_file else "would_download_then_build"
        for timeframe, loader in active.items():
            results.append(
                DayBuildResult(
                    symbol=loader.symbol,
                    timeframe=timeframe,
                    utc_day=day.isoformat(),
                    status=status,
                    rows_written=0,
                    chunks_read=0,
                    elapsed_seconds=time.time() - started,
                )
            )
        return results

    raw_file = base_loader._ensure_raw_trade_file(day)
    logger.info(
        "[PREBUILD-MULTI-DAY-START] symbol=%s timeframes=%s utc_day=%s raw=%s chunksize=%s",
        base_loader.symbol,
        ",".join(active),
        day,
        raw_file,
        chunksize,
    )
    carries: dict[str, pd.DataFrame | None] = {timeframe: None for timeframe in active}
    rows_written: dict[str, int] = {timeframe: 0 for timeframe in active}
    chunks_read = 0

    for raw in base_loader._iter_trade_csv_chunks(raw_file, chunksize=chunksize):
        chunks_read += 1
        # All loaders share symbol/contract-value/large-trade threshold, so the
        # normalized trade chunk is valid for every timeframe.
        chunk = base_loader._normalize_trade_chunk_fast(raw)
        for timeframe, loader in active.items():
            partial = loader._aggregate_trade_chunk_partial(chunk) if not chunk.empty else pd.DataFrame()
            carry = carries[timeframe]
            merged = merge_partial_bars([carry, partial] if carry is not None else [partial])
            if merged.empty:
                carries[timeframe] = None
                continue
            if len(merged) > 1:
                rows_written[timeframe] += finalize_and_write(loader, merged.iloc[:-1])
                carries[timeframe] = merged.iloc[[-1]]
            else:
                carries[timeframe] = merged

        if log_every_chunks > 0 and chunks_read % log_every_chunks == 0:
            logger.info(
                "[PREBUILD-MULTI-DAY-PROGRESS] symbol=%s utc_day=%s chunks=%s rows_by_timeframe=%s",
                base_loader.symbol,
                day,
                chunks_read,
                ",".join(f"{tf}:{rows_written[tf]}" for tf in active),
            )

    for timeframe, loader in active.items():
        carry = carries[timeframe]
        if carry is not None and not carry.empty:
            rows_written[timeframe] += finalize_and_write(loader, carry)

    # Coverage is deliberately published only after every active timeframe has
    # consumed the complete raw day. Partial rows remain harmless upserts when
    # a killed process reruns a day without a coverage marker.
    for timeframe, loader in active.items():
        loader._mark_coverage(day, rows=rows_written[timeframe])

    elapsed = time.time() - started
    for timeframe, loader in active.items():
        results.append(
            DayBuildResult(
                symbol=loader.symbol,
                timeframe=timeframe,
                utc_day=day.isoformat(),
                status="built",
                rows_written=rows_written[timeframe],
                chunks_read=chunks_read,
                elapsed_seconds=elapsed,
            )
        )
    logger.info(
        "[PREBUILD-MULTI-DAY-DONE] symbol=%s timeframes=%s utc_day=%s chunks=%s rows_by_timeframe=%s elapsed=%.2fs",
        base_loader.symbol,
        ",".join(active),
        day,
        chunks_read,
        ",".join(f"{tf}:{rows_written[tf]}" for tf in active),
        elapsed,
    )
    return results


def print_summary(results: list[DayBuildResult]) -> None:
    if not results:
        print("No work executed.")
        return
    total_built = sum(1 for r in results if r.status == "built")
    total_skipped = sum(1 for r in results if r.status.startswith("skipped"))
    total_failed = sum(1 for r in results if r.status == "failed")
    total_rows = sum(r.rows_written for r in results)

    # Multi-timeframe builds share one physical raw-file scan per UTC day.
    # Every per-timeframe result carries that day's shared chunk/elapsed value,
    # so summing results would overstate I/O and runtime by N timeframes.
    day_chunks: dict[str, int] = {}
    day_elapsed: dict[str, float] = {}
    for result in results:
        day_chunks[result.utc_day] = max(day_chunks.get(result.utc_day, 0), result.chunks_read)
        day_elapsed[result.utc_day] = max(day_elapsed.get(result.utc_day, 0.0), result.elapsed_seconds)
    total_chunks = sum(day_chunks.values())
    total_elapsed = sum(day_elapsed.values())

    print("=" * 100)
    print("OKX trade bar prebuild summary")
    print(f"days/timeframes processed : {len(results)}")
    print(f"built                     : {total_built}")
    print(f"skipped                   : {total_skipped}")
    print(f"failed                    : {total_failed}")
    print(f"rows written              : {total_rows}")
    print(f"raw chunks read           : {total_chunks}")
    print(f"elapsed seconds           : {total_elapsed:.2f}")
    print("=" * 100)
    for r in results:
        print(asdict(r))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = parse_day(args.start_date, "--start-date")
    end = parse_day(args.end_date, "--end-date")
    days = list(date_range(start, end))

    timeframes = list(dict.fromkeys(str(item) for item in args.timeframes))
    loaders = {timeframe: make_loader(args, timeframe) for timeframe in timeframes}
    logger.info(
        "[PREBUILD-START] symbol=%s timeframes=%s days=%s -> %s db=%s chunksize=%s force=%s mode=one_pass_multi_timeframe",
        args.symbol,
        ",".join(timeframes),
        start,
        end,
        next(iter(loaders.values())).db_path,
        args.chunksize,
        bool(args.force_rebuild),
    )
    results: list[DayBuildResult] = []
    for day in days:
        try:
            results.extend(
                prebuild_one_day_multi(
                    loaders,
                    day,
                    chunksize=int(args.chunksize),
                    force_rebuild=bool(args.force_rebuild),
                    dry_run=bool(args.dry_run),
                    log_every_chunks=int(args.log_every_chunks),
                )
            )
        except Exception as exc:
            logger.exception(
                "[PREBUILD-MULTI-DAY-FAILED] symbol=%s timeframes=%s utc_day=%s error=%s",
                args.symbol,
                ",".join(timeframes),
                day,
                exc,
            )
            results.extend(
                DayBuildResult(
                    symbol=args.symbol,
                    timeframe=timeframe,
                    utc_day=day.isoformat(),
                    status="failed",
                    rows_written=0,
                    chunks_read=0,
                    elapsed_seconds=0.0,
                    error=repr(exc),
                )
                for timeframe in timeframes
            )
            if not args.continue_on_error:
                print_summary(results)
                return 1
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    print_summary(results)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
