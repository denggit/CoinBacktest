#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build offline Books + Raw Trades liquidity-map artifacts.

The heavy raw L2 and tick files remain the source of truth. This tool creates
small, day-partitioned NPZ artifacts for analyze_tool and fast event studies.
It never loads more than one UTC day of raw input per worker.

The default execution mode uses conservative day-level parallelism. Each worker
owns all of its loaders/builders and publishes one day through atomic ``.part``
files, so an interrupted or failed worker cannot leave a completed checkpoint.

Example:
    python tools/prebuild_okx_offline_liquidity_map.py --symbol ETH-USDT-SWAP --start-date 2025-11-01 --end-date 2025-12-31 --books-depth 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_books_loader import OKXBooksLoader  # noqa: E402
from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader  # noqa: E402
from src.liquidity_map import LiquidityBuildStats, LiquidityFeatureStore, LiquidityMapConfig  # noqa: E402
from src.liquidity_map.builder import OfflineLiquidityMapBuilder  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prebuild offline OKX Books + Raw Trades liquidity heatmap and causal backtest features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", required=True, help="UTC raw-data day, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="UTC raw-data day, YYYY-MM-DD, inclusive")
    p.add_argument("--books-depth", type=int, default=400)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--map-profile", choices=["auto", "near", "broad"], default="auto", help="Near=400-level compact map; broad=5000-level CoinGlass-style map")
    p.add_argument("--price-step", type=float, default=1.0, help="Heatmap price bucket in quote currency")
    p.add_argument("--feature-seconds", type=int, default=1, help="Strategy-facing feature clock")
    p.add_argument("--heatmap-seconds", type=int, default=None, help="Canonical visualization bucket. Auto: 60s for near/400, 5s for broad/5000")
    p.add_argument("--contract-value-base", type=float, default=0.1, help="Base asset per contract; ETH-USDT-SWAP normally 0.1 ETH")
    p.add_argument("--max-distance-pct", type=float, default=None, help="Keep book levels within this fraction of mid price. Auto: 8%% near, 10%% broad")
    p.add_argument("--max-levels-per-side", type=int, default=None, help="Maximum retained price bins per side/time bucket. Auto: 60 near, unlimited broad")
    p.add_argument("--min-store-depth-base", type=float, default=None, help="Minimum stored ETH depth. Auto: 0.05 near, 0 broad")
    p.add_argument("--min-store-ratio", type=float, default=None, help="Drop levels below this fraction of current side maximum. Auto: 1%% near, 0 broad")
    p.add_argument("--large-depth-ratio", type=float, default=0.50, help="Causal current-book large-liquidity threshold")
    p.add_argument("--decision-delay-ms", type=int, default=1000, help="Feature availability delay used by backtests")
    p.add_argument("--max-book-staleness-seconds", type=int, default=30, help="Invalidate a reconstructed book after this many seconds without an event")
    p.add_argument("--trade-chunksize", type=int, default=500_000, help="Raw trade rows per streamed chunk; use 250000-300000 for very low-memory hosts")
    p.add_argument("--progress-every-events", type=int, default=250_000)
    p.add_argument("--allow-books-only", action="store_true", help="Build visualization without raw trades; flow attribution stays zero")
    p.add_argument("--no-strict-sequence", action="store_true", help="Do not invalidate on seqId/prevSeqId gaps")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--inspect-only", action="store_true", help="Print selected archive/schema and a bounded normalized-event probe, then exit")
    p.add_argument("--inspect-events", type=int, default=200, help="Normalized events sampled by --inspect-only")
    p.add_argument("--schema-probe", action="store_true", help="Read and print source samples before each build. Useful for diagnostics but disabled on the fast path")
    p.add_argument("--workers", type=int, default=0, help="Day workers. 0=CPU/memory-aware conservative auto; broad 5000-level is capped at 2")
    p.add_argument("--day-retries", type=int, default=1, help="Retry a failed day this many times in a fresh worker call")
    p.add_argument("--compression-level", type=int, default=1, choices=range(0, 10), metavar="0..9", help="NPZ DEFLATE level. 1 is fast; 0 stores without compression")
    return p.parse_args(argv)


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError("--end-date must be >= --start-date")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _resolve_profile_and_config(args: argparse.Namespace) -> tuple[str, LiquidityMapConfig]:
    profile = args.map_profile
    if profile == "auto":
        profile = "broad" if int(args.books_depth) >= 5000 else "near"
    heatmap_seconds = int(args.heatmap_seconds if args.heatmap_seconds is not None else (5 if profile == "broad" else 60))
    max_distance_pct = float(args.max_distance_pct if args.max_distance_pct is not None else (0.10 if profile == "broad" else 0.08))
    max_levels_per_side = int(args.max_levels_per_side if args.max_levels_per_side is not None else (0 if profile == "broad" else 60))
    min_store_depth_base = float(args.min_store_depth_base if args.min_store_depth_base is not None else (0.0 if profile == "broad" else 0.05))
    min_store_ratio = float(args.min_store_ratio if args.min_store_ratio is not None else (0.0 if profile == "broad" else 0.01))
    cfg = LiquidityMapConfig(
        symbol=args.symbol,
        books_depth=args.books_depth,
        price_step=args.price_step,
        feature_seconds=args.feature_seconds,
        heatmap_seconds=heatmap_seconds,
        contract_value_base=args.contract_value_base,
        max_distance_pct=max_distance_pct,
        max_levels_per_side=max_levels_per_side,
        min_store_depth_base=min_store_depth_base,
        min_store_ratio=min_store_ratio,
        large_depth_ratio=args.large_depth_ratio,
        decision_delay_ms=args.decision_delay_ms,
        max_book_staleness_seconds=args.max_book_staleness_seconds,
        strict_sequence=not args.no_strict_sequence,
    )
    cfg.validate()
    return profile, cfg


def _available_memory_bytes() -> int | None:
    """Best-effort available-memory probe without a mandatory dependency."""

    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except Exception:
        return None


def _auto_workers(books_depth: int, requested: int) -> int:
    if requested < 0:
        raise ValueError("--workers must be >= 0")
    if requested > 0:
        return requested
    cpus = max(1, os.cpu_count() or 1)
    available = _available_memory_bytes()
    gib = 1024 ** 3
    if int(books_depth) >= 5000:
        # A broad day can transiently own decompressed trade chunks, replay
        # state and several million output cells. Budget 6 GiB per worker and
        # never auto-select more than two, even on very large machines.
        by_memory = 1 if available is None else max(1, int(available // (6 * gib)))
        return min(2, cpus, by_memory)
    by_memory = 1 if available is None else max(1, int(available // (2 * gib)))
    return min(4, cpus, by_memory)


def _require_compatible_cached_config(
    metadata: dict[str, Any],
    config: LiquidityMapConfig,
    *,
    day: date,
) -> None:
    cached = metadata.get("config")
    requested = config.to_dict()
    if cached != requested:
        raise RuntimeError(
            f"cached offline liquidity-map config differs for {day}. "
            "Use --force-rebuild after reviewing the parameter change. "
            f"cached={cached} requested={requested}"
        )


def _day_task(
    *,
    day: date,
    cfg: LiquidityMapConfig,
    data_dir: str | None,
    trade_chunksize: int,
    progress_every_events: int,
    allow_books_only: bool,
    force_rebuild: bool,
    schema_probe: bool,
    compression_level: int,
) -> dict[str, Any]:
    return {
        "day": day.isoformat(),
        "config": cfg.to_dict(),
        "data_dir": data_dir,
        "trade_chunksize": int(trade_chunksize),
        "progress_every_events": int(progress_every_events),
        "allow_books_only": bool(allow_books_only),
        "force_rebuild": bool(force_rebuild),
        "schema_probe": bool(schema_probe),
        "compression_level": int(compression_level),
    }


def _build_one_day(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint; all heavy state is process-local and day-local."""

    day = parse_day(str(task["day"]))
    cfg = LiquidityMapConfig(**dict(task["config"]))
    data_dir = task.get("data_dir")
    books_loader = OKXBooksLoader(symbol=cfg.symbol, depth=cfg.books_depth, data_dir=data_dir)
    tick_loader = OKXTickLoader(
        symbol=cfg.symbol,
        data_dir=data_dir,
        trades_url_template=DEFAULT_OKX_TRADES_URL_TEMPLATE,
    )
    store = LiquidityFeatureStore(symbol=cfg.symbol, books_depth=cfg.books_depth, data_dir=data_dir)
    builder = OfflineLiquidityMapBuilder(cfg)
    prefix = f"[{day}]"

    if store.has_day(day) and not bool(task["force_rebuild"]):
        meta = store.load_metadata(day)
        _require_compatible_cached_config(meta, cfg, day=day)
        return {
            "status": "skipped",
            "day": day.isoformat(),
            "metadata": meta,
            "features": int(meta.get("stats", {}).get("book_feature_rows", 0)),
            "heatmap_cells": int(meta.get("stats", {}).get("heatmap_cells", 0)),
            "elapsed_seconds": 0.0,
        }

    started = time.perf_counter()
    stats = LiquidityBuildStats(day=day.isoformat())
    chunks_iter = None
    trade_by_price = None
    trade_by_time = None
    feature_rows = None
    heatmap_rows = None
    source_files = None
    try:
        book_files = books_loader.find_local_book_files(day)
        if not book_files:
            raise FileNotFoundError(
                f"missing local Books data for {day}; expected under {books_loader.raw_dir}"
            )
        print(f"{prefix} [stage] books ready | files={len(book_files)}", flush=True)
        if bool(task["schema_probe"]):
            schema = books_loader.inspect_day_schema(day, max_lines=2)
            first_sample = schema.get("files", [{}])[0].get("sample", []) if schema.get("files") else []
            if first_sample:
                print(f"{prefix} [schema] {first_sample[0][:300]}", flush=True)

        raw_trade_file = tick_loader.find_local_trade_file(
            day,
            template=DEFAULT_OKX_TRADES_URL_TEMPLATE,
        )
        trades_available = bool(raw_trade_file and raw_trade_file.exists() and raw_trade_file.stat().st_size > 0)
        if not trades_available:
            if bool(task["allow_books_only"]):
                print(f"{prefix} [warning] Raw Trades missing; Books-only flow attribution.", flush=True)
                stats.warnings.append(
                    "raw trades missing; cancellation/consumption/replenishment attribution is unavailable"
                )
                chunks_iter = iter(())
            else:
                raise FileNotFoundError(
                    f"missing local Raw Trades for {day}; expected under {tick_loader.raw_dir}. "
                    "Use --allow-books-only only for visual validation."
                )
        else:
            # Minimal mode skips timestamp objects, symbols, trade IDs and
            # row-wise raw_json generation. The four required primitive columns
            # retain exactly the same values and ordering.
            chunks_iter = tick_loader.read_zip(
                raw_trade_file,
                chunksize=int(task["trade_chunksize"]),
                minimal=True,
            )

        print(f"{prefix} [stage] aggregate raw trades", flush=True)
        trade_by_price, trade_by_time = builder.aggregate_trades(chunks_iter, stats=stats)
        print(
            f"{prefix} [trades] rows={stats.raw_trade_rows:,} price_buckets={stats.trade_buckets:,}",
            flush=True,
        )

        print(f"{prefix} [stage] replay books and build causal features", flush=True)
        feature_rows, heatmap_rows, stats, source_files = builder.build_day(
            day,
            book_events=books_loader.iter_book_events(day, files=book_files),
            trade_by_price=trade_by_price,
            trade_by_time=trade_by_time,
            stats=stats,
            progress_every_events=int(task["progress_every_events"]),
            trade_attribution_valid=trades_available,
        )
        if raw_trade_file is not None:
            source_files.add(str(raw_trade_file))
        if stats.snapshots == 0:
            stats.warnings.append(
                "no explicit snapshot flag was parsed; inspect source schema before trusting update replay"
            )
        valid_column = feature_rows.get("book_valid")
        has_valid_book = valid_column is not None and bool(np.any(valid_column))
        if stats.book_feature_rows == 0 or not has_valid_book:
            raise RuntimeError(
                "no valid reconstructed order-book feature rows were produced; "
                "run with --inspect-only and review the schema"
            )

        print(f"{prefix} [stage] persist compact derived artifacts", flush=True)
        paths = store.save_day(
            day,
            config=cfg,
            feature_rows=feature_rows,
            heatmap_rows=heatmap_rows,
            stats=stats,
            source_files=source_files,
            compression_level=int(task["compression_level"]),
        )
        elapsed = time.perf_counter() - started
        result = {
            "status": "built",
            "day": day.isoformat(),
            "elapsed_seconds": elapsed,
            "features": stats.book_feature_rows,
            "heatmap_cells": stats.heatmap_cells,
            "book_events": stats.book_events,
            "raw_trade_rows": stats.raw_trade_rows,
            "sequence_gaps": stats.sequence_gaps,
            "warnings": list(stats.warnings),
            "paths": {
                "features": str(paths.features),
                "heatmap": str(paths.heatmap),
                "metadata": str(paths.metadata),
            },
        }
        print(
            f"{prefix} [done-day] features={stats.book_feature_rows:,} "
            f"heatmap_cells={stats.heatmap_cells:,} events={stats.book_events:,} "
            f"sequence_gaps={stats.sequence_gaps:,} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return result
    except Exception as exc:
        return {
            "status": "error",
            "day": day.isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        close_chunks = getattr(chunks_iter, "close", None)
        if callable(close_chunks):
            close_chunks()
        chunks_iter = None
        trade_by_price = None
        trade_by_time = None
        feature_rows = None
        heatmap_rows = None
        source_files = None
        gc.collect()


def _run_inspection(
    days: list[date],
    *,
    symbol: str,
    books_depth: int,
    data_dir: str | None,
    inspect_events: int,
) -> int:
    books_loader = OKXBooksLoader(symbol=symbol, depth=books_depth, data_dir=data_dir)
    for day in days:
        payload = books_loader.inspect_day_schema(day)
        try:
            payload["event_probe"] = books_loader.probe_day_events(day, max_events=inspect_events)
        except Exception as exc:
            payload["event_probe_error"] = str(exc)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


def _print_progress(done: int, total: int, started: float, result: dict[str, Any]) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"[days] {done}/{total} ({done / total * 100:5.1f}%) "
        f"last={result.get('day')} status={result.get('status')} "
        f"elapsed={elapsed:.0f}s eta={eta:.0f}s rate={rate:.3f}/s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = parse_day(args.start_date)
    end = parse_day(args.end_date)
    days = list(iter_days(start, end))
    profile, cfg = _resolve_profile_and_config(args)
    workers = _auto_workers(cfg.books_depth, int(args.workers))
    available_memory = _available_memory_bytes()
    if int(args.day_retries) < 0:
        raise ValueError("--day-retries must be >= 0")

    memory_text = (
        f"{available_memory / 1024 ** 3:.1f}"
        if available_memory is not None
        else "unknown"
    )
    print(
        f"[run] offline liquidity map | {args.symbol} | {start} -> {end} | "
        f"depth={cfg.books_depth} profile={profile} feature={cfg.feature_seconds}s "
        f"heatmap={cfg.heatmap_seconds}s price_step={cfg.price_step} workers={workers} "
        f"compression={args.compression_level} available_memory_gib={memory_text}",
        flush=True,
    )
    print("[note] Input days are official raw UTC days; analyze_tool converts derived timestamps to project UTC+8 display time.", flush=True)
    print("[note] Books defines displayed liquidity. Raw trades only attribute removal/consumption/replenishment.", flush=True)
    print("[note] Day outputs use atomic .part files; metadata is published last as the completion checkpoint.", flush=True)
    print("[note] workers=0 selects a conservative memory-safe default; use --workers 1 for the lowest-memory mode.", flush=True)

    if args.inspect_only:
        return _run_inspection(
            days,
            symbol=args.symbol,
            books_depth=args.books_depth,
            data_dir=args.data_dir,
            inspect_events=args.inspect_events,
        )

    store = LiquidityFeatureStore(symbol=args.symbol, books_depth=args.books_depth, data_dir=args.data_dir)
    tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for day in days:
        if store.has_day(day) and not args.force_rebuild:
            meta = store.load_metadata(day)
            _require_compatible_cached_config(meta, cfg, day=day)
            result = {
                "status": "skipped",
                "day": day.isoformat(),
                "features": int(meta.get("stats", {}).get("book_feature_rows", 0)),
                "heatmap_cells": int(meta.get("stats", {}).get("heatmap_cells", 0)),
                "metadata": meta,
            }
            summaries.append(result)
            print(
                f"[{day}] [skip] derived artifacts already exist | "
                f"features={result['features']:,} heatmap={result['heatmap_cells']:,}",
                flush=True,
            )
            continue
        tasks.append(
            _day_task(
                day=day,
                cfg=cfg,
                data_dir=args.data_dir,
                trade_chunksize=args.trade_chunksize,
                progress_every_events=args.progress_every_events,
                allow_books_only=args.allow_books_only,
                force_rebuild=args.force_rebuild,
                schema_probe=args.schema_probe,
                compression_level=args.compression_level,
            )
        )

    if not tasks:
        print("[done] every requested day is already cached", flush=True)
        return 0

    started = time.perf_counter()
    completed = 0
    pending_tasks = list(tasks)
    attempts: dict[str, int] = {str(task["day"]): 0 for task in tasks}
    errors: list[dict[str, Any]] = []
    current_workers = workers

    # Re-create the pool for each retry wave. A fresh process state is useful
    # after MemoryError, decompressor failure or a poisoned parser state.
    while pending_tasks:
        retry_tasks: list[dict[str, Any]] = []
        wave_workers = min(current_workers, len(pending_tasks))
        if wave_workers == 1:
            results = [_build_one_day(task) for task in pending_tasks]
        else:
            context = mp.get_context("spawn")
            results = []
            with ProcessPoolExecutor(max_workers=wave_workers, mp_context=context) as executor:
                future_to_task: dict[Future[dict[str, Any]], dict[str, Any]] = {
                    executor.submit(_build_one_day, task): task for task in pending_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(
                            {
                                "status": "error",
                                "day": str(task["day"]),
                                "error": f"worker process failed: {exc}",
                                "traceback": traceback.format_exc(limit=12),
                            }
                        )

        for result in sorted(results, key=lambda item: str(item.get("day"))):
            day_text = str(result.get("day"))
            if result.get("status") == "error":
                attempts[day_text] += 1
                if attempts[day_text] <= int(args.day_retries):
                    print(
                        f"[{day_text}] [retry] {attempts[day_text]}/{args.day_retries} "
                        f"error={result.get('error')}",
                        flush=True,
                    )
                    retry_tasks.append(next(task for task in pending_tasks if str(task["day"]) == day_text))
                    continue
                errors.append(result)
                print(f"[{day_text}] [error] {result.get('error')}", flush=True)
                if result.get("traceback"):
                    print(result["traceback"], flush=True)
            else:
                summaries.append(result)
            completed += 1
            _print_progress(completed, len(tasks), started, result)
        if retry_tasks and current_workers > 1:
            current_workers -= 1
            print(
                f"[resilience] retry wave reduces workers to {current_workers} "
                "to lower peak memory and disk pressure",
                flush=True,
            )
        pending_tasks = retry_tasks

    print("\n[summary]", flush=True)
    summary_payload = {
        "requested_days": len(days),
        "built": sum(item.get("status") == "built" for item in summaries),
        "skipped": sum(item.get("status") == "skipped" for item in summaries),
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
        "workers_initial": workers,
        "workers_final": current_workers,
        "compression_level": int(args.compression_level),
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
