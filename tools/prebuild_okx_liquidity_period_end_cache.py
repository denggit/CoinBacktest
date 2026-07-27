#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild V2 compact period-end liquidity snapshots for fast Analyze Tool use."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader
from src.liquidity_map.aggregation import timeframe_to_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V2 one-last-snapshot-per-bar liquidity cache, day by day."
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", required=True, help="official raw UTC day, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="official raw UTC day, YYYY-MM-DD")
    parser.add_argument("--books-depth", type=int, default=5000)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--price-step", type=float, default=1.0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--workers", type=int, default=0, help="Day workers. 0=auto, capped at 4")
    parser.add_argument("--day-retries", type=int, default=1)
    return parser.parse_args()


def _auto_workers(requested: int) -> int:
    if requested < 0:
        raise ValueError("--workers must be >= 0")
    if requested:
        return requested
    return min(4, max(1, os.cpu_count() or 1))


def _build_day(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    day = pd.Timestamp(task["day"]).date()
    try:
        loader = OKXLiquidityMapLoader(
            symbol=task["symbol"],
            books_depth=int(task["books_depth"]),
            data_dir=task.get("data_dir"),
        )
        frame = loader.period_end_cache.load_or_build_day(
            day,
            target_seconds=int(task["target_seconds"]),
            target_price_step=float(task["price_step"]),
        )
        elapsed = time.perf_counter() - started
        if frame.empty:
            status = "missing"
            rows = 0
            cache_path = ""
        else:
            status = "hit" if bool(frame.attrs.get("cache_hit")) else "built"
            rows = len(frame)
            cache_path = str(frame.attrs.get("cache_path") or "")
        return {
            "day": day.isoformat(),
            "status": status,
            "rows": int(rows),
            "elapsed_seconds": elapsed,
            "path": cache_path,
        }
    except Exception as exc:
        return {
            "day": day.isoformat(),
            "status": "error",
            "rows": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "path": "",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if end < start:
        raise ValueError("end-date must be >= start-date")
    if args.day_retries < 0:
        raise ValueError("--day-retries must be >= 0")
    target_seconds = timeframe_to_seconds(args.timeframe)
    workers = _auto_workers(args.workers)
    days = list(pd.date_range(start, end, freq="D"))
    print(
        f"[run] liquidity period-end cache V2 | {args.symbol} | {start.date()} -> {end.date()} "
        f"| depth={args.books_depth} timeframe={args.timeframe} price_step={args.price_step:g} "
        f"workers={workers}",
        flush=True,
    )
    print("[note] Each day is an atomic independent cache; interrupted workers are safe to rerun.", flush=True)

    tasks = [
        {
            "day": day.date().isoformat(),
            "symbol": args.symbol,
            "books_depth": int(args.books_depth),
            "data_dir": args.data_dir,
            "target_seconds": int(target_seconds),
            "price_step": float(args.price_step),
        }
        for day in days
    ]
    attempts = {task["day"]: 0 for task in tasks}
    rows: list[dict[str, Any]] = []
    pending = tasks
    total_start = time.perf_counter()
    completed = 0

    while pending:
        if workers == 1:
            wave_results = [_build_day(task) for task in pending]
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = {executor.submit(_build_day, task): task for task in pending}
                wave_results = []
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        wave_results.append(future.result())
                    except Exception as exc:
                        wave_results.append(
                            {
                                "day": task["day"],
                                "status": "error",
                                "rows": 0,
                                "path": "",
                                "elapsed_seconds": 0.0,
                                "error": f"worker process failed: {exc}",
                                "traceback": traceback.format_exc(limit=12),
                            }
                        )
        retry: list[dict[str, Any]] = []
        for item in sorted(wave_results, key=lambda value: value["day"]):
            day = item["day"]
            if item["status"] == "error" and attempts[day] < args.day_retries:
                attempts[day] += 1
                print(f"[retry] {day} {attempts[day]}/{args.day_retries}: {item.get('error')}", flush=True)
                retry.append(next(task for task in pending if task["day"] == day))
                continue
            rows.append(item)
            completed += 1
            elapsed = max(time.perf_counter() - total_start, 1e-9)
            rate = completed / elapsed
            eta = (len(tasks) - completed) / rate if rate else 0.0
            print(
                f"[day {completed}/{len(tasks)}] {day} status={item['status']} rows={item['rows']:,} "
                f"elapsed={item['elapsed_seconds']:.2f}s eta={eta:.0f}s",
                flush=True,
            )
            if item["status"] == "error" and item.get("traceback"):
                print(item["traceback"], flush=True)
        pending = retry

    summary = {
        "built": sum(item["status"] == "built" for item in rows),
        "hits": sum(item["status"] == "hit" for item in rows),
        "missing": sum(item["status"] == "missing" for item in rows),
        "errors": sum(item["status"] == "error" for item in rows),
        "days": rows,
        "elapsed_seconds": time.perf_counter() - total_start,
        "workers": workers,
    }
    print("\n[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["missing"] or summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
