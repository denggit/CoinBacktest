#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild neutral relative-depth primitives from existing liquidity-map days.

This command does not download Books, rebuild the liquidity map or decide where
walls are. It only converts canonical day artifacts into fast NumPy research
inputs with atomic per-day checkpoints and causal rolling depth references.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader  # noqa: E402
from src.data_feed.okx_liquidity_primitives import (  # noqa: E402
    CausalPrimitiveReference,
    LiquidityPrimitiveConfig,
    OKXLiquidityPrimitiveStore,
    build_liquidity_primitive_day,
)
from src.research_common.progress import ProgressReporter  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prebuild low-semantic causal liquidity primitives; no wall labels are created.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--books-depth", type=int, default=5000)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cache-version", default="v1")
    p.add_argument("--reference-window-hours", type=float, default=24.0)
    p.add_argument("--denominator-floor-absolute", type=float, default=1e-9)
    p.add_argument("--denominator-floor-fraction-of-side-mean", type=float, default=0.02)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--compression-level",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0..9",
        help="NPZ DEFLATE level. 1 is fast; 0 stores without compression.",
    )
    p.add_argument(
        "--no-warmup-day",
        action="store_true",
        help="Do not read the preceding UTC day to warm the causal 24h reference.",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    start_day = pd.Timestamp(args.start_date).normalize()
    end_day = pd.Timestamp(args.end_date).normalize()
    if end_day < start_day:
        raise ValueError("--end-date must be >= --start-date")
    cfg = LiquidityPrimitiveConfig(
        reference_window_hours=float(args.reference_window_hours),
        denominator_floor_absolute=float(args.denominator_floor_absolute),
        denominator_floor_fraction_of_side_mean=float(
            args.denominator_floor_fraction_of_side_mean
        ),
    )
    cfg.validate()
    source = OKXLiquidityMapLoader(
        symbol=args.symbol,
        books_depth=int(args.books_depth),
        data_dir=args.data_dir,
    )
    store = OKXLiquidityPrimitiveStore(
        symbol=args.symbol,
        books_depth=int(args.books_depth),
        data_dir=args.data_dir,
        cache_version=args.cache_version,
    )
    reference = CausalPrimitiveReference(window_hours=cfg.reference_window_hours)
    scan_start = start_day if args.no_warmup_day else start_day - pd.Timedelta(days=1)
    dates = list(pd.date_range(scan_start, end_day, freq="D"))
    target_total = int((end_day - start_day).days) + 1
    progress = ProgressReporter(label="[liquidity-primitives] target UTC days", total=target_total, every=1)
    completed = 0
    print(
        f"[run] liquidity primitives | {args.symbol} | {start_day.date()} -> {end_day.date()} | "
        f"books={args.books_depth} cache={args.cache_version} compression={args.compression_level}",
        flush=True,
    )
    print(
        "[note] This reads existing liquidity-map NPZ files only. It creates no wall labels and no trade outcomes.",
        flush=True,
    )
    for day in dates:
        day_date = day.date()
        is_target = start_day.date() <= day_date <= end_day.date()
        if store.has_day(day_date) and not args.force:
            metadata = store.load_metadata(day_date)
            if metadata.get("config") != cfg.to_dict():
                raise RuntimeError(
                    f"Cached liquidity primitive config differs for {day_date}. "
                    "Use --force after reviewing the parameter change. "
                    f"cached={metadata.get('config')} requested={cfg.to_dict()}"
                )
            bucket_end_ms, snapshot_q95, snapshot_q99 = store.load_reference_arrays(day_date)
            reference.replay_arrays(bucket_end_ms, snapshot_q95, snapshot_q99)
            if is_target:
                completed += 1
                print(
                    f"[skip-day] {day_date} snapshots={int(metadata.get('snapshot_count', len(bucket_end_ms))):,} "
                    f"cells={int(metadata.get('cell_count', 0)):,} reference_only=1",
                    flush=True,
                )
                progress.update(completed)
            continue

        day_start_utc = pd.Timestamp(day_date, tz="UTC")
        day_end_utc = day_start_utc + pd.Timedelta(days=1)
        frames = list(source.iter_heatmap_days_raw(day_start_utc, day_end_utc))
        if not frames:
            if is_target:
                raise RuntimeError(
                    f"Missing offline liquidity-map day {day_date}. Run tools/prebuild_okx_offline_liquidity_map.py first."
                )
            print(f"[warmup-missing] {day_date}; causal reference starts with available data", flush=True)
            continue
        primitive_day = build_liquidity_primitive_day(frames[0], reference=reference, config=cfg)
        if is_target:
            paths = store.save_day(
                day_date,
                arrays=primitive_day.arrays,
                metadata=primitive_day.metadata,
                compression_level=int(args.compression_level),
            )
            completed += 1
            print(
                f"[done-day] {day_date} snapshots={primitive_day.snapshot_count:,} "
                f"cells={primitive_day.cell_count:,} path={paths.primitives}",
                flush=True,
            )
            progress.update(completed)
        else:
            print(
                f"[warmup-day] {day_date} snapshots={primitive_day.snapshot_count:,} (not written)",
                flush=True,
            )
    progress.close()
    print(f"[done] primitive_root={store.root}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
