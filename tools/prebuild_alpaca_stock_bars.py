#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild historical US-stock minute bars through src.data_feed.AlpacaStockLoader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader  # noqa: E402

NY_TZ = "America/New_York"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="SOXL")
    p.add_argument("--start-date", default="2019-01-01", help="New York calendar date")
    p.add_argument("--end-date", default="2026-06-30", help="New York calendar date")
    p.add_argument("--timeframe", default="1Min", choices=["1Min", "2Min", "5Min", "15Min"])
    p.add_argument("--feed", default="sip", choices=["sip", "iex", "boats"])
    p.add_argument("--adjustment", default="raw")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--chunk-months", type=int, default=1)
    return p.parse_args()


def _ny_date_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_date).tz_localize(NY_TZ)
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(NY_TZ) - pd.Timedelta(nanoseconds=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    cursor = start
    while cursor <= end:
        next_cursor = cursor + pd.DateOffset(months=max(1, int(months)))
        chunk_end = min(end, next_cursor - pd.Timedelta(nanoseconds=1))
        yield cursor, chunk_end
        cursor = next_cursor


def main() -> int:
    args = parse_args()
    start_utc, end_utc = _ny_date_bounds(args.start_date, args.end_date)
    loader = AlpacaStockLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        feed=args.feed,
        adjustment=args.adjustment,
        data_dir=args.data_dir,
    )
    chunks = list(_month_chunks(start_utc, end_utc, args.chunk_months))
    print(
        f"[run] Alpaca {args.symbol} {args.timeframe} feed={args.feed} adjustment={args.adjustment} "
        f"NY={args.start_date}->{args.end_date} chunks={len(chunks)}",
        flush=True,
    )
    total_rows = 0
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        frame = loader.fetch_remote(chunk_start, chunk_end, request_pause_seconds=0.35)
        if not frame.empty:
            loader.save_local_data(frame)
        total_rows += len(frame)
        print(
            f"[chunk] {i}/{len(chunks)} UTC={chunk_start}->{chunk_end} rows={len(frame):,} total_fetched={total_rows:,}",
            flush=True,
        )
    local = loader.load_local_data()
    print(
        f"[done] local_rows={len(local):,} db={loader.db_path} table={loader.table_name} "
        f"range={local.index.min() if not local.empty else None}->{local.index.max() if not local.empty else None}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
