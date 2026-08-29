#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prebuild/cache OKX OHLCV through the project data_feed loader.

This utility intentionally contains no exchange-specific data access logic;
all market-data I/O stays inside ``src.data_feed.OKXDataLoader``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prebuild/cache OKX OHLCV using src.data_feed.OKXDataLoader")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--data-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loader = OKXDataLoader(symbol=args.symbol, timeframe=args.timeframe, db_dir=args.data_dir)
    frame = loader.fetch_data_by_date_range(args.start_date, args.end_date)
    if frame.empty:
        raise SystemExit("No OHLCV rows returned; coverage is not ready")
    print(f"[prebuild-okx-ohlcv] symbol={args.symbol} timeframe={args.timeframe} rows={len(frame):,}")
    print(f"[prebuild-okx-ohlcv] range={frame.index.min()} -> {frame.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
