#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.edge_library import EdgeLibrary, EdgeRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update an ETH edge record.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--status", default="idea")
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--horizon", default="")
    parser.add_argument("--causal-timing", default="closed_bar_or_later")
    parser.add_argument("--data-required", nargs="*", default=[])
    parser.add_argument("--research-report", default="")
    parser.add_argument("--backtest-report", default="")
    parser.add_argument("--portfolio-report", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--registry", default="edge_library/registry.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    library = EdgeLibrary(args.registry)
    current = library.get(args.id)
    if current is None:
        record = EdgeRecord(
            id=args.id,
            name=args.name,
            family=args.family,
            status=args.status,
            symbol=args.symbol,
            horizon=args.horizon,
            causal_timing=args.causal_timing,
            data_required=tuple(args.data_required),
            research_report=args.research_report,
            backtest_report=args.backtest_report,
            portfolio_report=args.portfolio_report,
            notes=args.notes,
        )
    else:
        record = current.with_update(
            name=args.name,
            family=args.family,
            status=args.status,
            symbol=args.symbol,
            horizon=args.horizon,
            causal_timing=args.causal_timing,
            data_required=tuple(args.data_required),
            research_report=args.research_report or current.research_report,
            backtest_report=args.backtest_report or current.backtest_report,
            portfolio_report=args.portfolio_report or current.portfolio_report,
            notes=args.notes or current.notes,
        )
    library.upsert(record)
    print(f"edge registered | id={record.id} status={record.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
