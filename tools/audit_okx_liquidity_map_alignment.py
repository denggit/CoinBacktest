#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit Kline ↔ order-book heatmap time/price alignment on local artifacts.

This diagnostic uses the exact same period-end display path as Analyze Tool and
prints the structured alignment audit.  It does not download or modify data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tool.data_service import LoadRequest, load_dataframe  # noqa: E402
from analyze_tool.plugin_api import PluginRunContext  # noqa: E402
from analyze_tool.plugins.orderbook_liquidity_heatmap import (  # noqa: E402
    OrderBookLiquidityHeatmapPlugin,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local liquidity-map chart alignment")
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start", required=True, help="Project UTC+8 wall time/date")
    parser.add_argument("--end", required=True, help="Project UTC+8 wall time/date")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--data-type", choices=["normal", "trade_bar"], default="trade_bar")
    parser.add_argument("--books-depth", type=int, default=5000)
    parser.add_argument("--display-price-step", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=200000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = LoadRequest(
        data_type=args.data_type,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=str(args.start).replace("T", " "),
        end=str(args.end).replace("T", " "),
        limit=args.limit,
        local_only=True,
    )
    bars, meta = load_dataframe(request)
    if bars.empty:
        print(json.dumps({"status": "error", "error": "no local candles in requested range"}, ensure_ascii=False, indent=2))
        return 2
    context = PluginRunContext(
        display_df=bars,
        visible_df=bars,
        request={
            "data_type": request.data_type,
            "timeframe": request.timeframe,
            "start": request.start,
            "end": request.end,
        },
        meta=meta,
    )
    try:
        result = OrderBookLiquidityHeatmapPlugin().run_with_context(
            context,
            {
                "books_depth": args.books_depth,
                "display_mode": "period_end",
                "display_price_step": args.display_price_step,
                "normalization": "manual",
                "manual_max": 1.0,
                "min_intensity_pct": 0.0,
                "max_render_cells": 1_000_000,
                "large_window_hours": 24,
                "large_percentile": 95,
            },
        )
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "rebuild_required",
                    "error": str(exc),
                    "symbol": args.symbol,
                    "timeframe": args.timeframe,
                    "books_depth": args.books_depth,
                    "display_price_step": args.display_price_step,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    audit = dict(result.summary.get("alignment_audit") or {})
    audit["symbol"] = args.symbol
    audit["timeframe"] = args.timeframe
    audit["books_depth"] = args.books_depth
    audit["display_price_step"] = args.display_price_step
    audit["bar_rows"] = int(len(bars))
    audit["display_heatmap_cells"] = int(result.summary.get("display_heatmap_cells", 0))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
