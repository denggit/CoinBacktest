#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run all six ETH trend-following baselines on one shared data load."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.mf.trend_following import (
    donchian_breakout,
    ema_momentum,
    market_structure,
    orderflow_trend,
    trend_pullback,
    volatility_expansion,
)
from backtest.mf.trend_following.common import DEFAULT_REPORT_ROOT, load_bars, make_parser, run_spec

STRATEGIES = (
    donchian_breakout.SPEC,
    ema_momentum.SPEC,
    trend_pullback.SPEC,
    market_structure.SPEC,
    volatility_expansion.SPEC,
    orderflow_trend.SPEC,
)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = make_parser(__doc__ or "ETH trend following suite", "all")
    parser.set_defaults(out_dir=DEFAULT_REPORT_ROOT)
    args = parser.parse_args(argv)
    root = Path(args.out_dir)
    bars = load_bars(args)
    rows: list[dict[str, object]] = []
    total = len(STRATEGIES)
    for i, spec in enumerate(STRATEGIES, start=1):
        print(f"\n[suite] [{i}/{total}] {spec.strategy_name}", flush=True)
        child = type(args)(**vars(args))
        child.out_dir = str(root / spec.strategy_name)
        rows.append(run_spec(child, spec, bars=bars, emit_report=True))

    comparison = pd.DataFrame(rows)
    preferred = [
        "strategy", "total_trades", "long_trades", "short_trades", "win_rate",
        "profit_factor", "expectancy_pct", "total_return_pct", "max_drawdown_pct",
        "avg_mfe_r", "avg_mae_r", "avg_holding_hours", "total_fees", "signal_count",
    ]
    cols = [c for c in preferred if c in comparison.columns] + [c for c in comparison.columns if c not in preferred]
    comparison = comparison.loc[:, cols]
    root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(root / "trend_following_comparison.csv", index=False, encoding="utf-8-sig")
    with (root / "trend_following_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    print("\n" + "=" * 120)
    print("TREND FOLLOWING COMPARISON")
    print("=" * 120)
    show = [c for c in preferred if c in comparison.columns]
    print(comparison[show].to_string(index=False))
    print(f"\n[done] comparison={root / 'trend_following_comparison.csv'}")
    return comparison


if __name__ == "__main__":
    main()
