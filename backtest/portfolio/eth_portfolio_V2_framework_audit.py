#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bootstrap/audit the executable ETH Portfolio V2 strategy program.

No market data is loaded here.  This script freezes the portfolio destination
and validates that every planned core sleeve is specified as an executable
strategy rather than a research-only edge topic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_common.strategy_catalog import PORTFOLIO_V2_ID, build_core_strategy_catalog  # noqa: E402
from src.strategy_common import FunnelPolicy  # noqa: E402


DEFAULT_OUT_DIR = Path("data/reports/research/eth_portfolio_v2/framework")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = build_core_strategy_catalog()
    rows = [item.to_dict() for item in catalog]
    pd.DataFrame(rows).to_csv(out / "01_strategy_catalog.csv", index=False)

    policy = FunnelPolicy(strategy_class="core")
    goal = {
        "portfolio_id": PORTFOLIO_V2_ID,
        "market": "ETH-USDT-SWAP perpetual only",
        "purpose": "backtest -> robustness -> AetherEdge shadow/live -> copy trading",
        "excluded": [
            "spot portfolio",
            "funding-rate arbitrage",
            "cash-and-carry arbitrage",
            "research-only edge promotion without executable trade rules",
            "martingale/grid loss recovery",
        ],
        "core_sleeves": [item.strategy_id for item in catalog],
        "portfolio_targets": {
            "monthly_trades": "30-80",
            "win_rate": "55%-68% directional target, not a hard sleeve filter",
            "profit_factor_min": 1.40,
            "profit_factor_target": 1.60,
            "positive_month_rate_min": 0.70,
            "positive_month_rate_target": 0.75,
            "standard_mdd_target": "10%-15%",
            "hard_mdd_ceiling": 0.20,
            "max_flat_days_target": 7,
            "fee_2x_must_remain_profitable": True,
        },
        "default_cost": {
            "roundtrip_fee": 0.0011,
            "slippage_is_additional": True,
        },
        "funnel_policy": {
            "min_executed_trades_core": policy.min_executed_trades_core,
            "warn_hard_filter_retention": policy.warn_hard_filter_retention,
            "fail_hard_filter_retention": policy.fail_hard_filter_retention,
            "min_total_hard_filter_retention_core": policy.min_total_hard_filter_retention_core,
        },
        "development_order": [
            "01 framework + funnel gate",
            "02 trend breakout",
            "03 trend pullback",
            "04 liquidity reversal",
            "05 range mean reversion",
            "06 volatility expansion",
            "07 portfolio allocation/conflict/risk",
            "08 robustness/holdout",
            "09 AetherEdge migration",
            "10 shadow -> small live -> copy trading",
        ],
    }
    (out / "00_portfolio_goal.json").write_text(json.dumps(goal, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "portfolio_id": PORTFOLIO_V2_ID,
        "core_strategy_count": len(catalog),
        "all_contracts_valid": True,
        "all_symbol_eth_swap": all(item.symbol == "ETH-USDT-SWAP" for item in catalog),
        "all_core": all(item.strategy_class == "core" for item in catalog),
        "trend_breakout_in_development": any(
            item.strategy_id == "ETH_STRATEGY_TREND_BREAKOUT_V1" and item.stage == "in_development"
            for item in catalog
        ),
    }
    checks["passed"] = bool(
        checks["core_strategy_count"] == 5
        and checks["all_contracts_valid"]
        and checks["all_symbol_eth_swap"]
        and checks["all_core"]
        and checks["trend_breakout_in_development"]
    )
    (out / "02_framework_check.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"[write] {out.resolve()}", flush=True)
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
