#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH-USDT-SWAP long-only martingale limit-order backtest entrypoint.

Three frozen variants are included:
- midterm: 1% spacing, 1:0.94 initial/add ratio, 1.05x add growth,
  4.10% take profit, 8 additions, 10x leverage.
- aggressive: 0.53% spacing, 1:0.54 initial/add ratio, 1.10x add growth,
  4.10% take profit, 12 additions, 13x leverage.
- longterm: 1.37% spacing, 1:1.11 initial/add ratio, 1.05x add growth,
  5.00% take profit, 7 additions, 9x leverage.

Examples:
    python backtest/mf/eth_martingale_limit_long_backtest.py --data-source trade_bar
    python backtest/mf/eth_martingale_limit_long_backtest.py --data-source range_bar --range-pct 0.002
    python backtest/mf/eth_martingale_limit_long_backtest.py --data-source raw_trade
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from martingale_limit_long.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
