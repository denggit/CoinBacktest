#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Windows/Unix friendly entry point for ETH Trend Pullback V1."""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backtest.mf.eth_trend_pullback_v1.run import main
if __name__ == "__main__":
    main()
