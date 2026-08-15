#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility notice for the superseded standalone R01 audit.

The old ETH Q70 research has been archived under its named model directory.
The active historical R01 workflow was consolidated into
``01_trades_only_supervised_baseline.py`` and this file intentionally performs
no data audit or model work.
"""
from __future__ import annotations


def main() -> None:
    print(
        "superseded standalone R01 audit; see "
        "research/eth_ai_trading/eth_q70_reclaim_mf_long_v1/"
        "01_trades_only_supervised_baseline.py"
    )


if __name__ == "__main__":
    main()
