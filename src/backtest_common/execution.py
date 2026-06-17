#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execution-price helpers shared by backtest scripts."""

from __future__ import annotations


def apply_entry_slippage(price: float, side: int, slippage_pct: float) -> float:
    """Return conservative entry fill price after slippage.

    side: 1 for long, -1 for short.
    """
    return price * (1 + slippage_pct) if side == 1 else price * (1 - slippage_pct)


def apply_exit_slippage(price: float, side: int, slippage_pct: float) -> float:
    """Return conservative exit fill price after slippage.

    side is the original position direction: 1 for long, -1 for short.
    """
    return price * (1 - slippage_pct) if side == 1 else price * (1 + slippage_pct)
