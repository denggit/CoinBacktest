#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal async email sender placeholder used by legacy order-flow script.

CoinBacktest's backtest data layer does not require email.  This stub keeps the
legacy ``okx_ws_orderflow.py`` importable; replace it with a real SMTP sender
when you need live alerts.
"""

from __future__ import annotations

from src.utils.log import get_logger

logger = get_logger(__name__)


async def send_trading_signal_email(symbol: str, signal_type: str, price: float, details: str) -> bool:
    logger.warning(
        "Email sender is not configured. Skip signal email: symbol=%s type=%s price=%s",
        symbol,
        signal_type,
        price,
    )
    return False
