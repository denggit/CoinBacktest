#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH long martingale backtest package."""

from .config import EngineConfig, MartingaleVariant, VARIANTS
from .engine import MartingaleEngine

__all__ = [
    "EngineConfig",
    "MartingaleEngine",
    "MartingaleVariant",
    "VARIANTS",
]
