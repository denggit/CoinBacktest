#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH edge lifecycle registry.

For the single-market ETH portfolio, an edge is a verified market phenomenon,
not necessarily a directly tradable strategy. Edges graduate into backtest
candidates only after research evidence passes.
"""

from .models import EdgeRecord
from .registry import EdgeLibrary

__all__ = ["EdgeLibrary", "EdgeRecord"]
