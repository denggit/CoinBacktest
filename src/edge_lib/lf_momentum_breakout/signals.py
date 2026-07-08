#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Signal facade for ETH LF Momentum Breakout V3."""

from __future__ import annotations

from src.edge_lib.lf_momentum_breakout.config import EDGE_ID, PRESETS, MomentumConfig
from src.edge_lib.lf_momentum_breakout.features import build_features

__all__ = ["EDGE_ID", "PRESETS", "MomentumConfig", "build_features"]

