#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Signal facade for ETH LF Bear Short V3."""

from __future__ import annotations

from src.edge_lib.lf_bear_short.config import EDGE_ID, PRESETS, BearConfig
from src.edge_lib.lf_bear_short.features import build_bear_features

__all__ = ["EDGE_ID", "PRESETS", "BearConfig", "build_bear_features"]

