#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experiment lifecycle helpers for CoinBacktest.

The experiment layer is metadata-only. Research/backtest scripts remain entry
points; reusable logic should live under ``src`` and scripts should publish
standard manifests/decisions through this package.
"""

from .decision import build_decision, write_decision
from .models import ExperimentRecord
from .registry import ExperimentRegistry

__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "build_decision",
    "write_decision",
]
