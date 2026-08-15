#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1b target-consistency audit for latent-liquidity residualization."""

from .config import DEFAULT_CONFIG, TargetConsistencyConfig
from .pipeline import run_target_consistency_audit

__all__ = ["DEFAULT_CONFIG", "TargetConsistencyConfig", "run_target_consistency_audit"]
