#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.4.2.8A occupied-signal atlas and tranche eligibility gate."""

from .config import DEFAULT_TRANCHE_ELIGIBILITY_CONFIG, TrancheEligibilityConfig
from .pipeline import TrancheEligibilityResult, run_tranche_eligibility_audit

__all__ = [
    "DEFAULT_TRANCHE_ELIGIBILITY_CONFIG",
    "TrancheEligibilityConfig",
    "TrancheEligibilityResult",
    "run_tranche_eligibility_audit",
]
