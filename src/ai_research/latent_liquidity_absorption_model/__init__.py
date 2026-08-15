#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.3 absorption-completion and remaining-space learning."""
from .config import DEFAULT_CONFIG, AbsorptionModelConfig
from .pipeline import AbsorptionModelResult, run_absorption_remaining_space_model

__all__ = [
    "DEFAULT_CONFIG",
    "AbsorptionModelConfig",
    "AbsorptionModelResult",
    "run_absorption_remaining_space_model",
]
