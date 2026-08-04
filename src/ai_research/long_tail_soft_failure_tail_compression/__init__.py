#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.4.2.12 soft-failure attribution and real tail compression."""

from .config import (
    DEFAULT_TAIL_COMPRESSION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    TailCompressionConfig,
    TailCompressionPolicy,
)
from .pipeline import TailCompressionResult, run_tail_compression_audit

__all__ = [
    "DEFAULT_TAIL_COMPRESSION_CONFIG",
    "STAGE_ID",
    "STAGE_NAME",
    "TailCompressionConfig",
    "TailCompressionPolicy",
    "TailCompressionResult",
    "run_tail_compression_audit",
]
