#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Project-wide lightweight logger helper.

The original data loaders import ``src.utils.log.get_logger`` but the uploaded
project did not include that module.  Keep this helper dependency-free so data
collection scripts can run on a fresh machine without extra setup.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@lru_cache(maxsize=128)
def get_logger(name: str = "CoinBacktest") -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.getenv("COINBACKTEST_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(handler)
    return logger
