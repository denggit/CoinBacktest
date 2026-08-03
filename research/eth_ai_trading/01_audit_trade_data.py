#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the superseded standalone R01 audit.

R01 now performs a light public-loader preflight and immediately enters the
trades-only supervised baseline. Existing commands are forwarded to the new
complete research entrypoint.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("01_trades_only_supervised_baseline.py")
    runpy.run_path(str(target), run_name="__main__")
