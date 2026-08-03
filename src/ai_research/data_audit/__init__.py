#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deprecated standalone R01 data-audit package.

The platform-level audit was intentionally removed from the active ETH AI
research path. R01 now performs a small public-loader smoke check in
``src.ai_research.trades_baseline.dataset`` and immediately enters model
research. This namespace remains only so older imports fail clearly rather than
silently running the obsolete raw/SQLite implementation.
"""


def __getattr__(name: str):
    raise AttributeError(
        f"src.ai_research.data_audit.{name} was retired; use "
        "src.ai_research.trades_baseline instead"
    )
