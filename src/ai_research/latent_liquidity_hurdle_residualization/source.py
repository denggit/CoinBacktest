#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1 source adapter: exact-first-touch R02.2 cache with R02.3 quarantine semantics."""
from __future__ import annotations

import pandas as pd

from src.ai_research.latent_liquidity_excess_ranking.source import load_r02_2_exact_first_touch_dataset


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, gate = load_r02_2_exact_first_touch_dataset()
    frame = frame.copy()
    # Preserve the already-reviewed R02.3 quarantine semantics. R02.3.1 never
    # silently repairs old R02 1m/1s disagreements.
    frame["r02_3_1_upstream_eligible"] = frame["r02_3_source_eligible"].astype(bool)
    return frame, gate
