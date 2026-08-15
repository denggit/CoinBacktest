#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Residual target construction and within-snapshot relevance grades."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HurdleResidualizationConfig


def _add_group_relevance(
    out: pd.DataFrame,
    *,
    eligible: pd.Series,
    target: str,
    relevance_col: str,
    group_eligible_col: str,
    grades: int,
) -> None:
    out[relevance_col] = np.int16(-1)
    out[group_eligible_col] = False
    idx = out.index[eligible & pd.to_numeric(out[target], errors="coerce").notna()]
    if len(idx) == 0:
        return
    work = out.loc[idx, ["ranking_group", target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    stats = work.groupby("ranking_group", sort=False)[target].agg(["size", "min", "max"])
    good_groups = stats.index[(stats["size"] >= 2) & ((stats["max"] - stats["min"]) > 1e-12)]
    good = work["ranking_group"].isin(good_groups)
    if not good.any():
        return
    good_idx = work.index[good]
    pct = work.loc[good].groupby("ranking_group", sort=False)[target].rank(method="average", pct=True)
    bins = np.floor(np.clip(pct.to_numpy(dtype=float) - 1e-12, 0.0, 0.999999) * int(grades)).astype(np.int16)
    out.loc[good_idx, relevance_col] = bins
    out.loc[good_idx, group_eligible_col] = True


def attach_ranking_targets(frame: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    out = frame.copy()
    source_ok = out["r02_3_1_source_eligible"].astype(bool)
    _add_group_relevance(
        out,
        eligible=source_ok,
        target="excess_liquidity_residual",
        relevance_col="excess_residual_relevance",
        group_eligible_col="excess_residual_group_eligible",
        grades=config.rank_relevance_grades,
    )
    _add_group_relevance(
        out,
        eligible=source_ok & out["release_observed_180s"].astype(bool),
        target="reversal_quality_residual",
        relevance_col="reversal_residual_relevance",
        group_eligible_col="reversal_residual_group_eligible",
        grades=config.rank_relevance_grades,
    )
    return out
