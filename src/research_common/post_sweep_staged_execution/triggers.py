#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal structure triggers for R08."""
from __future__ import annotations
import pandas as pd

TRIGGER_NAMES = ("INITIAL", "EARLY_REJECTION", "CONFIRMED_REJECTION", "STRONG_RECLAIM")


def add_trigger_flags(path: pd.DataFrame) -> pd.DataFrame:
    out = path.copy()
    out["trigger_INITIAL"] = out.groupby("zone_event_id", sort=False).cumcount().eq(0)
    out["trigger_EARLY_REJECTION"] = out["no_new_low_3bars"].fillna(False) & out["micro_high_break_3bars"].fillna(False)
    out["trigger_CONFIRMED_REJECTION"] = out["no_new_low_5bars"].fillna(False) & out["micro_high_break_5bars"].fillna(False)
    out["trigger_STRONG_RECLAIM"] = (
        out["no_new_low_10bars"].fillna(False)
        & out["micro_high_break_10bars"].fillna(False)
        & out["zone_floor_reclaimed"].fillna(False)
    )
    return out


def earliest_trigger_rows(path: pd.DataFrame, max_deployment_minutes: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name in TRIGGER_NAMES:
        mask = path[f"trigger_{name}"]
        if name != "INITIAL":
            mask &= pd.to_numeric(path["elapsed_bars"], errors="coerce") <= max_deployment_minutes
        selected = path.loc[mask].sort_values(["zone_event_id", "elapsed_bars", "checkpoint_available_time", "checkpoint_id"], kind="mergesort").groupby("zone_event_id", sort=False).head(1).copy()
        selected["trigger_name"] = name
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)
