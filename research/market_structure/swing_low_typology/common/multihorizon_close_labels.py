#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-horizon next-open / future-close labels for research 13."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.market_structure.swing_low_typology.common.reversal_opportunity import (
    build_reversal_forward_labels,
)


def _rename_horizon(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    suffix = str(int(horizon))
    rename = {
        "entry_time": f"entry_time_{suffix}",
        "entry_price": f"entry_price_{suffix}",
        "label_end_time": f"label_end_time_{suffix}",
        "forward_horizon_bars": f"forward_horizon_bars_{suffix}",
        "tp_hit_1pct": f"tp_hit_{suffix}",
        "tp_first_touch_bar": f"tp_first_touch_bar_{suffix}",
        "mfe_pct": f"mfe_{suffix}_pct",
        "mae_horizon_pct": f"mae_{suffix}_pct",
        "mae_before_tp_pct": f"mae_before_tp_{suffix}_pct",
        "terminal_return_pct": f"terminal_return_{suffix}_pct",
        "tp_within_15": f"tp_within_15_h{suffix}",
        "tp_within_30": f"tp_within_30_h{suffix}",
        "tp_within_45": f"tp_within_45_h{suffix}",
    }
    for column in frame.columns:
        if column.startswith("adverse_hit_") or column.startswith("adverse_first_touch_bar_") or column.startswith("tp_before_adverse_"):
            rename[column] = f"{column}_h{suffix}"
    return frame.rename(columns=rename)


def build_multihorizon_close_labels(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    target_move_pct: float = 1.0,
    short_horizon: int = 60,
    long_horizon: int = 180,
    vectorized_chunk_size: int = 50_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    if not (1 <= int(short_horizon) < int(long_horizon)):
        raise ValueError("require 1 <= short_horizon < long_horizon")
    short = build_reversal_forward_labels(
        bars,
        decisions,
        horizon=int(short_horizon),
        target_move_pct=float(target_move_pct),
        vectorized_chunk_size=int(vectorized_chunk_size),
        show_progress=show_progress,
    )
    long = build_reversal_forward_labels(
        bars,
        decisions,
        horizon=int(long_horizon),
        target_move_pct=float(target_move_pct),
        vectorized_chunk_size=int(vectorized_chunk_size),
        show_progress=show_progress,
    )
    if short.empty or long.empty:
        return pd.DataFrame()
    short = _rename_horizon(short, int(short_horizon))
    long = _rename_horizon(long, int(long_horizon))
    merged = short.merge(long, on="event_id", how="inner", validate="one_to_one")
    # Entry is the same next 1m open for both horizons.
    if not pd.to_datetime(merged[f"entry_time_{short_horizon}"]).equals(pd.to_datetime(merged[f"entry_time_{long_horizon}"])):
        raise RuntimeError("multi-horizon labels disagree on next-open entry time")
    if not np.allclose(
        pd.to_numeric(merged[f"entry_price_{short_horizon}"], errors="coerce"),
        pd.to_numeric(merged[f"entry_price_{long_horizon}"], errors="coerce"),
        equal_nan=True,
    ):
        raise RuntimeError("multi-horizon labels disagree on next-open entry price")

    merged["entry_time"] = pd.to_datetime(merged[f"entry_time_{short_horizon}"])
    merged["entry_price"] = pd.to_numeric(merged[f"entry_price_{short_horizon}"], errors="coerce")
    merged["label_end_time"] = pd.to_datetime(merged[f"label_end_time_{long_horizon}"])
    merged["tp30"] = merged[f"tp_within_30_h{short_horizon}"].astype(bool)
    merged["tp60"] = merged[f"tp_hit_{short_horizon}"].astype(bool)
    merged["tp180"] = merged[f"tp_hit_{long_horizon}"].astype(bool)
    merged["clean30_0p5"] = merged["tp30"] & merged[f"tp_before_adverse_0p5pct_h{short_horizon}"].astype(bool)
    merged["clean60_0p5"] = merged[f"tp_before_adverse_0p5pct_h{short_horizon}"].astype(bool)
    merged["clean180_0p5"] = merged[f"tp_before_adverse_0p5pct_h{long_horizon}"].astype(bool)
    merged["clean180_1p0"] = merged[f"tp_before_adverse_1p0pct_h{long_horizon}"].astype(bool)
    merged["slow_success_180"] = (~merged["tp60"]) & merged["tp180"]
    merged["slow_clean_success_180"] = merged["slow_success_180"] & merged["clean180_0p5"]
    merged["deep_recovery_180"] = merged["tp180"] & (pd.to_numeric(merged[f"mae_{long_horizon}_pct"], errors="coerce") > 0.75)
    merged["persistent_failure_180"] = ~merged["tp180"]
    merged["path_class"] = np.select(
        [
            merged["tp60"] & merged["clean60_0p5"],
            merged["slow_clean_success_180"],
            merged["deep_recovery_180"],
            merged["tp180"],
        ],
        ["fast_clean", "slow_clean", "deep_recovery", "other_tp180"],
        default="persistent_failure",
    )
    return merged
