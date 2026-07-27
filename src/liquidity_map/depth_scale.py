#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal order-book depth scaling shared by display, research and live logic.

For every completed/latest snapshot we estimate a robust high-depth level across
its retained price bins.  The default is the cross-sectional P99.  The colour
and wall reference at time ``t`` is the rolling maximum of that statistic over
the completed past window, including the current available snapshot.

V2.5.2 keeps the exact same causal semantics but replaces the former
``groupby -> merge -> time rolling`` pipeline with compact NumPy arrays and a
monotonic deque.  This matters for last-snapshot maps where hundreds of
thousands of price cells may be normalised repeatedly inside Analyze Tool.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalDepthScaleConfig:
    window_hours: float = 24.0
    minimum_reference: float = 1e-12
    separate_sides: bool = False
    snapshot_reference_quantile: float = 0.99

    def validate(self) -> None:
        if self.window_hours <= 0:
            raise ValueError("window_hours must be > 0")
        if self.minimum_reference <= 0:
            raise ValueError("minimum_reference must be > 0")
        if not 0.5 <= self.snapshot_reference_quantile <= 1.0:
            raise ValueError("snapshot_reference_quantile must be in [0.5, 1.0]")


def infer_time_column(frame: pd.DataFrame) -> str:
    # Prefer the time at which a completed snapshot/cell is actually available.
    for name in (
        "source_bucket_end_ms",
        "bar_end_ms",
        "bucket_end_ms",
        "bar_start_ms",
        "bucket_start_ms",
    ):
        if name in frame.columns:
            return name
    raise ValueError("depth scaling requires a completed snapshot time column")


def _side_codes(frame: pd.DataFrame) -> np.ndarray:
    if "side_code" in frame.columns:
        values = pd.to_numeric(frame["side_code"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        return values
    if "side" in frame.columns:
        return frame["side"].astype("string").map({"bid": 1, "ask": -1}).fillna(0).to_numpy(dtype=np.int8)
    raise ValueError("depth scaling requires side or side_code")


def _rolling_max_by_group(
    times: np.ndarray,
    groups: np.ndarray,
    values: np.ndarray,
    *,
    window_ms: int,
    minimum_reference: float,
) -> np.ndarray:
    """Causal rolling maximum on sorted unique (group, timestamp) rows."""

    references = np.full(len(values), float(minimum_reference), dtype=np.float64)
    if len(values) == 0:
        return references
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        queue: deque[int] = deque()
        for position in positions:
            cutoff = int(times[position]) - int(window_ms)
            while queue and int(times[queue[0]]) < cutoff:
                queue.popleft()
            value = max(float(values[position]), float(minimum_reference))
            while queue and float(values[queue[-1]]) <= value:
                queue.pop()
            queue.append(int(position))
            references[position] = max(float(values[queue[0]]), float(minimum_reference))
    return references


def causal_depth_scale_arrays(
    frame: pd.DataFrame,
    *,
    depth_column: str,
    config: CausalDepthScaleConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ratio, reference, snapshot_high)`` in input-row order.

    The function is allocation-conscious and does not copy the complete input
    DataFrame. Bid and ask share one scale by default, matching the single
    heatmap legend.  Set ``separate_sides=True`` for diagnostics only.
    """

    cfg = config or CausalDepthScaleConfig()
    cfg.validate()
    count = 0 if frame is None else len(frame)
    ratio_out = np.zeros(count, dtype=np.float64)
    reference_out = np.full(count, float(cfg.minimum_reference), dtype=np.float64)
    snapshot_out = np.zeros(count, dtype=np.float64)
    if frame is None or frame.empty:
        return ratio_out, reference_out, snapshot_out
    if depth_column not in frame.columns:
        raise ValueError(f"depth scaling requires {depth_column}")

    time_column = infer_time_column(frame)
    times_float = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=np.float64)
    depths = pd.to_numeric(frame[depth_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    sides = _side_codes(frame)
    valid = np.isfinite(times_float) & np.isin(sides, (-1, 1))
    if not bool(valid.any()):
        return ratio_out, reference_out, snapshot_out

    valid_rows = np.flatnonzero(valid)
    times = times_float[valid].astype(np.int64, copy=False)
    groups = sides[valid].astype(np.int8, copy=False) if cfg.separate_sides else np.zeros(len(valid_rows), dtype=np.int8)
    valid_depths = depths[valid]

    key_dtype = np.dtype([("group", np.int8), ("time", np.int64)])
    keys = np.empty(len(valid_rows), dtype=key_dtype)
    keys["group"] = groups
    keys["time"] = times
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    sorted_codes = inverse[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1, len(order)]
    snapshot_high = np.empty(len(unique_keys), dtype=np.float64)
    quantile = float(cfg.snapshot_reference_quantile)
    for index in range(len(unique_keys)):
        rows = order[boundaries[index] : boundaries[index + 1]]
        values = valid_depths[rows]
        snapshot_high[index] = float(np.quantile(values, quantile)) if len(values) else 0.0

    unique_groups = unique_keys["group"].astype(np.int8, copy=False)
    unique_times = unique_keys["time"].astype(np.int64, copy=False)
    window_ms = max(1, int(round(float(cfg.window_hours) * 3_600_000.0)))
    references = _rolling_max_by_group(
        unique_times,
        unique_groups,
        snapshot_high,
        window_ms=window_ms,
        minimum_reference=float(cfg.minimum_reference),
    )

    row_references = references[inverse]
    row_snapshot = snapshot_high[inverse]
    row_ratios = np.divide(
        valid_depths,
        row_references,
        out=np.zeros_like(valid_depths, dtype=np.float64),
        where=row_references > 0,
    )
    ratio_out[valid_rows] = np.clip(row_ratios, 0.0, 1.0)
    reference_out[valid_rows] = np.maximum(row_references, float(cfg.minimum_reference))
    snapshot_out[valid_rows] = np.maximum(row_snapshot, 0.0)
    return ratio_out, reference_out, snapshot_out


def attach_causal_depth_scale(
    frame: pd.DataFrame,
    *,
    depth_column: str,
    config: CausalDepthScaleConfig | None = None,
    ratio_column: str = "causal_depth_ratio",
    reference_column: str = "causal_depth_reference",
    snapshot_max_column: str = "snapshot_side_max_depth",
) -> pd.DataFrame:
    """Attach causal depth-scale fields while preserving input order/index."""

    out = frame.copy() if frame is not None else pd.DataFrame()
    ratio, reference, snapshot = causal_depth_scale_arrays(
        out,
        depth_column=depth_column,
        config=config,
    )
    out[ratio_column] = ratio
    out[reference_column] = reference
    out[snapshot_max_column] = snapshot
    return out


__all__ = [
    "CausalDepthScaleConfig",
    "attach_causal_depth_scale",
    "causal_depth_scale_arrays",
    "infer_time_column",
]
