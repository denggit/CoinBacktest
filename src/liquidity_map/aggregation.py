#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable aggregation for offline order-book liquidity heatmap cells.

The prebuilder stores one canonical heatmap resolution (normally 60 seconds).
This module derives coarser display/research timeframes at query time so the
project never needs one artifact copy per chart timeframe.

Depth is a state variable and is therefore time-weighted averaged.  Flow fields
(added/removed/executed/cancelled/consumed/replenished) are interval totals and
are summed.  Color normalization deliberately stays outside this module because
it is a presentation concern rather than a backtest feature.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([smhdSMHD])\s*$")
_FLOW_COLUMNS = (
    "added_base",
    "removed_base",
    "executed_base",
    "cancelled_base",
    "consumed_base",
    "replenished_base",
)


def timeframe_to_seconds(value: Any) -> int:
    """Convert a chart timeframe such as ``5m`` or ``1H`` to seconds."""

    if isinstance(value, (int, np.integer)):
        seconds = int(value)
        if seconds <= 0:
            raise ValueError("timeframe seconds must be > 0")
        return seconds
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        seconds = int(value)
        if seconds <= 0:
            raise ValueError("timeframe seconds must be > 0")
        return seconds
    text = str(value or "").strip()
    aliases = {
        "1min": "1m",
        "3min": "3m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "1H",
        "1hour": "1H",
        "4hour": "4H",
    }
    text = aliases.get(text.lower(), text)
    match = _TIMEFRAME_RE.match(text)
    if not match:
        raise ValueError(f"unsupported timeframe: {value!r}")
    amount = int(match.group(1))
    unit = match.group(2).lower()
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    seconds = amount * multiplier
    if seconds <= 0:
        raise ValueError("timeframe seconds must be > 0")
    return seconds


def seconds_to_timeframe(seconds: int) -> str:
    seconds = int(seconds)
    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}H"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def infer_heatmap_seconds(frame: pd.DataFrame, fallback: int | None = None) -> int:
    """Infer the canonical source bucket width from frame attrs or columns."""

    attr_value = frame.attrs.get("heatmap_seconds")
    if attr_value is not None:
        seconds = int(attr_value)
        if seconds > 0:
            return seconds
    if {"bucket_start_ms", "bucket_end_ms"}.issubset(frame.columns) and not frame.empty:
        durations = (
            pd.to_numeric(frame["bucket_end_ms"], errors="coerce")
            - pd.to_numeric(frame["bucket_start_ms"], errors="coerce")
        )
        durations = durations.loc[durations > 0].dropna().astype("int64")
        if not durations.empty:
            # The store uses one canonical interval. Median is robust to a
            # partially clipped diagnostic row while still detecting the base.
            seconds = int(round(float(durations.median()) / 1000.0))
            if seconds > 0:
                return seconds
    if fallback is not None and int(fallback) > 0:
        return int(fallback)
    raise ValueError("cannot infer source heatmap interval")


def aggregate_heatmap_cells(
    frame: pd.DataFrame,
    *,
    target_seconds: int | str,
    source_price_step: float | None = None,
    target_price_step: float | None = None,
) -> pd.DataFrame:
    """Aggregate canonical heatmap cells to a coarser time/price grid.

    Parameters
    ----------
    frame:
        Canonical cells returned by :class:`OKXLiquidityMapLoader`.
    target_seconds:
        Requested output interval. It must be no finer than, and an integer
        multiple of, the canonical source interval.
    source_price_step / target_price_step:
        Price-bin widths. A target finer than the source is impossible and is
        therefore promoted to the source width rather than fabricating detail.
    """

    if frame is None or frame.empty:
        out = pd.DataFrame()
        if frame is not None:
            out.attrs.update(frame.attrs)
        return out

    target_seconds_int = timeframe_to_seconds(target_seconds)
    source_seconds = infer_heatmap_seconds(frame)
    if target_seconds_int < source_seconds:
        raise ValueError(
            f"target timeframe {seconds_to_timeframe(target_seconds_int)} is finer than "
            f"canonical heatmap {seconds_to_timeframe(source_seconds)}"
        )
    if target_seconds_int % source_seconds:
        raise ValueError(
            f"target timeframe {seconds_to_timeframe(target_seconds_int)} must be an integer multiple of "
            f"canonical heatmap {seconds_to_timeframe(source_seconds)}"
        )

    source_step = float(source_price_step or frame.attrs.get("price_step") or 1.0)
    if not math.isfinite(source_step) or source_step <= 0:
        raise ValueError("source_price_step must be > 0")
    requested_step = source_step if target_price_step is None else float(target_price_step)
    if not math.isfinite(requested_step) or requested_step <= 0:
        raise ValueError("target_price_step must be > 0")
    effective_step = max(source_step, requested_step)
    ratio = effective_step / source_step
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"target price step {effective_step:g} must be an integer multiple of source step {source_step:g}"
        )

    required = {
        "bucket_start_ms",
        "bucket_end_ms",
        "side_code",
        "depth_base",
        "depth_usd",
        "order_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"heatmap frame missing columns: {missing}")

    target_ms = target_seconds_int * 1000
    work = frame.copy()
    start_ms = pd.to_numeric(work["bucket_start_ms"], errors="coerce")
    end_ms = pd.to_numeric(work["bucket_end_ms"], errors="coerce")
    work = work.loc[start_ms.notna() & end_ms.notna()].copy()
    if work.empty:
        return work
    start_ms = pd.to_numeric(work["bucket_start_ms"], errors="raise").astype("int64")
    end_ms = pd.to_numeric(work["bucket_end_ms"], errors="raise").astype("int64")
    duration_ms = (end_ms - start_ms).clip(lower=0, upper=source_seconds * 1000)
    work["time_bin_ms"] = (start_ms // target_ms) * target_ms

    if "price_low" in work.columns:
        price_low = pd.to_numeric(work["price_low"], errors="coerce")
    elif "price_index" in work.columns:
        price_low = pd.to_numeric(work["price_index"], errors="coerce") * source_step
    else:
        raise ValueError("heatmap frame needs price_low or price_index")
    work = work.loc[price_low.notna()].copy()
    price_low = price_low.loc[work.index]
    work["price_index_out"] = np.floor(price_low / effective_step + 1e-12).astype("int64")

    for column in ("depth_base", "depth_usd", "order_count"):
        values = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        work[f"_{column}_weighted"] = values * duration_ms.loc[work.index] / target_ms
    for column in _FLOW_COLUMNS:
        if column not in work.columns:
            work[column] = 0.0
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    if "flow_valid" not in work.columns:
        work["flow_valid"] = 0
    work["flow_valid"] = pd.to_numeric(work["flow_valid"], errors="coerce").fillna(0).astype("int8")

    group_keys = ["time_bin_ms", "price_index_out", "side_code"]
    aggregations: dict[str, str] = {
        "_depth_base_weighted": "sum",
        "_depth_usd_weighted": "sum",
        "_order_count_weighted": "sum",
        "flow_valid": "min",
    }
    aggregations.update({column: "sum" for column in _FLOW_COLUMNS})
    grouped = work.groupby(group_keys, sort=True, observed=True).agg(aggregations).reset_index()
    grouped = grouped.rename(
        columns={
            "time_bin_ms": "bucket_start_ms",
            "price_index_out": "price_index",
            "_depth_base_weighted": "depth_base",
            "_depth_usd_weighted": "depth_usd",
            "_order_count_weighted": "order_count",
        }
    )
    grouped["bucket_start_ms"] = grouped["bucket_start_ms"].astype("int64")
    grouped["bucket_end_ms"] = grouped["bucket_start_ms"] + target_ms
    grouped["price_index"] = grouped["price_index"].astype("int64")
    grouped["side_code"] = grouped["side_code"].astype("int8")
    grouped["side"] = grouped["side_code"].map({1: "bid", -1: "ask"}).fillna("unknown")
    grouped["price_low"] = grouped["price_index"] * effective_step
    grouped["price_high"] = grouped["price_low"] + effective_step
    grouped["order_count"] = grouped["order_count"].round().clip(lower=0).astype("int64")
    side_max = grouped.groupby(["bucket_start_ms", "side_code"], observed=True)["depth_base"].transform("max")
    grouped["local_depth_ratio"] = np.where(side_max > 0, grouped["depth_base"] / side_max, 0.0)
    grouped = grouped.sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)
    grouped.attrs.update(frame.attrs)
    grouped.attrs["source_heatmap_seconds"] = source_seconds
    grouped.attrs["heatmap_seconds"] = target_seconds_int
    grouped.attrs["price_step"] = effective_step
    grouped.attrs["requested_price_step"] = requested_step
    return grouped
