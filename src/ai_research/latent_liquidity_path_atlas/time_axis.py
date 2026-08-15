#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Datetime-axis normalization for the latent-liquidity path atlas.

Pandas ``merge_asof`` requires both keys to have the exact same datetime
resolution. SQLite/Arrow-backed loaders can return ``datetime64[us]`` while
``date_range`` and generated event axes are commonly ``datetime64[ns]``.
All internal research axes are therefore normalized to timezone-naive UTC
``datetime64[ns]`` before alignment.
"""
from __future__ import annotations

import pandas as pd


def as_datetime_ns(
    values: pd.Series | pd.Index,
    *,
    errors: str = "raise",
) -> pd.Series | pd.DatetimeIndex:
    """Return timezone-naive UTC values with exact ``datetime64[ns]`` dtype."""
    converted = pd.to_datetime(values, errors=errors)
    if isinstance(converted, pd.DatetimeIndex):
        if converted.tz is not None:
            converted = converted.tz_convert("UTC").tz_localize(None)
        return pd.DatetimeIndex(
            converted.to_numpy(dtype="datetime64[ns]"),
            name=converted.name,
        )

    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.Series(
        converted.to_numpy(dtype="datetime64[ns]"),
        index=converted.index,
        name=converted.name,
    )
