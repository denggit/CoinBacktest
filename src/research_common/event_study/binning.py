#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature binning helpers for event-study condition research."""

from __future__ import annotations

import pandas as pd


def qcut_labels(values: pd.Series, *, q: int = 4, prefix: str = "Q", na_label: str = "NA") -> pd.Series:
    """Return stable quantile labels while tolerating small/constant samples."""
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series(na_label, index=values.index, dtype="object")
    valid = x.dropna()
    if len(valid) < int(q) or valid.nunique() < 2:
        return out
    try:
        binned = pd.qcut(valid, q=int(q), duplicates="drop")
    except ValueError:
        return out
    label_map = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(binned.cat.categories)}
    out.loc[valid.index] = binned.map(label_map).astype(str)
    return out


def fixed_threshold_labels(
    values: pd.Series,
    *,
    thresholds: list[float] | tuple[float, ...],
    labels: list[str] | tuple[str, ...] | None = None,
    na_label: str = "NA",
) -> pd.Series:
    """Label values by fixed thresholds using right-open buckets."""
    x = pd.to_numeric(values, errors="coerce")
    cuts = sorted(float(v) for v in thresholds)
    bucket_count = len(cuts) + 1
    if labels is None:
        labels = tuple(f"B{i + 1}" for i in range(bucket_count))
    if len(labels) != bucket_count:
        raise ValueError("labels length must be len(thresholds) + 1")
    out = pd.Series(na_label, index=values.index, dtype="object")
    valid = x.dropna()
    if valid.empty:
        return out
    bucket_ids = pd.cut(valid, bins=[float("-inf"), *cuts, float("inf")], labels=list(labels), include_lowest=True)
    out.loc[valid.index] = bucket_ids.astype(str)
    return out
