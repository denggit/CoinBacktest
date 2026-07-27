#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input quality checks for market-state research and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_state.models import DataQualityReport


REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def assess_market_state_input(df: pd.DataFrame) -> DataQualityReport:
    if df is None:
        df = pd.DataFrame()
    rows = int(len(df))
    warnings: list[str] = []
    missing_columns = [column for column in REQUIRED_OHLCV if column not in df.columns]
    if missing_columns:
        warnings.append(f"missing columns: {missing_columns}")
        return DataQualityReport(
            rows=rows,
            usable_rows=0,
            duplicate_timestamps=0,
            missing_ohlcv_rows=rows,
            invalid_price_rows=rows,
            monotonic_increasing=True,
            median_interval_seconds=None,
            irregular_interval_ratio=None,
            score=0.0,
            usable=False,
            warnings=tuple(warnings),
        )

    index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
    duplicate_timestamps = int(index.duplicated(keep=False).sum())
    monotonic = bool(index.is_monotonic_increasing)

    numeric = df.loc[:, REQUIRED_OHLCV].apply(pd.to_numeric, errors="coerce")
    missing_mask = numeric.isna().any(axis=1)
    invalid_price_mask = (
        (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
    )
    missing_rows = int(missing_mask.sum())
    invalid_rows = int(invalid_price_mask.sum())
    usable_rows = int((~missing_mask & ~invalid_price_mask).sum())

    valid_index = index[~index.isna()]
    median_interval: float | None = None
    irregular_ratio: float | None = None
    if len(valid_index) >= 3:
        diffs = pd.Series(valid_index).diff().dt.total_seconds().dropna()
        positive = diffs[diffs > 0]
        if not positive.empty:
            median_interval = float(positive.median())
            tolerance = max(1e-9, median_interval * 0.05)
            irregular_ratio = float((positive.sub(median_interval).abs() > tolerance).mean())

    penalties = 0.0
    if rows:
        penalties += 0.40 * min(1.0, missing_rows / rows)
        penalties += 0.35 * min(1.0, invalid_rows / rows)
        penalties += 0.15 * min(1.0, duplicate_timestamps / rows)
    if not monotonic:
        penalties += 0.10
    score = float(np.clip(1.0 - penalties, 0.0, 1.0))

    if duplicate_timestamps:
        warnings.append(f"duplicate timestamps: {duplicate_timestamps}")
    if not monotonic:
        warnings.append("timestamps are not monotonic increasing")
    if missing_rows:
        warnings.append(f"rows with missing OHLCV: {missing_rows}")
    if invalid_rows:
        warnings.append(f"rows with invalid OHLC geometry: {invalid_rows}")

    usable = rows > 0 and usable_rows >= 2 and score >= 0.50
    return DataQualityReport(
        rows=rows,
        usable_rows=usable_rows,
        duplicate_timestamps=duplicate_timestamps,
        missing_ohlcv_rows=missing_rows,
        invalid_price_rows=invalid_rows,
        monotonic_increasing=monotonic,
        median_interval_seconds=median_interval,
        irregular_interval_ratio=irregular_ratio,
        score=score,
        usable=usable,
        warnings=tuple(warnings),
    )
