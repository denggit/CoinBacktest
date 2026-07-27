#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalized data bundle used by the reusable market-state engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.market_state.causal_alignment import available_time_index
from src.market_state.data_quality import REQUIRED_OHLCV, assess_market_state_input
from src.market_state.models import DataQualityReport


@dataclass(frozen=True)
class MarketStateDataBundle:
    primary: pd.DataFrame
    available_times: pd.DatetimeIndex
    source: str
    timestamp_semantics: str
    bar_duration: pd.Timedelta | None
    data_quality: DataQualityReport
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        timestamp_semantics: str = "bar_end",
        bar_duration: pd.Timedelta | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MarketStateDataBundle":
        report = assess_market_state_input(df)
        if not report.usable:
            raise ValueError(f"market-state input is not usable: {report.warnings}")

        work = df.copy()
        work.index = pd.DatetimeIndex(pd.to_datetime(work.index))
        work = work[~work.index.duplicated(keep="last")].sort_index()
        for column in REQUIRED_OHLCV:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(subset=list(REQUIRED_OHLCV))
        valid_geometry = (
            (work[["open", "high", "low", "close"]] > 0).all(axis=1)
            & (work["high"] >= work["low"])
            & (work["high"] >= work[["open", "close"]].max(axis=1))
            & (work["low"] <= work[["open", "close"]].min(axis=1))
        )
        work = work.loc[valid_geometry]
        if len(work) < 2:
            raise ValueError("market-state input has fewer than two valid rows")

        delta = None if bar_duration is None else pd.Timedelta(bar_duration)
        available = available_time_index(
            work.index,
            bar_duration=delta,
            timestamp_semantics=timestamp_semantics,
        )
        return cls(
            primary=work,
            available_times=available,
            source=str(source),
            timestamp_semantics=str(timestamp_semantics),
            bar_duration=delta,
            data_quality=report,
            metadata=dict(metadata or {}),
        )
