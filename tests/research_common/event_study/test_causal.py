#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd

from src.research_common.event_study import add_available_time_index, causal_align_context


def test_add_available_time_index_shifts_bar_start_to_close_available_time() -> None:
    ctx = pd.DataFrame({"value": [10, 20]}, index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05"]))

    shifted = add_available_time_index(ctx, "5min")

    assert shifted.index.tolist() == [pd.Timestamp("2024-01-01 00:05"), pd.Timestamp("2024-01-01 00:10")]


def test_causal_align_context_does_not_forward_fill_unclosed_high_tf_bar() -> None:
    primary = pd.DataFrame(index=pd.date_range("2024-01-01 00:00", periods=7, freq="1min"))
    context = pd.DataFrame({"ctx_value": [100]}, index=pd.to_datetime(["2024-01-01 00:00"]))

    merged = causal_align_context(primary, context, timeframe="5min")

    assert pd.isna(merged.loc[pd.Timestamp("2024-01-01 00:04"), "ctx_value"])
    assert merged.loc[pd.Timestamp("2024-01-01 00:05"), "ctx_value"] == 100
    assert merged.loc[pd.Timestamp("2024-01-01 00:06"), "ctx_value"] == 100
