#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r20 import (
    R20Config,
    build_r20_gate,
    prepare_r20_trades,
    r20_causal_audit,
    summarize_r20_components,
)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2023-01-02 04:00", "2024-12-31 20:00", "2025-01-02 04:00"]),
            "exit_time": pd.to_datetime(["2023-01-03 04:00", "2025-01-02 00:00", "2025-01-03 04:00"]),
            "type": ["LONG", "LONG", "SHORT"],
            "engine": ["BULL_RECLAIM_V2", "BULL_RECLAIM_V2", "BEAR_V3_ONLY"],
            "avg_entry": [100.0, 100.0, 100.0],
            "exit": [102.0, 101.0, 98.0],
            "units": [1, 1, 1],
            "note": ["STRUCTURAL_STOP", "STRUCTURAL_STOP", "STRUCTURAL_STOP"],
        }
    )


def test_prepare_r20_trades_applies_boundaries_and_costs() -> None:
    trades = prepare_r20_trades(_raw())
    assert list(trades["path_status"]) == ["included", "boundary_censored", "included"]
    assert np.isclose(trades.iloc[0]["gross_return"], 0.02)
    assert np.isclose(trades.iloc[0]["net_return_cost2x"], 0.017)
    assert np.isclose(trades.iloc[2]["gross_return"], 0.02)
    assert not trades["research_split"].isin(["embargo", "holdout"]).any()


def test_component_summary_and_gate_do_not_use_censored_trade() -> None:
    trades = prepare_r20_trades(_raw())
    score = summarize_r20_components(trades)
    assert int(score["trades"].sum()) == 2
    gate = build_r20_gate(score)
    assert not gate.empty
    assert gate["forward_incubation_eligible"].eq(0).all()


def test_r20_causal_audit_matches_prior_signal_and_engine() -> None:
    trades = prepare_r20_trades(_raw())
    signal_times = pd.to_datetime(trades["signal_time"])
    features = pd.DataFrame(index=signal_times, data={"signal": [1, 1, -1], "selected_engine": list(trades["engine"])})
    audit = r20_causal_audit(trades, features, config=R20Config())
    assert int(audit["violations"].sum()) == 0

