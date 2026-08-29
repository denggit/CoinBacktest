from __future__ import annotations

import pandas as pd

from src.ai_research.rl_market_agent.causal import align_left_labeled_bars, causal_audit


def test_left_labeled_bar_is_not_visible_before_close():
    source = pd.DataFrame({"x": [10.0, 20.0]}, index=pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:05"]))
    decisions = pd.to_datetime(["2026-01-01 10:04", "2026-01-01 10:05", "2026-01-01 10:09", "2026-01-01 10:10"])
    out = align_left_labeled_bars(decisions, source, bar_duration="5min")
    assert pd.isna(out.features.iloc[0]["x"])
    assert out.features.iloc[1]["x"] == 10.0
    assert out.features.iloc[2]["x"] == 10.0
    assert out.features.iloc[3]["x"] == 20.0
    assert causal_audit(decisions, out.source_available_time)["passed"] is True


def test_tolerance_prevents_indefinitely_stale_regular_bar():
    source = pd.DataFrame({"x": [10.0]}, index=pd.to_datetime(["2026-01-01 10:00"]))
    decisions = pd.to_datetime(["2026-01-01 10:05", "2026-01-01 10:30"])
    out = align_left_labeled_bars(decisions, source, bar_duration="5min", tolerance="10min")
    assert out.features.iloc[0]["x"] == 10.0
    assert pd.isna(out.features.iloc[1]["x"])
