from __future__ import annotations

import numpy as np

from src.ai_research.rl_market_agent.opportunity import (
    TradeTemplate, conservative_template_returns, feature_groups,
)


def test_conservative_target_marks_both_hit_as_stop():
    template = TradeTemplate("T", 60, 0.01, 0.005)
    names = (
        "h60__final_return", "h60__long_mfe", "h60__long_mae",
        "h60__short_mfe", "h60__short_mae",
    )
    labels = np.array([[0.003, 0.02, -0.01, 0.02, -0.01]], dtype=np.float32)
    long_y, short_y = conservative_template_returns(labels, names, template, round_trip_cost=0.0011)
    assert np.isclose(long_y[0], -0.0061)
    assert np.isclose(short_y[0], -0.0061)


def test_feature_groups_exclude_availability_flags():
    groups = feature_groups([
        "kline_5m__x", "trade_1m__x", "range_r0020__x", "availability__trade_1m"
    ])
    assert groups["KLINE_ONLY"] == ("kline_5m__x",)
    assert groups["KLINE_TRADE"] == ("kline_5m__x", "trade_1m__x")
    assert "availability__trade_1m" not in groups["FULL"]
