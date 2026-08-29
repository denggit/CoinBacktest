from __future__ import annotations

import pandas as pd
import pytest

from src.ai_research.rl_market_agent.labels import build_forward_path_labels


def test_forward_labels_use_decision_open_and_future_path_only():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="1min")
    bars = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 105, 104, 106],
            "low": [99, 100, 98, 102],
            "close": [100.5, 102, 103, 105],
        },
        index=idx,
    )
    out = build_forward_path_labels(bars, pd.DatetimeIndex([idx[0]]), [3])
    row = out.iloc[0]
    assert row["entry_price"] == 100
    assert row["h3__final_return"] == pytest.approx(0.03)
    assert row["h3__long_mfe"] == pytest.approx(0.05)
    assert row["h3__long_mae"] == pytest.approx(-0.02)
    assert row["h3__short_mfe"] == pytest.approx(0.02)
    assert row["h3__short_mae"] == pytest.approx(-0.05)
