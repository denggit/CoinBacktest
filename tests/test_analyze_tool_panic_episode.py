from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_tool.plugins.panic_selloff_recovery import PanicSelloffRecoveryPlugin
from research.liquidity.panic_selloff_rejection_recovery_long.panic_episode import (
    PanicEpisodeConfig,
    detect_panic_episodes,
)


def _sample() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=120, freq="1min")
    close = np.full(120, 100.0)
    for i in range(1, 70):
        close[i] = close[i - 1] * (1 + (0.0001 if i % 2 else -0.00008))
    close[70:85] = [99.90, 99.75, 99.58, 99.40, 99.20, 98.95, 98.72, 98.66, 98.70, 98.86, 99.05, 99.24, 99.42, 99.55, 99.70]
    close[85:] = np.linspace(99.72, 100.4, 35)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0004
    low = np.minimum(open_, close) * 0.9996
    low[76] = 98.48
    volume = np.full(120, 100.0)
    volume[70:77] = [150, 170, 190, 220, 260, 320, 400]
    volume[77:85] = [300, 240, 180, 160, 150, 140, 130, 120]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_detector_builds_multi_stage_causal_episode() -> None:
    result = detect_panic_episodes(_sample(), PanicEpisodeConfig())
    assert result.signal_count == 1
    episode = result.episodes[0]
    kinds = [node.kind for node in episode.nodes]
    assert kinds == ["start", "acceleration", "low_candidate", "exhaustion", "signal"]
    assert episode.start_time < episode.signal_time
    signal = episode.nodes[-1]
    assert signal.fields["signal_is_causal"] is True
    assert signal.fields["outcome_return_15b"] > 0


def test_plugin_returns_regions_and_green_signal_marker() -> None:
    result = PanicSelloffRecoveryPlugin().run(_sample(), {})
    assert len(result.regions) == 1
    assert result.regions[0].status == "signal"
    signals = [marker for marker in result.markers if marker.role == "signal"]
    assert len(signals) == 1
    assert signals[0].color == "#22c55e"
    assert result.summary["signal_count"] == 1
