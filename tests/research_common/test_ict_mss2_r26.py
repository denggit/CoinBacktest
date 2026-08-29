from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r26 import (
    R26Config,
    build_positioning_leadership_events,
    build_positioning_leadership_paths,
    prepare_ratio_alignment,
)


def _bars(periods: int = 240, start: str = "2023-01-01 00:00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    frame = pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close.copy(),
            "volume": np.ones(periods),
        },
        index=index,
    )
    return frame


def _metrics(times: list[str], top: list[float], broad: list[float]) -> pd.DataFrame:
    timestamp = pd.to_datetime(times)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_time": timestamp + pd.Timedelta(minutes=1),
            "top_trader_position_long_share": top,
            "global_account_long_share": broad,
        }
    )


def test_r26_physically_rejects_embargo_metrics() -> None:
    bars = _bars(start="2025-06-30 20:00:00")
    metrics = _metrics(
        ["2025-06-30 23:55:00", "2025-07-01 00:00:00"],
        [0.49, 0.51],
        [0.50, 0.50],
    )
    with pytest.raises(RuntimeError, match="embargo or holdout metrics"):
        prepare_ratio_alignment(bars, metrics)


def test_r26_long_cross_requires_later_price_confirmation() -> None:
    bars = _bars()
    # A prior-hour high remains above the eventual confirmation entry target.
    bars.loc["2023-01-01 00:30:00", "high"] = 101.0
    # The 01:05-left-labelled five-minute bar ends at 01:10 and confirms above
    # the immediately prior five-minute high, after the 01:05 metric cross.
    bars.loc["2023-01-01 01:05:00":"2023-01-01 01:09:00", ["open", "close"]] = 100.30
    bars.loc["2023-01-01 01:05:00":"2023-01-01 01:09:00", "high"] = 100.40
    bars.loc["2023-01-01 01:05:00":"2023-01-01 01:09:00", "low"] = 99.95
    metrics = _metrics(
        ["2023-01-01 01:00:00", "2023-01-01 01:05:00", "2023-01-01 01:10:00"],
        [0.49, 0.51, 0.52],
        [0.50, 0.50, 0.50],
    )

    events, seal, engineering = build_positioning_leadership_events(bars, metrics)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["direction"] == "Long"
    assert event["cross_metric_time"] == pd.Timestamp("2023-01-01 01:05:00")
    assert event["confirmation_metric_time"] == pd.Timestamp("2023-01-01 01:10:00")
    assert event["signal_available_time"] == pd.Timestamp("2023-01-01 01:11:00")
    assert event["entry_time"] == pd.Timestamp("2023-01-01 01:11:00")
    assert int(seal.loc[seal["check"].eq("holdout_unsealed"), "value"].iloc[0]) == 0
    assert int(engineering.loc[engineering["check"].eq("raw_long_crosses"), "value"].iloc[0]) == 1


def test_r26_recross_cancels_pending_episode() -> None:
    bars = _bars()
    bars.loc["2023-01-01 01:05:00":"2023-01-01 01:09:00", "close"] = 100.50
    bars.loc["2023-01-01 01:05:00":"2023-01-01 01:09:00", "high"] = 100.60
    metrics = _metrics(
        ["2023-01-01 01:00:00", "2023-01-01 01:05:00", "2023-01-01 01:10:00"],
        [0.49, 0.51, 0.49],
        [0.50, 0.50, 0.50],
    )
    events, _, engineering = build_positioning_leadership_events(bars, metrics)
    assert events.empty
    assert int(engineering.loc[engineering["check"].eq("recrossed_episodes"), "value"].iloc[0]) == 1


def _event(setup_id: str, entry_time: str, *, target: float = 101.0, stop: float = 99.0) -> dict[str, object]:
    return {
        "setup_id": setup_id,
        "direction": "Long",
        "trade_direction": 1,
        "research_split": "discovery",
        "setup_status": "executable",
        "cross_available_time": pd.Timestamp(entry_time) - pd.Timedelta(minutes=10),
        "signal_available_time": pd.Timestamp(entry_time),
        "entry_time": pd.Timestamp(entry_time),
        "entry_price": 100.0,
        "stop_price": stop,
        "risk_distance_pct": (100.0 - stop) / 100.0,
        "structural_target_price": target,
        "structural_runway_pct": target / 100.0 - 1.0,
        "structural_reward_risk": (target - 100.0) / (100.0 - stop),
        "cross_relative_spread": 0.01,
        "confirmation_relative_spread": 0.02,
        "confirmation_delay_minutes": 5.0,
    }


def test_r26_same_bar_ambiguity_is_stop_first() -> None:
    bars = _bars(periods=30)
    bars.loc["2023-01-01 00:10:00", "high"] = 101.5
    bars.loc["2023-01-01 00:10:00", "low"] = 98.5
    events = pd.DataFrame([_event("A", "2023-01-01 00:10:00")])
    paths = build_positioning_leadership_paths(
        bars,
        events,
        config=R26Config(path_horizon_minutes=10),
    )
    primary = paths.loc[paths["target_model"].eq("H0_CROSS_TIME_1H_RANGE")].iloc[0]
    assert primary["outcome"] == "sl_first"
    assert primary["exit_time"] == pd.Timestamp("2023-01-01 00:10:00")
    assert primary["exit_price"] == 99.0


def test_r26_position_selection_skips_overlapping_signal() -> None:
    bars = _bars(periods=40)
    events = pd.DataFrame(
        [
            _event("A", "2023-01-01 00:10:00", target=110.0, stop=90.0),
            _event("B", "2023-01-01 00:11:00", target=110.0, stop=90.0),
        ]
    )
    paths = build_positioning_leadership_paths(
        bars,
        events,
        config=R26Config(path_horizon_minutes=10, max_stop_distance_pct=0.20),
    )
    primary = paths.loc[paths["target_model"].eq("H0_CROSS_TIME_1H_RANGE")].sort_values("entry_time")
    assert primary["position_selected"].tolist() == [True, False]
    assert primary.iloc[1]["overlap_skip_reason"] == "prior_same_direction_model_position_open"
