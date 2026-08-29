from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r25 import (
    build_range_run_events,
    range_source_audit,
    simulate_range_run_reversal,
)


def _range_frame(*, confirmation_end: str = "2023-01-01 00:00:50") -> pd.DataFrame:
    ends = [
        pd.Timestamp("2023-01-01 00:00:10"),
        pd.Timestamp("2023-01-01 00:00:20"),
        pd.Timestamp("2023-01-01 00:00:30"),
        pd.Timestamp("2023-01-01 00:00:40"),
        pd.Timestamp(confirmation_end),
    ]
    opens = [100.0, 99.8, 99.6, 99.4, 99.2]
    closes = [99.8, 99.6, 99.4, 99.2, 99.4]
    return pd.DataFrame(
        {
            "bar_id": [1, 2, 3, 4, 5],
            "start_ts": [value - pd.Timedelta(seconds=8) for value in ends],
            "end_ts": ends,
            "duration_seconds": [8.0] * 5,
            "open": opens,
            "high": [100.0, 99.8, 99.6, 99.4, 99.45],
            "low": [99.75, 99.55, 99.35, 99.15, 99.10],
            "close": closes,
            "direction": [-1, -1, -1, -1, 1],
        }
    )


def _minutes(index: list[str], opens: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": opens},
        index=pd.to_datetime(index),
    )


def test_r25_sequence_membership_uses_completed_order() -> None:
    frame = _range_frame()
    same_end = pd.Timestamp("2023-01-01 00:00:40")
    frame.loc[3, "end_ts"] = same_end
    frame.loc[4, "end_ts"] = same_end
    frame.loc[4, "start_ts"] = same_end
    events = build_range_run_events(frame)
    assert len(events) == 1
    row = events.iloc[0]
    assert int(row["run_bars"]) == 4
    assert int(row["run_end_bar_id"]) == 4
    assert int(row["confirmation_bar_id"]) == 5
    assert pd.Timestamp(row["signal_time"]) == same_end


def test_r25_invalid_range_row_resets_run() -> None:
    frame = _range_frame()
    frame.loc[2, "start_ts"] = frame.loc[2, "end_ts"] + pd.Timedelta(seconds=1)
    events = build_range_run_events(frame)
    assert events.empty
    audit = range_source_audit(frame)
    assert int(audit.set_index("check").loc["start_after_end", "value"]) == 1


def test_r25_entry_is_strictly_after_completed_signal() -> None:
    events = build_range_run_events(_range_frame(confirmation_end="2023-01-01 00:01:00"))
    bars = _minutes(
        ["2023-01-01 00:01:00", "2023-01-01 00:02:00", "2023-01-01 00:03:00"],
        [99.5, 99.5, 99.7], [99.7, 99.8, 100.1], [99.3, 99.3, 99.6],
    )
    result = simulate_range_run_reversal(
        bars, events, direction=1, split="discovery",
        split_start=pd.Timestamp("2023-01-01 00:00:00"),
        split_end=pd.Timestamp("2023-01-01 00:04:00"),
    )
    entered = result.loc[result["path_status"].eq("included")].iloc[0]
    assert pd.Timestamp(entered["entry_time"]) == pd.Timestamp("2023-01-01 00:02:00")


def test_r25_next_observed_minute_survives_gap() -> None:
    events = build_range_run_events(_range_frame())
    bars = _minutes(
        ["2023-01-01 00:02:00", "2023-01-01 00:03:00", "2023-01-01 00:04:00"],
        [99.5, 99.7, 99.7], [99.8, 100.1, 99.9], [99.3, 99.6, 99.5],
    )
    result = simulate_range_run_reversal(
        bars, events, direction=1, split="discovery",
        split_start=pd.Timestamp("2023-01-01 00:00:00"),
        split_end=pd.Timestamp("2023-01-01 00:05:00"),
    )
    assert pd.Timestamp(result.loc[result["path_status"].eq("included"), "entry_time"].iloc[0]) == pd.Timestamp("2023-01-01 00:02:00")


def test_r25_stop_first_on_same_minute_ambiguity() -> None:
    events = build_range_run_events(_range_frame())
    bars = _minutes(
        ["2023-01-01 00:01:00", "2023-01-01 00:02:00", "2023-01-01 00:03:00"],
        [99.5, 99.5, 99.5], [100.2, 99.8, 99.8], [99.0, 99.3, 99.3],
    )
    result = simulate_range_run_reversal(
        bars, events, direction=1, split="discovery",
        split_start=pd.Timestamp("2023-01-01 00:00:00"),
        split_end=pd.Timestamp("2023-01-01 00:04:00"),
    )
    trade = result.loc[result["path_status"].eq("included")].iloc[0]
    assert trade["exit_reason"] == "STOP"
    assert np.isclose(float(trade["exit_price"]), 99.1)


def test_r25_unresolved_path_is_censored_at_split_boundary() -> None:
    events = build_range_run_events(_range_frame())
    events.loc[:, "stop_price"] = 98.0
    events.loc[:, "target_price"] = 101.0
    bars = _minutes(
        ["2023-01-01 00:01:00", "2023-01-01 00:02:00", "2023-01-01 00:03:00"],
        [99.5, 99.5, 99.5], [99.7, 99.7, 99.7], [99.3, 99.3, 99.3],
    )
    result = simulate_range_run_reversal(
        bars, events, direction=1, split="discovery",
        split_start=pd.Timestamp("2023-01-01 00:00:00"),
        split_end=pd.Timestamp("2023-01-01 00:04:00"),
    )
    assert result.iloc[0]["path_status"] == "boundary_censored"
    assert pd.isna(result.iloc[0].get("exit_time", pd.NaT))
