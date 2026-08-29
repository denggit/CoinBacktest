from __future__ import annotations

import pandas as pd

from src.research_common.ict.premarket_mss_fvg import make_synthetic_ict_day
from src.research_common.ict.premarket_mss_fvg_v2 import (
    SweepEpisodeConfig,
    build_all_premarket_levels_v2,
    build_sweep_events_v2,
)
from src.research_common.ict import premarket_mss_fvg_v4 as v4


def _synthetic_inputs():
    bars = make_synthetic_ict_day()
    day = pd.Timestamp("2026-06-02").date()
    levels = build_all_premarket_levels_v2(
        bars,
        [day],
        pivot_left=2,
        pivot_right=2,
        episode_config=SweepEpisodeConfig(),
    )
    sweeps = build_sweep_events_v2(bars, levels)
    return bars, sweeps


def test_r05_does_not_gate_on_speed_ratio(monkeypatch) -> None:
    bars, sweeps = _synthetic_inputs()

    original = v4._select_inbound_anchor

    def deliberately_fast_inbound(*args, **kwargs):
        out = original(*args, **kwargs)
        # Make the inbound leg mechanically much faster/larger than the reversal
        # so displacement_speed_ratio falls below 1. R05 must still emit the
        # setup because speed is a discovery feature, not an entry condition.
        out = dict(out)
        out["anchor_price"] = 200.0
        return out

    monkeypatch.setattr(v4, "_select_inbound_anchor", deliberately_fast_inbound)
    attempts, _ = v4.build_signal_attempts_v4(
        bars,
        sweeps,
        config=v4.ICTDisplacementDiscoveryConfig(execution_timeframes=(1,)),
    )
    assert not attempts.empty
    assert (attempts["displacement_speed_ratio"] < 1.0).any()


def test_fvg_may_complete_after_mss_and_signal_waits_for_it() -> None:
    bars, sweeps = _synthetic_inputs()
    attempts, _ = v4.build_signal_attempts_v4(
        bars,
        sweeps,
        config=v4.ICTDisplacementDiscoveryConfig(execution_timeframes=(5,)),
    )
    assert not attempts.empty
    row = attempts.iloc[0]
    assert row["fvg_relation_to_mss"] == "mss_bar_is_fvg_middle"
    assert pd.Timestamp(row["fvg_available_time"]) > pd.Timestamp(row["mss_time"])
    assert pd.Timestamp(row["signal_time"]) == pd.Timestamp(row["fvg_available_time"])


def test_post_terminal_pivot_remains_valid_reference() -> None:
    tz = "America/New_York"
    pivots = pd.DataFrame(
        [
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 08:56", tz=tz),
                "pivot_price": 105.0,
                "confirmation_available_time": pd.Timestamp("2026-06-02 08:58", tz=tz),
            },
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 09:06", tz=tz),
                "pivot_price": 101.5,
                "confirmation_available_time": pd.Timestamp("2026-06-02 09:08", tz=tz),
            },
        ]
    )
    ref = v4._select_mss_reference(
        pivots,
        side="high",
        sweep_bar_start=pd.Timestamp("2026-06-02 09:00", tz=tz),
        terminal_available_time=pd.Timestamp("2026-06-02 09:03", tz=tz),
        signal_available_time=pd.Timestamp("2026-06-02 09:10", tz=tz),
    )
    assert ref is not None
    assert pd.Timestamp(ref["pivot_time"]) == pd.Timestamp("2026-06-02 09:06", tz=tz)
    assert ref["reference_relation"] == "post_terminal_dynamic"


def test_r05_datetime_resolution_does_not_drop_all_sweeps(monkeypatch) -> None:
    """Pandas 3 can use us-resolution indexes; ns/us must not be mixed."""
    bars, sweeps = _synthetic_inputs()
    original = v4.aggregate_closed_bars

    def aggregate_with_us_resolution(frame, timeframe_minutes):
        out = original(frame, timeframe_minutes).copy()
        out.index = pd.DatetimeIndex(out.index).as_unit("us")
        out["available_time"] = pd.DatetimeIndex(pd.to_datetime(out["available_time"])).as_unit("us")
        return out

    monkeypatch.setattr(v4, "aggregate_closed_bars", aggregate_with_us_resolution)
    attempts, funnel = v4.build_signal_attempts_v4(
        bars,
        sweeps,
        config=v4.ICTDisplacementDiscoveryConfig(execution_timeframes=(1,)),
    )
    assert not funnel.empty
    assert not attempts.empty
