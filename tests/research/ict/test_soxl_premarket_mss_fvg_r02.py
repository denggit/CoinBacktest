from __future__ import annotations

import pandas as pd

from src.research_common.ict.premarket_mss_fvg import ResearchConfig, make_synthetic_ict_day
from src.research_common.ict.premarket_mss_fvg_v2 import (
    _dynamic_reference,
    build_all_premarket_levels_v2,
    build_causal_audit_v2,
    build_signal_attempts_v2,
    build_sweep_events_v2,
)


def test_dynamic_reference_updates_after_sweep_when_newer_pivot_precedes_terminal() -> None:
    tz = "America/New_York"
    pivots = pd.DataFrame(
        [
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 08:40", tz=tz),
                "pivot_price": 105.0,
                "confirmation_available_time": pd.Timestamp("2026-06-02 08:42", tz=tz),
            },
            {
                "pivot_side": "high",
                "pivot_time": pd.Timestamp("2026-06-02 09:00", tz=tz),
                "pivot_price": 103.5,
                "confirmation_available_time": pd.Timestamp("2026-06-02 09:02", tz=tz),
            },
        ]
    )
    ref = _dynamic_reference(
        pivots,
        side="high",
        terminal_available_time=pd.Timestamp("2026-06-02 09:06", tz=tz),
        signal_available_time=pd.Timestamp("2026-06-02 09:10", tz=tz),
    )
    assert ref is not None
    assert pd.Timestamp(ref["pivot_time"]) == pd.Timestamp("2026-06-02 09:00", tz=tz)
    assert float(ref["pivot_price"]) == 103.5


def test_consumed_opposite_target_is_rejected_at_sweep() -> None:
    bars = make_synthetic_ict_day()
    # 08:35 NY occurs before the synthetic low sweep. Force the opposite
    # premarket high to be consumed first.
    premarket_high = float(bars.loc[(bars.index.hour < 8) | ((bars.index.hour == 8) & (bars.index.minute < 30)), "high"].max())
    bars.loc[pd.Timestamp("2026-06-02 08:35", tz="America/New_York"), "high"] = premarket_high + 1.0
    day = bars.index[0].date()
    levels = build_all_premarket_levels_v2(bars, [day], pivot_left=2, pivot_right=2)
    sweeps = build_sweep_events_v2(bars, levels)
    low_sweeps = sweeps.loc[sweeps["trade_side"] == "LONG"]
    assert not low_sweeps.empty
    assert (~low_sweeps["opposite_target_fresh_at_sweep"].astype(bool)).all()
    attempts = build_signal_attempts_v2(bars, sweeps, config=ResearchConfig())
    if not attempts.empty:
        assert not attempts["event_id"].isin(low_sweeps["event_id"]).any()


def test_r02_synthetic_pipeline_passes_dynamic_causal_audit() -> None:
    cfg = ResearchConfig()
    bars = make_synthetic_ict_day()
    day = bars.index[0].date()
    levels = build_all_premarket_levels_v2(bars, [day], pivot_left=2, pivot_right=2)
    sweeps = build_sweep_events_v2(bars, levels)
    attempts = build_signal_attempts_v2(bars, sweeps, config=cfg)
    base = attempts.loc[attempts["displacement_body_mult"].eq(cfg.displacement_body_mult)].copy()
    assert not base.empty
    assert (pd.to_datetime(base["mss_reference_time"]) < pd.to_datetime(base["episode_terminal_extreme_time"])).all()
    audit = build_causal_audit_v2(base, pd.DataFrame())
    assert audit["passed"].all(), audit.to_dict("records")
