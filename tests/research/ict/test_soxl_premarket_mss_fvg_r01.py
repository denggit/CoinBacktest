from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import (
    BASE_SCENARIO,
    NY_TZ,
    ResearchConfig,
    aggregate_closed_bars,
    build_all_premarket_levels,
    build_causal_audit,
    build_signal_attempts,
    build_sweep_events,
    eligible_ny_dates,
    enforce_single_lifecycle,
    make_synthetic_ict_day,
    replay_attempts,
    source_naive_to_new_york,
)


def _synthetic_pipeline():
    cfg = ResearchConfig()
    bars = make_synthetic_ict_day()
    day = bars.index[0].date()
    levels = build_all_premarket_levels(
        bars,
        [day],
        pivot_left=cfg.premarket_pivot_left,
        pivot_right=cfg.premarket_pivot_right,
    )
    sweeps = build_sweep_events(bars, levels)
    attempts = build_signal_attempts(bars, sweeps, config=cfg)
    base_1m = attempts.loc[
        attempts["displacement_body_mult"].eq(cfg.displacement_body_mult)
        & attempts["execution_tf_minutes"].eq(1)
    ].copy()
    lifecycle = replay_attempts(
        bars,
        base_1m,
        scenario=BASE_SCENARIO,
        round_trip_cost=cfg.round_trip_cost,
        risk_fraction=cfg.risk_fraction,
        max_notional_multiple=cfg.max_notional_multiple,
    )
    lifecycle, _ = enforce_single_lifecycle(lifecycle)
    return cfg, bars, levels, sweeps, attempts, lifecycle


def test_source_fixed_plus8_to_new_york_is_dst_aware() -> None:
    src = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-02 20:30:00")]),
    )
    ny = source_naive_to_new_york(src, source_offset_hours=8)
    assert ny.index.tz is not None
    assert str(ny.index.tz) == NY_TZ
    assert ny.index[0] == pd.Timestamp("2026-06-02 08:30:00", tz=NY_TZ)


def test_aggregate_closed_bars_exposes_available_time_only_after_close() -> None:
    idx = pd.date_range("2026-06-02 08:30", periods=5, freq="1min", tz=NY_TZ)
    bars = pd.DataFrame(
        {
            "open": np.arange(5.0),
            "high": np.arange(5.0) + 1.0,
            "low": np.arange(5.0) - 1.0,
            "close": np.arange(5.0) + 0.5,
        },
        index=idx,
    )
    out = aggregate_closed_bars(bars, 2)
    # 08:34 group is incomplete (only one source minute) and must be dropped.
    assert list(out.index) == [idx[0], idx[2]]
    assert list(pd.to_datetime(out["available_time"])) == [idx[0] + pd.Timedelta(minutes=2), idx[2] + pd.Timedelta(minutes=2)]


def test_weekends_and_us_equity_holidays_are_excluded_by_default() -> None:
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        index=pd.date_range("2026-07-02", "2026-07-06 23:59", freq="1h", tz=NY_TZ),
    )
    days = eligible_ny_dates(
        bars, start_date="2026-07-02", end_date="2026-07-06", exclude_equity_holidays=True
    )
    assert date(2026, 7, 3) not in days  # observed Independence Day holiday
    assert date(2026, 7, 4) not in days
    assert date(2026, 7, 5) not in days
    assert date(2026, 7, 2) in days
    assert date(2026, 7, 6) in days



def test_calendar_gate_does_not_hide_completely_missing_weekday() -> None:
    bars = make_synthetic_ict_day("2026-06-02")
    days = eligible_ny_dates(
        bars, start_date="2026-06-01", end_date="2026-06-03", exclude_equity_holidays=False
    )
    assert days == [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]

def test_synthetic_path_produces_strict_causal_mss_fvg_attempts() -> None:
    _, _, levels, sweeps, attempts, _ = _synthetic_pipeline()
    assert not levels.empty
    assert not sweeps.empty
    assert not attempts.empty
    assert attempts["strict_break_bar_fvg"].all()
    assert (pd.to_datetime(attempts["mss_reference_available_time"]) <= pd.to_datetime(attempts["sweep_time"])).all()
    assert (pd.to_datetime(attempts["signal_time"]) > pd.to_datetime(attempts["sweep_time"])).all()
    assert (attempts["planned_rr"] > 0).all()


def test_synthetic_lifecycle_and_causal_audit_pass() -> None:
    _, _, _, _, attempts, lifecycle = _synthetic_pipeline()
    assert not lifecycle.empty
    assert lifecycle["filled"].any()
    audit = build_causal_audit(attempts, lifecycle)
    assert not audit.empty
    assert audit["passed"].all(), audit.to_dict("records")


def test_research_implementation_does_not_bypass_data_feed() -> None:
    root = Path(__file__).resolve().parents[3]
    research_dir = root / "research" / "ict" / "soxl_premarket_mss_fvg"
    shared = root / "src" / "research_common" / "ict" / "premarket_mss_fvg.py"
    paths = list(research_dir.glob("*.py")) + [shared]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    forbidden = ["requests.get(", "sqlite3.connect(", "/api/v5/", "ccxt."]
    for token in forbidden:
        assert token not in combined
