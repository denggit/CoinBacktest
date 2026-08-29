from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.strategy_research.eth_source_locked_portfolio.config import SourceLockedConfig
from src.strategy_research.eth_source_locked_portfolio.engine import run_target_schedule, run_turtle_system2


def _minute(start="2023-01-01 00:00:00", periods=120, price0=100.0, drift=0.0001):
    idx = pd.date_range(start, periods=periods, freq="1min")
    op = price0 * (1 + drift) ** np.arange(periods)
    close = op * (1 + drift)
    return pd.DataFrame({"open": op, "high": np.maximum(op, close), "low": np.minimum(op, close), "close": close}, index=idx)


def _cfg(end="2023-01-01 01:59:00"):
    return SourceLockedConfig(research_start="2023-01-01 00:00:00", research_end=end, sealed_start="2026-01-01 00:00:00")


def test_target_executes_strictly_after_signal_time() -> None:
    bars = _minute()
    sched = pd.DataFrame({"signal_time": [pd.Timestamp("2023-01-01 00:10:00")], "raw_target": [1.0]})
    r = run_target_schedule(bars, sched, _cfg(), "X")
    assert r.events.iloc[0].execution_time == pd.Timestamp("2023-01-01 00:11:00")


def test_flip_charges_two_way_turnover() -> None:
    bars = _minute()
    sched = pd.DataFrame({"signal_time": [pd.Timestamp("2023-01-01 00:10:00"), pd.Timestamp("2023-01-01 00:20:00")], "raw_target": [1.0, -1.0]})
    r = run_target_schedule(bars, sched, _cfg(), "X")
    assert np.isclose(r.events.iloc[1].turnover, 2.0)
    assert np.isclose(r.events.iloc[1].fee_fraction, 2.0 * 0.0011 / 2.0)


def test_delay_moves_execution_later() -> None:
    bars = _minute()
    sched = pd.DataFrame({"signal_time": [pd.Timestamp("2023-01-01 00:10:00")], "raw_target": [1.0]})
    r = run_target_schedule(bars, sched, _cfg(), "X", extra_delay_minutes=2)
    assert r.events.iloc[0].execution_time == pd.Timestamp("2023-01-01 00:13:00")


def test_turtle_context_not_usable_at_exact_available_minute() -> None:
    bars = _minute(periods=10, drift=0.0)
    bars.loc[:, ["open", "high", "low", "close"]] = 100.0
    bars.loc[pd.Timestamp("2023-01-01 00:01:00"), "high"] = 120.0
    context = pd.DataFrame({
        "available_time": [pd.Timestamp("2023-01-01 00:01:00")],
        "entry_high": [110.0], "entry_low": [90.0], "exit_high": [105.0], "exit_low": [95.0], "N": [5.0],
    })
    cfg = _cfg(end="2023-01-01 00:09:00")
    r = run_turtle_system2(bars, context, cfg)
    # Threshold was hit exactly in the minute it became available; project rule forbids using it then.
    assert r.events.empty


def test_config_seal_is_not_opened_by_engine_audit() -> None:
    bars = _minute()
    sched = pd.DataFrame({"signal_time": [pd.Timestamp("2023-01-01 00:10:00")], "raw_target": [1.0]})
    r = run_target_schedule(bars, sched, _cfg(), "X")
    assert r.audit["sealed_2026_opened"] is False
    assert r.audit["future_visibility_violations"] == 0
