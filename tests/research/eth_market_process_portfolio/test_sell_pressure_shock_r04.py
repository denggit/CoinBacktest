from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.market_process.sell_pressure_shock_paths import (
    build_outcome_arrays,
    build_post_shock_events,
    build_sell_shock_arrays,
    build_sell_shock_pa,
    directional_outcomes,
    fixed_side_array,
    rolling_activity_ratio,
    rolling_pressure_ratio,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "eth_market_process_portfolio" / "order_flow" / "03_sell_pressure_shock_path_study.py"
spec = importlib.util.spec_from_file_location("sell_pressure_shock_r04", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _bars(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": close.copy(),
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close.copy(),
            "notional": np.full(n, 100.0),
            "delta_notional": np.zeros(n),
        },
        index=idx,
    )


def test_activity_ratio_excludes_current_window_from_baseline() -> None:
    values = np.full(20, 100.0)
    values[15:17] = 200.0
    ratio = rolling_activity_ratio(values, window=2, baseline_minutes=5)
    assert np.isclose(ratio[16], 2.0)
    # Changing a future value cannot change the earlier ratio.
    values[19] = 10_000.0
    changed = rolling_activity_ratio(values, window=2, baseline_minutes=5)
    assert np.isclose(changed[16], ratio[16])


def test_sell_shock_types_compare_non_overlapping_equal_windows() -> None:
    pressure = np.array([0.04, 0.04, -0.04, -0.04, -0.10, -0.10, -0.20, -0.20])
    shocks = build_sell_shock_arrays(pressure, window=2)
    # Buy-to-sell reversal is visible when current sell window is compared with prior buy window.
    assert shocks["buy_to_sell_reversal"].event_mask[2]
    # Strengthening sell is visible when current band is more negative than prior equal window.
    assert shocks["sell_strengthening"].event_mask[4]


def test_prior_low_reference_excludes_entire_shock_window() -> None:
    bars = _bars(120)
    window = 5
    pos = 80
    expected = bars["low"].iloc[pos - window - 29 : pos - window + 1].min()
    bars.iloc[pos - 2, bars.columns.get_loc("low")] = expected - 2.0
    bars.iloc[pos, bars.columns.get_loc("close")] = expected + 0.01
    pa = build_sell_shock_pa(bars, window)
    assert np.isclose(pa.prior_low_30[pos], expected)
    assert pa.prior_low_sweep[pos]
    assert pa.same_window_sweep_reclaim[pos]


def test_lower_wick_uses_aggregated_shock_window() -> None:
    bars = _bars(80)
    pos = 60
    bars.iloc[pos, bars.columns.get_loc("low")] = 98.0
    bars.iloc[pos, bars.columns.get_loc("close")] = 100.5
    bars.iloc[pos, bars.columns.get_loc("high")] = 100.7
    pa = build_sell_shock_pa(bars, 3)
    assert pa.lower_wick[pos]
    assert pa.close_recovery_fraction[pos] > 0.55


def test_delayed_reclaim_and_acceptance_are_stamped_on_confirmation_bar() -> None:
    close = np.array([100.0, 99.0, 98.5, 98.7, 100.2, 100.3, 99.0, 98.8, 98.7, 98.6])
    shock = np.zeros(len(close), dtype=bool)
    unresolved = np.zeros(len(close), dtype=bool)
    ref = np.full(len(close), np.nan)
    shock[1] = True
    unresolved[1] = True
    ref[1] = 100.0
    events = build_post_shock_events(close, shock, unresolved, ref, reclaim_waits=(3, 5), acceptance_bars=(2, 3))
    assert events.breakdown_acceptance[2][2]
    assert events.breakdown_acceptance[3][3]
    assert events.delayed_reclaim[3][4]
    assert events.delayed_reclaim[5][4]
    assert events.source_shock_index[4] == 1


def test_newer_unresolved_shock_supersedes_older_one() -> None:
    close = np.array([100.0, 99.0, 98.5, 97.5, 99.1])
    shock = np.array([False, True, False, True, False])
    unresolved = shock.copy()
    ref = np.array([np.nan, 100.0, np.nan, 99.0, np.nan])
    events = build_post_shock_events(close, shock, unresolved, ref, reclaim_waits=(3,), acceptance_bars=(2,))
    assert events.delayed_reclaim[3][4]
    assert events.source_shock_index[4] == 3


def test_outcome_enters_next_open_and_charges_cost() -> None:
    bars = _bars(30)
    bars.iloc[2, bars.columns.get_loc("open")] = 100.0
    bars.iloc[6, bars.columns.get_loc("close")] = 101.0
    labels = build_outcome_arrays(bars, 5)
    long_side = fixed_side_array(len(bars), 1)
    gross, net, _, _ = directional_outcomes(labels, long_side, 0.0011)
    assert np.isclose(gross[1], 0.01)
    assert np.isclose(net[1], 0.0089)


def test_future_changes_do_not_modify_earlier_pressure_or_pa() -> None:
    bars = _bars(150)
    pressure_before = rolling_pressure_ratio(bars["delta_notional"], bars["notional"], 10)
    pa_before = build_sell_shock_pa(bars, 10)
    bars.iloc[120:, bars.columns.get_loc("low")] -= 10.0
    bars.iloc[120:, bars.columns.get_loc("delta_notional")] = -100.0
    pressure_after = rolling_pressure_ratio(bars["delta_notional"], bars["notional"], 10)
    pa_after = build_sell_shock_pa(bars, 10)
    assert np.allclose(pressure_before[:110], pressure_after[:110], equal_nan=True)
    assert np.allclose(pa_before.prior_low_30[:110], pa_after.prior_low_30[:110], equal_nan=True)


def test_calendar_month_count_includes_inactive_months() -> None:
    assert module._month_count(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-06-30")) == 42
