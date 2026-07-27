from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.market_process.broad_order_flow_paths import (
    build_outcome_arrays,
    build_pa_context_arrays,
    build_transition_arrays,
    directional_outcomes,
    pressure_band_codes,
    rolling_pressure_ratio,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "eth_market_process_portfolio" / "order_flow" / "02_broad_order_flow_pa_path_atlas.py"
spec = importlib.util.spec_from_file_location("broad_order_flow_pa_r03", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _ohlc(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "notional": np.full(n, 100.0),
            "delta_notional": np.zeros(n),
        },
        index=idx,
    )


def test_rolling_pressure_uses_only_trailing_window() -> None:
    delta = np.array([10.0, 20.0, -10.0, 40.0])
    total = np.array([100.0, 100.0, 100.0, 100.0])
    result = rolling_pressure_ratio(delta, total, 2)
    assert np.isnan(result[0])
    assert np.allclose(result[1:], [0.15, 0.05, 0.15])


def test_fixed_pressure_bands_are_symmetric() -> None:
    values = np.array([-0.20, -0.10, -0.04, 0.0, 0.04, 0.10, 0.20, np.nan])
    assert pressure_band_codes(values).tolist() == [-4, -3, -2, 0, 2, 3, 4, 0]


def test_pa_prior_extremes_exclude_current_bar() -> None:
    bars = _ohlc(100)
    pos = 70
    prior_low = bars["low"].iloc[pos - 30 : pos].min()
    bars.iloc[pos, bars.columns.get_loc("low")] = prior_low - 1.0
    bars.iloc[pos, bars.columns.get_loc("close")] = prior_low + 0.01
    pa = build_pa_context_arrays(bars)
    assert abs(pa["prior_low_30"][pos] - prior_low) < 1e-12
    assert bool(pa["sweep_long"][pos])


def test_transition_paths_compare_adjacent_equal_windows() -> None:
    pressure = np.array([np.nan, 0.04, 0.04, 0.10, 0.10, -0.10, -0.10, -0.04, -0.04])
    transitions = build_transition_arrays(pressure, 2)
    # At index 3, current moderate buy is stronger than the mild buy two bars ago.
    assert bool(transitions["strengthening_follow"].event_mask[3])
    # At index 5, current sell pressure is opposite to buy pressure two bars ago.
    assert bool(transitions["reversal_follow"].event_mask[5])
    assert transitions["reversal_follow"].trade_side[5] == -1
    # Weakening fade trades opposite the current flow direction.
    weak_pos = np.flatnonzero(transitions["weakening_fade"].event_mask)
    if len(weak_pos):
        p = weak_pos[0]
        assert transitions["weakening_fade"].trade_side[p] == -transitions["weakening_fade"].flow_side[p]


def test_outcomes_enter_next_open_and_deduct_cost() -> None:
    bars = _ohlc(20)
    bars.iloc[2, bars.columns.get_loc("open")] = 100.0
    bars.iloc[6, bars.columns.get_loc("close")] = 101.0
    labels = build_outcome_arrays(bars, 5)
    side = np.zeros(len(bars), dtype=np.int8)
    side[1] = 1
    gross, net, mfe, mae = directional_outcomes(labels, side, 0.0011)
    assert abs(gross[1] - 0.01) < 1e-12
    assert abs(net[1] - 0.0089) < 1e-12
    assert np.isfinite(mfe[1])
    assert np.isfinite(mae[1])


def test_calendar_month_denominator_includes_inactive_months() -> None:
    assert module._month_count(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-06-30")) == 42
