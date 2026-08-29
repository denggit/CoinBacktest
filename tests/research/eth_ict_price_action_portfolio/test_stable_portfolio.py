from __future__ import annotations

import numpy as np
import pandas as pd

from research.eth_ict_price_action_portfolio import _breakout_robustness_bridge as robust
from research.eth_ict_price_action_portfolio import _stable_portfolio_bridge as stable


def synthetic_four_hour(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="4h")
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, rows)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, rows))
    notional = rng.uniform(1e6, 3e6, rows)
    delta = notional * rng.uniform(-0.25, 0.25, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "notional": notional, "delta_notional": delta, "large_delta_notional": delta * 0.25, "trades": 1000},
        index=index,
    )


def test_four_hour_signal_is_available_only_after_bar_close() -> None:
    four = synthetic_four_hour()
    result = robust.definition(four, 30, 0.20, 1.25, 6)
    assert result.index[0] == four.index[0] + pd.Timedelta(hours=4)
    assert np.array_equal(result["long_event"].to_numpy(), robust.definition(four, 30, 0.20, 1.25, 6)["long_event"].to_numpy())


def test_future_trade_mutation_cannot_change_past_signals() -> None:
    four = synthetic_four_hour()
    cutoff = four.index[220]
    base = robust.definition(four, 30, 0.20, 1.25, 6)
    changed = four.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "delta_notional"]] *= 1.7
    altered = robust.definition(changed, 30, 0.20, 1.25, 6)
    available_cutoff = cutoff + pd.Timedelta(hours=4)
    pd.testing.assert_frame_equal(base.loc[:available_cutoff], altered.loc[:available_cutoff])


def test_simulator_charges_open_and_close_cost() -> None:
    index = pd.date_range("2022-01-01", periods=4, freq="15min")
    bars = pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0]}, index=index)
    positions = pd.DataFrame({"sleeve": [0.5, 0.5, 0.0, 0.0]}, index=index)
    frame = stable.simulate(bars, positions, cost=0.0005)
    assert np.isclose(frame["trading_cost"].sum(), 0.5 * 0.0005 + 0.5 * 0.0005)
    assert np.isclose(frame["net_return"].sum(), -0.0005)


def test_daily_core_future_mutation_is_causal() -> None:
    index = pd.date_range("2019-01-01", periods=900, freq="1D")
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(index))))
    daily_like = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0},
        index=index,
    )
    cutoff = pd.Timestamp("2020-12-31")
    base = stable.build_daily_core(daily_like)
    changed = daily_like.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 2.0
    altered = stable.build_daily_core(changed)
    pd.testing.assert_series_equal(base.loc[:cutoff + pd.Timedelta(days=1), "position"], altered.loc[:cutoff + pd.Timedelta(days=1), "position"])
