from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.mf.trend_following import (
    donchian_breakout,
    ema_momentum,
    market_structure,
    orderflow_trend,
    trend_pullback,
    volatility_expansion,
)


def _bars(n: int = 1200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    x = np.arange(n, dtype=float)
    # Smooth trend + oscillation produces pullbacks, structure and breakouts
    # without any random fixture dependency.
    close = 2000.0 + 0.32 * x + 22.0 * np.sin(x / 17.0) + 8.0 * np.sin(x / 5.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 4.0 + 1.5 * np.sin(x / 9.0) ** 2
    low = np.minimum(open_, close) - 4.0 - 1.5 * np.cos(x / 11.0) ** 2
    volume = 1000.0 + 250.0 * (1.0 + np.sin(x / 13.0))
    delta = 100_000.0 * np.sin(x / 23.0) + 35_000.0 * np.sign(np.diff(np.r_[close[0], close]))
    buy_ratio = np.clip(0.5 + delta / 2_000_000.0, 0.35, 0.65)
    notional = 2_000_000.0 + volume * 1000.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "delta_notional": delta,
            "taker_buy_ratio": buy_ratio,
            "notional": notional,
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "builder",
    [
        donchian_breakout.build_features,
        ema_momentum.build_features,
        trend_pullback.build_features,
        market_structure.build_features,
        volatility_expansion.build_features,
        orderflow_trend.build_features,
    ],
)
def test_future_mutation_cannot_change_past_signal_or_stop(builder):
    bars = _bars()
    cutoff = 800
    base = builder(bars.copy())

    mutated = bars.copy()
    future = mutated.index[cutoff + 1 :]
    mutated.loc[future, "open"] *= 1.7
    mutated.loc[future, "high"] *= 1.9
    mutated.loc[future, "low"] *= 0.5
    mutated.loc[future, "close"] *= 1.8
    mutated.loc[future, "volume"] *= 11.0
    mutated.loc[future, "delta_notional"] *= -19.0
    mutated.loc[future, "taker_buy_ratio"] = 0.01
    mutated.loc[future, "notional"] *= 17.0
    changed = builder(mutated)

    past = bars.index[: cutoff + 1]
    pd.testing.assert_series_equal(base.loc[past, "signal"], changed.loc[past, "signal"], check_names=False)
    pd.testing.assert_series_equal(base.loc[past, "stop"], changed.loc[past, "stop"], check_names=False)


def test_market_structure_swing_is_only_available_after_right_confirmation():
    idx = pd.date_range("2024-01-01", periods=9, freq="15min")
    # Peak at position 3. With order=3 it may only be confirmed at position 6.
    high = pd.Series([1, 2, 3, 10, 4, 3, 2, 2, 2], index=idx, dtype=float)
    low = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    frame = pd.DataFrame({"high": high, "low": low}, index=idx)
    swings = market_structure._confirmed_swings(frame, order=3)
    assert swings["new_swing_high"].iloc[:6].isna().all()
    assert swings["new_swing_high"].iloc[6] == 10.0


def test_orderflow_requires_trade_bar_fields():
    bars = _bars().drop(columns=["delta_notional"])
    with pytest.raises(RuntimeError, match="requires OKX trade-bar fields"):
        orderflow_trend.build_features(bars)


def test_causal_trailing_stop_only_activates_on_next_bar():
    from backtest.mf.trend_following.common import run_causal_trend_backtest

    idx = pd.date_range("2024-02-01", periods=3, freq="15min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 109.0],
            "high": [101.0, 110.0, 110.0],
            "low": [99.0, 98.0, 104.0],
            "close": [100.0, 109.0, 105.0],
            "signal": [1, 0, 0],
            "stop": [95.0, np.nan, np.nan],
            "atr14": [2.0, 2.0, 2.0],
        },
        index=idx,
    )
    trades, _ = run_causal_trend_backtest(
        frame,
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        max_notional_mult=3.0,
        fee_rate_per_side=0.0,
        slippage_rate_per_side=0.0,
        min_stop_pct=0.001,
        max_stop_pct=0.10,
        max_hold_bars=100,
        trailing_atr_mult=3.0,
        trail_after_r=1.0,
    )
    assert len(trades) == 1
    # Bar 1 closes at 109, which raises the next-bar trail to 103.  Its own low
    # was 98, but that historical low must not be tested against the new trail.
    assert trades[0]["exit_time"] == idx[2]
    assert trades[0]["note"] == "FORCE_CLOSE_END"
