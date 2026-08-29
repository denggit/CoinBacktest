from __future__ import annotations

import numpy as np
import pandas as pd

from research.eth_market_process_portfolio.portfolio.clean_causal import (
    PortfolioConfig,
    build_desired_exposure,
    shock_survival_table,
    simulate_portfolio,
)


def _bars(periods: int = 900) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="4h")
    close = 100.0 * np.exp(np.arange(periods) * 0.0005)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(periods, 10.0),
        },
        index=index,
    )


def test_signal_is_executed_only_after_completed_signal_bar() -> None:
    bars = _bars()
    cfg = PortfolioConfig(
        momentum_days=(2, 3, 4),
        volatility_days=2,
        rebalance_bars=1,
        annual_carry_drag=0.0,
    )
    features = build_desired_exposure(bars, cfg)
    replay = simulate_portfolio(bars, cfg)
    first_signal = features["rebalanced_exposure_close"].first_valid_index()
    assert first_signal is not None
    signal_pos = bars.index.get_loc(first_signal)
    assert replay.loc[first_signal, "position"] == 0.0
    assert replay.loc[bars.index[signal_pos + 1], "position"] == features.loc[first_signal, "rebalanced_exposure_close"]


def test_one_bar_delay_adds_exactly_one_more_open() -> None:
    bars = _bars()
    base = PortfolioConfig(momentum_days=(2, 3, 4), volatility_days=2, rebalance_bars=1)
    delayed = PortfolioConfig(
        momentum_days=(2, 3, 4), volatility_days=2, rebalance_bars=1, execution_delay_bars=1
    )
    a = simulate_portfolio(bars, base)["position"]
    b = simulate_portfolio(bars, delayed)["position"]
    assert np.allclose(b.iloc[1:].to_numpy(), a.shift(1).fillna(0.0).iloc[1:].to_numpy())


def test_strategy_cap_is_below_exchange_cap_and_never_exceeded() -> None:
    bars = _bars()
    cfg = PortfolioConfig(
        momentum_days=(2, 3, 4),
        volatility_days=2,
        target_annual_volatility=0.50,
        strategy_notional_cap=1.5,
        exchange_leverage_cap=15.0,
        rebalance_bars=1,
    )
    replay = simulate_portfolio(bars, cfg)
    assert replay["gross_exposure"].max() <= 1.5
    assert cfg.strategy_notional_cap < cfg.exchange_leverage_cap


def test_fifty_percent_shock_survives_at_strategy_cap() -> None:
    cfg = PortfolioConfig(strategy_notional_cap=1.5, exchange_leverage_cap=15.0)
    table = shock_survival_table([cfg.strategy_notional_cap], cfg)
    worst = table[table["adverse_instantaneous_move"] == 0.50].iloc[0]
    assert bool(worst["survives_assumed_liquidation_rule"])
    assert worst["maintenance_headroom"] > 0


def test_invalid_cap_above_exchange_limit_is_rejected() -> None:
    cfg = PortfolioConfig(strategy_notional_cap=16.0, exchange_leverage_cap=15.0)
    try:
        cfg.validate()
    except ValueError as error:
        assert "exchange leverage cap" in str(error)
    else:
        raise AssertionError("configuration should have been rejected")


def test_conflicting_horizons_can_hold_long_and_short_together() -> None:
    bars = _bars(1_200)
    bars.loc[bars.index[-20]:, ["open", "high", "low", "close"]] *= np.linspace(1.0, 0.75, 20)[:, None]
    cfg = PortfolioConfig(momentum_days=(2, 20, 40), volatility_days=2, rebalance_bars=1)
    replay = simulate_portfolio(bars, cfg)
    overlap = replay[replay["long_short_overlap"]]
    assert not overlap.empty
    assert (overlap["long_gross_exposure"] > 0).all()
    assert (overlap["short_gross_exposure"] > 0).all()
    assert (overlap["gross_exposure"] >= overlap["net_exposure"].abs()).all()
