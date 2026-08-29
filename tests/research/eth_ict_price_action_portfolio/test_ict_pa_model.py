from __future__ import annotations

import numpy as np
import pandas as pd

from research.eth_ict_price_action_portfolio.ict_pa_model import (
    IctPaConfig,
    _swing_position,
    confirmed_pivots,
    shock_survival,
)


def _bars(count: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=count, freq="15min")
    close = np.linspace(100.0, 101.1, count)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def test_confirmed_pivot_is_not_visible_before_right_bars_close() -> None:
    bars = _bars(8)
    bars["low"] = [10.0, 9.0, 5.0, 8.0, 9.0, 9.5, 9.7, 9.8]
    pivots = confirmed_pivots(bars, left=2, right=2)
    assert pd.isna(pivots["confirmed_low"].iloc[3])
    assert pivots["confirmed_low"].iloc[4] == 5.0


def test_signal_close_executes_no_earlier_than_next_open() -> None:
    bars = _bars(12)
    signals = pd.DataFrame(
        {
            "long_signal": False,
            "short_signal": False,
            "long_sweep_extreme": np.nan,
            "short_sweep_extreme": np.nan,
            "atr": 1.0,
        },
        index=bars.index,
    )
    signals.loc[bars.index[3], "long_signal"] = True
    signals.loc[bars.index[3], "long_sweep_extreme"] = 98.0
    position = _swing_position(bars, signals, IctPaConfig(), side=1)
    assert position.loc[bars.index[3], "position"] == 0.0
    assert position.loc[bars.index[4], "position"] > 0.0


def test_user_fee_is_open_005_percent_and_close_005_percent() -> None:
    cfg = IctPaConfig()
    assert cfg.one_way_cost == 0.0005
    assert cfg.one_way_cost * 2 == 0.001


def test_default_portfolio_supports_counter_hedge_mode() -> None:
    cfg = IctPaConfig()
    assert cfg.core_mode == "daily_12m_blend"
    assert cfg.tactical_mode == "counter"


def test_strategy_cap_is_far_below_exchange_cap() -> None:
    cfg = IctPaConfig()
    assert cfg.gross_notional_cap <= 1.0
    assert cfg.exchange_leverage_cap == 15.0
    assert cfg.gross_notional_cap < cfg.exchange_leverage_cap


def test_shock_audit_distinguishes_safe_strategy_cap_from_15x() -> None:
    cfg = IctPaConfig()
    table = shock_survival([cfg.gross_notional_cap, cfg.exchange_leverage_cap], cfg)
    strategy_50 = table[(table["adverse_move"] == 0.50) & (table["gross_exposure"] == 1.0)].iloc[0]
    exchange_50 = table[(table["adverse_move"] == 0.50) & (table["gross_exposure"] == 15.0)].iloc[0]
    assert bool(strategy_50["survives_assumption"])
    assert not bool(exchange_50["survives_assumption"])
