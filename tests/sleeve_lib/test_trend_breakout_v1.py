from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_common.ohlcv_backtest import SignalBacktestParams, run_signal_backtest
from src.sleeve_lib.trend_breakout_v1 import TrendBreakoutConfig, build_features


def _bars(rows: int = 360) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="15min")
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.0025, rows)))
    # Inject causal structure breaks in both directions.
    close[150:170] += np.linspace(0.0, 8.0, 20)
    close[250:270] -= np.linspace(0.0, 9.0, 20)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0015
    low = np.minimum(open_, close) * 0.9985
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0},
        index=index,
    )


def test_breakout_features_are_future_mutation_invariant() -> None:
    cfg = TrendBreakoutConfig()
    bars = _bars()
    cutoff = bars.index[250]
    base = build_features(bars, cfg)
    changed = bars.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 1.8
    altered = build_features(changed, cfg)
    cols = ["prior_breakout_high", "prior_breakout_low", "signal", "stop", "risk_mult"]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], altered.loc[:cutoff, cols])


def test_quality_scoring_does_not_delete_structure_events() -> None:
    features = build_features(_bars(), TrendBreakoutConfig())
    events = features.loc[features["structure_event"]]
    assert len(events) > 0
    assert events["risk_mult"].notna().all()
    assert (events["risk_mult"] > 0).all()
    assert (events["signal"] != 0).all()


def test_signal_enters_next_bar_open() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="15min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 101.5, 102.0, 102.0],
            "high": [100.5, 102.0, 103.0, 102.5, 102.2],
            "low": [99.5, 100.5, 101.0, 101.5, 101.8],
            "close": [100.0, 101.5, 102.5, 102.0, 102.0],
            "signal": [1, 0, 0, 0, 0],
            "stop": [99.0, np.nan, np.nan, np.nan, np.nan],
            "risk_mult": [1.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    params = SignalBacktestParams(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        fee_rate=0.0,
        slippage_pct=0.0,
        target_r=1.0,
        min_stop_pct=0.0,
        max_stop_pct=0.10,
        max_hold_bars=4,
        exit_on_opposite_signal=False,
        risk_mult_col="risk_mult",
    )
    trades, _ = run_signal_backtest(frame, params)
    assert trades
    assert trades[0]["entry_time"] == index[1]
    assert trades[0]["entry"] == frame.loc[index[1], "open"]


def test_dynamic_risk_multiplier_reduces_position_size() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="15min")

    def replay(mult: float) -> float:
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 100.5, 101.0, 101.0],
                "low": [100.0, 99.5, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0, 100.0],
                "signal": [1, 0, 0, 0],
                "stop": [98.0, np.nan, np.nan, np.nan],
                "risk_mult": [mult, 0.0, 0.0, 0.0],
            },
            index=index,
        )
        params = SignalBacktestParams(
            initial_capital=1000.0,
            risk_per_trade=0.01,
            max_notional_mult=10.0,
            fee_rate=0.0,
            slippage_pct=0.0,
            target_r=10.0,
            min_stop_pct=0.0,
            max_stop_pct=0.10,
            max_hold_bars=1,
            exit_on_opposite_signal=False,
            risk_mult_col="risk_mult",
            min_risk_mult=0.0,
            max_risk_mult=1.0,
        )
        trades, _ = run_signal_backtest(frame, params)
        return float(trades[0]["qty"])

    assert np.isclose(replay(0.5), replay(1.0) * 0.5)


def test_v1_disables_close_known_exits() -> None:
    from backtest.mf.trend_breakout.eth_trend_breakout_v1_backtest import _params

    cfg = TrendBreakoutConfig()
    params = _params(cfg)
    assert params.exit_on_opposite_signal is False
    assert params.no_progress_bars == 0
    assert params.trailing_atr_col is None
    assert params.max_hold_bars >= 10**8
