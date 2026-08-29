from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.research_common.eth_dynamic_positioning import (
    DynamicPositionConfig,
    build_state_frame,
    prepare_execution_targets,
    simulate_dynamic_positioning,
)


def _bars(start: str = "2022-01-01", periods: int = 1200, drift: float = 0.0004) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h")
    x = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(drift * x + 0.01 * np.sin(x / 17.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


def _manual_state(
    index: pd.DatetimeIndex,
    medium: list[float],
    slow: list[float],
    *,
    decision: list[bool] | None = None,
    open_prices: list[float] | None = None,
) -> pd.DataFrame:
    n = len(index)
    if decision is None:
        decision = [True] * n
    if open_prices is None:
        open_prices = [100.0] * n
    frame = pd.DataFrame(index=index)
    frame["open"] = open_prices
    frame["high"] = np.asarray(open_prices) * 1.001
    frame["low"] = np.asarray(open_prices) * 0.999
    frame["close"] = open_prices
    frame["volume"] = 1000.0
    frame["medium_desired_close"] = medium
    frame["slow_desired_close"] = slow
    frame["decision_close"] = decision
    frame["state_ready"] = True
    frame["medium_trend"] = np.sign(medium)
    frame["slow_trend"] = np.sign(slow)
    frame["medium_extension"] = 0.0
    frame["slow_extension"] = 0.0
    frame["annual_vol"] = 0.5
    return frame


def _cfg(**kwargs) -> DynamicPositionConfig:
    base = DynamicPositionConfig(
        warmup_start="2022-01-01",
        trade_start="2023-01-01",
        trade_end="2026-06-30 23:59:59",
        decision_hours=1,
        no_trade_band=0.0,
        max_step_per_decision=10.0,
        fee_rate_per_side=0.00055,
        slippage_rate_per_side=0.0,
    )
    return replace(base, **kwargs)


def test_future_mutation_does_not_change_prior_state() -> None:
    bars = _bars(periods=1400)
    cfg = DynamicPositionConfig()
    first = build_state_frame(bars, cfg)
    cutoff = bars.index[1000]
    mutated = bars.copy()
    mutated.loc[mutated.index > cutoff, ["open", "high", "low", "close"]] *= 3.0
    second = build_state_frame(mutated, cfg)
    cols = [
        "medium_trend",
        "slow_trend",
        "medium_extension",
        "slow_extension",
        "annual_vol",
        "medium_desired_close",
        "slow_desired_close",
    ]
    pd.testing.assert_frame_equal(first.loc[:cutoff, cols], second.loc[:cutoff, cols])


def test_location_disabled_is_exactly_neutral_multiplier() -> None:
    bars = _bars(periods=1200)
    state = build_state_frame(bars, replace(DynamicPositionConfig(), location_strength=0.0))
    ready = state["state_ready"]
    assert ready.any()
    assert np.allclose(state.loc[ready, "medium_location_multiplier"], 1.0)
    assert np.allclose(state.loc[ready, "slow_location_multiplier"], 1.0)


def test_signal_executes_on_next_hour_open_not_same_bar() -> None:
    idx = pd.date_range("2023-01-01", periods=5, freq="1h")
    state = _manual_state(
        idx,
        medium=[0.8, 0.8, 0.8, 0.8, 0.8],
        slow=[0, 0, 0, 0, 0],
        decision=[True, False, False, False, False],
    )
    targets = prepare_execution_targets(state, _cfg())
    assert targets.iloc[0]["medium_raw_target"] == 0.0
    assert targets.iloc[1]["medium_raw_target"] == 0.8


def test_no_trade_band_blocks_small_churn() -> None:
    idx = pd.date_range("2023-01-01", periods=7, freq="1h")
    state = _manual_state(
        idx,
        medium=[0.50, 0.55, 0.57, 0.54, 0.58, 0.56, 0.55],
        slow=[0.0] * 7,
    )
    cfg = _cfg(no_trade_band=0.20, max_step_per_decision=10.0)
    replay = simulate_dynamic_positioning(state, cfg)
    # First executable target creates one adjustment; subsequent 5-8bp exposure
    # nudges are inside the 0.20x band and must not churn.
    assert int((replay["turnover"] > 1e-12).sum()) == 1


def test_opposite_sleeves_are_not_netted_before_fee_accounting() -> None:
    idx = pd.date_range("2023-01-01", periods=4, freq="1h")
    state = _manual_state(idx, medium=[0.5] * 4, slow=[-0.5] * 4)
    cfg = _cfg()
    replay = simulate_dynamic_positioning(state, cfg)
    active = replay.iloc[1]
    assert abs(active["net_exposure"]) < 1e-12
    assert np.isclose(active["gross_exposure"], 1.0)
    assert bool(active["long_short_overlap"])
    # 0 -> +0.5 and 0 -> -0.5 is 1.0x gross turnover, despite zero net.
    assert np.isclose(active["turnover"], 1.0)
    assert np.isclose(active["trading_cost"], 0.00055)


def test_positive_funding_long_pays_short_receives() -> None:
    idx = pd.date_range("2023-01-01", periods=5, freq="1h")
    funding = pd.DataFrame({"funding_rate": [0.001]}, index=[idx[2]])
    long_state = _manual_state(idx, medium=[1.0] * 5, slow=[0.0] * 5)
    short_state = _manual_state(idx, medium=[-1.0] * 5, slow=[0.0] * 5)
    cfg = _cfg(fee_rate_per_side=0.0)
    long_replay = simulate_dynamic_positioning(long_state, cfg, funding=funding)
    short_replay = simulate_dynamic_positioning(short_state, cfg, funding=funding)
    assert np.isclose(long_replay["funding_return"].sum(), -0.001)
    assert np.isclose(short_replay["funding_return"].sum(), 0.001)


def test_warmup_rows_can_never_enter_official_account() -> None:
    idx = pd.date_range("2022-12-31 20:00", periods=12, freq="1h")
    state = _manual_state(idx, medium=[1.0] * 12, slow=[0.0] * 12)
    cfg = _cfg(trade_start="2023-01-01 00:00:00", trade_end="2023-01-01 07:00:00")
    replay = simulate_dynamic_positioning(state, cfg)
    assert replay.index.min() >= pd.Timestamp("2023-01-01 00:00:00")
    assert not (replay.index.year == 2022).any()


def test_extra_execution_delay_is_applied_after_mandatory_next_open() -> None:
    idx = pd.date_range("2023-01-01", periods=8, freq="1h")
    state = _manual_state(idx, medium=[0.7] * 8, slow=[0.0] * 8, decision=[True] + [False] * 7)
    targets = prepare_execution_targets(state, _cfg(execution_delay_hours=2))
    assert np.allclose(targets["medium_raw_target"].iloc[:3], 0.0)
    assert np.isclose(targets["medium_raw_target"].iloc[3], 0.7)


def test_large_gap_does_not_trigger_hourly_followup_trades_between_decisions() -> None:
    idx = pd.date_range("2023-01-01", periods=9, freq="1h")
    state = _manual_state(
        idx,
        medium=[1.0] * 9,
        slow=[0.0] * 9,
        decision=[True, False, False, False, True, False, False, False, True],
    )
    cfg = _cfg(no_trade_band=0.0, max_step_per_decision=0.25)
    replay = simulate_dynamic_positioning(state, cfg)
    changed = replay.index[replay["turnover"] > 1e-12]
    # Signal at 00:00 can execute at 01:00; the still-large gap must not cause
    # 02:00/03:00/04:00 follow-up trades. Next allowed change is 05:00.
    assert list(changed[:2]) == [idx[1], idx[5]]
