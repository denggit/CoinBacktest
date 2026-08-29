from __future__ import annotations

import numpy as np
import pandas as pd

from research.eth_ict_price_action_portfolio import _okx_quarter_hour_bridge as quarter
from research.eth_ict_price_action_portfolio import _pinned_10s_bridge as pinned10
from research.eth_ict_price_action_portfolio import _multiscale_bos_bridge as multibos
from research.eth_ict_price_action_portfolio import _daily_reclaim_bridge as reclaim
from research.eth_ict_price_action_portfolio import _nr7_bridge as nr7
from research.eth_ict_price_action_portfolio import _multispeed_ema_bridge as multiema
from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as walk
from research.eth_ict_price_action_portfolio import _okx_weekly_regime_bridge as weekly


def synthetic_trade15(rows: int = 480) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="15min")
    rng = np.random.default_rng(31)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, rows)))
    open_ = np.r_[close[0], close[:-1]]
    notional = rng.uniform(1e6, 2e6, rows)
    delta = notional * rng.uniform(-0.2, 0.2, rows)
    return pd.DataFrame(
        {
            "open": open_, "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999, "close": close,
            "volume": notional / close, "notional": notional,
            "trades_count": 1000, "buy_notional": (notional + delta) / 2,
            "sell_notional": (notional - delta) / 2, "delta_notional": delta,
            "large_delta_notional": delta * 0.2, "max_trade_notional": notional * 0.01,
        },
        index=index,
    )


def test_hourly_features_are_available_after_completed_hour() -> None:
    trade = synthetic_trade15()
    features = walk.build_hourly_features(trade)
    assert features.index.min() == trade.index.min() + pd.Timedelta(hours=1)


def test_partial_hour_is_not_a_feature_observation() -> None:
    trade = synthetic_trade15().drop(pd.Timestamp("2022-01-02 10:15:00"))
    features = walk.build_hourly_features(trade)
    assert pd.Timestamp("2022-01-02 11:00:00") not in features.index


def test_future_trade_mutation_cannot_change_past_hourly_features() -> None:
    trade = synthetic_trade15()
    cutoff = pd.Timestamp("2022-01-04 00:00:00")
    base = walk.build_hourly_features(trade)
    changed = trade.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "delta_notional"]] *= 1.8
    altered = walk.build_hourly_features(changed)
    pd.testing.assert_frame_equal(base.loc[:cutoff], altered.loc[:cutoff])


def test_simulator_reports_post_cap_gross_and_two_sided_fee() -> None:
    index = pd.date_range("2022-01-01", periods=4, freq="1min")
    minute = pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0]}, index=index)
    positions = pd.DataFrame({"long": [1.0, 1.0, 0.0, 0.0]}, index=index)
    replay = walk.simulate_minute(minute, positions)
    assert np.isclose(replay["gross_exposure"].max(), 0.75)
    assert np.isclose(replay["trading_cost"].sum(), 2 * 0.75 * 0.0005)


def test_quarter_hour_signal_executes_after_observed_ten_seconds() -> None:
    index = pd.date_range("2022-04-01", periods=200, freq="15min")
    opening = pd.DataFrame(
        {
            "open": 100.0, "close": 100.01, "notional": 1e6,
            "delta_notional": 1e5, "large_delta_notional": 1e4,
            "high": 100.02, "low": 99.99, "trades_count": 100,
        },
        index=index,
    )
    hourly = pd.DataFrame(0.1, index=pd.date_range("2022-01-01", "2022-06-01", freq="1h"), columns=walk.FEATURE_COLUMNS)
    minute_index = pd.date_range("2022-01-01", "2022-06-01", freq="1min")
    minute = pd.DataFrame({"open": 100.0}, index=minute_index)
    sample = quarter.build_samples(opening, hourly, minute)
    assert (sample["available_time"] == sample.index + pd.Timedelta(seconds=10)).all()
    assert (sample["execution_time_1m"] == sample.index + pd.Timedelta(minutes=1)).all()
    assert (sample["execution_time_1m"] > sample["available_time"]).all()


def test_daily_features_use_positional_next_day_availability() -> None:
    days = 140
    minute_index = pd.date_range("2022-01-01", periods=days * 1440, freq="1min")
    minute_close = 100.0 * np.exp(np.arange(len(minute_index)) * 1e-7)
    minute = pd.DataFrame(
        {"open": minute_close, "high": minute_close, "low": minute_close, "close": minute_close, "volume": 1.0},
        index=minute_index,
    )
    trade = synthetic_trade15(days * 96)
    features = weekly.build_daily_features(minute, trade)
    assert features.index.min().hour == 0
    cutoff = pd.Timestamp("2022-04-15")
    changed_minute = minute.copy()
    changed_minute.loc[changed_minute.index >= cutoff, ["open", "high", "low", "close"]] *= 2.0
    altered = weekly.build_daily_features(changed_minute, trade)
    pd.testing.assert_frame_equal(features.loc[:cutoff], altered.loc[:cutoff])


def test_ten_second_pin_uses_only_prior_complete_minutes() -> None:
    minute_index = pd.date_range("2022-01-01", periods=121, freq="1min")
    minute = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.02,
            "low": 99.98,
            "close": 100.01,
        },
        index=minute_index,
    )
    pin_time = minute_index[90] + pd.Timedelta(seconds=20)
    extreme = pd.DataFrame(
        {
            "open": [100.0], "high": [100.005], "low": [99.995], "close": [100.001],
            "buy_trades_count": [5], "sell_trades_count": [45], "trades_count": [50],
        },
        index=pd.DatetimeIndex([pin_time]),
    )
    first = pinned10.filter_pinned_candidates(extreme, minute)
    changed = minute.copy()
    changed.loc[minute_index[90]:, ["open", "high", "low", "close"]] *= 3.0
    second = pinned10.filter_pinned_candidates(extreme, changed)
    assert first.at[pin_time, "available_time"] == pin_time + pd.Timedelta(seconds=10)
    assert bool(first.at[pin_time, "pinned"])
    for column in ("prior_60m_abs_return_median", "prior_60m_range_median", "return_gate_10s", "range_gate_10s"):
        assert np.isclose(first.at[pin_time, column], second.at[pin_time, column])


def test_ten_second_release_uses_full_minute_then_next_open() -> None:
    index = pd.date_range("2022-01-01 00:00:00", periods=200, freq="1min")
    minute = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0}, index=index)
    trade1 = pd.DataFrame({"buy_trades_count": 50, "sell_trades_count": 50}, index=index)
    pin_time = pd.Timestamp("2022-01-01 01:00:20")
    pins = pd.DataFrame(
        {
            "pinned": [True], "available_time": [pin_time + pd.Timedelta(seconds=10)],
            "dominant_sign": [-1], "sign_imbalance": [-0.8], "high": [100.1], "low": [99.9],
        },
        index=pd.DatetimeIndex([pin_time]),
    )
    minute.loc[pd.Timestamp("2022-01-01 01:01:00"), "close"] = 100.15
    releases = pinned10.build_release_events(pins, trade1, minute)
    assert releases.loc[0, "release_bar_time"] == pd.Timestamp("2022-01-01 01:01:00")
    assert releases.loc[0, "available_time"] == pd.Timestamp("2022-01-01 01:02:00")

    one_minute, _, _ = pinned10.build_side_positions(releases, minute, "long", 1)
    two_minute, _, _ = pinned10.build_side_positions(releases, minute, "long", 2)
    assert one_minute.at[pd.Timestamp("2022-01-01 01:02:00")] > 0.0
    assert two_minute.at[pd.Timestamp("2022-01-01 01:02:00")] == 0.0
    assert two_minute.at[pd.Timestamp("2022-01-01 01:03:00")] > 0.0


def test_multiscale_bos_is_available_next_day_and_executes_after_delay() -> None:
    days = 180
    index = pd.date_range("2021-01-01", periods=days * 1440, freq="1min")
    close = 100.0 * np.exp(np.arange(len(index)) * 1e-6)
    minute = pd.DataFrame(
        {"open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close},
        index=index,
    )
    features = multibos.build_daily_bos_features(minute)
    assert (features.index.hour == 0).all()
    assert features.index.min() == index.min().floor("D") + pd.Timedelta(days=1)
    pos1 = multibos.positions_from_features(features, index, 1)
    pos2 = multibos.positions_from_features(features, index, 2)
    event_day = features.index[100]
    assert np.allclose(pos1.loc[event_day].to_numpy(), pos1.loc[event_day - pd.Timedelta(minutes=1)].to_numpy())
    assert np.allclose(pos2.loc[event_day + pd.Timedelta(minutes=1)].to_numpy(), pos2.loc[event_day].to_numpy())
    assert np.allclose(pos1.loc[event_day + pd.Timedelta(minutes=1)].to_numpy(), features.loc[event_day, ["bos_7d", "bos_28d", "bos_91d"]].to_numpy(dtype=float) * features.at[event_day, "gross_budget"] / 3.0)


def test_future_daily_mutation_cannot_change_past_bos_features() -> None:
    days = 180
    index = pd.date_range("2021-01-01", periods=days * 1440, freq="1min")
    rng = np.random.default_rng(71)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0002, len(index))))
    minute = pd.DataFrame(
        {"open": close, "high": close * 1.0002, "low": close * 0.9998, "close": close},
        index=index,
    )
    cutoff = pd.Timestamp("2021-05-01")
    first = multibos.build_daily_bos_features(minute)
    changed = minute.copy()
    changed.loc[changed.index >= cutoff, ["open", "high", "low", "close"]] *= 2.5
    second = multibos.build_daily_bos_features(changed)
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])


def test_daily_reclaim_event_is_available_next_midnight_and_executes_later() -> None:
    days = 35
    index = pd.date_range("2021-01-01", periods=days * 1440, freq="1min")
    minute = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0}, index=index)
    event_day = pd.Timestamp("2021-01-25")
    minute.loc[event_day:event_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), "low"] = 95.0
    events = reclaim.build_daily_reclaim_events(minute)
    available = event_day + pd.Timedelta(days=1)
    event = events.loc[available]
    assert event["side"] == "long"
    assert event["event_day"] == event_day
    position1, _, _ = reclaim.build_side_positions(events, minute, "long", 1)
    position2, _, _ = reclaim.build_side_positions(events, minute, "long", 2)
    assert position1.at[available] == 0.0
    assert position1.at[available + pd.Timedelta(minutes=1)] > 0.0
    assert position2.at[available + pd.Timedelta(minutes=1)] == 0.0
    assert position2.at[available + pd.Timedelta(minutes=2)] > 0.0


def test_nr7_setup_and_breakout_are_both_observed_before_execution() -> None:
    days = 14
    index = pd.date_range("2021-01-01", periods=days * 1440, freq="1min")
    minute = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    setup_day = pd.Timestamp("2021-01-08")
    setup_slice = slice(setup_day, setup_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1))
    minute.loc[setup_slice, "high"] = 100.2
    minute.loc[setup_slice, "low"] = 99.8
    setups = nr7.build_nr7_setups(minute)
    available = setup_day + pd.Timedelta(days=1)
    assert available in setups.index
    breakout_bar = available + pd.Timedelta(hours=1)
    minute.loc[breakout_bar, "close"] = 100.3
    events = nr7.build_breakout_events(setups, minute)
    event = events.loc[events["setup_day"] == setup_day].iloc[0]
    assert event["release_bar_time"] == breakout_bar
    assert event["available_time"] == breakout_bar + pd.Timedelta(minutes=1)


def test_multispeed_ema_uses_completed_day_and_fixed_execution_delay() -> None:
    days = 200
    index = pd.date_range("2021-01-01", periods=days * 1440, freq="1min")
    close = 100.0 * np.exp(np.arange(len(index)) * 1e-6)
    minute = pd.DataFrame(
        {"open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close}, index=index
    )
    features = multiema.build_daily_ema_features(minute)
    event_day = features.index[150]
    pos1 = multiema.positions_from_features(features, index, 1)
    pos2 = multiema.positions_from_features(features, index, 2)
    assert np.allclose(pos1.loc[event_day].to_numpy(), pos1.loc[event_day - pd.Timedelta(minutes=1)].to_numpy())
    assert np.allclose(pos2.loc[event_day + pd.Timedelta(minutes=1)].to_numpy(), pos2.loc[event_day].to_numpy())
    assert (pos1.loc[event_day + pd.Timedelta(minutes=1)] > 0.0).all()
