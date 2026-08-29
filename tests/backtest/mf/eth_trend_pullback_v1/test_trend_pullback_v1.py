from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.mf.eth_trend_pullback_v1.engine import ExecutionConfig, run_backtest
from backtest.mf.eth_trend_pullback_v1.run import _slice_backtest
from backtest.mf.eth_trend_pullback_v1.strategy import build_features
from src.data_feed.binance_funding_archive_loader import BinanceFundingArchiveLoader


def _bars(n: int = 7000) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="15min")
    x = np.arange(n, dtype=float)
    close = 1200.0 + 0.08 * x + 25.0 * np.sin(x / 40.0) + 8.0 * np.sin(x / 9.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 3.0
    low = np.minimum(open_, close) - 3.0
    volume = 1000.0 + 100.0 * (1 + np.sin(x / 15.0))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_higher_timeframe_context_is_only_used_after_available_time():
    features = build_features(_bars())
    valid = features.dropna(subset=["used_h1_available_time", "used_h4_available_time"])
    assert not valid.empty
    assert (valid["used_h1_available_time"] <= valid["signal_available_time"]).all()
    assert (valid["used_h4_available_time"] <= valid["signal_available_time"]).all()


def test_future_mutation_cannot_change_past_signal_or_context():
    bars = _bars()
    cutoff = 5600
    base = build_features(bars)
    mutated = bars.copy()
    future = mutated.index[cutoff + 1 :]
    mutated.loc[future, "open"] *= 1.4
    mutated.loc[future, "high"] *= 1.7
    mutated.loc[future, "low"] *= 0.6
    mutated.loc[future, "close"] *= 1.5
    mutated.loc[future, "volume"] *= 9.0
    changed = build_features(mutated)
    past = bars.index[: cutoff + 1]
    for col in ("signal", "stop", "used_h1_timestamp", "used_h4_timestamp", "context_available_time_flag"):
        pd.testing.assert_series_equal(base.loc[past, col], changed.loc[past, col], check_names=False)


def _engine_frame(signal: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="15min")
    if signal > 0:
        stop = [95.0, np.nan, np.nan, np.nan]
        h4_long, h4_short = True, False
        h1_close, h1_ema = 101.0, 100.0
    else:
        stop = [105.0, np.nan, np.nan, np.nan]
        h4_long, h4_short = False, True
        h1_close, h1_ema = 99.0, 100.0
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "signal": [signal, 0, 0, 0],
            "stop": stop,
            "atr14": [2.0] * 4,
            "h1_ema20": [h1_ema] * 4,
            "h1_atr14": [2.0] * 4,
            "h1_close": [h1_close] * 4,
            "h4_regime_long": [h4_long] * 4,
            "h4_regime_short": [h4_short] * 4,
            "signal_available_time": idx + pd.Timedelta(minutes=15),
            "used_h1_timestamp": idx - pd.Timedelta(hours=1),
            "used_h1_available_time": idx,
            "used_h4_timestamp": idx - pd.Timedelta(hours=4),
            "used_h4_available_time": idx,
            "context_available_time_flag": [True] * 4,
        },
        index=idx,
    )


def test_entry_is_next_open_and_positive_funding_costs_long():
    frame = _engine_frame(1)
    funding = pd.DataFrame(
        {"funding_rate": [0.001], "mark_price": [100.0], "source": ["TEST"]},
        index=[frame.index[1]],
    )
    trades, _, ledger = run_backtest(
        frame,
        ExecutionConfig(fee_rate_per_side=0.0, slippage_rate_per_side=0.0, cooldown_bars=0, max_hold_bars=99, max_stop_pct=0.10),
        funding=funding,
    )
    assert len(trades) == 1
    assert trades[0]["entry_time"] == frame.index[1]
    assert trades[0]["entry"] == frame.iloc[1]["open"]
    assert trades[0]["funding_pnl"] < 0
    assert ledger[0]["funding_pnl"] < 0


def test_positive_funding_benefits_short_only_when_held():
    frame = _engine_frame(-1)
    # Use falling prices for a short so no accidental stop is hit.
    frame.loc[:, "open"] = [100.0, 100.0, 99.0, 98.0]
    frame.loc[:, "high"] = [101.0, 101.0, 100.0, 99.0]
    frame.loc[:, "low"] = [99.0, 98.0, 97.0, 96.0]
    frame.loc[:, "close"] = [100.0, 99.0, 98.0, 97.0]
    funding = pd.DataFrame(
        {"funding_rate": [0.001], "mark_price": [100.0], "source": ["TEST"]},
        index=[frame.index[2]],
    )
    trades, _, _ = run_backtest(
        frame,
        ExecutionConfig(fee_rate_per_side=0.0, slippage_rate_per_side=0.0, cooldown_bars=0, max_hold_bars=99, max_stop_pct=0.10),
        funding=funding,
    )
    assert len(trades) == 1
    assert trades[0]["funding_pnl"] > 0


def test_warmup_rows_are_never_tradable():
    idx = pd.date_range("2022-12-30", periods=20, freq="12h")
    frame = pd.DataFrame({"signal": np.ones(len(idx), dtype=int)}, index=idx)
    sliced = _slice_backtest(frame, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-05"))
    assert sliced.index.min() >= pd.Timestamp("2023-01-01")
    assert pd.Timestamp("2022-12-31 12:00") not in sliced.index


def test_binance_funding_loader_parses_mixed_fractional_timestamps(tmp_path):
    path = tmp_path / "funding.csv"
    pd.DataFrame(
        {
            "symbol": ["ETHUSDT", "ETHUSDT", "ETHUSDT"],
            "fundingTime": [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 08:00:00.006000+00:00",
                "2024-01-01 16:00:00+00:00",
            ],
            "fundingRate": [0.0001, -0.0002, 0.00005],
            "markPrice": [2000.0, 2010.0, 2020.0],
        }
    ).to_csv(path, index=False)
    frame = BinanceFundingArchiveLoader(path, timezone_offset_hours=8).load(
        "2024-01-01 07:00", "2024-01-02 01:00"
    )
    assert len(frame) == 3
    assert frame.index[0] == pd.Timestamp("2024-01-01 08:00:00")
    assert frame.index[1] == pd.Timestamp("2024-01-01 16:00:00")
    assert frame["funding_rate"].iloc[1] == -0.0002


def test_synthetic_trend_contains_at_least_one_causal_signal():
    features = build_features(_bars(10000))
    signals = features.loc[features["signal"].ne(0)]
    assert len(signals) > 0
    assert signals["context_available_time_flag"].all()
