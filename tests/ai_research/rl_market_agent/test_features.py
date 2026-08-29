from __future__ import annotations

import pandas as pd

from src.ai_research.rl_market_agent.features import build_fixed_bar_features, build_trade_bar_features


def _trade_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=20, freq="1min")
    n = len(idx)
    return pd.DataFrame({
        "open": [100+i for i in range(n)], "high": [101+i for i in range(n)],
        "low": [99+i for i in range(n)], "close": [100.5+i for i in range(n)],
        "notional": [1000.0]*n, "delta_notional": [100.0]*n, "trades_count": [10.0]*n,
        "buy_notional": [550.0]*n, "sell_notional": [450.0]*n,
        "large_buy_notional": [100.0]*n, "large_sell_notional": [50.0]*n,
        "large_delta_notional": [50.0]*n, "large_trades_count": [2.0]*n,
        "max_trade_notional": [200.0]*n, "vwap": [100.25+i for i in range(n)],
    }, index=idx)


def test_future_mutation_does_not_change_past_trade_features():
    frame = _trade_frame()
    base = build_trade_bar_features(frame, prefix="t", windows=["5min"])
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("delta_notional")] = 999999.0
    after = build_trade_bar_features(changed, prefix="t", windows=["5min"])
    pd.testing.assert_series_equal(base.iloc[-2], after.iloc[-2])


def test_fixed_bar_volume_baseline_uses_history_only():
    idx = pd.date_range("2026-01-01", periods=60, freq="5min")
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}, index=idx)
    base = build_fixed_bar_features(frame, prefix="k")
    changed = frame.copy(); changed.iloc[-1, changed.columns.get_loc("volume")] = 1e9
    after = build_fixed_bar_features(changed, prefix="k")
    pd.testing.assert_series_equal(base.iloc[-2], after.iloc[-2])


def test_range_event_features_accept_loader_schema_with_end_ts_as_index_and_column():
    """Mirror OKXRangeBarLoader.set_index('end_ts', drop=False)."""
    from src.ai_research.rl_market_agent.features import build_range_event_features

    end_ts = pd.date_range("2026-01-01 00:01:00", periods=3, freq="1min")
    frame = pd.DataFrame({
        "bar_id": [1, 2, 3],
        "end_ts": end_ts,
        "duration_seconds": [12.0, 18.0, 10.0],
        "direction": [1, -1, 1],
        "notional": [1000.0, 1200.0, 1500.0],
        "delta_notional": [100.0, -200.0, 300.0],
        "large_delta_notional": [50.0, -100.0, 150.0],
        "trades_count": [10.0, 12.0, 15.0],
        "taker_buy_ratio": [0.55, 0.42, 0.63],
        "max_trade_notional": [200.0, 250.0, 300.0],
    }).set_index("end_ts", drop=False)
    frame.index.name = "end_ts"

    result = build_range_event_features(frame, prefix="r", windows=["5min"])

    assert list(result.index) == list(end_ts)
    assert result.index.name == "end_ts"
    assert result.loc[end_ts[-1], "r__last_direction"] == 1
    assert result.loc[end_ts[-1], "r__5m__activity_per_min"] > 0


def test_footprint_summary_accepts_named_index_collision_without_changing_timestamp_column():
    """Defensively support future loaders that retain end_ts as an index too."""
    from src.ai_research.rl_market_agent.features import summarize_footprint_bars

    end_ts = pd.to_datetime(["2026-01-01 00:01:00", "2026-01-01 00:01:00"])
    frame = pd.DataFrame({
        "bar_id": [1, 1],
        "end_ts": end_ts,
        "price_bucket": [100.0, 101.0],
        "notional": [500.0, 700.0],
        "delta_notional": [-100.0, 200.0],
        "buy_notional": [200.0, 450.0],
        "sell_notional": [300.0, 250.0],
        "large_delta_notional": [-50.0, 100.0],
        "max_trade_notional": [150.0, 220.0],
    }).set_index("end_ts", drop=False)
    frame.index.name = "end_ts"

    result = summarize_footprint_bars(frame, prefix="fp")

    assert len(result) == 1
    assert result.index[0] == end_ts[0]
    assert result.loc[end_ts[0], "fp__bucket_count_log"] > 0


def test_official_1m_resample_builds_only_complete_left_labeled_bars():
    from src.ai_research.rl_market_agent.features import resample_ohlcv_from_1m_bars

    idx = pd.date_range("2026-01-01 00:00:00", periods=12, freq="1min")
    frame = pd.DataFrame({
        "open": range(100, 112),
        "high": range(101, 113),
        "low": range(99, 111),
        "close": [x + 0.5 for x in range(100, 112)],
        "volume": 1.0,
    }, index=idx)

    out = resample_ohlcv_from_1m_bars(frame, timeframe="5m")

    # 00:00-00:04 and 00:05-00:09 are complete; 00:10-00:14 is incomplete.
    assert list(out.index) == [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 00:05:00")]
    assert out.loc[pd.Timestamp("2026-01-01 00:00:00"), "open"] == 100
    assert out.loc[pd.Timestamp("2026-01-01 00:00:00"), "close"] == 104.5
    assert out.loc[pd.Timestamp("2026-01-01 00:00:00"), "volume"] == 5.0


def test_official_1m_daily_resample_matches_okx_local_utc_plus8_anchor():
    from src.ai_research.rl_market_agent.features import resample_ohlcv_from_1m_bars

    idx = pd.date_range("2026-01-01 08:00:00", periods=24 * 60, freq="1min")
    frame = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0,
    }, index=idx)

    out = resample_ohlcv_from_1m_bars(frame, timeframe="1D", daily_offset="8h")

    assert list(out.index) == [pd.Timestamp("2026-01-01 08:00:00")]
    assert out.iloc[0]["volume"] == 1440.0


def test_official_1m_daily_resample_uses_shift_not_pandas_offset_warning():
    """Daily +08 anchor must work without relying on pandas resample(offset=...)."""
    import warnings
    from src.ai_research.rl_market_agent.features import resample_ohlcv_from_1m_bars

    idx = pd.date_range("2026-01-01 07:59:00", periods=24 * 60 + 2, freq="1min")
    frame = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0,
    }, index=idx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = resample_ohlcv_from_1m_bars(frame, timeframe="1D", daily_offset="8h")

    assert list(out.index) == [pd.Timestamp("2026-01-01 08:00:00")]
    assert out.iloc[0]["volume"] == 1440.0
    assert not any("offset" in str(w.message).lower() and "resampl" in str(w.message).lower() for w in caught)
