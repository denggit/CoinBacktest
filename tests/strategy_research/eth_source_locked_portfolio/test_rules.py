from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy_research.eth_source_locked_portfolio.data import resample_daily_causal
from src.strategy_research.eth_source_locked_portfolio.rules import (
    _donchian_state,
    build_mop_tsmom,
    build_turtle_context,
    build_zarattini,
    turtle_n,
)


def _daily(n: int = 500, start: str = "2022-01-01 08:00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1D")
    close = 100.0 + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "available_time": idx + pd.Timedelta(days=1)}, index=idx)


def test_plus8_daily_anchor_and_available_time() -> None:
    idx = pd.date_range("2025-01-01 08:00:00", periods=1440, freq="1min")
    x = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}, index=idx)
    out = resample_daily_causal(x, 8)
    assert list(out.index) == [pd.Timestamp("2025-01-01 08:00:00")]
    assert out.iloc[0]["available_time"] == pd.Timestamp("2025-01-02 08:00:00")


def test_donchian_long_only_never_short() -> None:
    d = _daily(30)
    s = _donchian_state(d["close"], 5, allow_short=False)
    assert (s >= 0).all()
    assert s.iloc[-1] == 1


def test_donchian_long_short_can_short() -> None:
    d = _daily(30)
    d["close"] = d["close"].iloc[::-1].to_numpy()
    s = _donchian_state(d["close"], 5, allow_short=True)
    assert s.iloc[-1] == -1


def test_zarattini_has_nine_model_equal_weight_target() -> None:
    d = _daily(500)
    r = build_zarattini(d, allow_short=False)
    assert r.strategy_id == "SL01_ZARATTINI_LONG"
    assert r.schedule["raw_target"].max() <= 2.0 + 1e-12
    assert (r.schedule["raw_target"] >= 0).all()


def test_mop_rebalances_once_at_first_valid_observation_of_each_month() -> None:
    d = _daily(800)
    r = build_mop_tsmom(d)
    times = pd.DatetimeIndex(r.schedule["signal_time"])
    assert len(times) > 3
    months = times.to_period("M")
    assert not months.duplicated().any()
    # Once the 12-month warm-up has produced a valid signal, subsequent full months
    # rebalance on the first +8 daily availability timestamp. The first valid month
    # itself may begin later than day 1 because the lookback becomes valid mid-month.
    assert all(t.day == 1 for t in times[1:])


def test_turtle_n_matches_initial_20_day_average_then_recursive() -> None:
    d = _daily(30)
    n = turtle_n(d)
    assert np.isnan(n.iloc[18])
    assert np.isfinite(n.iloc[19])
    tr20 = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["close"].shift(1) - d["low"]).abs(),
    ], axis=1).max(axis=1).iloc[:20].mean()
    assert np.isclose(n.iloc[19], tr20)
    tr21 = max(d["high"].iloc[20] - d["low"].iloc[20], abs(d["high"].iloc[20] - d["close"].iloc[19]), abs(d["close"].iloc[19] - d["low"].iloc[20]))
    assert np.isclose(n.iloc[20], (19*n.iloc[19] + tr21)/20)


def test_turtle_context_uses_55_and_20_day_thresholds() -> None:
    d = _daily(100)
    ctx = build_turtle_context(d)
    row = ctx.iloc[-1]
    assert np.isclose(row.entry_high, d["high"].iloc[-55:].max())
    assert np.isclose(row.entry_low, d["low"].iloc[-55:].min())
    assert np.isclose(row.exit_high, d["high"].iloc[-20:].max())
    assert np.isclose(row.exit_low, d["low"].iloc[-20:].min())
