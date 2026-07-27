from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.market_state import MarketStateConfig, MarketStateDataBundle, MarketStateEngine


def _rich_sample(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(20260716)
    index = pd.date_range("2026-01-01", periods=rows, freq="1min")
    returns = rng.normal(0.0, 0.00005, rows)
    buy_start, buy_end = min(350, rows), min(650, rows)
    if buy_end > buy_start:
        returns[buy_start:buy_end] = 0.00025 + rng.normal(0.0, 0.00003, buy_end - buy_start)
    absorb_start, absorb_end = min(750, rows), min(950, rows)
    if absorb_end > absorb_start:
        returns[absorb_start:absorb_end] = 0.00003 + rng.normal(0.0, 0.000025, absorb_end - absorb_start)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    width = np.full(rows, 0.00035)
    if absorb_end > absorb_start:
        width[absorb_start:absorb_end] = 0.0011
    high = np.maximum(open_, close) * (1.0 + width)
    low = np.minimum(open_, close) * (1.0 - width)
    volume = rng.uniform(100, 150, rows)
    notional = close * volume * 10.0

    delta_ratio = rng.normal(0.0, 0.015, rows)
    if buy_end > buy_start:
        delta_ratio[buy_start:buy_end] = 0.18 + rng.normal(0.0, 0.015, buy_end - buy_start)
    # Aggressive selling with little/positive price damage: absorption episode.
    if absorb_end > absorb_start:
        delta_ratio[absorb_start:absorb_end] = -0.20 + rng.normal(0.0, 0.015, absorb_end - absorb_start)
    delta = notional * np.clip(delta_ratio, -0.45, 0.45)
    buy = (notional + delta) / 2.0
    sell = (notional - delta) / 2.0
    large_total = notional * 0.18
    large_delta = large_total * np.clip(np.sign(delta_ratio) * 0.75, -1.0, 1.0)

    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "trades_count": np.maximum(20, (volume * 2).astype(int)),
            "notional": notional, "buy_notional": buy, "sell_notional": sell,
            "delta_notional": delta,
            "large_buy_notional": (large_total + large_delta) / 2.0,
            "large_sell_notional": (large_total - large_delta) / 2.0,
            "large_delta_notional": large_delta,
        },
        index=index,
    )


def _config() -> MarketStateConfig:
    return MarketStateConfig(
        fast_trend_window=12,
        trend_window=48,
        slow_trend_window=160,
        volatility_window=24,
        activity_window=10,
        baseline_window=240,
        flow_fast_window=3,
        flow_window=12,
        flow_slow_window=30,
        location_window=48,
        structure_window=160,
        absorption_threshold=0.35,
    )


def _compute(df: pd.DataFrame) -> pd.DataFrame:
    bundle = MarketStateDataBundle.from_frame(
        df,
        source="trade_bar",
        timestamp_semantics="bar_start",
        bar_duration="1min",
    )
    return MarketStateEngine(_config()).compute(bundle).frame


def test_orderflow_distinguishes_effective_buying_from_absorbed_selling() -> None:
    frame = _compute(_rich_sample())
    buying = frame.iloc[450:620]
    absorbed = frame.iloc[800:930]

    assert buying["orderflow_available"].mean() > 0.95
    assert buying["flow_score"].median() > 0.12
    assert buying["flow_state"].isin(["buy_building", "buy_persistent", "buy_pressure"]).mean() > 0.90
    assert (buying["impact_state"] == "buy_effective").mean() > 0.55

    assert absorbed["flow_score"].median() < -0.12
    assert absorbed["sell_absorption_score"].median() > 0.35
    assert (absorbed["impact_state"] == "sell_absorbed").mean() > 0.90


def test_location_levels_exclude_current_bar_and_detect_downside_sweep_reclaim() -> None:
    df = _rich_sample(600)
    target = df.index[-1]
    prior_low = float(df["low"].iloc[-49:-1].min())
    df.loc[target, "open"] = prior_low * 1.001
    df.loc[target, "high"] = prior_low * 1.003
    df.loc[target, "low"] = prior_low * 0.995
    df.loc[target, "close"] = prior_low * 1.0015
    frame = _compute(df)

    assert frame.loc[target, "local_support"] == prior_low
    assert bool(frame.loc[target, "downside_sweep_reclaim"])
    assert frame.loc[target, "location_state"] == "downside_sweep_reclaim"


def test_appending_future_rows_does_not_change_orderflow_or_location_history() -> None:
    full = _rich_sample()
    prefix = full.iloc[:900]
    prefix_frame = _compute(prefix)
    full_frame = _compute(full).loc[prefix.index]

    numeric = [
        "flow_score", "flow_persistence", "flow_acceleration", "flow_price_effectiveness",
        "sell_absorption_score", "buy_absorption_score", "structural_location_score",
        "local_support", "local_resistance", "trade_context_score",
    ]
    pdt.assert_frame_equal(
        prefix_frame[numeric],
        full_frame[numeric],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    for column in ("flow_state", "impact_state", "location_state", "trade_context_state"):
        pdt.assert_series_equal(prefix_frame[column], full_frame[column])
