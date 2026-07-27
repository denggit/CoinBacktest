from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.market_state import MarketStateDataBundle, MarketStateEngine
from src.market_state.validity_audit import (
    ValidityAuditConfig,
    build_event_definitions,
    build_forward_path_frame,
    extract_event_rows,
    summarize_breakdowns,
    summarize_event_rows,
)


def rich_trade_bars(rows: int = 1800) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="1min")
    x = np.arange(rows, dtype=float)
    returns = 0.00004 + 0.00035 * np.sin(x / 37.0) + 0.00020 * np.sin(x / 11.0)
    close = 2000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * (0.0008 + 0.0002 * (1.0 + np.sin(x / 17.0)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    notional = 1_000_000.0 + 250_000.0 * (1.0 + np.sin(x / 23.0))
    delta_ratio = np.clip(0.25 * np.sin(x / 29.0) + 0.15 * np.sign(returns), -0.8, 0.8)
    delta = notional * delta_ratio
    buy = 0.5 * (notional + delta)
    sell = 0.5 * (notional - delta)
    large_delta = delta * 0.20
    large_buy = np.maximum(large_delta, 0.0)
    large_sell = np.maximum(-large_delta, 0.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": notional / close,
            "trades_count": 500.0 + 50.0 * np.sin(x / 13.0),
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "delta_notional": delta,
            "large_buy_notional": large_buy,
            "large_sell_notional": large_sell,
            "large_delta_notional": large_delta,
        },
        index=index,
    )


def compute_state(df: pd.DataFrame) -> pd.DataFrame:
    bundle = MarketStateDataBundle.from_frame(
        df,
        source="test",
        timestamp_semantics="bar_start",
        bar_duration="1min",
    )
    return MarketStateEngine().compute(bundle).frame


def test_market_state_prefix_is_append_invariant() -> None:
    df = rich_trade_bars(1900)
    prefix = compute_state(df.iloc[:1600])
    full = compute_state(df)
    columns = [
        "trend_score",
        "volatility_z",
        "flow_score",
        "sell_absorption_score",
        "buy_absorption_score",
        "trend_state",
        "flow_state",
        "impact_state",
        "location_state",
        "trade_context_state",
        "available_time",
    ]
    pdt.assert_frame_equal(prefix[columns], full.loc[prefix.index, columns], check_dtype=False)


def test_forward_path_uses_next_open_and_future_only() -> None:
    df = rich_trade_bars(1600)
    state = compute_state(df)
    cfg = ValidityAuditConfig(horizons_bars=(5,), trap_horizon_bars=5)
    path = build_forward_path_frame(state, cfg)
    pos = 1200
    entry = float(df["open"].iloc[pos + 1])
    exit_ = float(df["close"].iloc[pos + 5])
    assert path["entry_time"].iloc[pos] == df.index[pos + 1]
    assert path["entry_price"].iloc[pos] == entry
    assert path["exit_time_h5"].iloc[pos] == df.index[pos + 5]
    assert np.isclose(path["long_return_h5"].iloc[pos], exit_ / entry - 1.0)
    assert path["entry_time"].iloc[pos] >= path["available_time"].iloc[pos]


def test_event_rows_have_causal_entry_and_cost() -> None:
    df = rich_trade_bars(1800)
    state = compute_state(df)
    cfg = ValidityAuditConfig(horizons_bars=(5, 15), trap_horizon_bars=15, minimum_events=1)
    path = build_forward_path_frame(state, cfg)
    definitions = build_event_definitions(path, event_cooldown_bars=0)
    rows = extract_event_rows(path, definitions, cfg, profile="base")
    assert not rows.empty
    assert (pd.to_datetime(rows["entry_time"]) > pd.to_datetime(rows["signal_time"])).all()
    assert (pd.to_datetime(rows["entry_time"]) >= pd.to_datetime(rows["available_time"])).all()
    assert np.allclose(rows["net_return"], rows["gross_return"] - cfg.round_trip_cost)
    summary = summarize_event_rows(rows, cfg)
    yearly, period = summarize_breakdowns(rows)
    assert not summary.empty
    assert not yearly.empty
    assert set(period["period"]).issubset({"pre_holdout", "holdout", "all"})


def test_transition_event_is_defined_without_future_values() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="1min")
    frame = pd.DataFrame(
        {
            "data_ready": True,
            "orderflow_available": True,
            "location_available": True,
            "trend_state": "balanced",
            "flow_state": "balanced",
            "impact_state": "neutral",
            "location_state": "middle_zone",
            "trade_context_state": "wait",
        },
        index=index,
    )
    frame.loc[index[5:8], "impact_state"] = "sell_effective"
    frame.loc[index[8], "impact_state"] = "sell_absorbed"
    definitions = build_event_definitions(
        frame,
        transition_lookback_bars=5,
        event_cooldown_bars=0,
    )
    target = next(d for d in definitions if d.event_name == "transition_sell_effective_to_absorbed")
    assert target.mask.loc[index[8]]
    assert int(target.mask.sum()) == 1
