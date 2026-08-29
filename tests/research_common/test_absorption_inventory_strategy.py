from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.absorption_inventory_strategy import (
    AccountConfig,
    StrategyConfig,
    build_scale_states,
    simulate_cross_inventory,
)


def _feature_frame(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    f = pd.DataFrame(index=idx)
    f["feature_ready"] = True
    f["flow_side"] = -1
    f["pressure_z"] = 2.0
    f["flow_persistence"] = 0.8
    f["price_response_norm"] = -0.2
    f["same_side_adjacent_window"] = False
    f["pressure_retention"] = 1.0
    f["response_retention"] = 1.0
    f["prior_defense_count_long"] = 0
    f["prior_defense_count_short"] = 0
    f["hold_ratio_long"] = 0.0
    f["hold_ratio_short"] = 0.0
    f["floor_stability_atr"] = 9.0
    f["ceiling_stability_atr"] = 9.0
    f["spring_reclaim_long"] = False
    f["spring_reclaim_short"] = False
    return f


def test_failed_sell_pressure_creates_one_fresh_long_vote_not_bar_spam() -> None:
    out = build_scale_states(_feature_frame(), StrategyConfig())
    assert out["vote"].tolist()[0] == 1
    assert int((out["vote"] == 1).sum()) == 1
    assert out.iloc[0]["family"] == "pressure_failed"


def test_effective_sell_pressure_votes_short() -> None:
    f = _feature_frame()
    f["price_response_norm"] = 1.0
    out = build_scale_states(f, StrategyConfig())
    assert out.iloc[0]["vote"] == -1
    assert out.iloc[0]["family"] == "pressure_effective"


def _market(signals: list[int], opens: list[float] | None = None, closes: list[float] | None = None) -> pd.DataFrame:
    n = len(signals)
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    opens = opens or [100.0] * n
    closes = closes or opens
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "signal": signals,
            "signal_scale": ["1H"] * n,
            "signal_family": ["pressure_failed"] * n,
        },
        index=idx,
    )


def test_vote_executes_next_open() -> None:
    f = _market([1, 0, -1, 0])
    path, orders, _ = simulate_cross_inventory(f, account=AccountConfig(fee_rate_per_fill=0.0))
    assert len(orders) == 2
    assert pd.Timestamp(orders.iloc[0]["source_signal_time"]) == f.index[0]
    assert pd.Timestamp(orders.iloc[0]["execution_time"]) == f.index[1]
    assert path.iloc[0]["net_qty"] == 0
    assert path.iloc[1]["net_qty"] > 0


def test_long_vote_can_never_turn_into_sell_when_already_over_cap() -> None:
    # Build a long position, then sharply reduce equity so existing exposure is
    # above the new 10x cap. A fresh LONG must be blocked, never sold down.
    f = _market([1] * 12, closes=[100.0] * 10 + [91.0, 91.0])
    path, orders, _ = simulate_cross_inventory(
        f,
        account=AccountConfig(initial_equity=1000, leverage=1.0, vote_margin_fraction=0.5, fee_rate_per_fill=0.0),
    )
    long_orders = orders[orders["signal"] > 0]
    assert not long_orders.empty
    assert (long_orders["delta_notional"] >= 0).all()


def test_short_vote_can_never_turn_into_buy_when_already_over_cap() -> None:
    f = _market([-1] * 12, closes=[100.0] * 10 + [109.0, 109.0])
    _, orders, _ = simulate_cross_inventory(
        f,
        account=AccountConfig(initial_equity=1000, leverage=1.0, vote_margin_fraction=0.5, fee_rate_per_fill=0.0),
    )
    short_orders = orders[orders["signal"] < 0]
    assert not short_orders.empty
    assert (short_orders["delta_notional"] <= 0).all()


def test_signal_builder_has_no_position_input() -> None:
    # Regression guard for the design: signal classification is pure market state.
    import inspect

    params = inspect.signature(build_scale_states).parameters
    assert "position" not in params
    assert "pnl" not in params


def test_htf_fresh_vote_is_not_forward_filled_on_1m_axis() -> None:
    from src.research_common.absorption_inventory_strategy import ScaleSpec, build_multiscale_votes
    from src.research_common.multiscale_absorption import AbsorptionFeatureConfig

    idx = pd.date_range("2026-01-01", periods=60 * 36, freq="1min")
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0, 0.0005, len(idx))
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    buy = rng.lognormal(11, 0.7, len(idx))
    sell = rng.lognormal(11, 0.7, len(idx))
    raw = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0002,
            "low": np.minimum(open_, close) * 0.9998,
            "close": close,
            "volume": 100.0,
            "notional": buy + sell,
            "buy_notional": buy,
            "sell_notional": sell,
            "delta_notional": buy - sell,
            "trades_count": 100,
        },
        index=idx,
    )

    def tiny() -> AbsorptionFeatureConfig:
        return AbsorptionFeatureConfig(
            process_window=1,
            baseline_bars=4,
            baseline_min_periods=2,
            floor_lookback=3,
            defense_lookback=4,
            reclaim_bars=1,
            atr_lookback=3,
        )

    specs = (
        ScaleSpec("5m", "5min", pd.Timedelta(minutes=5), tiny()),
        ScaleSpec("15m", "15min", pd.Timedelta(minutes=15), tiny()),
        ScaleSpec("1H", "1h", pd.Timedelta(hours=1), tiny()),
        ScaleSpec("4H", "4h", pd.Timedelta(hours=4), tiny()),
    )
    frame, _ = build_multiscale_votes(raw, scale_specs=specs)
    active = frame.index[frame["signal"].ne(0)]
    assert len(active) > 0
    # HTF available times land on their bar boundaries. If a vote were asof/ffill
    # leaked, we would see signals on arbitrary adjacent minutes.
    for ts, row in frame.loc[active].iterrows():
        if row["signal_scale"] == "15m":
            assert ts.minute % 15 == 0
        elif row["signal_scale"] == "1H":
            assert ts.minute == 0
        elif row["signal_scale"] == "4H":
            assert ts.minute == 0 and ts.hour % 4 == 0


def test_slippage_is_charged_to_equity() -> None:
    f = _market([1, 0, 0])
    p0, _, _ = simulate_cross_inventory(
        f, account=AccountConfig(initial_equity=1000, leverage=1, vote_margin_fraction=0.1, fee_rate_per_fill=0, slippage_bps_per_fill=0)
    )
    p1, _, _ = simulate_cross_inventory(
        f, account=AccountConfig(initial_equity=1000, leverage=1, vote_margin_fraction=0.1, fee_rate_per_fill=0, slippage_bps_per_fill=10)
    )
    assert p1.iloc[-1]["equity"] < p0.iloc[-1]["equity"]


def test_intrabar_adverse_extreme_can_liquidate_cross_account() -> None:
    f = _market([1, 0, 0], opens=[100, 100, 100], closes=[100, 100, 100])
    f.loc[f.index[1], "low"] = 1.0
    path, _, summary = simulate_cross_inventory(
        f,
        account=AccountConfig(
            initial_equity=1000,
            leverage=10,
            vote_margin_fraction=1.0,
            fee_rate_per_fill=0.0,
            maintenance_margin_rate=0.005,
        ),
    )
    assert summary["liquidated"] is True
    assert path.iloc[-1]["equity"] == 0.0
