#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

import src.research_common.flow_pa_accumulation as flow_pa_module
from src.research_common.flow_impact import regularize_trade_bar_axis
from src.research_common.flow_pa_accumulation import (
    AccumulatedPAConfig,
    build_accumulated_features,
    build_causal_pivots,
    detect_accumulated_pa_setups,
    resolve_position_conflicts,
    simulate_structural_exits,
)


def _bars(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    returns = rng.normal(0.0, 0.00018, n)
    delta = rng.normal(0.0, 35_000.0, n)
    # Sustained buy pressure with early impact and late decay.
    delta[800:810] += 450_000.0
    returns[800:805] += 0.00055
    returns[805:810] += 0.00005
    notional = 1_200_000.0 + rng.lognormal(12.0, 0.25, n)
    buy = np.maximum((notional + delta) / 2.0, 1.0)
    sell = np.maximum((notional - delta) / 2.0, 1.0)
    close = 1800.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.00002, 0.00015, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.00002, 0.00015, n))
    trades = np.maximum(50, np.round(notional / 5000.0).astype(int))
    buy_trades = np.clip(np.round(trades * (0.5 + 0.35 * delta / notional)), 1, trades - 1).astype(int)
    sell_trades = trades - buy_trades
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": notional / close,
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "delta_notional": buy - sell,
            "trades_count": trades,
            "buy_trades_count": buy_trades,
            "sell_trades_count": sell_trades,
            "large_buy_notional": np.maximum(delta, 0.0),
            "large_sell_notional": np.maximum(-delta, 0.0),
            "large_delta_notional": delta,
            "large_trades_count": np.full(n, 5),
            "max_trade_notional": np.maximum(np.abs(delta) * 0.25, 1000.0),
        },
        index=idx,
    )


def test_causal_pivots_are_available_after_confirmation_plus_one() -> None:
    bars = _bars(100)
    pivots = build_causal_pivots(bars, left=2, right=2)
    assert not pivots.empty
    assert (pivots["available_pos"] == pivots["pivot_pos"] + 3).all()
    assert (pd.to_datetime(pivots["available_time"]) > pd.to_datetime(pivots["pivot_time"])).all()


def test_accumulated_features_do_not_change_before_future_perturbation() -> None:
    raw = regularize_trade_bar_axis(_bars(), bar_delta=pd.Timedelta(minutes=1))
    cfg = AccumulatedPAConfig(
        accumulation_windows=(5, 10),
        baseline_bars=240,
        baseline_min_periods=120,
    )
    base = build_accumulated_features(raw, cfg)
    changed = raw.copy()
    cutoff = raw.index[850]
    future = changed.index > cutoff
    changed.loc[future, ["open", "high", "low", "close", "delta_notional"]] *= 3.0
    alt = build_accumulated_features(changed, cfg)
    cols = [c for c in base.columns if c.startswith(("pressure_", "impact_decay", "late_directional", "early_impact", "late_impact"))]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], alt.loc[:cutoff, cols])


def test_structural_exit_uses_conservative_same_bar_order() -> None:
    bars = _bars(100)
    pos = 50
    entry = float(bars.iloc[pos]["open"])
    setups = pd.DataFrame(
        [
            {
                "setup_id": 1,
                "spec_id": "x",
                "branch": "continuation",
                "profile": "x",
                "pressure_window_bars": 5,
                "pressure_side": 1,
                "trade_side": 1,
                "side_name": "LONG",
                "event_pos": pos - 2,
                "signal_pos": pos - 1,
                "entry_pos": pos,
                "event_time": bars.index[pos - 2],
                "signal_time": bars.index[pos - 1],
                "entry_time": bars.index[pos],
                "entry_price": entry,
                "break_level": entry,
                "opposite_level": entry * 0.99,
                "attack_high": entry * 1.01,
                "attack_low": entry * 0.99,
                "stop_price": entry * 0.999,
                "target_price": entry * 1.001,
                "risk_bps": 10.0,
                "reward_bps": 10.0,
                "reward_risk": 1.0,
                "pressure_z": 2.0,
                "accumulated_notional": 1e6,
                "flow_ratio": 0.3,
                "activity_z": 1.0,
                "flow_persistence": 0.8,
                "pressure_effectiveness": 1.0,
                "impact_decay_ratio": 1.0,
                "late_directional_flow_share": 0.5,
                "price_response": 0.002,
            }
        ]
    )
    bars = bars.copy()
    bars.iloc[pos, bars.columns.get_loc("low")] = entry * 0.998
    bars.iloc[pos, bars.columns.get_loc("high")] = entry * 1.002
    trades = simulate_structural_exits(
        bars,
        setups,
        normal_cost=0.0015,
        fee_only_cost=0.0011,
        max_holding_bars=20,
    )
    assert trades.loc[0, "exit_reason"] == "stop_same_bar_conservative"
    assert np.isclose(trades.loc[0, "exit_price"], entry * 0.999)


def test_conflict_resolution_keeps_highest_rr_at_same_entry() -> None:
    trades = pd.DataFrame(
        {
            "entry_pos": [10, 10, 30],
            "exit_pos": [20, 15, 35],
            "reward_risk": [1.2, 2.0, 1.5],
            "pressure_z": [3.0, 2.5, 2.0],
        }
    )
    out = resolve_position_conflicts(trades)
    assert len(out) == 2
    assert float(out.iloc[0]["reward_risk"]) == 2.0


def test_setup_detection_converts_full_columns_only_once_per_window(monkeypatch) -> None:
    bars = regularize_trade_bar_axis(_bars(1600), bar_delta=pd.Timedelta(minutes=1))
    cfg = AccumulatedPAConfig(
        accumulation_windows=(5, 10),
        baseline_bars=240,
        baseline_min_periods=120,
        min_accumulation_z=0.5,
        min_risk_bps=1.0,
        max_risk_bps=300.0,
        min_reward_risk=0.2,
    )
    features = build_accumulated_features(bars, cfg)
    pivots = build_causal_pivots(bars, left=2, right=2)
    original_numeric = flow_pa_module._numeric
    calls = {"count": 0}

    def counted_numeric(*args, **kwargs):
        calls["count"] += 1
        return original_numeric(*args, **kwargs)

    monkeypatch.setattr(flow_pa_module, "_numeric", counted_numeric)
    detect_accumulated_pa_setups(
        bars,
        features,
        pivots,
        cfg,
        progress_enabled=False,
    )

    # Four OHLC conversions plus ten feature conversions per accumulation
    # window. The count must not grow with the number of detected events.
    assert calls["count"] <= 4 + 10 * len(cfg.accumulation_windows)
