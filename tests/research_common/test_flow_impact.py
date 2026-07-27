#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.flow_impact import (
    FlowImpactConfig,
    assign_pressure_event_clusters,
    build_flow_impact_features,
    detect_pressure_events,
    regularize_trade_bar_axis,
    response_state_labels,
    validate_flow_input,
)


def _bars(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2026-01-01", periods=n, freq="1min")
    delta = rng.normal(0.0, 20_000.0, n)
    delta[700:703] += np.array([500_000.0, 800_000.0, 600_000.0])
    delta[780:783] -= np.array([600_000.0, 900_000.0, 700_000.0])
    notional = 900_000.0 + rng.lognormal(mean=11.0, sigma=0.35, size=n)
    buy = (notional + delta) / 2.0
    sell = (notional - delta) / 2.0
    ret = rng.normal(0.0, 0.0001, n)
    ret[700:703] += 0.0005
    ret[780:783] -= 0.0005
    close = 1800.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    trades = np.maximum(20, np.round(notional / rng.uniform(3500.0, 6500.0, size=n)).astype(int))
    buy_trades = np.clip(np.round(trades * (0.5 + 0.4 * delta / notional)), 1, trades - 1).astype(int)
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
            "delta_notional": delta,
            "trades_count": trades,
            "buy_trades_count": buy_trades,
            "sell_trades_count": sell_trades,
            "large_buy_notional": np.maximum(delta, 0.0),
            "large_sell_notional": np.maximum(-delta, 0.0),
            "large_delta_notional": delta,
            "large_trades_count": np.full(n, 5),
            "max_trade_notional": np.maximum(np.abs(delta) * 0.2, 1000.0),
        },
        index=index,
    )


def test_rejects_plain_ohlcv_fallback() -> None:
    bars = _bars()[["open", "high", "low", "close", "volume"]]
    with pytest.raises(RuntimeError, match="requires populated OKX trade-bar"):
        validate_flow_input(bars)


def test_regularized_gap_is_explicit_and_not_feature_ready() -> None:
    raw = _bars().drop(_bars().index[650:653])
    regular = regularize_trade_bar_axis(raw, bar_delta=pd.Timedelta(minutes=1))
    assert int((~regular["source_bar_observed_flag"]).sum()) == 3
    features = build_flow_impact_features(
        regular,
        FlowImpactConfig(pressure_windows=(3,), baseline_bars=240, baseline_min_periods=120),
    )
    assert not features.loc[regular.index[652], "feature_ready_w3"]


def test_future_perturbation_does_not_change_past_features() -> None:
    bars = regularize_trade_bar_axis(_bars(), bar_delta=pd.Timedelta(minutes=1))
    cfg = FlowImpactConfig(pressure_windows=(1, 3, 5), baseline_bars=240, baseline_min_periods=120)
    base = build_flow_impact_features(bars, cfg)
    changed = bars.copy()
    cutoff = bars.index[750]
    changed.loc[changed.index > cutoff, ["close", "high", "low", "delta_notional"]] *= 10.0
    alt = build_flow_impact_features(changed, cfg)
    columns = [column for column in base.columns if column.startswith(("pressure_", "flow_", "price_response", "activity_", "notional_"))]
    pd.testing.assert_frame_equal(base.loc[:cutoff, columns], alt.loc[:cutoff, columns])


def test_detects_symmetric_events_and_clusters_cross_window_duplicates() -> None:
    bars = regularize_trade_bar_axis(_bars(), bar_delta=pd.Timedelta(minutes=1))
    cfg = FlowImpactConfig(pressure_windows=(1, 3, 5), baseline_bars=240, baseline_min_periods=120, min_pressure_z=1.0)
    features = build_flow_impact_features(bars, cfg)
    events = detect_pressure_events(features, windows=cfg.pressure_windows, min_pressure_z=1.0)
    assert {1, -1}.issubset(set(events["side"].unique()))
    clustered = assign_pressure_event_clusters(events, cluster_gap_bars=5)
    assert clustered["event_cluster_id"].nunique() <= len(clustered)
    assert int(clustered["cluster_primary_flag"].sum()) == clustered["event_cluster_id"].nunique()
    assert clustered.groupby("event_cluster_id")["side"].nunique().max() == 1


def test_response_state_labels_are_predeclared() -> None:
    labels = response_state_labels(pd.Series([-0.1, 0.0, 0.3, 0.9, np.nan]))
    assert labels.tolist() == [
        "opposite_or_absorbed",
        "flat_0_0.25",
        "moderate_0.25_0.75",
        "effective_ge_0.75",
        "NA",
    ]
