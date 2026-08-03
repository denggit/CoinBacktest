from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.future_process_forecast.config import DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
from src.ai_research.future_process_forecast.events import discover_events
from src.ai_research.future_process_forecast.micro_features import build_micro_decision_features


def _path(closes: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="15min")
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.0005
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.9995
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def test_r033_contract_keeps_2026_sealed_and_requires_micro() -> None:
    config = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    config.validate()
    assert pd.Timestamp(config.research_end) < pd.Timestamp(config.sealed_holdout_start)
    assert config.micro_required
    assert config.forecast_horizons_hours == (6, 12, 24)


def test_directional_event_atlas_finds_launch_not_every_following_bar() -> None:
    config = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    flat = np.full(160, 2000.0)
    launch = 2000.0 * np.exp(np.linspace(0.0, np.log(1.065), 48))
    after = np.full(160, launch[-1])
    path = _path(np.concatenate([flat, launch, after]))
    events = discover_events(path, config)
    up = events.loc[events["process"] == "up_expansion"]
    assert 1 <= len(up) <= 3
    assert up["target_move"].min() >= config.directional.target_floor
    assert up["start_time"].min() >= path.index[140]


def test_volatile_range_event_is_independent_from_directional_expansion() -> None:
    config = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    flat = np.full(160, 2000.0)
    oscillation = 2000.0 * (1.0 + 0.028 * np.sin(np.linspace(0, 8 * np.pi, 64)))
    after = np.full(120, 2000.0)
    events = discover_events(_path(np.concatenate([flat, oscillation, after])), config)
    assert (events["process"] == "volatile_range").any()


def _micro_bars(end: str = "2024-01-01 02:00:00") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", end, freq="5s", inclusive="left")
    ret = np.sin(np.arange(len(index)) / 17.0) * 0.00002
    close = 2000.0 * np.exp(np.cumsum(ret))
    delta = np.sin(np.arange(len(index)) / 11.0) * 1000.0
    notional = np.full(len(index), 5000.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.00002,
            "low": close * 0.99998,
            "close": close,
            "notional": notional,
            "delta_notional": delta,
            "large_delta_notional": delta * 0.2,
            "trades_count": np.full(len(index), 3.0),
            "large_trades_count": np.full(len(index), 0.2),
            "max_trade_notional": np.full(len(index), 800.0),
        },
        index=index,
    )


def test_micro_features_do_not_change_before_future_perturbation() -> None:
    config = DEFAULT_FUTURE_PROCESS_FORECAST_CONFIG
    bars = _micro_bars()
    decisions = pd.date_range("2024-01-01 01:00:00", "2024-01-01 02:00:00", freq="15min")
    base = build_micro_decision_features(bars, decisions, config)
    changed = bars.copy()
    mask = changed.index >= pd.Timestamp("2024-01-01 01:31:00")
    changed.loc[mask, "delta_notional"] *= -20.0
    changed.loc[mask, "close"] *= 1.05
    future = build_micro_decision_features(changed, decisions, config)
    pd.testing.assert_series_equal(base.loc[pd.Timestamp("2024-01-01 01:30:00")], future.loc[pd.Timestamp("2024-01-01 01:30:00")])
