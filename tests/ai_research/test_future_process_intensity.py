from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.future_process_forecast.intensity_config import (
    CURRENT_STATE_DEFINITION,
    DEFAULT_FUTURE_INTENSITY_CONFIG,
)
from src.ai_research.future_process_forecast.intensity_targets import build_intensity_target_frame


def _path(closes: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:15:00", periods=len(closes), freq="15min")
    close = pd.Series(closes, index=index, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def test_r0332_contract_keeps_current_state_causal_and_2026_sealed() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    config.validate()
    assert config.horizons_hours == (6, 12)
    assert pd.Timestamp(config.base.research_end) < pd.Timestamp(config.base.sealed_holdout_start)
    assert "strategic_structure" in CURRENT_STATE_DEFINITION
    assert "future" not in " ".join(
        str(value) for group in CURRENT_STATE_DEFINITION.values() for value in group.values()
    ).lower()


def test_future_target_excludes_current_decision_bar() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    path = _path(np.full(60, 100.0))
    decision = path.index[5:6]
    path.loc[decision[0], "high"] = 150.0
    path.loc[decision[0], "low"] = 50.0
    target = build_intensity_target_frame(
        path,
        decision,
        np.array([100.0]),
        np.array([0.01]),
        config,
    )
    assert target.iloc[0]["future_range_pct_h6"] == 0.0
    assert target.iloc[0]["future_max_directional_pct_h6"] == 0.0


def test_monotonic_future_has_directional_space_but_little_two_sided_space() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    flat = np.full(8, 100.0)
    trend = np.linspace(100.0, 108.0, 52)
    path = _path(np.concatenate([flat, trend]))
    decision = path.index[7:8]
    target = build_intensity_target_frame(
        path,
        decision,
        np.array([100.0]),
        np.array([0.01]),
        config,
    )
    assert target.iloc[0]["future_max_directional_pct_h6"] > 0.03
    assert target.iloc[0]["future_two_sided_pct_h6"] < 0.005
    assert target.iloc[0]["future_range_atr_multiple_h6"] > 1.0


def test_oscillating_future_has_two_sided_opportunity() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    flat = np.full(8, 100.0)
    oscillation = 100.0 * (1.0 + 0.03 * np.sin(np.linspace(0, 6 * np.pi, 52)))
    path = _path(np.concatenate([flat, oscillation]))
    decision = path.index[7:8]
    target = build_intensity_target_frame(
        path,
        decision,
        np.array([100.0]),
        np.array([0.01]),
        config,
    )
    assert target.iloc[0]["future_two_sided_pct_h6"] > 0.02
    assert target.iloc[0]["future_range_pct_h6"] > 0.05


def test_year_boundary_decision_before_first_available_bar_is_supported() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    path = _path(np.linspace(100.0, 106.0, 60))
    decision = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00")])
    target = build_intensity_target_frame(
        path,
        decision,
        np.array([100.0]),
        np.array([0.01]),
        config,
    )
    assert np.isfinite(target.iloc[0]["future_range_pct_h6"])
    assert target.iloc[0]["future_max_directional_pct_h6"] > 0.0


def test_exact_decision_timestamp_still_excludes_bar_available_at_decision() -> None:
    config = DEFAULT_FUTURE_INTENSITY_CONFIG
    path = _path(np.full(60, 100.0))
    decision = path.index[5:6]
    path.loc[decision[0], "high"] = 160.0
    path.loc[decision[0], "low"] = 40.0
    path.loc[path.index[6], "high"] = 101.0
    path.loc[path.index[6], "low"] = 99.0
    target = build_intensity_target_frame(
        path,
        decision,
        np.array([100.0]),
        np.array([0.01]),
        config,
    )
    assert target.iloc[0]["future_range_pct_h6"] < 0.05
