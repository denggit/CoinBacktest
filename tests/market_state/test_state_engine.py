from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.market_state import MarketStateConfig, MarketStateDataBundle, MarketStateEngine
from src.market_state.transitions import stabilize_direction_scores


def _sample(rows: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = pd.date_range("2026-01-01", periods=rows, freq="1min")
    returns = np.r_[
        rng.normal(0.0, 0.00008, 400),
        np.full(500, 0.00028) + rng.normal(0.0, 0.00005, 500),
        rng.normal(0.0, 0.00110, 300),
        np.full(max(0, rows - 1200), -0.00026) + rng.normal(0.0, 0.00005, max(0, rows - 1200)),
    ][:rows]
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    width = np.r_[
        np.full(min(rows, 900), 0.00025),
        np.full(max(0, min(rows - 900, 300)), 0.00230),
        np.full(max(0, rows - 1200), 0.00035),
    ][:rows]
    high = np.maximum(open_, close) * (1.0 + width)
    low = np.minimum(open_, close) * (1.0 - width)
    volume = np.r_[
        rng.uniform(80, 120, min(rows, 900)),
        rng.uniform(500, 1000, max(0, min(rows - 900, 300))),
        rng.uniform(120, 180, max(0, rows - 1200)),
    ][:rows]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _config() -> MarketStateConfig:
    return MarketStateConfig(
        fast_trend_window=12,
        trend_window=48,
        slow_trend_window=160,
        volatility_window=30,
        activity_window=12,
        baseline_window=240,
        trend_confirm_bars=3,
        min_state_bars=15,
    )


def _result(df: pd.DataFrame):
    bundle = MarketStateDataBundle.from_frame(
        df,
        source="normal",
        timestamp_semantics="bar_start",
        bar_duration="1min",
    )
    return MarketStateEngine(_config()).compute(bundle)


def test_engine_detects_structural_uptrend_and_keeps_quality_separate() -> None:
    result = _result(_sample())
    middle = result.frame.iloc[600:850]
    shock = result.frame.iloc[920:1080]

    assert (middle["trend_state"] == "up").mean() > 0.85
    assert middle["medium_trend_score"].median() > 0.30
    assert middle["slow_trend_score"].median() > 0.20
    assert middle["trend_alignment_score"].median() > 0.80
    assert set(middle["trend_quality_state"]) <= {"high_order", "normal", "noisy"}
    assert middle["trend_quality_state"].nunique() >= 2
    assert shock["volatility_state"].isin(["expansion", "shock"]).mean() > 0.55
    assert shock["activity_score"].median() > 0.75
    assert result.frame["available_time"].iloc[700] == result.frame.index[700] + pd.Timedelta(minutes=1)
    assert result.ready_rows > 1000
    assert result.segments


def test_direction_hysteresis_respects_exit_and_minimum_duration() -> None:
    scores = np.r_[np.zeros(5), np.full(3, 0.40), np.full(5, 0.16), np.full(4, 0.05), np.full(8, -0.40)]
    raw, stable, ages, candidates, progress = stabilize_direction_scores(
        scores,
        np.ones(len(scores), dtype=bool),
        enter_threshold=0.24,
        exit_threshold=0.10,
        confirm_bars=3,
        min_duration_bars=10,
    )

    assert stable[7] == "up"  # entered after three confirmations
    assert all(value == "up" for value in stable[8:15])  # 0.16 does not trigger exit
    assert stable[15] == "up"  # confirmation exists but minimum duration is not yet met
    assert stable[16] == "balanced"  # exits after < 0.10, confirmation and minimum duration
    assert stable[-1] == "down"
    assert max(progress) <= 1.0
    assert len(raw) == len(ages) == len(candidates) == len(scores)


def test_appending_future_rows_does_not_change_existing_state() -> None:
    full = _sample()
    prefix = full.iloc[:1200]
    prefix_result = _result(prefix).frame
    full_result = _result(full).frame.loc[prefix.index]

    numeric_columns = [
        "trend_score",
        "fast_trend_score",
        "medium_trend_score",
        "slow_trend_score",
        "trend_alignment_score",
        "orderliness_score",
        "orderliness_percentile",
        "volatility_score",
        "volatility_z",
        "activity_score",
        "activity_z",
    ]
    pdt.assert_frame_equal(
        prefix_result[numeric_columns],
        full_result[numeric_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    for column in ("trend_state", "trend_quality_state", "trend_phase", "volatility_state"):
        pdt.assert_series_equal(prefix_result[column], full_result[column])


def test_range_bar_end_timestamp_is_available_immediately() -> None:
    df = _sample(700)
    bundle = MarketStateDataBundle.from_frame(df, source="range_bar", timestamp_semantics="bar_end")
    config = MarketStateConfig(
        fast_trend_window=8,
        trend_window=24,
        slow_trend_window=80,
        volatility_window=16,
        activity_window=8,
        baseline_window=120,
    )
    result = MarketStateEngine(config).compute(bundle)
    assert list(pd.DatetimeIndex(result.frame["available_time"])) == list(result.frame.index)
