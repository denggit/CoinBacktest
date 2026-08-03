from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.config import DEFAULT_SWING_BASELINE_CONFIG
from src.ai_research.swing_baseline.dataset import cache_signature
from src.ai_research.swing_baseline.features import (
    BASE_FEATURE_PROFILE,
    LONG_CONTEXT_PROFILE,
    build_timeframe_features,
)
from src.ai_research.swing_long_context.config import (
    DEFAULT_SWING_LONG_CONTEXT_CONFIG,
    validate_long_context_contract,
)


def _bars(rows: int, freq: str) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq=freq)
    trend = np.linspace(100.0, 180.0, rows)
    wave = 2.0 * np.sin(np.arange(rows) / 17.0)
    close = trend + wave
    open_ = close * (1.0 - 0.001 * np.cos(np.arange(rows) / 9.0))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    notional = 1_000_000.0 + 50_000.0 * np.sin(np.arange(rows) / 13.0)
    delta = notional * 0.1 * np.sin(np.arange(rows) / 11.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": notional / close,
            "notional": notional,
            "delta_notional": delta,
            "large_delta_notional": delta * 0.4,
            "buy_notional": (notional + delta) / 2.0,
            "sell_notional": (notional - delta) / 2.0,
            "large_buy_notional": (notional * 0.2 + delta * 0.4) / 2.0,
            "large_sell_notional": (notional * 0.2 - delta * 0.4) / 2.0,
        },
        index=index,
    )


def test_long_context_daily_contains_months_of_history_and_process_features() -> None:
    features = build_timeframe_features(
        _bars(520, "1D"),
        "1d",
        structural_swing_bars_4h=8,
        feature_profile=LONG_CONTEXT_PROFILE,
    )
    required = {
        "tf1d_ret_365",
        "tf1d_range_pos_365",
        "tf1d_bars_since_high_365",
        "tf1d_bars_since_low_365",
        "tf1d_trend_age_above_ema50",
        "tf1d_vol_lifecycle_10_365",
        "tf1d_close_rel_ema200",
        "tf1d_ema100_200",
    }
    assert required.issubset(features.columns)
    assert features.index[0] == pd.Timestamp("2021-01-02")
    assert features["tf1d_ret_365"].notna().any()


def test_baseline_profile_remains_frozen() -> None:
    features = build_timeframe_features(
        _bars(520, "1D"),
        "1d",
        structural_swing_bars_4h=8,
        feature_profile=BASE_FEATURE_PROFILE,
    )
    assert "tf1d_ret_50" in features.columns
    assert "tf1d_ret_365" not in features.columns
    assert "tf1d_close_rel_ema200" not in features.columns


def test_long_context_feature_at_time_is_unchanged_by_future_mutation() -> None:
    bars = _bars(520, "1D")
    cutoff = bars.index[430]
    base = build_timeframe_features(
        bars,
        "1d",
        structural_swing_bars_4h=8,
        feature_profile=LONG_CONTEXT_PROFILE,
    )
    mutated = bars.copy()
    mutated.loc[mutated.index > cutoff, ["open", "high", "low", "close"]] *= 5.0
    changed = build_timeframe_features(
        mutated,
        "1d",
        structural_swing_bars_4h=8,
        feature_profile=LONG_CONTEXT_PROFILE,
    )
    available = cutoff + pd.Timedelta(days=1)
    pd.testing.assert_series_equal(base.loc[available], changed.loc[available])


def test_long_context_cache_is_isolated_and_signature_differs() -> None:
    config = DEFAULT_SWING_LONG_CONTEXT_CONFIG
    validate_long_context_contract(config)
    assert config.base.feature_lookback_days == 420
    assert "r03_2_long_context" in config.base.cache_dir
    assert "r03_2_exact_outcomes" in config.exact_label_cache_dir
    assert cache_signature(config.base, feature_profile=LONG_CONTEXT_PROFILE) != cache_signature(
        DEFAULT_SWING_BASELINE_CONFIG,
        feature_profile=BASE_FEATURE_PROFILE,
    )


def test_long_context_contract_rejects_short_warmup() -> None:
    bad_base = replace(DEFAULT_SWING_LONG_CONTEXT_CONFIG.base, feature_lookback_days=180)
    bad = replace(DEFAULT_SWING_LONG_CONTEXT_CONFIG, base=bad_base)
    try:
        validate_long_context_contract(bad)
    except ValueError as exc:
        assert "400" in str(exc)
    else:
        raise AssertionError("short long-context warmup should fail")


def test_r032_wrapper_freezes_profile_and_isolated_rebuild_flag(monkeypatch) -> None:
    from src.ai_research.swing_entry_mvp.pipeline import SwingEntryMvpResult
    from src.ai_research.swing_long_context import pipeline as module

    captured = {}

    def fake_run_entry_pipeline(**kwargs):
        captured.update(kwargs)
        return SwingEntryMvpResult("FAIL_VALIDATION", kwargs["config"].report_path, None, None)

    monkeypatch.setattr(module, "run_entry_pipeline", fake_run_entry_pipeline)
    result = module.run_pipeline(force_rebuild_long_context_cache=True, progress=False)
    assert result.decision == "FAIL_VALIDATION"
    assert captured["feature_profile"] == LONG_CONTEXT_PROFILE
    assert captured["force_rebuild_base_cache"] is True
    assert captured["pass_decision"] == "PASS_SWING_LONG_CONTEXT_MVP"
    assert captured["stage_id"] == "R03.2"
