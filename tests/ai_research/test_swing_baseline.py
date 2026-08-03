from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.backtest import MarketPath, simulate_structural_portfolio
from src.ai_research.swing_baseline.config import (
    DEFAULT_SWING_BASELINE_CONFIG,
    SwingTargetSpec,
)
from src.ai_research.swing_baseline.dataset import _build_labels, cache_signature
from src.ai_research.swing_baseline.features import (
    FLOW_MAX_COLUMNS,
    FLOW_SUM_COLUMNS,
    aggregate_timeframe,
    build_timeframe_features,
)
from src.ai_research.swing_baseline.modeling import PeriodData, default_folds


def synthetic_minute_bars(start: str, periods: int, *, drift: float = 0.00002) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    x = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(drift * x + 0.001 * np.sin(x / 30.0))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.0005,
            "low": np.minimum(open_, close) * 0.9995,
            "close": close,
        },
        index=index,
    )
    for column in FLOW_SUM_COLUMNS:
        frame[column] = 1.0
    frame["notional"] = 1000.0
    frame["buy_notional"] = 550.0
    frame["sell_notional"] = 450.0
    frame["delta_notional"] = 100.0
    frame["large_buy_notional"] = 100.0
    frame["large_sell_notional"] = 50.0
    frame["large_delta_notional"] = 50.0
    for column in FLOW_MAX_COLUMNS:
        frame[column] = 10.0
    return frame


def test_4h_feature_is_available_only_after_bar_close() -> None:
    minute = synthetic_minute_bars("2025-01-01", 12 * 60)
    bars_4h = aggregate_timeframe(minute, "4h")
    features = build_timeframe_features(bars_4h, "4h", structural_swing_bars_4h=2)
    assert features.index[0] == pd.Timestamp("2025-01-01 04:00:00")
    assert pd.Timestamp("2025-01-01 03:59:00") not in features.index

    changed = minute.copy()
    changed.loc[changed.index >= pd.Timestamp("2025-01-01 04:00:00"), ["open", "high", "low", "close"]] *= 2.0
    changed_features = build_timeframe_features(
        aggregate_timeframe(changed, "4h"),
        "4h",
        structural_swing_bars_4h=2,
    )
    common_columns = features.columns.intersection(changed_features.columns)
    pd.testing.assert_series_equal(
        features.loc[pd.Timestamp("2025-01-01 04:00:00"), common_columns],
        changed_features.loc[pd.Timestamp("2025-01-01 04:00:00"), common_columns],
        check_names=False,
    )


def test_low_mae_path_label_uses_future_path_not_fixed_close_only() -> None:
    minute = synthetic_minute_bars("2025-01-01", 180, drift=0.0)
    decision = pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00:00")])
    entry_time = pd.Timestamp("2025-01-01 00:01:00")
    entry = float(minute.loc[entry_time, "open"])
    minute.loc[pd.Timestamp("2025-01-01 00:30:00"), "high"] = entry * 1.04
    minute.loc[entry_time : pd.Timestamp("2025-01-01 01:00:00"), "low"] = np.maximum(
        minute.loc[entry_time : pd.Timestamp("2025-01-01 01:00:00"), "low"],
        entry * 0.995,
    )
    target = SwingTargetSpec("test3", target_move=0.03, max_adverse_move=0.01, horizon_hours=1)
    config = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        target_specs=(target,),
        max_hold_hours=120,
    )
    labels, entries = _build_labels(minute, decision, config)
    assert entries.iloc[0]["entry_price"] == entry
    assert labels.iloc[0]["test3_long_quality"] == 1.0
    assert labels.iloc[0]["test3_long_mfe"] >= 0.03
    assert labels.iloc[0]["test3_long_mae"] <= 0.01


def test_trailing_stop_updates_after_bar_and_applies_next_bar() -> None:
    decision_times = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:15:00"])
    feature_columns = ("tf1h_close_rel_ema20", "tf4h_ema20_slope3")
    context_columns = (
        "ctx_recent_low_4h",
        "ctx_recent_high_4h",
        "ctx_atr_abs_4h",
        "ctx_atr_pct_4h",
        "ctx_atr_pct_15m",
        "ctx_close_1h",
        "ctx_ema20_1h",
        "ctx_close_4h",
        "ctx_ema20_4h",
    )
    context = np.array(
        [
            [99.5, 101.0, 0.1, 0.005, 0.004, 100, 99, 100, 99],
            [99.5, 103.0, 0.1, 0.005, 0.004, 102, 101, 102, 101],
        ],
        dtype=float,
    )
    period = PeriodData(
        timestamps_ns=decision_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        high_x=np.zeros((2, 2), dtype=np.float32),
        full_x=np.zeros((2, 2), dtype=np.float32),
        labels={},
        context=context,
        context_columns=context_columns,
        entry_times_ns=(decision_times + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entry_prices=np.array([100.0, 102.0]),
    )
    minute_times = pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00", "2025-01-01 00:03:00"])
    market = MarketPath(
        times_ns=minute_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        ohlc=np.array(
            [
                [100.0, 103.0, 99.5, 102.0],
                [102.0, 102.5, 102.0, 102.2],
                [102.2, 102.4, 102.1, 102.3],
            ],
            dtype=float,
        ),
    )
    config = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        min_initial_stop_pct=0.006,
        max_initial_stop_pct=0.02,
        max_hold_hours=120,
    )
    result = simulate_structural_portfolio(
        fold_id="TEST",
        architecture="high_logistic",
        target=SwingTargetSpec("test", 0.03, 0.01, 72),
        quantile=0.9,
        delay_minutes=1,
        period=period,
        feature_columns=feature_columns,
        score_long=np.array([0.9, 0.9]),
        score_short=np.array([0.1, 0.1]),
        thresholds=(0.8, 0.8),
        market_path=market,
        config=config,
    )
    assert len(result.records) == 1
    record = result.records[0]
    assert record.exit_reason == "trailing_stop"
    assert record.exit_time == pd.Timestamp("2025-01-01 00:02:00")


def test_swing_folds_have_full_horizon_embargo() -> None:
    config = DEFAULT_SWING_BASELINE_CONFIG
    for fold in default_folds(config):
        assert fold.fit_end + pd.Timedelta(hours=config.max_horizon_hours + 1) <= fold.calibration_start


def test_swing_cache_signature_ignores_model_and_report_settings() -> None:
    base = DEFAULT_SWING_BASELINE_CONFIG
    changed = replace(
        base,
        lightgbm_n_estimators=999,
        train_sample_cap=12_345,
        report_dir="tmp/other_report",
        signal_quantiles=(0.8, 0.9, 0.99),
    )
    assert cache_signature(base) == cache_signature(changed)


def test_swing_package_uses_public_loader_not_raw_or_sqlite() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "ai_research" / "swing_baseline"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "sqlite3" not in text
    assert "zipfile" not in text
    assert "pd.read_csv" not in text
    assert "OKXTickLoader" not in text


def test_pipeline_dependency_preflight_happens_before_loader(monkeypatch, tmp_path: Path) -> None:
    import src.ai_research.swing_baseline.modeling as modeling
    import src.ai_research.swing_baseline.pipeline as pipeline

    monkeypatch.setattr(modeling, "LGBMClassifier", None)
    monkeypatch.setattr(pipeline, "LGBMClassifier", None, raising=False)

    def forbidden_loader(*args, **kwargs):  # pragma: no cover
        raise AssertionError("loader must not be touched before dependency check")

    monkeypatch.setattr(pipeline, "create_loader", forbidden_loader)
    config = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        cache_dir=str(tmp_path / "cache"),
        report_dir=str(tmp_path / "report"),
    )
    try:
        pipeline.run_pipeline(
            config=config,
            architectures=("high_lightgbm",),
            progress=False,
        )
    except RuntimeError as exc:
        assert "startup dependency check failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing LightGBM must fail before data work")
    assert not (tmp_path / "cache").exists()
