from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.backtest import MarketPath
from src.ai_research.swing_baseline.config import DEFAULT_SWING_BASELINE_CONFIG, SwingTargetSpec
from src.ai_research.swing_baseline.modeling import PeriodData
from src.ai_research.swing_entry_mvp.backtest import simulate_entry_portfolio
from src.ai_research.swing_entry_mvp.config import ExitPolicySpec, SwingEntryMvpConfig
from src.ai_research.swing_entry_mvp.outcomes import _first_hit_kernel


def _period(times: pd.DatetimeIndex) -> PeriodData:
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
    context = np.tile(np.array([99.0, 101.0, 0.2, 0.01, 0.005, 100, 99, 100, 99], dtype=float), (len(times), 1))
    return PeriodData(
        timestamps_ns=times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        high_x=np.zeros((len(times), 1), dtype=np.float32),
        full_x=np.zeros((len(times), 1), dtype=np.float32),
        labels={},
        context=context,
        context_columns=context_columns,
        entry_times_ns=(times + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entry_prices=np.full(len(times), 100.0),
    )


def test_exact_label_requires_target_before_adverse_and_is_conservative_same_bar() -> None:
    positions = np.array([0, 0], dtype=np.int64)
    entries = np.array([100.0, 100.0])
    high = np.array([103.5, 100.0, 103.5], dtype=float)
    low = np.array([98.5, 99.5, 99.5], dtype=float)
    quality, event, bars = _first_hit_kernel(positions, entries, high, low, 0.03, 0.01, 3, 1)
    assert quality[0] == 0.0
    assert event[0] == 3
    assert bars[0] == 0

    high2 = np.array([100.5, 103.5, 103.5], dtype=float)
    low2 = np.array([99.5, 99.5, 98.5], dtype=float)
    quality2, event2, bars2 = _first_hit_kernel(positions[:1], entries[:1], high2, low2, 0.03, 0.01, 3, 1)
    assert quality2[0] == 1.0
    assert event2[0] == 1
    assert bars2[0] == 1


def test_entry_mvp_has_no_minimum_hold_and_exits_on_early_target() -> None:
    decision = pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00:00")])
    period = _period(decision)
    minute_times = pd.date_range("2025-01-01 00:01:00", periods=4, freq="1min")
    market = MarketPath(
        times_ns=minute_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        ohlc=np.array(
            [
                [100.0, 100.5, 99.7, 100.2],
                [100.2, 103.2, 100.0, 103.0],
                [103.0, 103.1, 102.8, 103.0],
                [103.0, 103.0, 102.9, 103.0],
            ],
            dtype=float,
        ),
    )
    config = SwingEntryMvpConfig(
        base=replace(DEFAULT_SWING_BASELINE_CONFIG, target_specs=(SwingTargetSpec("t3", 0.03, 0.01, 72),)),
        exit_policies=(ExitPolicySpec("fixed_adverse_target", False, False),),
    )
    result = simulate_entry_portfolio(
        fold_id="TEST",
        architecture="high_lightgbm",
        target=config.base.target_specs[0],
        direction_name="long",
        policy=config.exit_policies[0],
        quantile=0.95,
        delay_minutes=1,
        period=period,
        score_long=np.array([0.9]),
        score_short=np.array([0.1]),
        threshold=0.8,
        market_path=market,
        config=config,
    )
    assert len(result.records) == 1
    record = result.records[0]
    assert record.exit_reason == "target_hit"
    assert record.target_hit
    assert record.hold_hours < 1.0
    assert np.isclose(record.gross_return, 0.03)


def test_profit_protection_becomes_active_only_after_completed_minute() -> None:
    decision = pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00:00")])
    period = _period(decision)
    minute_times = pd.date_range("2025-01-01 00:01:00", periods=3, freq="1min")
    market = MarketPath(
        times_ns=minute_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        ohlc=np.array(
            [
                [100.0, 101.7, 99.5, 101.5],
                [101.5, 101.6, 100.8, 101.0],
                [101.0, 101.1, 100.9, 101.0],
            ],
            dtype=float,
        ),
    )
    config = SwingEntryMvpConfig(
        base=replace(DEFAULT_SWING_BASELINE_CONFIG, target_specs=(SwingTargetSpec("t3", 0.03, 0.01, 72),)),
        exit_policies=(ExitPolicySpec("protected", False, True),),
        protection_trigger_fraction=0.5,
        locked_profit_fraction=0.3,
    )
    result = simulate_entry_portfolio(
        fold_id="TEST",
        architecture="high_lightgbm",
        target=config.base.target_specs[0],
        direction_name="long",
        policy=config.exit_policies[0],
        quantile=0.95,
        delay_minutes=1,
        period=period,
        score_long=np.array([0.9]),
        score_short=np.array([0.1]),
        threshold=0.8,
        market_path=market,
        config=config,
    )
    record = result.records[0]
    assert record.exit_reason == "protected_stop"
    assert record.exit_time == pd.Timestamp("2025-01-01 00:02:00")
    assert np.isclose(record.exit_price, 100.9)


def test_short_return_is_measured_against_entry_not_exit() -> None:
    decision = pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00:00")])
    period = _period(decision)
    minute_times = pd.date_range("2025-01-01 00:01:00", periods=2, freq="1min")
    market = MarketPath(
        times_ns=minute_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        ohlc=np.array([[100.0, 100.2, 94.8, 95.0], [95.0, 95.1, 94.9, 95.0]], dtype=float),
    )
    config = SwingEntryMvpConfig(
        base=replace(DEFAULT_SWING_BASELINE_CONFIG, target_specs=(SwingTargetSpec("t5", 0.05, 0.0175, 120),)),
        exit_policies=(ExitPolicySpec("fixed", False, False),),
    )
    result = simulate_entry_portfolio(
        fold_id="TEST",
        architecture="high_lightgbm",
        target=config.base.target_specs[0],
        direction_name="short",
        policy=config.exit_policies[0],
        quantile=0.95,
        delay_minutes=1,
        period=period,
        score_long=np.array([0.1]),
        score_short=np.array([0.9]),
        threshold=0.8,
        market_path=market,
        config=config,
    )
    assert np.isclose(result.records[0].gross_return, 0.05)


def test_r031_package_does_not_read_raw_zip_or_sqlite() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "ai_research" / "swing_entry_mvp"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "sqlite3" not in text
    assert "zipfile" not in text
    assert "pd.read_csv" not in text
    assert "OKXTickLoader" not in text


def test_exact_overlay_reuses_frozen_feature_cache_without_rebuilding_features(tmp_path: Path) -> None:
    import json

    from src.ai_research.swing_entry_mvp.outcomes import build_outcome_overlay, collect_exact_period_data

    shard = tmp_path / "samples_2025"
    shard.mkdir()
    decision = pd.date_range("2025-01-01 00:00:00", periods=4, freq="15min")
    minute = pd.date_range("2025-01-01 00:00:00", periods=180, freq="1min")
    entry_times = decision + pd.Timedelta(minutes=1)
    minute_ohlc = np.column_stack(
        [
            np.full(len(minute), 100.0),
            np.full(len(minute), 100.5),
            np.full(len(minute), 99.5),
            np.full(len(minute), 100.0),
        ]
    )
    # First decision reaches +3% before -1%; others time out.
    minute_ohlc[20, 1] = 103.2
    np.save(shard / "decision_times_ns.npy", decision.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    np.save(shard / "features.npy", np.zeros((4, 2), dtype=np.float32))
    np.save(shard / "context.npy", np.zeros((4, 1), dtype=np.float64))
    np.save(shard / "labels.npy", np.zeros((4, 2), dtype=np.float32))
    np.save(shard / "entry_times_ns.npy", entry_times.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    np.save(shard / "entry_prices.npy", np.full(4, 100.0))
    np.save(shard / "minute_times_ns.npy", minute.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    np.save(shard / "minute_ohlc.npy", minute_ohlc)
    target = SwingTargetSpec("t3", 0.03, 0.01, 1)
    manifest = {
        "schema_version": 1,
        "cache_signature": "synthetic",
        "high_feature_columns": ["tf4h_x"],
        "full_feature_columns": ["tf4h_x", "tf1m_x"],
        "context_columns": ["ctx_x"],
        "label_columns": ["t3_long_quality", "t3_short_quality"],
    }
    (shard / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        target_specs=(target,),
        max_hold_hours=120,
        cache_dir=str(tmp_path / "base_cache"),
    )
    config = SwingEntryMvpConfig(base=base, exact_label_cache_dir=str(tmp_path / "outcomes"))
    overlay = build_outcome_overlay(shard, config)
    data = collect_exact_period_data(
        [shard],
        [overlay],
        decision[0],
        decision[-1],
        target=target,
    )
    assert data.labels["t3_long_quality"][0] == 1.0
    assert data.labels["t3_long_quality"].sum() >= 1.0
    assert data.full_x.shape == (4, 2)


def test_shared_model_fitter_accepts_exact_period_labels() -> None:
    from src.ai_research.swing_baseline.modeling import fit_model_bundle_from_period

    times = pd.date_range("2024-01-01", periods=40, freq="15min")
    x = np.linspace(-1.0, 1.0, 40, dtype=np.float32)[:, None]
    target = SwingTargetSpec("t3", 0.03, 0.01, 72)
    data = PeriodData(
        timestamps_ns=times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        high_x=x,
        full_x=np.column_stack([x, x**2]).astype(np.float32),
        labels={
            "t3_long_quality": (x[:, 0] > 0).astype(np.float32),
            "t3_short_quality": (x[:, 0] < 0).astype(np.float32),
        },
        context=np.zeros((40, 1), dtype=float),
        context_columns=("ctx_x",),
        entry_times_ns=(times + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]").astype(np.int64),
        entry_prices=np.full(40, 100.0),
    )
    bundle, metadata = fit_model_bundle_from_period(
        "high_logistic",
        target,
        data,
        high_columns=("tf4h_x",),
        full_columns=("tf4h_x", "tf1m_x"),
        config=replace(DEFAULT_SWING_BASELINE_CONFIG, train_sample_cap=100),
    )
    scores = bundle.predict(data.high_x, data.full_x)
    assert scores["score_long"].shape == (40,)
    assert metadata["train_rows"] == 40


def test_r031_dependency_preflight_happens_before_loader(monkeypatch, tmp_path: Path) -> None:
    import src.ai_research.swing_entry_mvp.outcomes as outcomes
    import src.ai_research.swing_entry_mvp.pipeline as pipeline

    monkeypatch.setattr(outcomes, "njit", None)

    def forbidden_loader(*args, **kwargs):  # pragma: no cover
        raise AssertionError("loader must not be touched before dependency checks")

    monkeypatch.setattr(pipeline, "create_loader", forbidden_loader)
    base = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        cache_dir=str(tmp_path / "base"),
        report_dir=str(tmp_path / "report"),
    )
    config = SwingEntryMvpConfig(
        base=base,
        exact_label_cache_dir=str(tmp_path / "outcomes"),
        report_dir=str(tmp_path / "report"),
        architectures=("high_logistic",),
    )
    try:
        pipeline.run_pipeline(config=config, progress=False)
    except RuntimeError as exc:
        assert "numba is not installed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing numba must fail before data work")
    assert not (tmp_path / "base").exists()
