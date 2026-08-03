from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.trades_baseline.backtest import ScenarioAccumulator, select_validation_champion
from src.ai_research.trades_baseline.config import DEFAULT_TRADES_BASELINE_CONFIG
from src.ai_research.trades_baseline.modeling import default_folds, validate_model_dependencies
from src.ai_research.trades_baseline.dataset import (
    FLOW_COLUMNS,
    REQUIRED_COLUMNS,
    build_day_samples,
    cache_signature,
    feature_columns,
    run_public_loader_preflight,
)


class FakeTradeBarLoader:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars

    def fetch_data_by_date_range(self, start, end, **kwargs):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return self.bars.loc[(self.bars.index >= start_ts) & (self.bars.index <= end_ts)].copy()


def synthetic_bars() -> pd.DataFrame:
    index = pd.date_range("2022-12-31 23:54:00", "2023-01-01 00:31:00", freq="1s", inclusive="left")
    x = np.arange(len(index), dtype=float)
    close = 1200.0 + x * 0.001 + np.sin(x / 30.0) * 0.2
    open_ = close - np.sin(x / 7.0) * 0.01
    size = 10.0 + (x % 5)
    side = np.where((x.astype(int) % 2) == 0, 1.0, -1.0)
    notional = close * size * 0.1
    buy_notional = np.where(side > 0, notional, 0.0)
    sell_notional = np.where(side < 0, notional, 0.0)
    buy_volume = np.where(side > 0, size, 0.0)
    sell_volume = np.where(side < 0, size, 0.0)
    data = {
        "open": open_,
        "high": np.maximum(open_, close) + 0.02,
        "low": np.minimum(open_, close) - 0.02,
        "close": close,
        "volume": size,
        "trades_count": np.ones(len(index)),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "notional": notional,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "buy_trades_count": (side > 0).astype(float),
        "sell_trades_count": (side < 0).astype(float),
        "delta_volume": buy_volume - sell_volume,
        "delta_notional": buy_notional - sell_notional,
        "large_buy_notional": np.zeros(len(index)),
        "large_sell_notional": np.zeros(len(index)),
        "large_delta_notional": np.zeros(len(index)),
        "large_trades_count": np.zeros(len(index)),
        "large_buy_trades_count": np.zeros(len(index)),
        "large_sell_trades_count": np.zeros(len(index)),
        "max_trade_notional": notional,
        "max_trade_size": size,
    }
    frame = pd.DataFrame(data, index=index)
    for col in REQUIRED_COLUMNS:
        assert col in frame.columns
    return frame


def test_public_loader_preflight_is_small_and_uses_public_interface() -> None:
    bars = synthetic_bars()
    loader = FakeTradeBarLoader(bars)
    config = replace(DEFAULT_TRADES_BASELINE_CONFIG, research_start="2023-01-01 00:00:00")
    result = run_public_loader_preflight(loader, config, sample_dates=("2023-01-01 00:00:00",))
    assert result.status == "PASS"
    assert result.sample_windows[0]["rows"] == 31 * 60


def test_day_samples_are_causal_and_include_labels() -> None:
    bars = synthetic_bars()
    loader = FakeTradeBarLoader(bars)
    config = replace(
        DEFAULT_TRADES_BASELINE_CONFIG,
        horizons_seconds=(60, 180),
        latency_scenarios_seconds=(0.5, 1.0),
        base_latency_seconds=0.5,
    )
    samples = build_day_samples(loader, pd.Timestamp("2023-01-01"), config)
    assert not samples.empty
    assert samples.index.is_monotonic_increasing
    assert set(feature_columns(config)) <= set(samples.columns)
    assert "gross_ret_h60_lat500" in samples.columns
    assert "mfe_h180" in samples.columns
    assert samples[feature_columns(config)].notna().all().all()

    changed = bars.copy()
    changed.loc[changed.index >= pd.Timestamp("2023-01-01 00:10:00"), "close"] *= 1.2
    changed.loc[changed.index >= pd.Timestamp("2023-01-01 00:10:00"), "open"] *= 1.2
    changed.loc[changed.index >= pd.Timestamp("2023-01-01 00:10:00"), "high"] *= 1.2
    changed.loc[changed.index >= pd.Timestamp("2023-01-01 00:10:00"), "low"] *= 1.2
    changed_samples = build_day_samples(FakeTradeBarLoader(changed), pd.Timestamp("2023-01-01"), config)
    cutoff = pd.Timestamp("2023-01-01 00:09:55")
    common = samples.index.intersection(changed_samples.index)
    common = common[common <= cutoff]
    pd.testing.assert_frame_equal(
        samples.loc[common, feature_columns(config)],
        changed_samples.loc[common, feature_columns(config)],
    )




def test_day_samples_normalize_microsecond_datetime_index_to_nanoseconds() -> None:
    bars = synthetic_bars().copy()
    # SQLite/Arrow/Pandas combinations may preserve timestamps as datetime64[us].
    # The research pipeline must not treat the raw integer representation as ns.
    if hasattr(bars.index, "as_unit"):
        bars.index = bars.index.as_unit("us")
    else:  # pragma: no cover - compatibility with older pandas
        bars.index = pd.DatetimeIndex(bars.index.to_numpy(dtype="datetime64[us]"))
    assert str(bars.index.dtype) == "datetime64[us]"

    config = replace(
        DEFAULT_TRADES_BASELINE_CONFIG,
        horizons_seconds=(60, 180),
        latency_scenarios_seconds=(0.5, 1.0),
        base_latency_seconds=0.5,
    )
    samples = build_day_samples(FakeTradeBarLoader(bars), pd.Timestamp("2023-01-01"), config)
    assert not samples.empty
    assert samples["gross_ret_h60_lat500"].notna().all()
    timestamps = samples["decision_time_ns"].to_numpy(dtype=np.int64)
    assert timestamps[1] - timestamps[0] == 5_000_000_000


def test_scenario_accumulator_prevents_overlap() -> None:
    acc = ScenarioAccumulator("WF_2025", "lightgbm", 60, 0.99, 0.5, 1.0, 10_000.0)
    start = pd.Timestamp("2025-01-01").value
    acc.consume(decision_ns=start, direction=1, prediction=0.01, signed_gross_return=0.02, cost_rate=0.001)
    acc.consume(decision_ns=start + 5_000_000_000, direction=1, prediction=0.01, signed_gross_return=0.02, cost_rate=0.001)
    acc.consume(decision_ns=start + 61_000_000_000, direction=-1, prediction=-0.01, signed_gross_return=0.01, cost_rate=0.001)
    assert len(acc.records) == 2
    assert acc.summary()["trades"] == 2


def test_champion_selection_uses_validation_robustness() -> None:
    rows = []
    for cost, latency, total in ((1.0, 0.5, 0.5), (2.0, 0.5, 0.2), (1.0, 1.0, 0.3)):
        rows.append(
            {
                "fold_id": "WF_2025",
                "model": "lightgbm",
                "horizon_seconds": 300,
                "quantile": 0.995,
                "latency_seconds": latency,
                "cost_multiplier": cost,
                "trades": 500,
                "mean_net_return": 0.001,
                "profit_factor": 1.3,
                "total_return": total,
                "max_drawdown": 0.1,
                "top10_removed_total_return": 0.2,
            }
        )
    rows.append(
        {
            "fold_id": "WF_2025",
            "model": "lightgbm",
            "horizon_seconds": 300,
            "quantile": 0.990,
            "latency_seconds": 0.5,
            "cost_multiplier": 1.0,
            "trades": 400,
            "mean_net_return": 0.0005,
            "profit_factor": 1.1,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "top10_removed_total_return": 0.1,
        }
    )
    champion = select_validation_champion(pd.DataFrame(rows), DEFAULT_TRADES_BASELINE_CONFIG)
    assert champion is not None
    assert champion["model"] == "lightgbm"


def test_baseline_package_does_not_parse_raw_files_or_sqlite() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "ai_research" / "trades_baseline"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "sqlite3" not in text
    assert "zipfile" not in text
    assert "pd.read_csv" not in text
    assert "OKXTickLoader" not in text


def test_walk_forward_embargo_exceeds_max_label_horizon() -> None:
    config = DEFAULT_TRADES_BASELINE_CONFIG
    for fold in default_folds(config):
        assert fold.fit_end + pd.Timedelta(seconds=config.max_future_seconds) <= fold.calibration_start
        assert fold.calibration_end + pd.Timedelta(seconds=config.max_future_seconds) <= fold.test_start


def test_cache_signature_ignores_model_training_cap() -> None:
    base = DEFAULT_TRADES_BASELINE_CONFIG
    changed = replace(base, train_sample_cap=12345, report_dir="tmp/report")
    assert cache_signature(base) == cache_signature(changed)


def test_month_cache_is_memory_mapped_and_resumable(tmp_path: Path) -> None:
    from src.ai_research.trades_baseline.dataset import (
        build_monthly_sample_cache,
        load_month_shard,
    )

    config = replace(
        DEFAULT_TRADES_BASELINE_CONFIG,
        research_start="2023-01-01 00:00:00",
        research_end="2023-01-01 00:20:00",
        horizons_seconds=(60, 180),
        latency_scenarios_seconds=(0.5, 1.0),
        base_latency_seconds=0.5,
        cache_dir=str(tmp_path / "cache"),
    )
    loader = FakeTradeBarLoader(synthetic_bars())
    paths = build_monthly_sample_cache(
        loader,
        config,
        start=pd.Timestamp(config.research_start),
        end=pd.Timestamp(config.research_end),
        progress=False,
    )
    assert len(paths) == 1
    shard = load_month_shard(paths[0])
    assert len(shard.timestamps_ns) > 0
    assert isinstance(shard.features, np.memmap)
    assert isinstance(shard.labels, np.memmap)
    paths_again = build_monthly_sample_cache(
        loader,
        config,
        start=pd.Timestamp(config.research_start),
        end=pd.Timestamp(config.research_end),
        progress=False,
    )
    assert paths_again == paths


def test_missing_lightgbm_fails_before_expensive_pipeline_work(monkeypatch) -> None:
    import src.ai_research.trades_baseline.modeling as modeling

    monkeypatch.setattr(modeling, "LGBMRegressor", None)
    try:
        modeling.validate_model_dependencies(("ridge", "lightgbm"))
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing LightGBM must fail startup preflight")
    assert "python -m pip install lightgbm" in message
    assert "do not pass --force-rebuild-cache" in message


def test_ridge_only_dependency_preflight_does_not_require_lightgbm(monkeypatch) -> None:
    import src.ai_research.trades_baseline.modeling as modeling

    monkeypatch.setattr(modeling, "LGBMRegressor", None)
    assert modeling.validate_model_dependencies(("ridge",)) == {"ridge": "available"}


def test_pipeline_dependency_check_precedes_loader_and_cache(monkeypatch, tmp_path: Path) -> None:
    import src.ai_research.trades_baseline.modeling as modeling
    import src.ai_research.trades_baseline.pipeline as pipeline

    monkeypatch.setattr(modeling, "LGBMRegressor", None)

    def forbidden_loader(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("loader must not be touched before dependency preflight")

    monkeypatch.setattr(pipeline, "create_loader", forbidden_loader)
    config = replace(
        DEFAULT_TRADES_BASELINE_CONFIG,
        report_dir=str(tmp_path / "report"),
        cache_dir=str(tmp_path / "cache"),
    )
    try:
        pipeline.run_pipeline(config, models=("ridge", "lightgbm"), progress=False)
    except RuntimeError as exc:
        assert "startup dependency check failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("pipeline must fail before data work")
    assert not (tmp_path / "cache").exists()
