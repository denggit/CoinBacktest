from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.config import StablePathExecutionAuditConfig
from src.ai_research.latent_liquidity_execution_audit.pipeline import run_stable_path_execution_audit
from src.ai_research.latent_liquidity_execution_audit.replay import (
    _safe_columnwise_nanquantile,
    _simulate_confirmation,
    confirmation_offsets,
    replay_samples,
)
from src.ai_research.latent_liquidity_execution_audit.source import scan_source_tables
from src.ai_research.latent_liquidity_execution_audit.statistics import (
    block_bootstrap_ci,
    cluster_feature_profiles,
    daily_stability,
    stability_scorecard,
)


def _write_source(root, rows: int = 240) -> None:
    event_time = pd.date_range("2024-01-01", periods=rows, freq="1min")
    sides = np.where(np.arange(rows) % 2 == 0, "DOWN", "UP")
    clusters = np.where(np.arange(rows) % 4 < 2, 10, 8)
    periods = np.where(np.arange(rows) < rows // 3, "TRAIN_2023_2024", np.where(np.arange(rows) < 2 * rows // 3, "VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"))
    episode = [f"EP_{i:04d}" for i in range(rows)]
    feature = pd.DataFrame(
        {
            "event_id": [f"E{i:04d}" for i in range(rows)],
            "event_time": event_time,
            "event_side": sides,
            "period": periods,
            "release_episode_id": episode,
            "release_episode_ordinal": 1,
            "release_episode_size": 1,
            "release_episode_weight": 1.0,
            "path_efficiency_5s": np.where(clusters == 10, 0.3, 0.8),
            "path_notional_intensity_60s": np.where(clusters == 10, 1.5, 0.8),
            "macro_pressure_without_progress_60m": np.where(clusters == 10, 0.4, 0.1),
            "unswept_nearest_distance_bp": np.linspace(10, 200, rows),
        }
    )
    favorable = clusters == 10
    label = pd.DataFrame(
        {
            "event_id": feature["event_id"],
            "event_time": event_time,
            "event_side": sides,
            "event_reference_price": 100.0,
            "future_extension_bp": np.where(favorable, 20.0, 30.0),
            "future_time_to_extreme_seconds": 60,
            "future_immediate_reversal_bp": 5.0,
            "future_reversal_after_extreme_bp": np.where(favorable, 35.0, 10.0),
            "future_acceptance_fraction_60s": np.where(favorable, 0.2, 0.9),
            "future_stable_after_extreme": favorable,
            "outcome_type": np.where(favorable, "EXTEND_STABILIZE_REVERSAL", "ACCEPT_CONTINUATION"),
            "favorable_reversal": favorable,
        }
    )
    assignment = pd.DataFrame(
        {
            "event_id": feature["event_id"],
            "event_time": event_time,
            "event_side": sides,
            "period": periods,
            "path_cluster": clusters,
            "cluster_distance": 1.0,
        }
    )
    feature.to_csv(root / "12_feature_table.csv.gz", index=False, compression="gzip")
    label.to_csv(root / "13_label_table.csv.gz", index=False, compression="gzip")
    assignment.to_csv(root / "14_cluster_assignment.csv.gz", index=False, compression="gzip")
    (root / "00_manifest.json").write_text(json.dumps({"stage_id": "R01.1"}), encoding="utf-8")
    pd.DataFrame({"check": ["causal"], "violations": [0], "status": ["PASS"]}).to_csv(root / "09_causal_audit.csv", index=False)


def _config(tmp_path, **kwargs) -> StablePathExecutionAuditConfig:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    report = tmp_path / "eth_latent_liquidity_path_v1" / "report"
    report.mkdir(parents=True, exist_ok=True)
    base = StablePathExecutionAuditConfig(
        source_report_dir=str(source),
        report_dir=str(report),
        csv_read_chunk_rows=1000,
        profile_sample_per_stratum=100,
        replay_sample_per_stratum=20,
        bootstrap_repetitions=200,
        minimum_period_episodes=1,
    )
    return replace(base, **kwargs)


def test_source_scan_aligns_large_tables_and_keeps_episode_first_rows(tmp_path) -> None:
    cfg = _config(tmp_path)
    _write_source(cfg.source_report_path)
    result = scan_source_tables(cfg, progress=False)
    assert result.scanned_rows == 240
    assert len(result.episode_rows) == 240
    assert set(result.episode_rows["path_cluster"]) == {8, 10}
    assert "path_efficiency_5s" in result.feature_columns
    assert not result.replay_samples.empty


def test_source_scan_rejects_event_id_misalignment(tmp_path) -> None:
    cfg = _config(tmp_path)
    _write_source(cfg.source_report_path)
    labels = pd.read_csv(cfg.source_report_path / "13_label_table.csv.gz")
    labels.loc[0, "event_id"] = "BROKEN"
    labels.to_csv(cfg.source_report_path / "13_label_table.csv.gz", index=False, compression="gzip")
    try:
        scan_source_tables(cfg, progress=False)
    except RuntimeError as exc:
        assert "alignment mismatch" in str(exc)
    else:
        raise AssertionError("expected alignment failure")


def test_episode_stability_and_day_block_bootstrap_use_episode_rows(tmp_path) -> None:
    cfg = _config(tmp_path)
    _write_source(cfg.source_report_path)
    scan = scan_source_tables(cfg, progress=False)
    score = stability_scorecard(scan.episode_rows, cfg)
    cluster10 = score.loc[score["path_cluster"].eq(10)]
    assert cluster10["favorable_reversal_rate"].eq(1.0).all()
    daily = daily_stability(scan.episode_rows)
    bootstrap = block_bootstrap_ci(daily, cfg)
    assert bootstrap.loc[bootstrap["path_cluster"].eq(10), "positive_gap_ci"].all()
    assert not bootstrap.loc[bootstrap["path_cluster"].eq(8), "positive_gap_ci"].any()


def test_cluster_profiles_are_liquidity_first_and_swing_is_only_one_family(tmp_path) -> None:
    cfg = _config(tmp_path)
    _write_source(cfg.source_report_path)
    scan = scan_source_tables(cfg, progress=False)
    profile = cluster_feature_profiles(scan.profile_samples, scan.feature_columns, cfg)
    target = profile.loc[(profile["path_cluster"].eq(10)) & (profile["feature"].eq("path_efficiency_5s"))]
    assert not target.empty
    assert target["robust_effect"].lt(0).all()
    swing = profile.loc[profile["feature"].str.startswith("unswept_")]
    assert not swing.empty
    assert swing["feature_family"].eq("SWING_INVENTORY_SUPPLEMENT").all()


def _down_reversal_path() -> pd.DataFrame:
    n = 620
    close = np.full(n, 100.0)
    close[:20] = np.linspace(99.98, 99.70, 20)
    close[20:40] = 99.70
    close[40:100] = np.linspace(99.70, 100.20, 60)
    close[100:] = 100.20
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "notional": 1_000_000.0,
            "trades_count": 100.0,
            "delta_notional": -100_000.0,
        }
    )


def test_confirmations_use_completed_path_and_next_second_entry() -> None:
    cfg = StablePathExecutionAuditConfig(bootstrap_repetitions=200)
    path = _down_reversal_path()
    rules = confirmation_offsets(path, "DOWN", cfg)
    assert rules["NO_NEW_EXTREME_15S"] is not None
    assert rules["RECLAIM_5BP"] is not None
    pos = int(rules["RECLAIM_5BP"])
    result = _simulate_confirmation(path, "DOWN", pos, 1, 11.0, 300, cfg)
    assert result is not None
    assert result["entry_pos_seconds"] == pos + 2
    assert result["stop_distance_bp"] > 0
    assert result["mfe_bp"] > 0


def test_stop_is_realized_even_after_one_r_was_seen() -> None:
    cfg = StablePathExecutionAuditConfig(bootstrap_repetitions=200)
    close = np.r_[np.linspace(100.0, 100.3, 20), np.linspace(100.3, 99.0, 80), np.full(520, 99.0)]
    path = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close})
    result = _simulate_confirmation(path, "DOWN", 0, 1, 11.0, 180, cfg)
    assert result is not None
    assert result["one_r_before_stop"]
    assert result["stopped_before_horizon"]
    assert result["realized_gross_bp"] < 0


def test_pipeline_writes_cumulative_compact_report_without_micro_replay(tmp_path) -> None:
    cfg = _config(tmp_path)
    _write_source(cfg.source_report_path)
    result = run_stable_path_execution_audit(
        config=cfg,
        progress=False,
        skip_review_pack=True,
        skip_micro_replay=True,
    )
    assert result.report_dir.exists()
    assert (result.report_dir / "00_manifest.json").exists()
    assert (result.report_dir / "07_cluster_feature_profile.csv").exists()
    assert (result.report_dir / "17_decision.md").exists()



def test_safe_columnwise_nanquantile_preserves_all_nan_offsets_without_warning() -> None:
    import warnings

    values = np.array([[np.nan, 1.0, 3.0], [np.nan, 2.0, 5.0]], dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _safe_columnwise_nanquantile(values, (0.5, 0.9))
    assert not [item for item in caught if "All-NaN slice" in str(item.message)]
    assert np.isnan(result[:, 0]).all()
    assert result[0, 1] == 1.5


def _sparse_replay_bars(event_time: pd.Timestamp, missing_seconds: list[int]) -> pd.DataFrame:
    index = pd.date_range(event_time - pd.Timedelta(seconds=310), event_time + pd.Timedelta(seconds=610), freq="1s")
    close = 100.0 + np.linspace(-0.1, 0.2, len(index))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "notional": 1_000_000.0,
            "trades_count": 100.0,
            "delta_notional": -100_000.0,
        },
        index=index,
    )
    drop_index = [event_time + pd.Timedelta(seconds=value) for value in missing_seconds]
    return frame.drop(index=drop_index)


def test_replay_fills_short_no_trade_gaps_and_reports_stratum_coverage(monkeypatch) -> None:
    import src.ai_research.latent_liquidity_execution_audit.replay as replay_module

    event_time = pd.Timestamp("2025-01-01 12:00:00")
    bars = _sparse_replay_bars(event_time, [10, 11])

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_data_by_date_range(self, *args, **kwargs):
            return bars

    monkeypatch.setattr(replay_module, "OKXTradeBarLoader", DummyLoader)
    sample = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_time": [event_time],
            "event_side": ["DOWN"],
            "period": ["VALIDATION_2025Q1_Q3"],
            "path_cluster": [10],
            "event_reference_price": [100.0],
        }
    )
    result = replay_samples(sample, StablePathExecutionAuditConfig(bootstrap_repetitions=200), progress=False)
    quality = result.replay_quality.set_index("check")
    assert quality.loc["complete_replay_episodes", "value"] == 1
    assert quality.loc["replay_completion_rate", "value"] == 1.0
    assert quality.loc["completion_rate_cluster_10_DOWN_VALIDATION_2025Q1_Q3", "value"] == 1.0


def test_replay_rejects_long_data_gaps(monkeypatch) -> None:
    import src.ai_research.latent_liquidity_execution_audit.replay as replay_module

    event_time = pd.Timestamp("2025-01-01 12:00:00")
    bars = _sparse_replay_bars(event_time, list(range(10, 16)))

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_data_by_date_range(self, *args, **kwargs):
            return bars

    monkeypatch.setattr(replay_module, "OKXTradeBarLoader", DummyLoader)
    sample = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_time": [event_time],
            "event_side": ["DOWN"],
            "period": ["VALIDATION_2025Q1_Q3"],
            "path_cluster": [10],
            "event_reference_price": [100.0],
        }
    )
    result = replay_samples(sample, StablePathExecutionAuditConfig(bootstrap_repetitions=200), progress=False)
    quality = result.replay_quality.set_index("check")
    assert quality.loc["complete_replay_episodes", "value"] == 0
    assert quality.loc["missing_replay_episodes", "value"] == 1
