#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research_common.post_sweep_supervised import (
    FeatureModuleResult,
    PostSweepSupervisedConfig,
    build_oi_features,
    build_range_features,
    build_trade_1s_features,
    causal_audit,
    checkpoint_index,
    load_r13_data,
    module_coverage_report,
    run_supervised_ablation,
)
from src.research_common.post_sweep_supervised.data import _derive_m0_labels


def _write_minimal_reports(root: Path) -> tuple[Path, Path]:
    r09 = root / "r09"
    r12 = root / "r12"
    r09.mkdir(parents=True)
    r12.mkdir(parents=True)
    dates = pd.DatetimeIndex(np.concatenate([
        pd.date_range("2023-01-01 00:01:00", "2024-12-01 00:01:00", periods=12).to_numpy(),
        pd.date_range("2025-01-05 00:01:00", "2025-09-25 00:01:00", periods=12).to_numpy(),
        pd.date_range("2025-10-05 00:01:00", "2026-06-20 00:01:00", periods=12).to_numpy(),
    ]))
    ids = [f"Z{i}" for i in range(len(dates))]
    features = pd.DataFrame({
        "zone_event_id": ids + ["CONTROL"],
        "event_kind": ["swing_zone_sweep"] * len(ids) + ["non_zone_downside_control"],
        "event_bar_time": list(dates - pd.Timedelta(minutes=1)) + [pd.Timestamp("2023-01-01")],
        "event_available_time": list(dates) + [pd.Timestamp("2023-01-01 00:01:00")],
        "zone_primary_timeframe": ["15m"] * (len(ids) + 1),
        "zone_timeframe_count": [1 + (i % 2) for i in range(len(ids) + 1)],
        "sweep_low": [100.0] * (len(ids) + 1),  # absolute price must never enter the contract
        "pre_return_60m": np.linspace(-0.02, 0.02, len(ids) + 1),
        "current_delta_ratio": np.linspace(-0.2, 0.2, len(ids) + 1),
    })
    label_rows = []
    for i, (event_id, when) in enumerate(zip(ids, dates)):
        label_rows.append({
            "zone_event_id": event_id,
            "event_kind": "swing_zone_sweep",
            "r09_entry_time": when,
            "tp15_sl15_outcome": "TP" if i % 2 == 0 else "SL",
            "tp15_sl15_same_bar_both_flag": i == 0,
            "tp15_sl15_gross_return": 0.0015 if i % 2 == 0 else -0.0015,
            "tp15_sl15_net_return_1x_cost": 0.0002 if i % 2 == 0 else -0.0028,
            "tp15_sl15_net_return_2x_cost": -0.0011 if i % 2 == 0 else -0.0041,
            "release_event_bar_downside_bp": 10.0 + i,
            "release_sell_notional_1m_vs_prior60": 2.0 + i,
            "release_sell_notional_5m_vs_prior60": 3.0 + i,
            "release_sell_notional_15m_vs_prior60": 4.0 + i,
            "stop_release_score": 1.0 + i,
            "high_stop_release_label": i % 2 == 0,
        })
    labels = pd.DataFrame(label_rows)
    features.to_csv(r09 / "16_zone_feature_table.csv.gz", index=False, compression="gzip")
    labels.to_csv(r09 / "17_zone_label_table.csv.gz", index=False, compression="gzip")

    checkpoint_rows = []
    outcome_rows = []
    for i, (event_id, event_time) in enumerate(zip(ids, dates)):
        for minutes in (3, 5, 10):
            decision = event_time + pd.Timedelta(minutes=minutes + 1)
            checkpoint_rows.append({
                "zone_event_id": event_id,
                "checkpoint_minutes": minutes,
                "checkpoint_available_time": decision,
                "entry_time": decision,
                "checkpoint_close": 2_000.0,  # absolute price must be excluded
                "first_floor_reclaim_pos_visible": 123456,  # global bar position must be excluded
                "second_wave_new_low_visible": minutes == 3,
                "post_close_below_floor_share": 0.25,
                "state": "REJECT",
                "state_direction": "LONG",
                "high_release": True,
            })
            for direction in ("LONG", "SHORT"):
                favorable = (i % 2 == 0) if direction == "LONG" else (i % 2 == 1)
                outcome_rows.append({
                    "zone_event_id": event_id,
                    "checkpoint_minutes": minutes,
                    "checkpoint_available_time": decision,
                    "entry_time": decision,
                    "trade_direction": direction,
                    "natural_stop_distance_bp": 30.0,
                    "r2p0_outcome": "TARGET" if favorable else "STOP",
                    "r2p0_target_before_stop": favorable,
                    "r2p0_gross_r": 2.0 if favorable else -1.0,
                    "r2p0_net_1x_r": 1.55 if favorable else -1.45,
                    "r2p0_net_2x_r": 1.1 if favorable else -1.9,
                })
    pd.DataFrame(checkpoint_rows).to_csv(r12 / "14_checkpoint_feature_table.csv.gz", index=False, compression="gzip")
    pd.DataFrame(outcome_rows).to_csv(r12 / "15_outcome_label_table.csv.gz", index=False, compression="gzip")
    return r09, r12


def _small_config() -> PostSweepSupervisedConfig:
    return replace(
        PostSweepSupervisedConfig(),
        minimum_train_events=5,
        minimum_validation_events=5,
        minimum_holdout_events=5,
        minimum_holdout_trades=1,
        hgb_min_samples_leaf=2,
        hgb_max_iter=30,
        sample_rows=100,
    ).validate()


def test_r09_controls_and_future_release_fields_are_excluded(tmp_path: Path) -> None:
    r09, r12 = _write_minimal_reports(tmp_path)
    bundle = load_r13_data(r09, r12, _small_config())
    assert all(frame["zone_event_id"].ne("CONTROL").all() for frame in bundle.datasets.values())
    assert "release_sell_notional_1m_vs_prior60" in bundle.base_columns[0]
    assert "release_sell_notional_5m_vs_prior60" not in bundle.base_columns[0]
    assert "stop_release_score" not in bundle.base_columns[3]
    assert "release_sell_notional_5m_vs_prior60" in bundle.base_columns[5]
    assert "stop_release_score" in bundle.base_columns[5]
    assert all("_15m" not in name for columns in bundle.base_columns.values() for name in columns if "release_" in name)
    assert "sweep_low" not in bundle.base_columns[0]
    assert "checkpoint_close" not in bundle.dynamic_columns[5]
    assert "first_floor_reclaim_pos_visible" not in bundle.dynamic_columns[5]
    assert "second_wave_new_low_visible" in bundle.dynamic_columns[5]
    assert bundle.audit["release_availability_violations"].eq(0).all()


def test_m0_same_bar_both_is_missing_for_both_directions(tmp_path: Path) -> None:
    r09, _ = _write_minimal_reports(tmp_path)
    labels = pd.read_csv(r09 / "17_zone_label_table.csv.gz")
    derived = _derive_m0_labels(labels)
    ambiguous = derived.loc[derived["zone_event_id"].eq("Z0")].iloc[0]
    assert pd.isna(ambiguous["long_net_1x_r"])
    assert pd.isna(ambiguous["short_net_1x_r"])


def _checkpoint_frame(decision: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "checkpoint_id": ["Z__M3"],
        "zone_event_id": ["Z"],
        "checkpoint_minutes": [3],
        "decision_time": [decision],
        "event_available_time": [decision - pd.Timedelta(minutes=3)],
        "event_bar_time": [decision - pd.Timedelta(minutes=4)],
        "period": ["TRAIN"],
        "split": ["TRAIN"],
    })


def test_trade_1s_excludes_the_decision_second(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = pd.Timestamp("2023-01-01 00:01:00")
    index = pd.date_range(decision - pd.Timedelta(seconds=60), decision, freq="1s")
    bars = pd.DataFrame(index=index)
    bars["open"] = 100.0
    bars["high"] = 100.0
    bars["low"] = 100.0
    bars["close"] = 100.0
    for name in (
        "notional", "buy_notional", "sell_notional", "delta_notional", "trades_count",
        "buy_trades_count", "sell_trades_count", "large_buy_notional", "large_sell_notional",
        "large_delta_notional", "large_buy_trades_count", "large_sell_trades_count", "large_trades_count",
    ):
        bars[name] = 1.0
    bars["max_trade_notional"] = 1.0
    bars.loc[decision, "notional"] = 1_000_000_000.0

    class FakeLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load_local_data(self, *_: object, **__: object) -> pd.DataFrame:
            return bars

    monkeypatch.setattr("src.research_common.post_sweep_supervised.features.OKXTradeBarLoader", FakeLoader)
    result = build_trade_1s_features(
        _checkpoint_frame(decision), symbol="ETH-USDT-SWAP", data_dir=None,
        db_name="unused.db", config=replace(_small_config(), trade_chunk_days=1), progress=False,
    )
    row = result.features.iloc[0]
    assert bool(row["trade1s_causal_valid"])
    assert pd.Timestamp(row["trade1s_latest_bar_time"]) == decision - pd.Timedelta(seconds=1)
    assert row["trade_recent_60s_notional"] < 1_000_000_000.0


def test_range_features_use_only_completed_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = pd.Timestamp("2023-01-01 00:10:00")
    bars = pd.DataFrame({
        "bar_id": [1, 2, 3],
        "end_ts": pd.to_datetime(["2023-01-01 00:09:00", "2023-01-01 00:10:00", "2023-01-01 00:11:00"]),
        "direction": [-1, 1, -100],
        "duration_seconds": [10.0, 20.0, 1.0],
        "open": [100.0, 99.8, 99.9],
        "close": [99.8, 99.9, 90.0],
        "notional": [1_000.0, 1_000.0, 1_000_000_000.0],
        "buy_notional": [400.0, 600.0, 0.0],
        "sell_notional": [600.0, 400.0, 1_000_000_000.0],
        "delta_notional": [-200.0, 200.0, -1_000_000_000.0],
        "max_trade_notional": [100.0, 100.0, 1_000_000_000.0],
    })

    class FakeLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load_local_data(self, *_: object, **__: object) -> pd.DataFrame:
            return bars

    monkeypatch.setattr("src.research_common.post_sweep_supervised.features.OKXRangeBarLoader", FakeLoader)
    result = build_range_features(
        _checkpoint_frame(decision), symbol="ETH-USDT-SWAP", data_dir=None,
        db_name="unused.db", config=replace(_small_config(), range_chunk_days=1), progress=False,
    )
    row = result.features.iloc[0]
    assert bool(row["range_causal_valid"])
    assert pd.Timestamp(row["range_last_end_time"]) == decision
    assert row["range_last3_direction_sum"] == 0.0
    assert row["range_last3_notional"] < 1_000_000_000.0


def test_cached_false_strings_do_not_count_as_module_coverage() -> None:
    checkpoints = pd.DataFrame({
        "checkpoint_id": ["a", "b"], "checkpoint_minutes": [3, 3],
        "split": ["TRAIN", "HOLDOUT"], "zone_event_id": ["z1", "z2"],
        "decision_time": pd.to_datetime(["2023-01-01", "2025-11-01"]),
    })
    module = FeatureModuleResult(
        "trade_1s",
        pd.DataFrame({"checkpoint_id": ["a", "b"], "trade1s_causal_valid": ["False", "False"]}),
        pd.DataFrame(),
    )
    empty = FeatureModuleResult("range_r0020", pd.DataFrame({"checkpoint_id": ["a", "b"], "range_causal_valid": False}), pd.DataFrame())
    modules = {
        "trade_1s": module,
        "range_r0020": empty,
        "footprint": FeatureModuleResult("footprint", pd.DataFrame({"checkpoint_id": ["a", "b"], "fp_causal_valid": False}), pd.DataFrame()),
        "oi": FeatureModuleResult("oi", pd.DataFrame({"checkpoint_id": ["a", "b"], "oi_context_present": False}), pd.DataFrame()),
    }
    coverage = module_coverage_report(checkpoints, modules)
    trade = coverage.loc[coverage["module"].eq("trade_1s")]
    assert trade["coverage"].eq(0.0).all()


def test_validation_thresholds_do_not_depend_on_holdout(tmp_path: Path) -> None:
    r09, r12 = _write_minimal_reports(tmp_path)
    cfg = _small_config()
    bundle = load_r13_data(r09, r12, cfg)
    checkpoints = checkpoint_index(bundle.datasets)
    modules = {
        "trade_1s": FeatureModuleResult("trade_1s", pd.DataFrame({"checkpoint_id": checkpoints["checkpoint_id"], "trade1s_causal_valid": False}), pd.DataFrame()),
        "range_r0020": FeatureModuleResult("range_r0020", pd.DataFrame({"checkpoint_id": checkpoints["checkpoint_id"], "range_causal_valid": False}), pd.DataFrame()),
        "footprint": FeatureModuleResult("footprint", pd.DataFrame({"checkpoint_id": checkpoints["checkpoint_id"], "fp_causal_valid": False}), pd.DataFrame()),
        "oi": FeatureModuleResult("oi", pd.DataFrame({"checkpoint_id": checkpoints["checkpoint_id"], "oi_context_present": False}), pd.DataFrame()),
    }
    first = run_supervised_ablation(bundle.datasets, bundle.base_columns, bundle.dynamic_columns, modules, cfg)
    changed = {minute: frame.copy() for minute, frame in bundle.datasets.items()}
    for frame in changed.values():
        mask = frame["split"].eq("HOLDOUT")
        frame.loc[mask, "pre_return_60m"] = 999.0
        frame.loc[mask, "long_net_1x_r"] = -99.0
        frame.loc[mask, "short_net_1x_r"] = 99.0
    second = run_supervised_ablation(changed, bundle.base_columns, bundle.dynamic_columns, modules, cfg)
    keys = ["checkpoint_minutes", "ablation", "model", "score_quantile", "split"]
    one = first.selection_summary.loc[first.selection_summary["split"].eq("VALIDATION"), keys + ["long_score_threshold", "short_score_threshold"]]
    two = second.selection_summary.loc[second.selection_summary["split"].eq("VALIDATION"), keys + ["long_score_threshold", "short_score_threshold"]]
    merged = one.merge(two, on=keys, suffixes=("_one", "_two"), validate="one_to_one")
    assert np.allclose(merged["long_score_threshold_one"], merged["long_score_threshold_two"], equal_nan=True)
    assert np.allclose(merged["short_score_threshold_one"], merged["short_score_threshold_two"], equal_nan=True)
    audit = causal_audit(checkpoints, modules, first)
    assert audit.loc[audit["status"].eq("FAIL")].empty


def test_numeric_prefix_paths_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "research/liquidity/13_post_sweep_supervised_meta_labeling_study.py").exists()



def test_oi_features_keep_series_semantics_when_optional_columns_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = pd.DataFrame({
        "checkpoint_id": ["Z__M5"],
        "decision_time": [pd.Timestamp("2025-01-01 00:05:00")],
    })
    lookup = pd.DataFrame({"checkpoint_id": ["Z__M5"]})
    captured: dict[str, pd.DataFrame] = {}

    def fake_load_binance_oi_context(source: pd.DataFrame, **_: object) -> pd.DataFrame:
        captured["source"] = source.copy()
        return pd.DataFrame({
            "checkpoint_id": source["checkpoint_id"].astype(str),
            "oi_context_present": [False] * len(source),
        })

    monkeypatch.setattr(
        "src.research_common.post_sweep_supervised.features.load_binance_oi_context",
        fake_load_binance_oi_context,
    )
    result = build_oi_features(
        checkpoints,
        lookup,
        symbol="ETHUSDT",
        data_dir=None,
        db_name="unused.db",
    )
    assert len(result.features) == 1
    assert result.features["oi_context_present"].eq(False).all()
    source = captured["source"]
    assert source["price_change_5m_bp"].isna().all()
    assert source["delta_ratio_5m"].isna().all()



def test_r13_censors_time_outcomes_instead_of_learning_time_exits(tmp_path: Path) -> None:
    r09, r12 = _write_minimal_reports(tmp_path)
    outcomes = pd.read_csv(r12 / "15_outcome_label_table.csv.gz")
    target = outcomes.index[0]
    outcomes.loc[target, "r2p0_outcome"] = "TIME"
    outcomes.loc[target, "r2p0_target_before_stop"] = False
    outcomes.loc[target, "r2p0_net_1x_r"] = 1.25
    event_id = str(outcomes.loc[target, "zone_event_id"])
    minutes = int(outcomes.loc[target, "checkpoint_minutes"])
    direction = str(outcomes.loc[target, "trade_direction"]).lower()
    outcomes.to_csv(r12 / "15_outcome_label_table.csv.gz", index=False, compression="gzip")

    bundle = load_r13_data(r09, r12, _small_config())
    row = bundle.datasets[minutes].loc[bundle.datasets[minutes]["zone_event_id"].eq(event_id)].iloc[0]
    assert pd.isna(row[f"{direction}_profitable_label"])
