from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_first_touch_ranking.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_first_touch_ranking.labels import (
    _aggregate_episode_windows_for_group,
    _episode_arrays,
    _exact_touch_seconds,
    _first_touch_minute_positions,
    _flow_labels,
    _prepare_second_arrays,
    add_relative_relevance,
)
from src.ai_research.latent_liquidity_first_touch_ranking.modeling import (
    feature_columns,
    fit_models,
    predict,
    ranking_metrics,
    top_zone_summary,
)
from src.ai_research.latent_liquidity_first_touch_ranking.reports import causal_audit


def test_first_touch_minute_positions_down_and_strict_zero() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="1min")
    minute = pd.DataFrame({"low": [100.0, 99.0, 97.0], "high": [101.0, 100.0, 99.0]}, index=idx)
    out = _first_touch_minute_positions(
        minute,
        side="DOWN",
        thresholds=np.array([100.0, 99.5, 98.0, 96.0]),
        zero_distance=np.array([True, False, False, False]),
    )
    assert out.tolist() == [1, 1, 2, -1]


def test_exact_touch_and_flow_use_dense_second_arrays() -> None:
    idx = pd.date_range("2026-01-01 00:00:00", periods=500, freq="1s")
    second = pd.DataFrame(
        {
            "low": np.full(500, 100.0),
            "high": np.full(500, 101.0),
            "notional": np.ones(500),
            "trades_count": np.ones(500),
            "delta_notional": np.ones(500),
            "unsafe_gap": np.zeros(500, dtype=np.int8),
        },
        index=idx,
    )
    second.loc[idx[70]:, "low"] = 98.0
    second.loc[idx[70]:idx[129], "notional"] = 2.0
    arrays = _prepare_second_arrays(second)
    touch, pos = _exact_touch_seconds(
        arrays,
        np.array([np.datetime64("2026-01-01 00:01:00")]),
        side="DOWN",
        thresholds=np.array([99.0]),
        strict=np.array([False]),
    )
    assert pd.Timestamp(touch[0]) == idx[70]
    assert pos[0] == 70
    flow = _flow_labels(arrays, pos, windows=(30, 60, 180, 300), pre_seconds=60)
    assert flow["ft_micro_label_complete"][0]
    assert np.isclose(flow["ft_notional_ratio_60s"][0], 2.0)


def test_episode_windows_are_anchored_after_first_touch() -> None:
    episodes = pd.DataFrame(
        {
            "event_time": pd.to_datetime([
                "2026-01-01 00:00:09",
                "2026-01-01 00:00:10",
                "2026-01-01 00:00:40",
                "2026-01-01 00:03:20",
            ]),
            "event_side": ["DOWN"] * 4,
            "event_reference_price": [99.0] * 4,
            "release_density_proxy": [100.0, 2.0, 3.0, 4.0],
            "release_episode_size": [1, 1, 1, 1],
            "release_score": [1.0, 1.0, 1.0, 1.0],
            "favorable_reversal": [False, True, False, True],
            "outcome_type": ["MIXED", "EXTENSION_STABILIZE_REVERSAL", "ACCEPT_CONTINUATION", "MIXED"],
            "future_extension_bp": [1.0, 10.0, 20.0, 30.0],
            "future_reversal_after_extreme_bp": [1.0, 20.0, 10.0, 40.0],
        }
    )
    arrays = _episode_arrays(episodes)["DOWN"]
    rows = pd.DataFrame({"zone_near_price": [100.0], "zone_far_price": [98.0], "zone_distance_bp": [100.0], "zone_near_distance_bp": [0.0], "zone_far_distance_bp": [200.0], "current_price": [100.0]})
    labels = _aggregate_episode_windows_for_group(
        rows,
        np.array([np.datetime64("2026-01-01 00:00:10")]),
        arrays,
        side="DOWN",
        windows=(30, 60, 180, 300),
    )
    assert labels["ft_release_density_sum_30s"][0] == 2.0
    assert labels["ft_release_density_sum_60s"][0] == 5.0
    assert labels["ft_release_density_sum_180s"][0] == 5.0
    assert labels["ft_release_density_sum_300s"][0] == 9.0
    assert labels["ft_favorable_episode_count_60s"][0] == 1.0
    assert labels["ft_continuation_episode_count_60s"][0] == 1.0


def test_relative_relevance_is_within_group_not_absolute_threshold() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2025-01-01"] * 4),
            "zone_side": ["DOWN"] * 4,
            "first_touch_label_complete": [True] * 4,
            "ft_release_density_sum_180s": [0.0, 1.0, 3.0, 10.0],
        }
    )
    out = add_relative_relevance(frame, replace(DEFAULT_CONFIG, rank_relevance_grades=4))
    assert out["ranking_group_eligible"].all()
    assert out["ranking_relevance"].tolist() == sorted(out["ranking_relevance"].tolist())
    assert out["ranking_relevance"].max() == 3


def _ranking_frame(groups: int = 12, zones: int = 5) -> pd.DataFrame:
    rows = []
    for side in ("DOWN", "UP"):
        for g in range(groups):
            period = "TRAIN_2023_2024" if g < 8 else ("VALIDATION_2025Q1_Q3" if g < 10 else "HOLDOUT_2025Q4_2026H1")
            dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=g)
            for j in range(zones):
                target = float(j)
                rows.append(
                    {
                        "zone_id": f"{side}-{g}-{j}",
                        "decision_time": dt,
                        "feature_available_time": dt,
                        "period": period,
                        "zone_side": side,
                        "zone_distance_bp": float(10 + 20 * j),
                        "side_is_down": int(side == "DOWN"),
                        "micro_signal": target + g * 0.001,
                        "swing_count_25bp": float(4 - j),
                        "first_touch_observed": True,
                        "first_touch_label_complete": True,
                        "first_touch_time": dt + pd.Timedelta(minutes=30 + j),
                        "touch_720m": True,
                        "ranking_target": target,
                        "ranking_group": f"{dt}|{side}",
                        "ranking_relevance": min(j, 4),
                        "ranking_group_eligible": True,
                        "ft_release_episode_count_180s": int(j > 0),
                        "ft_favorable_episode_count_180s": int(j >= 3),
                        "ft_continuation_episode_count_180s": int(j == 1),
                        "ft_notional_ratio_60s": 1.0 + j / 10.0,
                        "current_price": 100.0,
                        "zone_price": 99.0,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_features_exclude_swing_and_future_labels() -> None:
    frame = _ranking_frame()
    cols = feature_columns(frame, include_swing=False)
    assert "micro_signal" in cols
    assert not any(c.startswith("swing_") for c in cols)
    assert not any(c.startswith("ft_") for c in cols)
    assert "ranking_target" not in cols
    assert "first_touch_time" not in cols


def test_ranker_learns_relative_order_and_reports_metrics() -> None:
    frame = _ranking_frame()
    cfg = replace(
        DEFAULT_CONFIG,
        minimum_rank_groups=2,
        model_train_cap_rows_per_side=1000,
        model_n_estimators=30,
        model_min_child_samples=2,
    )
    models = fit_models(frame, cfg)
    pred = predict(frame, models)
    metrics = ranking_metrics(pred, cfg)
    hold = metrics.loc[
        metrics["period"].eq(cfg.holdout_period)
        & metrics["zone_side"].eq("DOWN")
        & metrics["model"].eq("PATH_NO_SWING")
    ]
    assert not hold.empty
    assert float(hold.iloc[0]["mean_group_spearman"]) > 0.5
    top = top_zone_summary(pred, cfg)
    assert not top.empty
    assert top["top1_touched"].gt(0).all()


def test_causal_audit_rejects_no_primary_swing_or_future_leak() -> None:
    frame = _ranking_frame(groups=2, zones=25)
    cfg = replace(DEFAULT_CONFIG, minimum_rank_groups=1, model_n_estimators=5, model_min_child_samples=1)
    models = fit_models(frame, cfg)
    source_gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    audit = causal_audit(frame, models, source_gate, cfg)
    assert audit["status"].eq("PASS").all()


def test_build_first_touch_dataset_integration_uses_equal_windows(monkeypatch, tmp_path) -> None:
    from src.ai_research.latent_liquidity_first_touch_ranking import labels as label_mod

    decision = pd.Timestamp("2026-01-01 00:01:00")
    minute_idx = pd.date_range("2025-12-31 23:59:00", periods=20, freq="1min")
    minute = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 100.0,
            "close": 100.0,
            "notional": 1.0,
            "trades_count": 1.0,
            "delta_notional": 0.0,
        },
        index=minute_idx,
    )
    minute.loc[pd.Timestamp("2026-01-01 00:01:00"), "low"] = 99.75
    minute.loc[pd.Timestamp("2026-01-01 00:02:00"), "low"] = 99.55

    second_idx = pd.date_range("2025-12-31 23:59:00", periods=900, freq="1s")
    second = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 100.0,
            "close": 100.0,
            "notional": 1.0,
            "trades_count": 1.0,
            "delta_notional": 0.0,
        },
        index=second_idx,
    )
    second.loc[pd.Timestamp("2026-01-01 00:01:10"), "low"] = 99.75
    second.loc[pd.Timestamp("2026-01-01 00:02:20"), "low"] = 99.55

    class FakeLoader:
        def __init__(self, symbol, timeframe, **kwargs):
            self.timeframe = timeframe

        def fetch_data_by_date_range(self, *args, **kwargs):
            return minute.copy() if self.timeframe == "1m" else second.copy()

    monkeypatch.setattr(label_mod, "OKXTradeBarLoader", FakeLoader)
    audit = pd.DataFrame(
        {
            "zone_id": ["z30", "z50"],
            "decision_time": [decision, decision],
            "feature_available_time": [decision, decision],
            "period": ["HOLDOUT_2025Q4_2026H1"] * 2,
            "zone_side": ["DOWN", "DOWN"],
            "zone_distance_bp": [30.0, 50.0],
            "zone_near_distance_bp": [20.0, 40.0],
            "zone_far_distance_bp": [40.0, 60.0],
            "zone_near_price": [99.8, 99.6],
            "zone_far_price": [99.6, 99.4],
            "zone_price": [99.7, 99.5],
            "current_price": [100.0, 100.0],
            "touch_720m": [True, True],
            "full_lattice_audit_group": [True, True],
            "side_is_down": [1, 1],
        }
    )
    episodes = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-01 00:01:15", "2026-01-01 00:02:30"]),
            "event_side": ["DOWN", "DOWN"],
            "event_reference_price": [99.7, 99.5],
            "release_density_proxy": [2.0, 5.0],
            "release_episode_size": [1, 1],
            "release_score": [1.0, 1.0],
            "favorable_reversal": [True, False],
            "outcome_type": ["EXTEND_STABILIZE_REVERSAL", "ACCEPT_CONTINUATION"],
            "future_extension_bp": [20.0, 30.0],
            "future_reversal_after_extreme_bp": [40.0, 10.0],
        }
    )
    cfg = replace(DEFAULT_CONFIG, cache_dir=str(tmp_path / "cache"), touch_replay_chunk_days=1)
    built = label_mod.build_first_touch_dataset(audit, episodes, cfg, use_cache=False, progress=False)
    out = built.frame.sort_values("zone_distance_bp").reset_index(drop=True)
    assert out["first_touch_time"].tolist() == [pd.Timestamp("2026-01-01 00:01:10"), pd.Timestamp("2026-01-01 00:02:20")]
    assert out["first_touch_label_complete"].all()
    assert out["ft_release_density_sum_180s"].tolist() == [2.0, 5.0]


def test_causal_audit_treats_same_second_bar_start_as_available_one_second_later() -> None:
    frame = _ranking_frame(groups=8, zones=25)
    idx = frame.index[0]
    frame.loc[idx, "first_touch_time"] = frame.loc[idx, "decision_time"]
    frame.loc[idx, "first_touch_observed"] = True
    frame.loc[idx, "first_touch_label_complete"] = True
    cfg = replace(DEFAULT_CONFIG, minimum_rank_groups=2, model_train_cap_rows_per_side=10000, model_n_estimators=10, model_min_child_samples=2)
    models = fit_models(frame, cfg)
    source_gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    audit = causal_audit(frame, models, source_gate, cfg)
    row = audit.loc[audit["check"].eq("first_touch_bar_available_strictly_after_decision")].iloc[0]
    assert row["status"] == "PASS"
