from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_pool_forecast.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_pool_forecast.labels import (
    attach_release_labels,
    attach_touch_labels,
    deterministic_control_sample,
)
from src.ai_research.latent_liquidity_pool_forecast.modeling import (
    feature_columns,
    fit_models,
    metric_table,
    predict,
)
from src.ai_research.latent_liquidity_pool_forecast.reports import causal_audit, decide
from src.ai_research.latent_liquidity_pool_forecast.spatial import (
    add_zone_path_features,
    attach_swing_spatial_features,
    expand_zone_lattice,
)


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2025-01-01 12:00:00")],
            "feature_available_time": [pd.Timestamp("2025-01-01 12:00:00")],
            "period": ["VALIDATION_2025Q1_Q3"],
            "current_price": [100.0],
            "macro_bar_start_time": [pd.Timestamp("2025-01-01 11:59:00")],
            "macro_drawdown_from_high_15m": [-0.01],
            "macro_rally_from_low_15m": [0.01],
            "macro_notional_15m": [10_000_000.0],
            "macro_ret_15m": [0.005],
            "macro_notional_intensity_15m": [1.5],
            "macro_delta_share_15m": [0.1],
        }
    )


def test_zone_lattice_is_symmetric_and_not_swing_gated() -> None:
    config = replace(DEFAULT_CONFIG, zone_offsets_bp=(10.0, 20.0), macro_windows_minutes=(15,))
    zones = expand_zone_lattice(_snapshot(), config)
    assert len(zones) == 4
    assert zones["zone_side"].tolist() == ["DOWN", "DOWN", "UP", "UP"]
    assert np.isclose(zones.iloc[0]["zone_price"], 99.9)
    assert np.isclose(zones.iloc[-1]["zone_price"], 100.2)
    assert "zone_boundary_nesting_count" in zones


def test_touch_labels_use_only_future_completed_minutes() -> None:
    config = replace(DEFAULT_CONFIG, zone_offsets_bp=(10.0,), touch_horizons_minutes=(60, 240, 720), primary_horizon_minutes=720, macro_windows_minutes=(15,))
    zones = expand_zone_lattice(_snapshot(), config)
    idx = pd.date_range("2025-01-01 11:59:00", periods=722, freq="min")
    close = np.full(len(idx), 100.0)
    low = close.copy(); high = close.copy()
    # after decision, touch 99.9 on the downside but never 100.1 upside
    low[3] = 99.85
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    labeled = attach_touch_labels(zones, bars, config)
    down = labeled.loc[labeled["zone_side"].eq("DOWN")].iloc[0]
    up = labeled.loc[labeled["zone_side"].eq("UP")].iloc[0]
    assert bool(down["touch_60m"])
    assert not bool(up["touch_60m"])


def test_release_label_maps_future_episode_to_spatial_zone() -> None:
    config = replace(DEFAULT_CONFIG, zone_offsets_bp=(10.0, 20.0), zone_half_width_bp=4.0, macro_windows_minutes=(15,))
    zones = expand_zone_lattice(_snapshot(), config)
    for h in config.touch_horizons_minutes:
        zones[f"touch_{h}m"] = True
    episodes = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2025-01-01 12:30:00")],
            "event_side": ["DOWN"],
            "event_reference_price": [99.90],
            "favorable_reversal": [True],
            "outcome_type": ["EXTEND_STABILIZE_REVERSAL"],
            "release_density_proxy": [3.0],
            "release_episode_size": [10],
            "release_score": [4.0],
            "path_cluster": [10],
            "future_extension_bp": [18.0],
            "future_reversal_after_extreme_bp": [40.0],
            "future_time_to_extreme_seconds": [95.0],
        }
    )
    labeled = attach_release_labels(zones, episodes, config)
    target = labeled.loc[(labeled["zone_side"].eq("DOWN")) & labeled["zone_distance_bp"].eq(10.0)].iloc[0]
    other = labeled.loc[(labeled["zone_side"].eq("DOWN")) & labeled["zone_distance_bp"].eq(20.0)].iloc[0]
    assert bool(target["release_within_horizon"])
    assert bool(target["favorable_release"])
    assert target["release_path_cluster"] == 10
    assert np.isclose(target["sweep_depth_bp"], 18.0)
    assert not bool(other["release_within_horizon"])


def test_all_old_unswept_15m_plus_swings_are_kept_until_swept() -> None:
    config = replace(DEFAULT_CONFIG, zone_offsets_bp=(10.0,), swing_band_bp=(5.0, 10.0, 25.0, 50.0), macro_windows_minutes=(15,))
    zones = expand_zone_lattice(_snapshot(), config)
    lifecycle = pd.DataFrame(
        {
            "level_id": [1, 2, 3],
            "level_side": ["LOW", "LOW", "LOW"],
            "source_timeframe": ["15m", "4H", "5m"],
            "source_timeframe_min": [15, 240, 5],
            "level_price": [99.90, 99.86, 99.90],
            "initial_available_time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01 11:00")],
            "sweep_available_time": [pd.NaT, pd.Timestamp("2024-06-01"), pd.NaT],
        }
    )
    attached = attach_swing_spatial_features(zones, lifecycle, config)
    down = attached.loc[attached["zone_side"].eq("DOWN")].iloc[0]
    # only the old unswept 15m level survives.  Swept 4H and forbidden 5m are excluded.
    assert down["swing_count_5bp"] == 1
    assert down["swing_nearest_age_minutes"] > 100_000


def test_control_sampling_keeps_all_touched_and_weights_background() -> None:
    config = replace(DEFAULT_CONFIG, touched_control_keep_fraction=0.5, untouched_control_keep_fraction=0.5)
    frame = pd.DataFrame(
        {
            "zone_id": [f"z{i}" for i in range(100)],
            "decision_time": [pd.Timestamp("2025-01-01 12:00:00") + pd.Timedelta(minutes=15 * (i // 10)) for i in range(100)],
            "zone_side": ["DOWN" if (i % 20) < 10 else "UP" for i in range(100)],
            "touch_720m": [True] * 10 + [False] * 90,
            "release_within_horizon": [True] + [False] * 99,
        }
    )
    sampled = deterministic_control_sample(frame, config)
    assert "z0" in set(sampled["zone_id"])  # every release survives control sampling
    model = sampled["model_sample_keep"].astype(bool)
    assert sampled.loc[model & sampled["touch_720m"] & ~sampled["release_within_horizon"], "sample_weight"].eq(2.0).all()
    assert sampled.loc[model & ~sampled["touch_720m"], "sample_weight"].eq(2.0).all()
    assert sampled.loc[~model, "sample_weight"].eq(0.0).all()


def _model_frame(n_per_period: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = []
    for period in DEFAULT_CONFIG.periods:
        n = n_per_period
        side = rng.choice(["DOWN", "UP"], size=n)
        dist = rng.choice(DEFAULT_CONFIG.zone_offsets_bp, size=n)
        path = rng.normal(size=n)
        swing = rng.normal(size=n)
        touch_prob = 1.0 / (1.0 + np.exp((dist - 100) / 50))
        touch = rng.random(n) < touch_prob
        release_prob = 1.0 / (1.0 + np.exp(-(1.2 * path - 0.006 * dist)))
        release = touch & (rng.random(n) < release_prob)
        fav_prob = 1.0 / (1.0 + np.exp(-(0.9 * path + 0.05 * swing)))
        fav = release & (rng.random(n) < fav_prob)
        decision = pd.date_range("2023-01-01", periods=n, freq="15min")
        frames.append(pd.DataFrame({
            "zone_id": [f"{period}-{i}" for i in range(n)],
            "decision_time": decision,
            "feature_available_time": decision,
            "period": period,
            "zone_side": side,
            "zone_price": 100.0,
            "current_price": 100.0,
            "zone_distance_bp": dist,
            "side_is_down": (side == "DOWN").astype(np.int8),
            "zone_boundary_nesting_count": rng.integers(0, 5, size=n),
            "zone_untouched_window_count": rng.integers(0, 5, size=n),
            "macro_notional_intensity_60m": np.exp(path / 3),
            "macro_pressure_without_progress_60m": path,
            "micro_path_efficiency_60s": rng.uniform(size=n),
            "swing_count_25bp": np.maximum(0, (swing + 1).astype(int)),
            "swing_nearest_age_minutes": np.abs(swing) * 1000,
            "touch_720m": touch,
            "primary_touch_label_complete": True,
            "release_within_horizon": release,
            "release_on_touch": release,
            "favorable_release": fav,
            "favorable_on_release": fav,
            "continuation_release": release & ~fav,
            "sweep_depth_bp": np.where(release, np.maximum(1, 20 - 5 * path + rng.normal(0, 2, size=n)), np.nan),
            "reversal_after_extreme_bp": np.where(release, np.maximum(1, 25 + 7 * path + rng.normal(0, 3, size=n)), np.nan),
            "sample_weight": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_model_schema_has_liquidity_path_and_optional_swing_increment() -> None:
    frame = _model_frame(100)
    full = feature_columns(frame, include_swing=True)
    path = feature_columns(frame, include_swing=False)
    assert "swing_count_25bp" in full
    assert "swing_count_25bp" not in path
    assert "macro_pressure_without_progress_60m" in path
    assert not any(name.startswith(("touch_", "release_", "favorable_", "sweep_depth")) for name in full)


def test_fit_predict_reports_full_vs_no_swing_vs_distance_baseline() -> None:
    config = replace(DEFAULT_CONFIG, model_n_estimators=30, minimum_train_rows=100, minimum_class_rows=10, model_train_cap_rows=20_000)
    frame = _model_frame(1600)
    models = fit_models(frame, config)
    pred = predict(frame, models)
    assert pred["pool_score"].between(0, 1).all()
    metrics = metric_table(pred, config)
    assert set(metrics["task"]) >= {"RELEASE_FULL", "RELEASE_PATH_NO_SWING", "RELEASE_DISTANCE_BASELINE", "FAVORABLE_FULL", "SWEEP_DEPTH"}


def test_causal_audit_allows_15m_plus_swing_only_as_supplement() -> None:
    frame = _model_frame(100)
    full = feature_columns(frame, include_swing=True)
    gate = pd.DataFrame([{"check": "source", "status": "PASS"}])
    audit = causal_audit(frame, full, gate, DEFAULT_CONFIG)
    assert not audit["status"].eq("FAIL").any()
    assert audit.loc[audit["check"].eq("swing_is_supplement_not_admission"), "status"].iloc[0] == "PASS"


def test_decision_promotes_only_when_path_beats_distance_and_quality_holds() -> None:
    from src.ai_research.latent_liquidity_pool_forecast.reports import decide

    rows = []
    for side in ("DOWN", "UP"):
        rows += [
            {"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "task": "RELEASE_FULL", "roc_auc": 0.66},
            {"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "task": "RELEASE_PATH_NO_SWING", "roc_auc": 0.65},
            {"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "task": "RELEASE_DISTANCE_BASELINE", "roc_auc": 0.60},
            {"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "task": "FAVORABLE_FULL", "roc_auc": 0.62},
            {"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "task": "SWEEP_DEPTH", "spearman": 0.30},
        ]
    metrics = pd.DataFrame(rows)
    top = pd.DataFrame([{"period": DEFAULT_CONFIG.holdout_period, "zone_side": side, "release_count": 100, "release_rate": 0.2, "favorable_given_release": 0.5} for side in ("DOWN", "UP")])
    decision, reasons = decide(metrics, top, pd.DataFrame([{"status": "PASS"}]), DEFAULT_CONFIG)
    assert decision == "PROMOTE_DOWN_AND_UP_TO_R02_1_LIMIT_PLACEMENT_STUDY"
    assert any("Swing uplift" in reason for reason in reasons)


def test_incomplete_primary_touch_window_is_marked_not_silently_negative() -> None:
    config = replace(
        DEFAULT_CONFIG,
        zone_offsets_bp=(10.0,),
        touch_horizons_minutes=(60,),
        primary_horizon_minutes=60,
        macro_windows_minutes=(15,),
    )
    zones = expand_zone_lattice(_snapshot(), config)
    # Only ten future minutes exist although the frozen label requires sixty.
    idx = pd.date_range("2025-01-01 11:59:00", periods=11, freq="min")
    close = np.full(len(idx), 100.0)
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    labeled = attach_touch_labels(zones, bars, config)
    assert not labeled["primary_touch_label_complete"].any()
    assert not labeled["touch_60m"].astype(bool).any()


def test_primary_touch_completeness_flag_is_never_a_model_feature() -> None:
    frame = _model_frame(100)
    frame["primary_touch_label_complete"] = True
    assert "primary_touch_label_complete" not in feature_columns(frame, include_swing=True)


def test_score_deciles_uses_configured_primary_horizon_not_hardcoded_720() -> None:
    from src.ai_research.latent_liquidity_pool_forecast.reports import score_deciles

    config = replace(DEFAULT_CONFIG, touch_horizons_minutes=(60,), primary_horizon_minutes=60)
    frame = _model_frame(100).drop(columns=["touch_720m"]).rename(columns={"release_within_horizon": "release_within_horizon"})
    frame["touch_60m"] = np.arange(len(frame)) % 2 == 0
    frame["pool_score"] = np.linspace(0.0, 1.0, len(frame), endpoint=False)
    out = score_deciles(frame, config)
    assert not out.empty
    assert out["touch_rate"].notna().all()


def test_full_lattice_audit_sampling_keeps_entire_decision_side_group() -> None:
    config = replace(
        DEFAULT_CONFIG,
        zone_offsets_bp=(10.0, 30.0, 50.0),
        full_lattice_audit_group_fraction=1.0,
        touched_control_keep_fraction=0.1,
        untouched_control_keep_fraction=0.1,
    )
    snapshots = pd.concat([_snapshot(), _snapshot().assign(decision_time=pd.Timestamp("2025-01-01 12:15:00"), feature_available_time=pd.Timestamp("2025-01-01 12:15:00"))], ignore_index=True)
    zones = expand_zone_lattice(snapshots, config)
    zones["touch_720m"] = False
    zones["release_within_horizon"] = False
    sampled = deterministic_control_sample(zones, config)
    assert sampled["full_lattice_audit_group"].all()
    sizes = sampled.groupby(["decision_time", "zone_side"]).size()
    assert sizes.eq(len(config.zone_offsets_bp)).all()


def test_touch_means_entering_price_cell_not_reaching_cell_center() -> None:
    config = replace(
        DEFAULT_CONFIG,
        zone_offsets_bp=(10.0, 30.0, 50.0),
        zone_half_width_bp=10.0,
        touch_horizons_minutes=(60,),
        primary_horizon_minutes=60,
        macro_windows_minutes=(15,),
    )
    zones = expand_zone_lattice(_snapshot(), config)
    idx = pd.date_range("2025-01-01 11:59:00", periods=62, freq="min")
    close = np.full(len(idx), 100.0)
    low = close.copy(); high = close.copy()
    # 21bp down enters the 20-40bp cell centered at 30bp, but never reaches its 30bp center.
    low[3] = 99.79
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    labeled = attach_touch_labels(zones, bars, config)
    down30 = labeled.loc[labeled["zone_side"].eq("DOWN") & labeled["zone_distance_bp"].eq(30.0)].iloc[0]
    down50 = labeled.loc[labeled["zone_side"].eq("DOWN") & labeled["zone_distance_bp"].eq(50.0)].iloc[0]
    assert bool(down30["touch_60m"])
    assert not bool(down50["touch_60m"])


def test_release_horizon_excludes_exact_right_endpoint() -> None:
    config = replace(DEFAULT_CONFIG, zone_offsets_bp=(10.0,), zone_half_width_bp=5.0, macro_windows_minutes=(15,), primary_horizon_minutes=720)
    zones = expand_zone_lattice(_snapshot(), config)
    for h in config.touch_horizons_minutes:
        zones[f"touch_{h}m"] = False
    decision = pd.Timestamp("2025-01-01 12:00:00")
    episodes = pd.DataFrame({
        "event_time": [decision + pd.Timedelta(minutes=720)],
        "event_side": ["DOWN"],
        "event_reference_price": [99.90],
        "favorable_reversal": [True],
        "outcome_type": ["EXTEND_STABILIZE_REVERSAL"],
        "release_density_proxy": [3.0], "release_episode_size": [10], "release_score": [4.0],
        "path_cluster": [10], "future_extension_bp": [18.0], "future_reversal_after_extreme_bp": [40.0],
        "future_time_to_extreme_seconds": [95.0],
    })
    labeled = attach_release_labels(zones, episodes, config)
    target = labeled.loc[labeled["zone_side"].eq("DOWN")].iloc[0]
    assert not bool(target["release_within_horizon"])
