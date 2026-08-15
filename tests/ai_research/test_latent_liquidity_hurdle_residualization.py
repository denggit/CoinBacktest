from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_hurdle_residualization.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_hurdle_residualization.labels import attach_ranking_targets
from src.ai_research.latent_liquidity_hurdle_residualization.modeling import (
    feature_importance,
    fit_models,
    predict,
    ranking_metrics,
    residual_feature_columns,
    geometry_feature_columns,
)
from src.ai_research.latent_liquidity_hurdle_residualization.nuisance import (
    attach_past_only_nuisance_predictions,
    nuisance_feature_columns,
)
from src.ai_research.latent_liquidity_hurdle_residualization.reports import causal_audit, residual_stability


def _frame(groups_per_month: int = 4, zones: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("TRAIN_2023_2024", pd.Timestamp("2023-01-01"), 12),
        ("VALIDATION_2025Q1_Q3", pd.Timestamp("2025-01-01"), 4),
        ("HOLDOUT_2025Q4_2026H1", pd.Timestamp("2025-10-01"), 4),
    ]
    seq = 0
    for period, start, months in specs:
        for m in range(months):
            month = start + pd.DateOffset(months=m)
            for g in range(groups_per_month):
                dt = month + pd.Timedelta(days=g * 3, hours=(g * 5) % 24)
                activity = 0.8 + 0.35 * np.sin(seq * 0.7)
                seq += 1
                for side in ("DOWN", "UP"):
                    for j in range(zones):
                        distance = float(10 + 20 * j)
                        hidden = np.sin((j + 1) * 0.83 + g * 0.31 + (0.4 if side == "UP" else 0.0))
                        boundary = hidden * 3.0 + 0.01 * distance
                        base_logit = 1.2 - 0.0100 * distance + 0.9 * activity + 1.15 * hidden
                        p_release = 1.0 / (1.0 + np.exp(-base_logit))
                        # deterministic pseudo-uniform, independent of feature scale
                        u = (np.sin((seq + j * 17 + (1 if side == "UP" else 0)) * 12.9898) + 1.0) / 2.0
                        released = bool(u < p_release)
                        base_log_mag = 0.55 - 0.0004 * distance + 0.35 * activity
                        # hidden path signal is the true abnormal component beyond nuisance.
                        log_mag = max(0.02, base_log_mag + 0.65 * hidden)
                        density = float(np.expm1(log_mag)) if released else 0.0
                        raw_rev = (0.0018 * distance - 0.2 * activity + 0.65 * hidden) if released else 0.0
                        fav = density * max(raw_rev, 0.0)
                        cont = density * max(-raw_rev, 0.0)
                        rows.append({
                            "zone_id": f"{period}-{side}-{dt}-{j}",
                            "decision_time": dt,
                            "feature_available_time": dt,
                            "period": period,
                            "zone_side": side,
                            "zone_distance_bp": distance,
                            "side_is_down": int(side == "DOWN"),
                            "current_price": 2000.0,
                            "zone_boundary_gap_60m_bp": boundary,
                            "zone_buildup_log_notional_60m": 1.0 + 0.4 * hidden,
                            "zone_untouched_window_count": int(j > 10),
                            "macro_notional_intensity_60m": activity,
                            "macro_trades_intensity_60m": 1.0 + activity * 0.5,
                            "macro_realized_vol_60m": 0.002 + activity * 0.001,
                            "macro_range_bp_60m": 20.0 + activity * 8.0,
                            "swing_count_25bp": float((j + g) % 4),
                            "first_touch_observed": True,
                            "first_touch_label_complete": True,
                            "first_touch_time": dt + pd.Timedelta(seconds=1 + (j % 5)),
                            "first_touch_available_time": dt + pd.Timedelta(seconds=2 + (j % 5)),
                            "touch_720m": True,
                            "r02_touch_consistent": True,
                            "r02_3_source_eligible": True,
                            "r02_3_1_upstream_eligible": True,
                            "split_purge_eligible": True,
                            "ft_release_density_sum_180s": density,
                            "ft_release_episode_count_180s": float(released),
                            "ft_favorable_density_sum_180s": fav,
                            "ft_continuation_density_sum_180s": cont,
                            "ft_favorable_episode_count_180s": float(fav > 0),
                            "ft_continuation_episode_count_180s": float(cont > 0),
                            "ft_sweep_depth_weighted_bp_180s": (12.0 + 10.0 * max(hidden, -0.5)) if released else np.nan,
                            "ft_reversal_room_weighted_bp_180s": (25.0 + 15.0 * hidden) if released else np.nan,
                        })
    return pd.DataFrame(rows)


def _cfg():
    return replace(
        DEFAULT_CONFIG,
        nuisance_initial_train_months=3,
        nuisance_forward_block_months=2,
        nuisance_purge_hours=13,
        nuisance_model_n_estimators=35,
        nuisance_model_min_child_samples=20,
        nuisance_min_rows=150,
        nuisance_min_class_rows=20,
        nuisance_min_positive_rows=30,
        nuisance_train_cap_rows_per_side=50_000,
        minimum_rank_groups=30,
        model_n_estimators=45,
        model_min_child_samples=10,
        model_train_cap_rows_per_side=50_000,
        minimum_regression_rows=100,
    )


def test_hurdle_expected_density_is_nonzero_under_zero_inflation() -> None:
    result = attach_past_only_nuisance_predictions(_frame(), _cfg())
    sf = result.frame.loc[result.frame["r02_3_1_source_eligible"]]
    assert (sf["raw_release_density"] == 0).mean() > 0.10
    assert (sf["nuisance_expected_density"] > 0).mean() > 0.99
    assert sf["excess_liquidity_residual"].notna().all()


def test_train_nuisance_predictions_are_expanding_oos_and_future_is_train_frozen() -> None:
    result = attach_past_only_nuisance_predictions(_frame(), _cfg())
    frame = result.frame
    train_eligible = frame["period"].eq("TRAIN_2023_2024") & frame["r02_3_1_source_eligible"]
    future_eligible = frame["period"].isin(["VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"]) & frame["r02_3_1_source_eligible"]
    assert frame.loc[train_eligible, "nuisance_prediction_source"].eq("TRAIN_EXPANDING_OOS").all()
    assert frame.loc[future_eligible, "nuisance_prediction_source"].eq("TRAIN_FULL_FROZEN").all()
    assert result.fold_audit["causal_fit_before_prediction"].all()
    expanding = result.fold_audit[result.fold_audit["prediction_source"].eq("TRAIN_EXPANDING_OOS")]
    assert (pd.to_datetime(expanding["fit_end_exclusive"]) < pd.to_datetime(expanding["block_start"])).all()


def test_nuisance_features_are_distance_plus_group_level_activity_only() -> None:
    result = attach_past_only_nuisance_predictions(_frame(), _cfg())
    cols = nuisance_feature_columns(result.frame)
    assert "zone_distance_bp" in cols
    assert "macro_notional_intensity_60m" in cols
    assert "zone_boundary_gap_60m_bp" not in cols
    assert "swing_count_25bp" not in cols
    assert result.feature_audit["status"].eq("PASS").all()


def test_primary_residual_features_exclude_distance_nuisance_and_swing() -> None:
    result = attach_past_only_nuisance_predictions(_frame(), _cfg())
    data = attach_ranking_targets(result.frame, _cfg())
    cols = residual_feature_columns(data, include_swing=False)
    assert "zone_boundary_gap_60m_bp" in cols
    assert "zone_distance_bp" not in cols
    assert "macro_notional_intensity_60m" not in cols
    assert not any(c.startswith("swing_") for c in cols)
    assert not any(c.startswith("nuisance_") for c in cols)
    assert not any(c.startswith("ft_") for c in cols)
    assert "r02_touch_consistent" not in cols
    assert "r02_3_source_eligible" not in cols
    assert "split_purge_eligible" not in cols
    geom = geometry_feature_columns(data)
    assert "r02_touch_consistent" not in geom
    assert "r02_3_source_eligible" not in geom
    assert "split_purge_eligible" not in geom
    assert not any(c.startswith("nuisance_") for c in geom)
    assert not any(c.startswith("r02_3_1_") for c in geom)


def test_residualization_reduces_mechanical_distance_bias_out_of_sample() -> None:
    cfg = _cfg()
    result = attach_past_only_nuisance_predictions(_frame(groups_per_month=6), cfg)
    stability = residual_stability(result.frame, cfg)
    future = stability[stability["period"].isin([cfg.calibration_period, cfg.holdout_period])]
    assert not future.empty
    # Hurdle expected density should remove a material share of raw distance bias.
    assert (future["distance_vs_excess_residual_spearman"].abs() < future["distance_vs_raw_density_spearman"].abs()).mean() >= 0.75


def test_residual_ranker_learns_hidden_zone_structure_beyond_nuisance() -> None:
    cfg = _cfg()
    result = attach_past_only_nuisance_predictions(_frame(groups_per_month=6), cfg)
    data = attach_ranking_targets(result.frame, cfg)
    models = fit_models(data, cfg)
    pred = predict(data, models)
    metrics = ranking_metrics(pred, cfg)
    hold = metrics[
        metrics["period"].eq(cfg.holdout_period)
        & metrics["task"].eq("EXCESS_RESIDUAL")
        & metrics["model"].eq("PATH_NO_SWING")
    ]
    assert len(hold) == 2
    assert hold["mean_group_spearman"].mean() > 0.08
    imp = feature_importance(models)
    assert "zone_boundary_gap_60m_bp" in set(imp["feature"])


def test_causal_audit_passes_expanding_oos_and_strict_feature_boundaries() -> None:
    cfg = _cfg()
    result = attach_past_only_nuisance_predictions(_frame(groups_per_month=6), cfg)
    data = attach_ranking_targets(result.frame, cfg)
    models = fit_models(data, cfg)
    pred = predict(data, models)
    gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    audit = causal_audit(pred, models, gate, result.feature_audit, result.fold_audit, cfg)
    assert audit["status"].eq("PASS").all(), audit.to_string(index=False)


def test_pipeline_synthetic_end_to_end_writes_reviewable_report(tmp_path, monkeypatch) -> None:
    from src.ai_research.latent_liquidity_hurdle_residualization import pipeline

    source = _frame(groups_per_month=5)
    gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    monkeypatch.setattr(pipeline, "load_source", lambda: (source.copy(), gate.copy()))
    cfg = replace(
        _cfg(),
        report_dir=str(tmp_path / "eth_latent_liquidity_path_v1" / "report"),
        cache_dir=str(tmp_path / "eth_latent_liquidity_path_v1" / "cache"),
    )
    result = pipeline.run_hurdle_residualization(
        config=cfg,
        use_cache=False,
        skip_review_pack=True,
        progress=False,
    )
    assert result.rows == len(source)
    report = tmp_path / "eth_latent_liquidity_path_v1" / "report"
    assert (report / "07_residualization_stability.csv").exists()
    assert (report / "14_causal_audit.csv").exists()
    assert (report / "16_decision.md").exists()
