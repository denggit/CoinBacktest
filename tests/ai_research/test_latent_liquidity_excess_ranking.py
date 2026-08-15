from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_excess_ranking.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_excess_ranking.labels import (
    DistanceNormalizer,
    attach_excess_and_reversal_targets,
    fit_distance_normalizer,
)
from src.ai_research.latent_liquidity_excess_ranking.modeling import (
    feature_columns,
    fit_models,
    predict,
    ranking_metrics,
    regression_metrics,
    top_zone_summary,
)
from src.ai_research.latent_liquidity_excess_ranking.reports import causal_audit


def _base_frame(groups_per_period: int = 8, zones: int = 5) -> pd.DataFrame:
    rows = []
    periods = ["TRAIN_2023_2024", "VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"]
    for period_i, period in enumerate(periods):
        for side in ("DOWN", "UP"):
            for g in range(groups_per_period):
                dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=period_i * 100 + g)
                for j in range(zones):
                    distance = float(10 + 20 * j)
                    # Raw density has a strong mechanical near-distance bias.
                    mechanical = 8.0 / (1.0 + j)
                    hidden = float(j) + 0.05 * g
                    density = mechanical * np.exp(0.35 * hidden)
                    fav = density * (0.75 if j >= 3 else 0.15)
                    cont = density * (0.10 if j >= 3 else 0.55)
                    rows.append({
                        "zone_id": f"{period}-{side}-{g}-{j}",
                        "decision_time": dt,
                        "feature_available_time": dt,
                        "period": period,
                        "zone_side": side,
                        "zone_distance_bp": distance,
                        "side_is_down": int(side == "DOWN"),
                        "zone_boundary_gap_60m_bp": hidden,
                        "zone_buildup_log_notional_60m": hidden * 0.5,
                        "macro_notional_intensity_60m": 1.0 + 0.1 * hidden,
                        "swing_count_25bp": float(4 - j),
                        "first_touch_observed": True,
                        "first_touch_label_complete": True,
                        "first_touch_time": dt + pd.Timedelta(seconds=1 + j),
                        "first_touch_available_time": dt + pd.Timedelta(seconds=2 + j),
                        "touch_720m": True,
                        "r02_touch_consistent": True,
                        "r02_3_source_eligible": True,
                        "ft_release_density_sum_180s": density,
                        "ft_release_episode_count_180s": 1.0,
                        "ft_favorable_density_sum_180s": fav,
                        "ft_continuation_density_sum_180s": cont,
                        "ft_favorable_episode_count_180s": float(fav > 0),
                        "ft_continuation_episode_count_180s": float(cont > 0),
                        "ft_sweep_depth_weighted_bp_180s": 10.0 + 5.0 * hidden,
                        "ft_reversal_room_weighted_bp_180s": 20.0 + 8.0 * hidden,
                    })
    return pd.DataFrame(rows)


def test_train_distance_normalizer_uses_robust_distance_baseline() -> None:
    frame = _base_frame(groups_per_period=20)
    cfg = replace(DEFAULT_CONFIG, normalizer_min_rows_per_distance=5)
    normalizer = fit_distance_normalizer(frame, cfg)
    out = attach_excess_and_reversal_targets(frame, normalizer, cfg)
    train = out.loc[out["period"].eq(cfg.train_period)]
    # Each distance's train median is approximately neutral after normalization.
    med = train.groupby(["zone_side", "zone_distance_bp"])["excess_log_density"].median().abs().max()
    assert float(med) < 1e-9
    assert train["excess_liquidity_z"].notna().all()


def test_source_mismatch_is_quarantined_from_targets() -> None:
    frame = _base_frame(groups_per_period=5)
    frame.loc[0, "r02_touch_consistent"] = False
    frame.loc[0, "r02_3_source_eligible"] = False
    cfg = replace(DEFAULT_CONFIG, normalizer_min_rows_per_distance=3)
    norm = fit_distance_normalizer(frame, cfg)
    out = attach_excess_and_reversal_targets(frame, norm, cfg)
    assert not bool(out.loc[0, "excess_group_eligible"])


def test_primary_feature_columns_exclude_swing_and_all_new_future_targets() -> None:
    frame = _base_frame()
    cfg = replace(DEFAULT_CONFIG, normalizer_min_rows_per_distance=3)
    frame = attach_excess_and_reversal_targets(frame, fit_distance_normalizer(frame, cfg), cfg)
    cols = feature_columns(frame, include_swing=False)
    assert "zone_boundary_gap_60m_bp" in cols
    assert "zone_distance_bp" not in cols
    assert not any(c.startswith("swing_") for c in cols)
    assert not any(c.startswith("excess_") for c in cols)
    assert not any(c.startswith("reversal_quality") for c in cols)
    assert not any(c.startswith("ft_") for c in cols)


def test_rankers_learn_excess_and_reversal_order() -> None:
    frame = _base_frame(groups_per_period=8)
    cfg = replace(
        DEFAULT_CONFIG,
        normalizer_min_rows_per_distance=3,
        minimum_rank_groups=3,
        model_train_cap_rows_per_side=10_000,
        model_n_estimators=40,
        model_min_child_samples=2,
        minimum_regression_rows=20,
    )
    norm = fit_distance_normalizer(frame, cfg)
    data = attach_excess_and_reversal_targets(frame, norm, cfg)
    models = fit_models(data, cfg)
    pred = predict(data, models)
    metrics = ranking_metrics(pred, cfg)
    hold_ex = metrics.loc[
        metrics["period"].eq(cfg.holdout_period)
        & metrics["zone_side"].eq("DOWN")
        & metrics["task"].eq("EXCESS_LIQUIDITY")
        & metrics["model"].eq("PATH_NO_SWING")
    ]
    hold_rv = metrics.loc[
        metrics["period"].eq(cfg.holdout_period)
        & metrics["zone_side"].eq("DOWN")
        & metrics["task"].eq("REVERSAL_QUALITY")
        & metrics["model"].eq("PATH_NO_SWING")
    ]
    assert float(hold_ex.iloc[0]["mean_group_spearman"]) > 0.5
    assert float(hold_rv.iloc[0]["mean_group_spearman"]) > 0.5
    reg = regression_metrics(pred, cfg)
    assert reg.loc[reg["task"].eq("SWEEP_DEPTH"), "spearman"].dropna().min() > 0.5
    top = top_zone_summary(pred, cfg)
    assert top["top1_touched"].gt(0).all()


def test_causal_audit_uses_first_touch_available_time_and_quarantine() -> None:
    frame = _base_frame(groups_per_period=8, zones=25)
    # Bar starts at decision time but becomes available one second later: causal.
    idx = frame.index[0]
    frame.loc[idx, "first_touch_time"] = frame.loc[idx, "decision_time"]
    frame.loc[idx, "first_touch_available_time"] = frame.loc[idx, "decision_time"] + pd.Timedelta(seconds=1)
    cfg = replace(
        DEFAULT_CONFIG,
        normalizer_min_rows_per_distance=3,
        minimum_rank_groups=3,
        model_train_cap_rows_per_side=100_000,
        model_n_estimators=10,
        model_min_child_samples=2,
        minimum_regression_rows=20,
    )
    norm = fit_distance_normalizer(frame, cfg)
    data = attach_excess_and_reversal_targets(frame, norm, cfg)
    models = fit_models(data, cfg)
    gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    audit = causal_audit(data, models, gate, norm, cfg)
    assert audit["status"].eq("PASS").all()


def test_true_noncausal_touch_available_time_fails_audit() -> None:
    frame = _base_frame(groups_per_period=8, zones=25)
    idx = frame.index[0]
    frame.loc[idx, "first_touch_available_time"] = frame.loc[idx, "decision_time"]
    # Keep it marked eligible deliberately so the audit must catch it.
    frame.loc[idx, "r02_3_source_eligible"] = True
    cfg = replace(
        DEFAULT_CONFIG,
        normalizer_min_rows_per_distance=3,
        minimum_rank_groups=3,
        model_train_cap_rows_per_side=100_000,
        model_n_estimators=10,
        model_min_child_samples=2,
        minimum_regression_rows=20,
    )
    norm = fit_distance_normalizer(frame, cfg)
    data = attach_excess_and_reversal_targets(frame, norm, cfg)
    models = fit_models(data, cfg)
    gate = pd.DataFrame({"check": ["source"], "status": ["PASS"]})
    audit = causal_audit(data, models, gate, norm, cfg)
    row = audit.loc[audit["check"].eq("first_touch_available_strictly_after_decision")].iloc[0]
    assert row["status"] == "FAIL"
