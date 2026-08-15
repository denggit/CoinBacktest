from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_target_consistency.audit import (
    causal_audit,
    distance_cell_audit,
    target_consistency_stability,
)
from src.ai_research.latent_liquidity_target_consistency.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_target_consistency.nuisance import (
    attach_past_only_target_consistency_predictions,
    construct_target_consistency_columns,
    nuisance_feature_columns,
    nuisance_metric_table,
)
from src.ai_research.latent_liquidity_target_consistency.reports import _decision, write_reports


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
                        base_logit = 1.2 - 0.0100 * distance + 0.9 * activity
                        p_release = 1.0 / (1.0 + np.exp(-base_logit))
                        u = (np.sin((seq + j * 17 + (1 if side == "UP" else 0)) * 12.9898) + 1.0) / 2.0
                        released = bool(u < p_release)
                        log_mag = max(0.02, 0.55 - 0.0004 * distance + 0.35 * activity + 0.20 * hidden)
                        density = float(np.expm1(log_mag)) if released else 0.0
                        rows.append({
                            "zone_id": f"{period}-{side}-{dt}-{j}",
                            "decision_time": dt,
                            "feature_available_time": dt,
                            "period": period,
                            "zone_side": side,
                            "zone_distance_bp": distance,
                            "zone_boundary_gap_60m_bp": hidden * 3.0,
                            "zone_buildup_log_notional_60m": 1.0 + 0.4 * hidden,
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
                            "r02_3_1b_upstream_eligible": True,
                            "ft_release_density_sum_180s": density,
                            "ft_release_episode_count_180s": float(released),
                        })
    return pd.DataFrame(rows)


def _cfg():
    return replace(
        DEFAULT_CONFIG,
        nuisance_initial_train_months=3,
        nuisance_forward_block_months=2,
        nuisance_purge_hours=13,
        nuisance_model_n_estimators=30,
        nuisance_model_min_child_samples=20,
        nuisance_min_rows=150,
        nuisance_min_class_rows=20,
        nuisance_min_positive_rows=30,
        nuisance_train_cap_rows_per_side=50_000,
    )


def test_same_scale_hurdle_expectation_is_not_log_of_raw_expectation() -> None:
    frame = pd.DataFrame({"raw_log_release_density": [0.0, np.log1p(3.0)]})
    scored = construct_target_consistency_columns(
        frame,
        p_release=np.array([0.25, 0.25]),
        huber_log_if_release=np.array([np.log1p(3.0), np.log1p(3.0)]),
        mean_log_if_release=np.array([np.log1p(3.0), np.log1p(3.0)]),
        huber_smearing_factor=1.0,
    )
    expected_log = 0.25 * np.log1p(3.0)
    assert np.allclose(scored["mean_aligned_expected_log_density"], expected_log)
    assert not np.allclose(scored["legacy_expected_log_proxy"], expected_log)


def test_nuisance_schema_excludes_zone_path_and_swing() -> None:
    cols = nuisance_feature_columns(_frame(groups_per_month=1))
    assert "zone_distance_bp" in cols
    assert "macro_notional_intensity_60m" in cols
    assert "zone_boundary_gap_60m_bp" not in cols
    assert "zone_buildup_log_notional_60m" not in cols
    assert "swing_count_25bp" not in cols


def test_predictions_are_expanding_oos_and_future_train_frozen_with_l2_primary() -> None:
    result = attach_past_only_target_consistency_predictions(_frame(), _cfg())
    frame = result.frame
    train = frame["period"].eq("TRAIN_2023_2024") & frame["r02_3_1b_source_eligible"]
    future = frame["period"].isin(["VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"]) & frame["r02_3_1b_source_eligible"]
    assert frame.loc[train, "nuisance_prediction_source"].eq("TRAIN_EXPANDING_OOS").all()
    assert frame.loc[future, "nuisance_prediction_source"].eq("TRAIN_FULL_FROZEN").all()
    assert result.fold_audit["causal_fit_before_prediction"].all()
    assert set(result.fold_audit["positive_log_primary_objective"].astype(str)) == {"regression_l2"}


def test_primary_expected_log_identity_is_exact() -> None:
    result = attach_past_only_target_consistency_predictions(_frame(), _cfg())
    sf = result.frame.loc[result.frame["r02_3_1b_source_eligible"]]
    lhs = sf["mean_aligned_expected_log_density"].to_numpy(dtype=float)
    rhs = (
        sf["nuisance_p_release"].to_numpy(dtype=float)
        * sf["nuisance_mean_log_density_if_release"].to_numpy(dtype=float)
    )
    assert np.allclose(lhs, rhs, rtol=0.0, atol=1e-12)
    assert sf["target_consistent_excess_residual"].notna().all()


def test_distance_cell_audit_preserves_all_frozen_distance_cells() -> None:
    result = attach_past_only_target_consistency_predictions(_frame(), _cfg())
    cells = distance_cell_audit(result.frame)
    for period in ("VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"):
        for side in ("DOWN", "UP"):
            sf = cells.loc[cells["period"].eq(period) & cells["zone_side"].eq(side)]
            assert sf["zone_distance_bp"].nunique() == 25
            assert sf["target_consistent_residual_mean"].notna().all()


def test_causal_audit_passes_on_clean_synthetic_frame() -> None:
    result = attach_past_only_target_consistency_predictions(_frame(), _cfg())
    source_gate = pd.DataFrame([{"check": "synthetic", "status": "PASS"}])
    causal = causal_audit(result.frame, source_gate, result.feature_audit, result.fold_audit, _cfg())
    assert causal["status"].eq("PASS").all(), causal.to_dict(orient="records")


def test_decision_blocks_when_corrected_residual_still_distance_contaminated() -> None:
    cfg = _cfg()
    rows = []
    for period in (cfg.calibration_period, cfg.holdout_period):
        for side in ("DOWN", "UP"):
            rows.append({
                "period": period,
                "zone_side": side,
                "distance_vs_raw_log_density_spearman": -0.17,
                "distance_vs_legacy_residual_spearman": -0.16,
                "distance_vs_formula_only_residual_spearman": -0.14,
                "distance_vs_target_consistent_residual_spearman": -0.13,
            })
    causal = pd.DataFrame([{"check": "ok", "status": "PASS"}])
    metrics = pd.DataFrame()
    decision, _ = _decision(pd.DataFrame(rows), metrics, causal, cfg)
    assert decision == "BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED"


def test_stability_reports_all_three_residual_definitions() -> None:
    result = attach_past_only_target_consistency_predictions(_frame(), _cfg())
    stability = target_consistency_stability(result.frame)
    assert {
        "distance_vs_legacy_residual_spearman",
        "distance_vs_formula_only_residual_spearman",
        "distance_vs_target_consistent_residual_spearman",
    }.issubset(stability.columns)


def test_report_writer_smoke_is_audit_only(tmp_path) -> None:
    cfg = replace(_cfg(), report_dir=str(tmp_path / "eth_latent_liquidity_path_v1" / "report"), cache_dir=str(tmp_path / "eth_latent_liquidity_path_v1" / "cache"))
    result = attach_past_only_target_consistency_predictions(_frame(), cfg)
    source_gate = pd.DataFrame([{"check": "synthetic", "status": "PASS"}])
    metrics = nuisance_metric_table(result.frame, cfg)
    causal = causal_audit(result.frame, source_gate, result.feature_audit, result.fold_audit, cfg)
    report_dir, _ = write_reports(
        config=cfg,
        source_gate=source_gate,
        frame=result.frame,
        nuisance_feature_audit=result.feature_audit,
        nuisance_fold_audit=result.fold_audit,
        nuisance_metrics=metrics,
        causal=causal,
        skip_review_pack=True,
    )
    assert (report_dir / "00_manifest.json").exists()
    assert (report_dir / "07_target_consistency_stability.csv").exists()
    assert (report_dir / "12_decision.md").exists()
    summary = pd.read_csv(report_dir / "06_dataset_summary.csv")
    assert not bool(summary.loc[0, "path_ranker_trained"])
    assert not bool(summary.loc[0, "new_data_family_added"])
