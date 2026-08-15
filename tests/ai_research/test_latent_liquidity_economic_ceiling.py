from dataclasses import replace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_economic_ceiling.audit import (
    attach_oracle_metrics,
    ceiling_distribution,
    decision_from_episode_metrics,
    fixed_r_performance,
)
from src.ai_research.latent_liquidity_economic_ceiling.config import DEFAULT_CONFIG


def _episodes(n_per_period: int = 6, favorable_bp: float = 60.0, adverse_bp: float = 10.0) -> pd.DataFrame:
    rows = []
    periods = ["VALIDATION_2025Q1_Q3", "HOLDOUT_2025Q4_2026H1"]
    for p_i, period in enumerate(periods):
        for i in range(n_per_period):
            row = {
                "event_id": f"{period}-{i}",
                "release_episode_id": f"ep-{period}-{i}",
                "event_time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=p_i * 300 + i),
                "event_side": "DOWN" if i % 2 == 0 else "UP",
                "period": period,
                "release_episode_size": 1,
                "event_reference_price": 2000.0,
                "future_extension_bp": adverse_bp,
                "future_time_to_extreme_seconds": 10,
                "future_reversal_after_extreme_bp": favorable_bp,
                "future_acceptance_fraction_60s": 0.1,
                "outcome_type": "EXTEND_STABILIZE_REVERSAL",
                "favorable_reversal": True,
                "path_cluster": 10,
                "cluster_distance": 0.5,
            }
            for horizon in DEFAULT_CONFIG.horizons_seconds:
                row[f"future_same_direction_extension_{horizon}s_bp"] = adverse_bp
                row[f"future_opposite_excursion_{horizon}s_bp"] = favorable_bp
                row[f"future_close_return_{horizon}s_bp"] = 20.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_attach_oracle_metrics_uses_future_mae_plus_buffer_and_fixed_r_target():
    frame = attach_oracle_metrics(_episodes(1, favorable_bp=40.0, adverse_bp=10.0), DEFAULT_CONFIG)
    row = frame.iloc[0]
    assert row["oracle_risk_bp_300s"] == 13.0
    assert row["oracle_target_bp_300s_r1p5"] == 19.5
    assert bool(row["oracle_target_hit_300s_r1p5"])
    assert row["oracle_net_bp_300s_r1p5_c11"] == 8.5
    assert row["oracle_net_mfe_bp_300s_c11"] == 29.0


def test_fixed_r_uses_terminal_close_when_target_not_hit():
    episodes = _episodes(1, favorable_bp=5.0, adverse_bp=10.0)
    for horizon in DEFAULT_CONFIG.horizons_seconds:
        episodes[f"future_close_return_{horizon}s_bp"] = -4.0
    frame = attach_oracle_metrics(episodes, DEFAULT_CONFIG)
    assert not bool(frame.iloc[0]["oracle_target_hit_300s_r1p5"])
    assert frame.iloc[0]["oracle_net_bp_300s_r1p5_c11"] == -15.0


def test_ceiling_distribution_and_performance_include_frozen_universes():
    frame = attach_oracle_metrics(_episodes(), DEFAULT_CONFIG)
    distribution = ceiling_distribution(frame, DEFAULT_CONFIG)
    performance = fixed_r_performance(frame, DEFAULT_CONFIG)
    assert {"ALL_RELEASE_EPISODES", "FAVORABLE_REVERSAL_ORACLE", "FROZEN_R01_REVERSAL_CLUSTERS"}.issubset(set(distribution["universe"]))
    assert not performance.loc[performance["cost_bp"].eq(11.0)].empty


def test_decision_passes_only_when_oracle_economics_are_thick_in_validation_and_holdout():
    cfg = replace(
        DEFAULT_CONFIG,
        minimum_oracle_episodes_per_period=10,
        gate_min_base_mean_net_bp=1.0,
        gate_min_base_profit_factor=1.01,
        gate_min_stress_mean_net_bp=0.0,
        gate_min_stress_profit_factor=1.0,
        gate_min_base_positive_mfe_rate=0.5,
    )
    strong = attach_oracle_metrics(_episodes(20, favorable_bp=60.0, adverse_bp=10.0), cfg)
    decision, gate = decision_from_episode_metrics(strong, cfg)
    assert decision.startswith("CONTINUE_")
    assert gate["status"].eq("PASS").all()

    weak = attach_oracle_metrics(_episodes(20, favorable_bp=8.0, adverse_bp=25.0), cfg)
    decision, gate = decision_from_episode_metrics(weak, cfg)
    assert decision.startswith("STOP_")
    assert gate["status"].eq("FAIL").any()


def test_oracle_metrics_do_not_use_model_scores_or_swing_columns():
    frame = attach_oracle_metrics(_episodes(), DEFAULT_CONFIG)
    forbidden = [c for c in frame.columns if c.startswith("pred_") or c.startswith("p_") or "swing" in c.lower()]
    assert forbidden == []
    assert np.isfinite(frame["oracle_gross_reward_risk_300s"]).all()
