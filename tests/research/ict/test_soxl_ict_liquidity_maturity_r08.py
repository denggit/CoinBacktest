from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.liquidity_maturity import (
    LiquidityMaturityConfig,
    attach_liquidity_maturity_features,
    build_maturity_causal_audit,
    build_maturity_feature_atlas,
)
from src.research_common.ict.premarket_mss_fvg import make_synthetic_ict_day
from src.research_common.ict.premarket_mss_fvg_v2 import SweepEpisodeConfig, build_all_premarket_levels_v2, build_sweep_events_v2
from src.research_common.ict.premarket_mss_fvg_v4 import ICTDisplacementDiscoveryConfig, build_signal_attempts_v4
from src.research_common.ict.semantic_gap import attach_causal_semantic_features


def _attempts_and_bars():
    bars = make_synthetic_ict_day()
    day = pd.Timestamp("2026-06-02").date()
    levels = build_all_premarket_levels_v2(bars, [day], pivot_left=2, pivot_right=2, episode_config=SweepEpisodeConfig())
    sweeps = build_sweep_events_v2(bars, levels)
    attempts, _ = build_signal_attempts_v4(bars, sweeps, config=ICTDisplacementDiscoveryConfig(execution_timeframes=(1, 5)))
    attempts = attach_causal_semantic_features(attempts, bars)
    return bars, attempts


def test_r08_maturity_features_do_not_filter_candidates_and_final_reclaim_non_negative() -> None:
    bars, attempts = _attempts_and_bars()
    assert not attempts.empty
    ids = attempts["attempt_id"].tolist()
    enriched = attach_liquidity_maturity_features(attempts, bars)
    assert enriched["attempt_id"].tolist() == ids
    assert len(enriched) == len(attempts)
    vals = pd.to_numeric(enriched["maturity_first_reclaim_after_final_terminal_minutes"], errors="coerce").dropna()
    assert (vals >= 0).all()
    audit = build_maturity_causal_audit(enriched)
    assert bool(audit["passed"].all())


def test_r08_discovery_edges_ignore_2025_2026_extremes() -> None:
    rows = []
    for period, dates, values in [
        ("disc", pd.date_range("2023-07-01", periods=60, freq="D"), np.linspace(1, 5, 60)),
        ("fwd", pd.date_range("2025-01-01", periods=20, freq="D"), np.linspace(100, 200, 20)),
        ("hold", pd.date_range("2026-01-01", periods=20, freq="D"), np.linspace(300, 400, 20)),
    ]:
        for i, (d, value) in enumerate(zip(dates, values)):
            rows.append({
                "ny_date": str(d.date()), "execution_tf": "1m", "liquidity_family": "premarket_extreme",
                "filled": True, "net_return": 0.001 if i % 2 == 0 else -0.001,
                "net_r": 0.5 if i % 2 == 0 else -0.5, "mfe_r": 1.0, "mae_r": -0.5,
                "outcome_reached_0_5r": True, "outcome_reached_1r": True,
                "outcome_reached_2r": False, "outcome_reached_3r": False,
                "outcome_target_hit": False, "outcome_stop_hit": False,
                "outcome_immediate_failure_15m": False,
                "outcome_favorable_then_failed_1r": False,
                "outcome_favorable_then_failed_2r": False,
                "maturity_initial_sweep_depth_bp": float(value),
            })
    df = pd.DataFrame(rows)
    edges, perf = build_maturity_feature_atlas(
        df,
        config=LiquidityMaturityConfig(min_discovery_samples=40, quantile_bins=5),
        features=("maturity_initial_sweep_depth_bp",),
    )
    assert not edges.empty
    finite_edges = pd.to_numeric(edges.filter(like="edge_").stack(), errors="coerce").dropna()
    assert finite_edges.max() <= 5.0
    assert {"2025_forward", "2026_late_holdout"}.issubset(set(perf["analysis_period"]))
